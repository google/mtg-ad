# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemini API helpers: video references and retrying generate_content calls.

Error handling for generate_content calls:
  * Transient errors (HTTP 429, 5xx, network errors) are retried with
    exponential backoff.
  * Configuration errors that would make every call fail (unknown model name ->
    404 NOT_FOUND, invalid API key -> 400/401/403) raise FatalApiError right
    away, so a batch run stops instead of silently skipping every video.
  * Other client errors (e.g. 400 INVALID_ARGUMENT for an unavailable or
    private YouTube video) are not retried: an ERROR is printed and the call
    returns None, so batch scripts can record the failure and move on.
"""

import os
import time

from common import local_video
from common import usage
from google import genai
from google.genai import errors
from google.genai import types

# Default model for AD generation, transcripts and visual info.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
# Default model for LLM-judge evaluation metrics.
DEFAULT_JUDGE_MODEL = "gemini-3.8-flash"

_RETRYABLE_CLIENT_CODES = (408, 429)
# Uploaded videos are polled every 5 s until ACTIVE (up to 10 minutes).
_UPLOAD_POLL_ATTEMPTS = 120


class FatalApiError(RuntimeError):
  """A Gemini API error that will recur on every call (e.g. unknown model)."""


def _is_fatal_client_error(e: errors.ClientError) -> bool:
  """Returns True for client errors caused by the model name or API key."""
  message = str(e).lower()
  if e.code == 404 and "models/" in message:
    return True
  # 403 alone is not fatal: it can also mean a single video is inaccessible.
  return e.code == 401 or "api key" in message


def make_video_part(
    client: genai.Client,
    video_id: str,
    mp4_path: str | None = None,
) -> types.Part:
  """Returns a Gemini Part referencing a video.

  Args:
    client: Gemini client (used only to upload `mp4_path`).
    video_id: YouTube video ID; streamed by URL when `mp4_path` is not given.
    mp4_path: Optional local MP4 to upload to the Gemini Files API instead.

  Returns:
    A types.Part with file_data pointing at the video.

  Raises:
    FileNotFoundError: If `mp4_path` does not exist.
    RuntimeError: If the uploaded file fails or never becomes ACTIVE.
  """
  if not mp4_path:
    return types.Part(
        file_data=types.FileData(
            file_uri=f"https://www.youtube.com/watch?v={video_id}"
        )
    )
  if not os.path.isfile(mp4_path):
    raise FileNotFoundError(f"Video file not found: {mp4_path}")
  print(f"  Uploading {mp4_path} to Gemini Files API...")
  # Upload from an open file with an ASCII display name: file names with
  # non-ASCII characters (e.g. full-width '｜' in titles) can break the upload
  # request headers when a path is passed.
  with open(mp4_path, "rb") as fh:
    uploaded = client.files.upload(
        file=fh,
        config=types.UploadFileConfig(
            display_name=local_video.video_id_from_path(mp4_path),
            mime_type="video/mp4",
        ),
    )
  for i in range(_UPLOAD_POLL_ATTEMPTS):
    f = client.files.get(name=uploaded.name)
    if f.state.name == "ACTIVE":
      print(f"  Video file active: {f.name}")
      return types.Part(
          file_data=types.FileData(file_uri=f.uri, mime_type="video/mp4")
      )
    if f.state.name == "FAILED":
      raise RuntimeError(f"Gemini failed to process {mp4_path}: {f.error}")
    if (i + 1) % 6 == 0:
      print(f"  Still processing video ({(i + 1) * 5}s elapsed)...")
    time.sleep(5)
  raise RuntimeError(f"Uploaded file {uploaded.name} never became ACTIVE.")


def _generate_with_retries(
    client: genai.Client,
    model: str,
    contents: types.Content | str,
    retries: int,
    delay: float,
) -> str | None:
  """Calls generate_content with exponential backoff; returns text or None.

  Raises:
    FatalApiError: For errors that will recur on every call (unknown model
      name, invalid API key).
  """
  for attempt in range(retries):
    try:
      response = client.models.generate_content(model=model, contents=contents)
      # Billed even if the answer turns out empty, so record it first.
      usage.record_response(model, response)
      if response.text:
        return response.text
      finish_reason = _finish_reason(response)
      if finish_reason.endswith("STOP"):
        # A completed but empty answer (e.g. no speech to transcribe).
        return ""
      print(
          f"    Gemini attempt {attempt + 1}/{retries}: empty response"
          f" (finish_reason={finish_reason})."
      )
    except errors.ClientError as e:
      if _is_fatal_client_error(e):
        raise FatalApiError(
            f"Gemini API error {e.code} for model '{model}': {str(e)[:300]}\n"
            "This affects every request; check the model name (see"
            " client.models.list()) and the API key."
        ) from e
      if e.code not in _RETRYABLE_CLIENT_CODES:
        print(
            f"    ERROR: Gemini request rejected ({e.code}), not retrying:"
            f" {str(e)[:300]}"
        )
        return None
      print(
          f"    Gemini attempt {attempt + 1}/{retries} failed: {str(e)[:200]}"
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      # ServerError (5xx), timeouts and connection errors are transient.
      print(
          f"    Gemini attempt {attempt + 1}/{retries} failed: {str(e)[:200]}"
      )
    if attempt < retries - 1:
      time.sleep(delay * (2**attempt))
  print(f"    ERROR: Gemini call failed after {retries} attempts.")
  return None


def _finish_reason(response: types.GenerateContentResponse) -> str:
  """Returns the first candidate's finish reason, or 'unknown'."""
  try:
    return str(response.candidates[0].finish_reason)
  except (AttributeError, IndexError, TypeError):
    return "unknown"


def generate_with_video(
    client: genai.Client,
    model: str,
    video_part: types.Part,
    prompt: str,
    retries: int = 5,
    delay: float = 5.0,
) -> str | None:
  """Prompts Gemini with a video and text; returns the response text or None."""
  return _generate_with_retries(
      client,
      model,
      types.Content(parts=[video_part, types.Part(text=prompt)]),
      retries,
      delay,
  )


def generate_text(
    client: genai.Client,
    model: str,
    prompt: str,
    retries: int = 5,
    delay: float = 5.0,
) -> str | None:
  """Prompts Gemini with text only; returns the response text or None."""
  return _generate_with_retries(client, model, prompt, retries, delay)
