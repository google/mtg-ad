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

"""Text-to-speech rendering of AD scripts (Gemini TTS or Google Cloud TTS).

Each AD line is synthesized to its own WAV file, named by a content hash of the
description text:
  {output_dir}/{video_id}_tts_{md5(text)}.wav
The same text always maps to the same file, so WAVs are reused across runs and
the evaluator can match WAVs to AD lines without relying on file order. This is
also the naming scheme expected for pre-synthesized WAV submissions.

A per-video summary is written alongside the WAVs:
  {output_dir}/{video_id}_tts_metadata.json
as a list of {timestamp, start, text, hashcode, duration_tts, end_tts,
wav_path} dicts, which evaluation uses for TTS-based temporal metrics.
"""

import hashlib
import io
import json
import os
import re
import time
from typing import Any
import wave

from common import ad_format
from common import usage
from google import genai
from google.genai import types

try:
  from google.cloud import texttospeech  # pylint: disable=g-import-not-at-top
except ImportError:
  texttospeech = None

# Default backend: Gemini TTS, which only needs the Gemini API key.
TTS_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_VOICE = "Kore"
# Optional backend (--cloud_tts): Google Cloud Text-to-Speech at a fixed 1.2x
# speaking rate, the setting used for the paper's temporal metrics. 1.2x is
# roughly 180 WPM (average human speech is ~130-150 WPM). Requires the optional
# google-cloud-texttospeech package and Application Default Credentials.
CLOUD_TTS_SPEAKING_RATE = 1.2
CLOUD_TTS_VOICE = "en-US-Neural2-C"


class TtsError(RuntimeError):
  """Raised when TTS output is missing or incomplete."""


def require_cloud_tts() -> None:
  """Raises TtsError if the optional Cloud TTS package is not installed."""
  if texttospeech is None:
    raise TtsError(
        "--cloud_tts needs the optional google-cloud-texttospeech package:"
        " pip install google-cloud-texttospeech (and set up Application"
        " Default Credentials: gcloud auth application-default login)."
    )


def parse_ad_file(ad_text: str) -> list[tuple[str, str]]:
  """Parses AD text (canonical lines or raw model output) into segments.

  Uses the same parser as evaluation (ad_format.parse_ad_lines), so TTS
  segments line up with the AD segments that are scored.

  Args:
    ad_text: Contents of an output_{VIDEO_ID}.txt AD file.

  Returns:
    List of (canonical H:MM:SS.mmm timestamp, description) tuples, in order.
  """
  return [
      (ad_format.format_timestamp(start), text)
      for start, text in ad_format.parse_ad_lines(ad_text)
  ]


def timestamp_to_seconds(timestamp_str: str) -> float | None:
  """Converts an AD timestamp (e.g. 'H:MM:SS.mmm') to seconds."""
  return ad_format.timestamp_to_seconds(timestamp_str)


def text_hash(text: str) -> str:
  """Returns the MD5 hex digest used to name a description's WAV file."""
  return hashlib.md5(text.encode()).hexdigest()


def wav_filename(video_id: str, text: str) -> str:
  """Returns the canonical WAV filename (no directory) for an AD line."""
  return f"{video_id}_tts_{text_hash(text)}.wav"


def metadata_path(video_id: str, tts_dir: str) -> str:
  """Returns the path of a video's TTS metadata JSON."""
  return os.path.join(tts_dir, f"{video_id}_tts_metadata.json")


def load_metadata(video_id: str, tts_dir: str) -> list[dict[str, Any]] | None:
  """Returns a video's saved TTS metadata, or None if it doesn't exist.

  Raises:
    TtsError: If the metadata exists but some segment has no TTS duration
      (an incomplete TTS run); re-run generate_tts.py to fill it in.
  """
  path = metadata_path(video_id, tts_dir)
  if not os.path.exists(path):
    return None
  with open(path, "r", encoding="utf-8") as f:
    metadata = json.load(f)
  missing = [
      i
      for i, seg in enumerate(metadata)
      if seg.get("duration_tts") is None or seg.get("start") is None
  ]
  if missing:
    raise TtsError(
        f"TTS metadata {path} is incomplete: segments {missing} have no TTS"
        " duration. Re-run generate_tts.py for this video."
    )
  return metadata


def pcm_to_wav(
    pcm_data: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
    sampwidth: int = 2,
) -> bytes:
  """Wraps raw PCM bytes in a WAV container."""
  buf = io.BytesIO()
  with wave.open(buf, "wb") as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(sample_rate)
    wf.writeframes(pcm_data)
  return buf.getvalue()


def wav_duration_secs(wav_bytes: bytes) -> float:
  """Returns the duration in seconds of in-memory WAV bytes."""
  with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
    return wf.getnframes() / float(wf.getframerate())


def synthesize_text(
    client: genai.Client,
    text: str,
    voice_name: str = DEFAULT_VOICE,
    model: str = TTS_MODEL,
    retries: int = 3,
    delay: float = 2.0,
) -> bytes | None:
  """Synthesizes one text segment with Gemini TTS.

  Args:
    client: Gemini client.
    text: Text to synthesize.
    voice_name: Prebuilt Gemini voice name.
    model: Gemini TTS model name.
    retries: Number of attempts.
    delay: Seconds to wait between attempts.

  Returns:
    WAV bytes (16-bit, 24 kHz, mono), or None on failure.
  """
  for attempt in range(retries):
    try:
      response = client.models.generate_content(
          model=model,
          contents=[{"parts": [{"text": text}]}],
          config=types.GenerateContentConfig(
              response_modalities=[types.Modality.AUDIO],
              speech_config=types.SpeechConfig(
                  voice_config=types.VoiceConfig(
                      prebuilt_voice_config=types.PrebuiltVoiceConfig(
                          voice_name=voice_name,
                      )
                  )
              ),
          ),
      )
      usage.record_response(model, response)
      for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("audio/"):
          return pcm_to_wav(part.inline_data.data)
      print(f"    TTS attempt {attempt + 1}/{retries}: no audio in response.")
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"    TTS attempt {attempt + 1}/{retries} failed: {str(e)[:300]}")
    if attempt < retries - 1:
      time.sleep(delay * (2**attempt))
  return None


def synthesize_text_cloud_tts(
    text: str,
    language_code: str = "en-US",
    voice_name: str = CLOUD_TTS_VOICE,
    speaking_rate: float = CLOUD_TTS_SPEAKING_RATE,
    retries: int = 3,
    delay: float = 2.0,
) -> bytes | None:
  """Synthesizes one text segment with Google Cloud Text-to-Speech.

  Runs at a fixed speaking rate (1.2x by default, as in the paper) so results
  are comparable across systems. Requires the optional
  google-cloud-texttospeech package and Application Default Credentials or
  GOOGLE_APPLICATION_CREDENTIALS.

  Args:
    text: Text to synthesize.
    language_code: BCP-47 language code.
    voice_name: Cloud TTS voice name.
    speaking_rate: Speaking-rate multiplier.
    retries: Number of attempts.
    delay: Seconds to wait between attempts.

  Returns:
    WAV bytes (LINEAR16), or None on failure.
  """
  require_cloud_tts()

  for attempt in range(retries):
    try:
      response = texttospeech.TextToSpeechClient().synthesize_speech(
          input=texttospeech.SynthesisInput(text=text),
          voice=texttospeech.VoiceSelectionParams(
              language_code=language_code, name=voice_name
          ),
          audio_config=texttospeech.AudioConfig(
              audio_encoding=texttospeech.AudioEncoding.LINEAR16,
              speaking_rate=speaking_rate,
          ),
      )
      return response.audio_content
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"    Cloud TTS attempt {attempt + 1}/{retries} failed: {e}")
      if attempt < retries - 1:
        time.sleep(delay)
  return None


def synthesize_ad_file(
    client: genai.Client | None,
    video_id: str,
    ad_text: str,
    output_dir: str,
    voice_name: str = DEFAULT_VOICE,
    use_cloud_tts: bool = False,
    tts_model: str = TTS_MODEL,
    speaking_rate: float = CLOUD_TTS_SPEAKING_RATE,
) -> list[dict[str, Any]]:
  """Synthesizes every AD line of a video and writes its TTS metadata JSON.

  WAVs that already exist in `output_dir` are reused (only their duration is
  measured), so re-runs only synthesize new or changed lines. Use a separate
  output_dir per TTS backend, since WAVs are keyed by the description text.

  Args:
    client: Gemini client. Not used when use_cloud_tts=True.
    video_id: YouTube video ID.
    ad_text: AD script text (canonical lines or raw model output).
    output_dir: Directory for WAVs and the metadata JSON.
    voice_name: Gemini TTS voice (ignored for Cloud TTS).
    use_cloud_tts: Use Google Cloud TTS at `speaking_rate` instead of Gemini.
    tts_model: Gemini TTS model (ignored for Cloud TTS).
    speaking_rate: Cloud TTS speaking rate (ignored for Gemini TTS).

  Returns:
    The list of per-segment metadata dicts.

  Raises:
    TtsError: If the AD has no parseable lines or any segment failed to
      synthesize. The metadata JSON is only written when every segment
      succeeded, so a re-run retries the failed segments.
  """
  segments = parse_ad_file(ad_text)
  if not segments:
    raise TtsError(f"No AD segments could be parsed for {video_id}.")
  if use_cloud_tts:
    require_cloud_tts()
    backend = f"cloud_tts:{CLOUD_TTS_VOICE}@{speaking_rate}x"
  else:
    backend = f"gemini_tts:{tts_model}:{voice_name}"

  os.makedirs(output_dir, exist_ok=True)
  segment_info = []
  failed = []
  for i, (timestamp, text) in enumerate(segments):
    wav_path = os.path.join(output_dir, wav_filename(video_id, text))
    start_secs = timestamp_to_seconds(timestamp)
    duration_secs = None

    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 44:
      with open(wav_path, "rb") as f:
        duration_secs = wav_duration_secs(f.read())
    else:
      if use_cloud_tts:
        wav_bytes = synthesize_text_cloud_tts(text, speaking_rate=speaking_rate)
      else:
        wav_bytes = synthesize_text(
            client, text, voice_name=voice_name, model=tts_model
        )
      if wav_bytes:
        duration_secs = wav_duration_secs(wav_bytes)
        with open(wav_path, "wb") as f:
          f.write(wav_bytes)
      else:
        failed.append(i)
        print(
            f"  ERROR: TTS failed for segment {i} of {video_id}: '{text[:50]}'"
        )

    segment_info.append({
        "timestamp": timestamp,
        "start": start_secs,
        "text": text,
        "hashcode": text_hash(text),
        "duration_tts": duration_secs,
        "end_tts": (
            start_secs + duration_secs
            if start_secs is not None and duration_secs is not None
            else None
        ),
        "wav_path": wav_path,
        "tts_backend": backend,
    })

  if failed:
    raise TtsError(
        f"TTS failed for {len(failed)}/{len(segments)} segments of {video_id}"
        f" (segments {failed}); metadata not written. Re-run to retry."
    )
  with open(metadata_path(video_id, output_dir), "w", encoding="utf-8") as f:
    json.dump(segment_info, f, indent=2)
  return segment_info
