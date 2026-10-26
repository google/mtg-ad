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

"""File helpers: API keys, video ID lists, durations, cached text outputs."""

import json
import os
import sys

# Durations (M:SS) of all 433 dataset videos, keyed by YouTube video ID.
DEFAULT_VIDEO_DURATIONS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metadata",
    "video_durations.json",
)


def fix_dash_prefixed_values(
    argv: list[str], flags: tuple[str, ...] = ("--video_id",)
) -> list[str]:
  """Joins `--flag VALUE` into `--flag=VALUE` when VALUE starts with '-'.

  YouTube IDs may start with '-' (e.g. --T4zFh82Ao), which argparse would
  otherwise read as an option and fail with "expected one argument". Quoting
  does not help because the shell strips the quotes before Python sees them.

  Args:
    argv: Command-line arguments (without the program name).
    flags: Flags whose value may start with '-'.

  Returns:
    A new argument list.
  """
  fixed = []
  i = 0
  while i < len(argv):
    if argv[i] in flags and i + 1 < len(argv):
      fixed.append(f"{argv[i]}={argv[i + 1]}")
      i += 2
    else:
      fixed.append(argv[i])
      i += 1
  return fixed


def load_video_durations(path: str | None) -> dict[str, str]:
  """Loads a {video_id: duration} JSON map; returns {} if path is empty/missing.

  Durations may be seconds or M:SS / H:MM:SS strings.

  Args:
    path: Path to the JSON file.

  Returns:
    Dict mapping video ID to its duration string.
  """
  if not path:
    return {}
  if not os.path.exists(path):
    print(f"WARNING: video durations file not found: {path}")
    return {}
  with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
  return {str(k).strip(): str(v).strip() for k, v in data.items()}


def report_failures(failures: dict[str, str], total: int, step: str) -> None:
  """Prints a per-video failure summary and exits non-zero if any failed.

  Args:
    failures: Map of video ID to failure reason.
    total: Number of videos attempted.
    step: Name of the step, for the message.
  """
  print(
      f"\n=== {step}: {total - len(failures)}/{total} succeeded,"
      f" {len(failures)} failed ==="
  )
  if failures:
    for video_id, reason in failures.items():
      print(f"  FAILED {video_id}: {reason}")
    sys.exit(1)


def get_api_key(api_key_file_path: str) -> str:
  """Returns the first non-empty line of a file, or "" on error."""
  try:
    with open(api_key_file_path, "r", encoding="utf-8") as f:
      return [line.strip() for line in f if line.strip()][0]
  except FileNotFoundError:
    print(f"Error: The API key file '{api_key_file_path}' was not found.")
    return ""
  except (OSError, IndexError) as e:
    print(f"Error reading API key file '{api_key_file_path}': {e}")
    return ""


def read_video_ids_file(video_ids_file: str) -> list[tuple[str, str | None]]:
  """Parses a whitespace-delimited file of video IDs and optional durations.

  Each non-empty line is `<video_id> [<duration>]`. The duration is returned as
  the raw string (seconds or a timestamp); parsing_lib.
  estimate_video_duration_seconds() accepts it as-is.

  Args:
    video_ids_file: Path to the video IDs file.

  Returns:
    List of (video_id, duration_or_None) tuples, in file order.
  """
  entries = []
  with open(video_ids_file, "r", encoding="utf-8") as f:
    for line in f:
      parts = line.strip().split()
      if not parts:
        continue
      entries.append((parts[0], parts[1] if len(parts) >= 2 else None))
  return entries


def read_text_if_nonempty(path: str) -> str | None:
  """Returns the file's contents if it exists and is non-empty, else None."""
  if os.path.exists(path) and os.path.getsize(path) > 0:
    with open(path, "r", encoding="utf-8") as f:
      return f.read()
  return None


def write_text(path: str, content: str) -> None:
  """Writes text to `path`, creating parent directories as needed."""
  parent = os.path.dirname(path)
  if parent:
    os.makedirs(parent, exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    f.write(content)
