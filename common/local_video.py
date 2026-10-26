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

"""Helpers for local MP4 files used instead of YouTube video IDs.

Every output file of the pipeline is keyed by a video ID (output_{id}.txt,
{id}_tts_metadata.json, ...). For a local MP4 the ID is derived from the file
name with video_id_from_path(): characters other than letters, digits, '_' and
'-' become '_'. A YouTube ID is left unchanged, so `{youtube_id}.mp4` files keep
their dataset ID.
"""

import os
import re
import subprocess

VIDEO_EXTENSIONS = (".mp4",)


def video_id_from_path(path: str) -> str:
  """Returns a file-name-safe video ID derived from a video file's name.

  Examples:
    "videos/dQw4w9WgXcQ.mp4" -> "dQw4w9WgXcQ"
    "How to build a catio ｜ DIY.mp4" -> "How_to_build_a_catio_DIY"

  Args:
    path: Path to a video file.

  Returns:
    The video ID ("video" if nothing usable remains).
  """
  stem = os.path.splitext(os.path.basename(path))[0]
  video_id = re.sub(r"[^A-Za-z0-9_-]+", "_", stem)
  video_id = re.sub(r"_+", "_", video_id).strip("_")
  return video_id or "video"


def list_local_videos(video_dir: str) -> list[tuple[str, str]]:
  """Returns (video_id, path) for every MP4 in `video_dir`, sorted by ID.

  Args:
    video_dir: Directory of MP4 files.

  Returns:
    (video_id, path) pairs.

  Raises:
    ValueError: If two files map to the same video ID.
  """
  videos = {}
  for name in sorted(os.listdir(video_dir)):
    path = os.path.join(video_dir, name)
    if not name.lower().endswith(VIDEO_EXTENSIONS) or not os.path.isfile(path):
      continue
    video_id = video_id_from_path(name)
    if video_id in videos:
      raise ValueError(
          f"{videos[video_id]!r} and {path!r} both map to video ID"
          f" {video_id!r}; rename one of them."
      )
    videos[video_id] = path
  return sorted(videos.items())


def find_local_video(video_dir: str | None, video_id: str) -> str | None:
  """Returns the MP4 in `video_dir` for `video_id`, or None.

  Looks for `{video_id}.mp4` first, then for any MP4 whose
  video_id_from_path() equals `video_id`.

  Args:
    video_dir: Directory of MP4 files (None or "" -> None).
    video_id: Video ID.

  Returns:
    Path to the MP4 file, or None if there is none.
  """
  if not video_dir or not os.path.isdir(video_dir):
    return None
  exact = os.path.join(video_dir, f"{video_id}.mp4")
  if os.path.isfile(exact):
    return exact
  for name in sorted(os.listdir(video_dir)):
    if name.lower().endswith(VIDEO_EXTENSIONS) and (
        video_id_from_path(name) == video_id
    ):
      return os.path.join(video_dir, name)
  return None


def probe_duration_seconds(path: str) -> float | None:
  """Returns a video file's duration in seconds, or None if unknown.

  Uses ffprobe if installed, else OpenCV (optional dependency).

  Args:
    path: Path to the video file.

  Returns:
    Duration in seconds, or None.
  """
  try:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    duration = float(result.stdout.strip())
    if duration > 0:
      return duration
  except (OSError, subprocess.CalledProcessError, ValueError):
    pass
  try:
    import cv2  # pylint: disable=g-import-not-at-top
  except ImportError:
    cv2 = None
  if cv2 is not None:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0 and frames > 0:
      return float(frames / fps)
  print(
      f"WARNING: could not read the duration of {path} (install ffprobe or"
      " opencv-python, or pass the duration explicitly)."
  )
  return None
