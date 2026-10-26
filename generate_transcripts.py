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

r"""Generates ASR transcripts for YouTube videos or local MP4s using Gemini.

Usage:
  python generate_transcripts.py \
    --api_key_path /path/to/gemini_api_key.txt \
    --video_ids_file sample_video_ids.txt \
    --output_dir transcript/

  # Every MP4 in a folder (IDs derived from the file names):
  python generate_transcripts.py \
    --api_key_path /path/to/gemini_api_key.txt \
    --input_video_dir videos/ \
    --output_dir transcript/

The output transcript format is one line per utterance:
  H:MM:SS.mmm--H:MM:SS.mmm spoken text here

This is the format expected by the evaluation script (run_ad_eval.py).
"""

import argparse
import os
import sys

from common import file_utils
from common import gemini_utils
from common import local_video
from common import transcripts
from common import usage
from google import genai


def main():
  """Parses CLI arguments and generates transcripts for all listed videos."""
  parser = argparse.ArgumentParser(
      description="Generate ASR transcripts for YouTube videos using Gemini."
  )
  parser.add_argument(
      "--api_key_path",
      type=str,
      required=True,
      help="Path to Gemini API key file.",
  )
  parser.add_argument(
      "--video_ids_file",
      type=str,
      default="",
      help=(
          "Path to file with one YouTube video ID per line. May be omitted"
          " with --input_video_dir to transcribe every MP4 in it."
      ),
  )
  parser.add_argument(
      "--input_video_dir",
      type=str,
      default="",
      help=(
          "Optional folder of local MP4 files ({video_id}.mp4, or any MP4"
          " whose name maps to the ID, see common/local_video.py), uploaded"
          " instead of streaming the YouTube URL."
      ),
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Output directory for transcript files (output_{VIDEO_ID}.txt).",
  )
  parser.add_argument(
      "--model",
      type=str,
      default=gemini_utils.DEFAULT_GEMINI_MODEL,
      help="Gemini model to use for transcription.",
  )
  parser.add_argument(
      "--skip_existing",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Skip videos that already have a transcript file (default). Use"
          " --no-skip_existing to regenerate them."
      ),
  )
  parser.add_argument(
      "--pricing_file",
      type=str,
      default=usage.DEFAULT_PRICING_FILE,
      help=(
          "JSON of Gemini prices per 1M tokens for the cost estimates in"
          " {output_dir}/usage/. Default: metadata/gemini_pricing.json."
      ),
  )
  args = parser.parse_args()
  usage.set_pricing_file(args.pricing_file)
  if not args.video_ids_file and not args.input_video_dir:
    parser.error("pass --video_ids_file and/or --input_video_dir.")

  api_key = file_utils.get_api_key(args.api_key_path)
  if not api_key:
    sys.exit(1)

  if args.video_ids_file:
    video_ids = [
        video_id
        for video_id, _ in file_utils.read_video_ids_file(args.video_ids_file)
    ]
  else:
    video_ids = [
        video_id
        for video_id, _ in local_video.list_local_videos(args.input_video_dir)
    ]
  if not video_ids:
    return

  os.makedirs(args.output_dir, exist_ok=True)
  client = genai.Client(api_key=api_key)

  skipped = 0
  failures = {}
  for i, video_id in enumerate(video_ids):
    out_path = os.path.join(args.output_dir, f"output_{video_id}.txt")

    if args.skip_existing and os.path.exists(out_path):
      skipped += 1
      continue

    print(f"[{i + 1}/{len(video_ids)}] Transcribing {video_id}...")
    recorder = usage.UsageRecorder()
    try:
      video_part = gemini_utils.make_video_part(
          client,
          video_id,
          local_video.find_local_video(args.input_video_dir, video_id),
      )
      with usage.recording(recorder), usage.step("transcript"):
        transcript = transcripts.generate_transcript(
            client, video_part, model=args.model
        )
    except gemini_utils.FatalApiError as e:
      sys.exit(f"FATAL: {e}")
    finally:
      # Gemini token usage and estimated cost, summed over runs.
      usage.save(
          os.path.join(args.output_dir, "usage", f"output_{video_id}.json"),
          recorder,
          extra={"video_id": video_id, "transcript_model": args.model},
      )

    if transcript is None:
      print(f"  ERROR: Failed to transcribe {video_id}")
      failures[video_id] = "transcription failed"
      continue
    if not transcript.strip():
      print(f"  Note: no speech found in {video_id}; writing an empty file.")
    file_utils.write_text(out_path, transcript)

  if skipped:
    print(f"Skipped {skipped} videos with existing transcripts.")
  file_utils.report_failures(failures, len(video_ids), "Transcripts")


if __name__ == "__main__":
  main()
