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

r"""Renders audio description scripts to speech using Gemini TTS or Cloud TTS.

Converts an AD text file (one timestamped line per description, or raw model
output) into per-segment WAV files. The resulting audio durations are used for
the temporal metrics (overlap, coverage, silence duration); evaluation does not
estimate durations from word counts.

Two TTS backends are supported:
  1. Gemini TTS (default, gemini-2.5-flash-preview-tts): needs only the Gemini
     API key.
  2. Google Cloud TTS (--cloud_tts): fixed speaking rate (1.2x by default), the
     setting used for the paper's temporal metrics. Needs the optional
     google-cloud-texttospeech package and Application Default Credentials.

Use a separate --output_dir per backend: WAV files are keyed by the description
text, so a shared directory would mix voices and speaking rates.

Any failed segment is reported as an error, the video's metadata JSON is not
written (so a re-run retries it), and the script exits with status 1.

Usage (Gemini TTS):
  python generate_tts.py \
    --api_key_path /path/to/gemini_api_key.txt \
    --input_dir auto_ad/ \
    --output_dir tts_gemini/

Usage (Cloud TTS at 1.2x):
  pip install google-cloud-texttospeech
  gcloud auth application-default login
  python generate_tts.py \
    --cloud_tts \
    --input_dir auto_ad/ \
    --output_dir tts_cloud/

Available Gemini voices: Kore, Charon, Fenrir, Aoede, Puck (see Gemini TTS
docs).
"""

import argparse
import glob
import os
import sys

from common import file_utils
from common import tts
from common import usage
from google import genai


def main():
  """Parses CLI arguments and synthesizes TTS audio for AD files."""
  parser = argparse.ArgumentParser(
      description="Render audio description scripts to speech."
  )
  parser.add_argument(
      "--api_key_path",
      type=str,
      default="",
      help="Path to Gemini API key file. Not needed with --cloud_tts.",
  )
  parser.add_argument(
      "--input_dir",
      type=str,
      required=True,
      help="Directory containing AD files (output_{VIDEO_ID}.txt).",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Output directory for TTS audio and metadata.",
  )
  parser.add_argument(
      "--video_ids_file",
      type=str,
      default="",
      help=(
          "Optional path to a file listing video IDs to process (one per line)."
          " If not provided, all output_*.txt files in --input_dir are"
          " processed."
      ),
  )
  parser.add_argument(
      "--voice_name",
      type=str,
      default=tts.DEFAULT_VOICE,
      help=f"Gemini TTS voice name. Default: {tts.DEFAULT_VOICE}.",
  )
  parser.add_argument(
      "--cloud_tts",
      action="store_true",
      default=False,
      help=(
          "Use Google Cloud TTS at {:.1f}x speaking rate instead of Gemini TTS."
          " Requires the optional google-cloud-texttospeech package and"
          " Application Default Credentials."
      ).format(tts.CLOUD_TTS_SPEAKING_RATE),
  )
  parser.add_argument(
      "--tts_model",
      type=str,
      default=tts.TTS_MODEL,
      help=f"Gemini TTS model. Default: {tts.TTS_MODEL}.",
  )
  parser.add_argument(
      "--speaking_rate",
      type=float,
      default=tts.CLOUD_TTS_SPEAKING_RATE,
      help=(
          "Cloud TTS speaking rate (only with --cloud_tts). Default:"
          f" {tts.CLOUD_TTS_SPEAKING_RATE}."
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

  client = None
  if args.cloud_tts:
    try:
      tts.require_cloud_tts()
    except tts.TtsError as e:
      sys.exit(f"ERROR: {e}")
  else:
    api_key = file_utils.get_api_key(args.api_key_path)
    if not api_key:
      sys.exit("ERROR: --api_key_path is required when not using --cloud_tts.")
    client = genai.Client(api_key=api_key)

  # Determine which video IDs to process
  if args.video_ids_file:
    video_ids = [
        video_id
        for video_id, _ in file_utils.read_video_ids_file(args.video_ids_file)
    ]
  else:
    paths = glob.glob(os.path.join(args.input_dir, "output_*.txt"))
    video_ids = sorted(
        os.path.basename(p)[len("output_") : -len(".txt")] for p in paths
    )

  os.makedirs(args.output_dir, exist_ok=True)
  backend = (
      f"Cloud TTS ({args.speaking_rate}x)"
      if args.cloud_tts
      else f"Gemini TTS ({args.tts_model}, {args.voice_name})"
  )
  failures = {}
  for i, video_id in enumerate(video_ids):
    prefix = f"[{i+1}/{len(video_ids)}]"
    ad_path = os.path.join(args.input_dir, f"output_{video_id}.txt")
    if not os.path.exists(ad_path):
      print(f"{prefix} ERROR: No AD file for {video_id}: {ad_path}")
      failures[video_id] = "no AD file"
      continue
    if os.path.exists(tts.metadata_path(video_id, args.output_dir)):
      print(f"{prefix} SKIP (already done): {video_id}")
      continue

    print(f"{prefix} Synthesizing {backend} for {video_id}...")
    with open(ad_path, "r", encoding="utf-8") as f:
      ad_text = f.read()
    with usage.recording() as recorder, usage.step("tts"):
      try:
        tts.synthesize_ad_file(
            client,
            video_id,
            ad_text,
            args.output_dir,
            voice_name=args.voice_name,
            use_cloud_tts=args.cloud_tts,
            tts_model=args.tts_model,
            speaking_rate=args.speaking_rate,
        )
      except tts.TtsError as e:
        print(f"{prefix} ERROR: {e}")
        failures[video_id] = str(e)
    if not args.cloud_tts:
      usage.save(
          os.path.join(args.output_dir, "usage", f"output_{video_id}.json"),
          recorder,
          extra={"video_id": video_id},
      )

  file_utils.report_failures(failures, len(video_ids), "TTS")


if __name__ == "__main__":
  main()
