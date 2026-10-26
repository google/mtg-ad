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

r"""CLI entry point for generating inline Audio Descriptions (AD).

Uses the same defaults and Gemini generation steps as run_end_to_end.py
(prompt timing_asr, gemini-3.8-flash; local MP4s; missing transcripts
generated; token usage and cost in {output_dir}/usage/).

Usage (YouTube video IDs, with existing transcripts):
  python run_ad_generation.py \
    --genai_model gemini \
    --gemini_model gemini-3.8-flash \
    --prompt_type timing_asr \
    --api_key_path /path/to/gemini_api_key.txt \
    --video_ids_file sample_video_ids.txt \
    --input_transcript_dir transcript/ \
    --output_dir auto_ad/

Usage (single local MP4, agentic prompt):
  python run_ad_generation.py \
    --api_key_path /path/to/gemini_api_key.txt \
    --mp4_path /path/to/video.mp4 \
    --prompt_type AGENTIC_GEMINI_3P7_UP \
    --output_dir auto_ad/

Usage (every MP4 in a folder; transcripts are generated if missing):
  python run_ad_generation.py \
    --api_key_path /path/to/gemini_api_key.txt \
    --video_dir videos/ \
    --output_dir auto_ad/
"""

from generation import generation

if __name__ == "__main__":
  generation.main()
