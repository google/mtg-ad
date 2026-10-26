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

r"""CLI entry point for evaluating generated Audio Descriptions (AD).

Usage (LLM judge defaults to gemini-3.8-flash; Quality-RF video_asr runs by
default, VideoA11y-Ref when GT AD is given):
  python run_ad_eval.py \
    --api_key_path /path/to/gemini_api_key.txt \
    --video_ids_file sample_video_ids.txt \
    --output_dir eval_results/ \
    --input_transcript_dir transcript/ \
    --input_gt_ad_dir gt_ad/ \
    --input_visual_info_dir visual_info/ \
    --input_auto_ad_dir auto_ad/ \
    --input_tts_dir tts_cloud/
"""

from evaluation import evaluation

if __name__ == "__main__":
  evaluation.main()
