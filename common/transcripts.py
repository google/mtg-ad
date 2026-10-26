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

"""ASR transcript generation with Gemini.

Transcripts are one utterance per line:
  H:MM:SS.mmm--H:MM:SS.mmm spoken text
which is the format expected by AD generation and evaluation.
"""

from common import gemini_utils
from google import genai
from google.genai import types

TRANSCRIPT_PROMPT = """\
Please transcribe all spoken dialogue in this video with precise timestamps.
Format each line EXACTLY as: H:MM:SS.mmm--H:MM:SS.mmm spoken_text
Where the first timestamp is the start time and the second is the end time.
Include ALL dialogue. Output ONLY the transcript lines, nothing else.\
"""


def generate_transcript(
    client: genai.Client,
    video_part: types.Part,
    model: str = gemini_utils.DEFAULT_GEMINI_MODEL,
) -> str | None:
  """Returns a timestamped transcript of the video's dialogue, or None."""
  return gemini_utils.generate_with_video(
      client, model, video_part, TRANSCRIPT_PROMPT
  )
