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

"""Generates the dense visual-information reference used by evaluation.

The visual info lists time-stamped visual events, on-screen text, people, and a
narrative summary. Evaluation uses it for precision, timeliness, and on-screen
text metrics. The dataset ships pre-computed visual info for every video; this
module regenerates it for new videos.
"""

from common import gemini_utils
from google import genai
from google.genai import types

VISUAL_INFO_PROMPT = """
Please watch this video and list out all the important visual events
that happen in the video narrative with timestamps. Please be very
descriptive with time stamps.

  The output format must be exactly like this:
      * One scene description per line.
      * Each description follows the format <timestamp> <description>
        * The timestamp must be in the format 00:00:00.000 (hh:mm:ss.ms) followed by a "-".
      * The descriptions must be sorted by timestamp in ascending order.
      * **CRUCIAL**: The descriptions must follow this format and cover the entire video.

Example output:
00:00:00.000 visual_event_description1
00:00:07.123 visual_event_description2
00:00:23.745 visual_event_description3

Afterwards please watch the video and list all the text on screen and when it
appeared on the screen. (in **Text on screen:** section)

Please list all people in the video if there are any and what we know about
the people. (in **People in the video:** section)

Finally, please provide a summary of the video narrative.
(in **Summary of the video narrative:** section)
"""


def generate_visual_info(
    client: genai.Client,
    video_part: types.Part,
    model: str = gemini_utils.DEFAULT_GEMINI_MODEL,
) -> str | None:
  """Returns the visual-information reference for a video, or None."""
  return gemini_utils.generate_with_video(
      client, model, video_part, VISUAL_INFO_PROMPT
  )
