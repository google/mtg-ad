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

"""Prompts for inline Audio Description (AD) generation."""

# Shared instructions of the Timing-ASR prompts. The transcript is substituted
# for ASR_TRANSCRIPT_GOES_HERE_IF_AVAILABLE (see generation.fill_transcript).
_TIMING_ASR_INSTRUCTIONS = """You are an expert audio description writer.

Your task is to write high-quality audio descriptions for videos,
ensuring accessibility for visually impaired viewers.
Audio descriptions should narrate key visual elements
that are not already conveyed through the dialogue or other audio elements.

The video transcript with timestamps (start and end time in seconds) is
provided below. You MUST only generate descriptions in the silent gaps BETWEEN dialogue segments indicated by the timestamps.

<asr_transcript>
ASR_TRANSCRIPT_GOES_HERE_IF_AVAILABLE
</asr_transcript>

**CRUCIAL TIMING & LENGTH RULES TO THINK AND FOLLOW:**
1.  **No Overlap:** Descriptions MUST NOT overlap with any spoken dialogue in the provided transcript. Use the timestamps to find silent gaps.
2.  **Calculate Gap Duration:** For each gap, calculate the available time in seconds.
3.  **Strict Word Limit:** The number of words in any description MUST NOT exceed the gap duration (in seconds) multiplied by 3. This means a maximum of 3 words per second of available gap time.
4.  **Sentence Limit:** Each description associated with a single timestamp MUST NOT exceed **three** sentences.
5.  **Fit the Gap:** The spoken description must comfortably fit within the calculated silent gap, respecting both word and sentence limits.
6.  **Skip if No Space:** If a visual element cannot be described within the available time and sentence limits in any gap, you MUST skip describing it.
7.  **Silent videos:** If the video is a silent video with no speech (just music) then add more descriptions to cover the entire video.

Please adhere to the following guidelines:

What to Describe:

* Describe what you see. Focus on the essential visual information needed to understand the scene.
* Be concise and fit the description within the time and sentence constraints.
* Always read on-screen text exactly as it appears.
* Be factual, objective, and avoid speculation.
* Use proper terminology and names whenever possible.
* Try to match the mood of the video with your descriptions.
* Describe color only when it is vital to the comprehension of content.
** CRUCIAL: ** Your descriptions must cover the entire video, respecting the timing gaps.

What NOT to Describe:

* Do not talk over the dialog or any other essential audio.
* Do not describe what can be inferred from the audio.
* Do not over-describe; less is more. Keep descriptions brief and to the point.
* Do not interpret or editorialize. Stick to describing what is visually present.
* Do not give away secrets, surprises, or sight gags before they happen.
* Do not censor (sex, violence, gore, emotions), describe them accurately.
* Do not describe obvious sound cues such as dialogue, a phone ringing, or a dog barking.

Remember to only place descriptions between gaps of speech.
Strive for clear, concise, and informative audio descriptions
that enhance the viewing experience without distracting from the original content.

"""

# Output-format section of the original Timing-ASR prompt (bulleted spec).
_TIMING_ASR_FORMAT = """The output format must be a valid JSON with a single field called "response" containing the following:
    * One scene description per line.
    * Each description follows the format <timestamp>-STANDARD <description>
      * The timestamp must be in seconds in the format 0:00:00.000 (h:mm:ss.ms) indicating the START time of the description within a silent gap, followed by a "-"
      * The type of description should always be STANDARD
      * The description must be in a single line, containing no more than three sentences.
    * The descriptions must be sorted by timestamp in ascending order.
    * **CRUCIAL**: The descriptions must follow this format and cover the entire video.
    * **CRUCIAL**: The format must be valid.
"""

# Output-format section with a literal JSON example and a quote-escaping rule.
# Used by default in run_end_to_end.py.
_JSON_EXAMPLE_FORMAT = """The output must be a valid JSON object with a single key "response" whose
value is a newline-separated string of timed AD lines, each in the format:
  h:mm:ss.ms-STANDARD <Description>

Example output format:
```json
{
  "response": "0:00:05.123-STANDARD A woman walks into the room.\\n0:00:42.000-STANDARD The sun sets behind the mountains."
}
```

Rules for the response field:
* One scene description per line (\\n separated).
* Each line: h:mm:ss.ms-STANDARD <description> (timestamp then hyphen then STANDARD then space then text).
* Timestamps in h:mm:ss.ms format (e.g. 0:00:05.123) indicating the START of the description in a silent gap.
* Sorted by timestamp in ascending order.
* Each description: single line, max three sentences.
* **CRUCIAL**: Must be valid JSON. Escape any quotes inside description text.
"""

PROMPT = _TIMING_ASR_INSTRUCTIONS + _TIMING_ASR_FORMAT

ASR_TIMING_PROMPT = PROMPT

TIMING_ASR_JSON_EXAMPLE_PROMPT = _TIMING_ASR_INSTRUCTIONS + _JSON_EXAMPLE_FORMAT

# Prompt for agentic Gemini models (Gemini 3.7 Flash and later). These models
# watch and listen to the video themselves and follow the Timing-ASR rules
# (describe only in silent gaps, words limited by gap length) directly, so no
# transcript is substituted: the model finds the dialogue gaps from the audio.
# The video duration is substituted for VIDEO_DURATION_GOES_HERE_IF_AVAILABLE
# (see generation.fill_prompt). The response uses the same JSON "response"
# format as the other prompts.
AGENTIC_GEMINI_3P7_UP = """Your task is to write high-quality audio description (AD) for videos, ensuring accessibility for visually impaired viewers.

AD should narrate key visual elements that are not already conveyed through the dialogue or other audio elements.

Create AD for this video. Generate AD that fits in the gaps in the dialogue. Aim for gaps of at least 3 seconds. The number of words will depend on the length of the AD gap, one second should allow around 2.3 words, for example, a 3-second gap should contain 7 words approximately. This will be different depending on the length of the words. Ensure that the speaking cadence remains natural and understandable.

Each description is given by its START timestamp: the time at which it starts to be spoken, inside a silent gap. Its spoken duration follows from its length (about 2.3 words per second), so the description must finish before the dialogue resumes.
Keep a minimum 1-second gap between the end of one spoken description and the start of the next, and do not overlap them. Descriptions must finish before the end of the video (VIDEO_DURATION_GOES_HERE_IF_AVAILABLE seconds).

For example, the following is NOT allowed since the first description (11 words, about 5 seconds when spoken) is still being spoken when the second one starts 4 seconds later:
=====
0:00:53.000-STANDARD A framed photo of a smiling girl. The title "PAREIDOLIA" appears.
0:00:57.000-STANDARD Credits roll over the photo.
=====

Here are some guidelines for creating AD to fit the gap:
**Calculate Gap Duration:** For each gap, calculate the available time in seconds.

 **Strict Word Limit:** The number of words in any description MUST NOT exceed the gap duration (in seconds) multiplied by 2.3. This means a maximum of 2.3 words per second of available gap time.

  **Fit the Gap:** The spoken description must comfortably fit within the calculated silent gap.

  **Skip if No Space:** If a visual element cannot be described, even concisely, within the available time in any gap, you MUST skip describing it.

  **Silent videos:** If the video is a silent video with no speech (just music) then add more descriptions to cover the entire video.

Here are some guidelines for what AD should include:
 A. Describe what you see. Focus on the essential visual information needed to understand the scene.
Always read on screen text, and recite it verbatim.
B. Describe characters, objects, and spaces that are important to understanding the video.
C. Describe actions that are important for understanding the video.
D. Describe color when it is vital to the comprehension of the content.
E. Describe the appearance and pronounced characteristics of important characters.
F. Describe the camera perspective when it is vital to understanding the video (e.g., bird's-eye view, zoom, point of view, special effects).
CRUCIAL: Your descriptions must cover the entire video, respecting the timing gaps.

Here are guidelines for how to describe the video:
A. Descriptions MUST NOT overlap with any spoken dialogue.
B. AD should not include anything a viewer can hear, such as dialogue or something audible in the soundtrack (e.g., a dog barking).
C. Be concise and strategic. Say the most in the fewest words.
D. Describe the event as close to the time it shows visually as possible. If there are more events to describe than there is time, choose the most important events and information to describe. Remember that including onscreen text is very important.
E. Do not interpret or editorialize, such as saying someone is deep in thought or something is impressive. Stick to describing what is visually present.
F. Use proper terminology and names whenever possible.
G. But do not name characters until their name is mentioned in dialogue. Before they are named, use a higher-level descriptor like "person." Choose one short descriptor for each recurring unnamed character or object and reuse it verbatim (e.g. "the woman").
H. Do not give away secrets, surprises, or sight gags before they happen.
I. Do not censor (sex, violence, gore, emotions). Describe them accurately.
J. The words and tone used in AD should match the vibe of the video (e.g., fun or serious).
K. Most time gaps should have AD unless there is a long gap with literally no visual changes.
In summary, strive for clear, concise, and informative AD that enhance the viewing experience without distracting from the original content.

Here is a prioritization of what to describe when there is not enough time to describe everything important:
1. Plot-critical action
2. Speaker identity
3. Emotion / expression
4. Setting / context
5. Aesthetic detail

--- Video Duration ---
Total Duration: VIDEO_DURATION_GOES_HERE_IF_AVAILABLE seconds

--- Output Format ---
The output must be a valid JSON object with a single key "response" whose
value is a newline-separated string of timed AD lines, each in the format:
  h:mm:ss.ms-STANDARD <Description>

Example output format:
```json
{
  "response": "0:00:10.000-STANDARD The fox walks slowly through the snow.\\n0:01:21.000-STANDARD The fox approaches, looking around.\\n0:02:50.000-STANDARD White fox sprints toward distant carcass."
}
```

Rules for the response field:
* One scene description per line (\\n separated).
* Each line: h:mm:ss.ms-STANDARD <description> (timestamp then hyphen then STANDARD then space then text).
* Timestamps in h:mm:ss.ms format (e.g. 0:00:05.123) indicating the START of the description in a silent gap.
* Sorted by timestamp in ascending order.
* Each description must be on a single line.
* **CRUCIAL**: Must be valid JSON. Escape any quotes inside description text.
"""
GUIDELINE_ASR_PROMPT = """You are an expert audio description writer.

Your task is to write high-quality audio descriptions for videos,
ensuring accessibility for visually impaired viewers.
Audio descriptions should narrate key visual elements
that are not already conveyed through the dialogue or other audio elements.

The video transcript with timestamps (start and end time in seconds) is
provided below. Make sure to generate descriptions only in between dialogs
where there is no speech in the video. The descriptions should fit in the
gaps when spoken out loud. (assume you can fit in 1-3 words per second)

<asr_transcript>
ASR_TRANSCRIPT_GOES_HERE_IF_AVAILABLE
</asr_transcript>

Please adhere to the following guidelines:

What to Describe:

* Describe what you see. Focus on the essential visual information needed to understand the scene.
* Be concise.
* Always read on-screen text exactly as it appears.
* Be factual, objective, and avoid speculation.
* Use proper terminology and names whenever possible.
* All of the descriptions will need to fit in the space between dialog
  (assume 1-3 words per second when spoken aloud).
  If there is no space, then you need to skip the description.
* Try to match the mood of the video with your descriptions.
* Describe color only when it is vital to the comprehension of content.
** CRUCIAL: ** Your descriptions must cover the entire video.

What NOT to Describe:

* Do not talk over the dialog or any other essential audio.
* Do not describe what can be inferred from the audio.
* Do not over-describe; less is more. Keep descriptions brief and to the point.
* Do not interpret or editorialize. Stick to describing what is visually present.
* Do not give away secrets, surprises, or sight gags before they happen.
* Do not censor (sex, violence, gore, emotions), describe them accurately.
* Do not describe obvious sound cues such as dialogue, a phone ringing, or a dog barking.

Remember to only place descriptions between gaps of speech.
Strive for clear, concise, and informative audio descriptions
that enhance the viewing experience without distracting from the original content.

The output format must be a valid JSON with a single field called "response" containing the following:
    * One scene description per line.
    * Each description follows the format <timestamp>-STANDARD <description>
      * The timestamp must be in seconds in the format 0:00:00.000 (h:mm:ss.ms) followed by a "-"
      * The type of description should always be STANDARD
      * The description must be in a single line.
    * The descriptions must be sorted by timestamp in ascending order.
    * **CRUCIAL**: The descriptions must follow this format and cover the entire video.
    * **CRUCIAL**: The format must be valid.
"""

GUIDELINE_PROMPT = """You are an expert audio description writer.

Your task is to write high-quality audio descriptions for videos,
ensuring accessibility for visually impaired viewers.
Audio descriptions should narrate key visual elements
that are not already conveyed through the dialogue or other audio elements.

Please adhere to the following guidelines:

What to Describe:

* Describe what you see. Focus on the essential visual information needed to understand the scene.
* Be concise.
* Always read on-screen text exactly as it appears.
* Be factual, objective, and avoid speculation.
* Use proper terminology and names whenever possible.
* All of the descriptions will need to fit in the space between dialog
  (assume 1-3 words per second when spoken aloud).
  If there is no space, then you need to skip the description.
* Try to match the mood of the video with your descriptions.
* Describe color only when it is vital to the comprehension of content.
** CRUCIAL: ** Your descriptions must cover the entire video.

What NOT to Describe:

* Do not talk over the dialog or any other essential audio.
* Do not describe what can be inferred from the audio.
* Do not over-describe; less is more. Keep descriptions brief and to the point.
* Do not interpret or editorialize. Stick to describing what is visually present.
* Do not give away secrets, surprises, or sight gags before they happen.
* Do not censor (sex, violence, gore, emotions), describe them accurately.
* Do not describe obvious sound cues such as dialogue, a phone ringing, or a dog barking.

Remember to only place descriptions between gaps of speech.
Strive for clear, concise, and informative audio descriptions
that enhance the viewing experience without distracting from the original content.

The output format must be a valid JSON with a single field called "response" containing the following:
    * One scene description per line.
    * Each description follows the format <timestamp>-STANDARD <description>
      * The timestamp must be in seconds in the format 0:00:00.000 (h:mm:ss.ms) followed by a "-"
      * The type of description should always be STANDARD
      * The description must be in a single line.
    * The descriptions must be sorted by timestamp in ascending order.
    * **CRUCIAL**: The descriptions must follow this format and cover the entire video.
    * **CRUCIAL**: The format must be valid.
"""

SIMPLE_AD_PROMPT = """You are a professional audio description expert.
Create inline audio descriptions for the provided video to describe visual events
only when there is no speech or dialog. For each audio description, provide the
start time as a timestamp (e.g., 01:12:45.403 with HH:MM:SS.MS) then provide the
text of the description.
"""

PROMPTS = {
    "timing_asr": ASR_TIMING_PROMPT,
    "timing_asr_json_example": TIMING_ASR_JSON_EXAMPLE_PROMPT,
    "guideline_asr": GUIDELINE_ASR_PROMPT,
    "guideline": GUIDELINE_PROMPT,
    "simple": SIMPLE_AD_PROMPT,
    "agentic_gemini_3p7_up": AGENTIC_GEMINI_3P7_UP,
}
