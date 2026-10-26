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

"""Reference-free qualitative metrics (Quality-RF).

Quality-RF scores a generated AD on the four VideoA11y dimensions
(Descriptive, Objective, Accurate, Clear; 1-5) without a human reference: the
MLLM judge watches the video, optionally with the ASR transcript and/or the
dense visual information, summarizes the video, gives a rationale, and then
rates the AD.

Variants (keys of PROMPTS), named after the judge's inputs besides the video:
  video_asr (default): video + ASR transcript.
  video_only: video only.
  video_visual_info: video + dense visual information.
  video_visual_info_asr: video + visual information + ASR transcript.

Output keys are prefixed with the variant name, e.g. video_asr_descriptive,
video_asr_objective, video_asr_accurate, video_asr_clear, plus
video_asr_summary and video_asr_rationale.
"""

import json
import re
from typing import Any, Callable

# Substituted into a prompt when an optional input is absent.
NOT_AVAILABLE = "NOT AVAILABLE"

DEFAULT_VARIANT = "video_asr"

VIDEO_ONLY_PROMPT = """
You are an expert evaluator with a deep understanding of audio description quality, particularly for making video content accessible to blind and low vision (BLV) individuals.

Your role is to assess and rate audio descriptions based on specific metrics. Audio descriptions should describe important visual events that are not already captured in the existing dialog or audio of the video.

Task: I have a video and a model-generated audio description. Additionally, I have four evaluation metrics: Descriptive, Objective, Accurate, and Clear.

Please evaluate the model-generated audio description using the above metrics by comparing it to the video.

Before providing the ratings, please provide a brief summary of the video and your rationale for the ratings.

Metrics Definition:

1. Descriptive: A descriptive description should provide vivid details about objects, people, and settings while maintaining a concise narrative flow. It should include essential information about the appearance of individuals, such as their clothing, facial expressions, and actions, and visual properties of objects, such as color and shape. For example, "A smiling woman, wearing a loose white dress, types on a laptop in a softly lit room."

2. Objective: An objective description should report what is visible without adding personal opinions or assumptions. For instance, describe two people as "a woman and a man" without adding any relationship context unless it is mentioned. It should also avoid guessing personal attributes like racial or gender identities unless explicitly clear.

3. Accurate: An accurate description should aim for precision in describing visible elements without imagination. It should ensure that all details, such as colors and spatial arrangements, are reported correctly. For instance, "Blue sky with white clouds" instead of "White sky with blue clouds" if that is what appears. Additionally, it should avoid adding unnecessary details that do not contribute to a deeper understanding of the scene.

4. Clear: A clear description should present information in the videos in a way that is easy to follow for blind and low vision individuals. It should describe the object or character's properties before the actions or relationships with other objects or characters. For example, "A man wearing sunglasses plays the piano." Pronouns should only be used when it is clear who they refer to, and the description should include any on-screen text if it is central to understanding. For instance, "A man in a black polo shirt is speaking. He is in a studio setting with a red couch and a blue background featuring the text 'Talk of the Town'".

Input:

Model-generated audio description:
<generated_desc>
GENERATED_DESC_GOES_HERE_IF_AVAILABLE
</generated_desc>

Output Format: Return the result as a string format dictionary with the following structure. Please output the fields in this exact order:

{
  "Summary": "Your brief summary of the video here",
  "Rationale": "Your rationale for the ratings here",
  "Descriptive": [Rating out of 5],
  "Objective": [Rating out of 5],
  "Accurate": [Rating out of 5],
  "Clear": [Rating out of 5]
}

"""

VIDEO_ASR_PROMPT = """
You are an expert evaluator with a deep understanding of audio description quality, particularly for making video content accessible to blind and low vision (BLV) individuals.

Your role is to assess and rate audio descriptions based on specific metrics. Audio descriptions should describe important visual events that are not already captured in the existing dialog or audio of the video. The provided ASR transcript represents the spoken content in the video.

Task: I have a video, an ASR transcript of the video, and a model-generated audio description. Additionally, I have four evaluation metrics: Descriptive, Objective, Accurate, and Clear.

Please evaluate the model-generated audio description using the above metrics by comparing it to the video and the ASR transcript.

Before providing the ratings, please provide a brief summary of the video and your rationale for the ratings.

Metrics Definition:

1. Descriptive: A descriptive description should provide vivid details about objects, people, and settings while maintaining a concise narrative flow. It should include essential information about the appearance of individuals, such as their clothing, facial expressions, and actions, and visual properties of objects, such as color and shape. For example, "A smiling woman, wearing a loose white dress, types on a laptop in a softly lit room."

2. Objective: An objective description should report what is visible without adding personal opinions or assumptions. For instance, describe two people as "a woman and a man" without adding any relationship context unless it is mentioned. It should also avoid guessing personal attributes like racial or gender identities unless explicitly clear.

3. Accurate: An accurate description should aim for precision in describing visible elements without imagination. It should ensure that all details, such as colors and spatial arrangements, are reported correctly. For instance, "Blue sky with white clouds" instead of "White sky with blue clouds" if that is what appears. Additionally, it should avoid adding unnecessary details that do not contribute to a deeper understanding of the scene.

4. Clear: A clear description should present information in the videos in a way that is easy to follow for blind and low vision individuals. It should describe the object or character's properties before the actions or relationships with other objects or characters. For example, "A man wearing sunglasses plays the piano." Pronouns should only be used when it is clear who they refer to, and the description should include any on-screen text if it is central to understanding. For instance, "A man in a black polo shirt is speaking. He is in a studio setting with a red couch and a blue background featuring the text 'Talk of the Town'".

Input:

ASR transcript:
<asr_transcript>
ASR_TRANSCRIPT_GOES_HERE_IF_AVAILABLE
</asr_transcript>

Model-generated audio description:
<generated_desc>
GENERATED_DESC_GOES_HERE_IF_AVAILABLE
</generated_desc>

Output Format: Return the result as a string format dictionary with the following structure. Please output the fields in this exact order:

{
  "Summary": "Your brief summary of the video here",
  "Rationale": "Your rationale for the ratings here",
  "Descriptive": [Rating out of 5],
  "Objective": [Rating out of 5],
  "Accurate": [Rating out of 5],
  "Clear": [Rating out of 5]
}

"""

VIDEO_VISUAL_INFO_PROMPT = """
You are an expert evaluator with a deep understanding of audio description quality, particularly for making video content accessible to blind and low vision (BLV) individuals.

Your role is to assess and rate audio descriptions based on specific metrics. Audio descriptions should describe important visual events that are not already captured in the existing dialog or audio of the video.

Task: I have a video, visual information extracted from the video, and a model-generated audio description. Additionally, I have four evaluation metrics: Descriptive, Objective, Accurate, and Clear.

Please evaluate the model-generated audio description using the above metrics by comparing it to the video and the provided visual information.

Before providing the ratings, please provide a brief summary of the video and your rationale for the ratings.

Metrics Definition:

1. Descriptive: A descriptive description should provide vivid details about objects, people, and settings while maintaining a concise narrative flow. It should include essential information about the appearance of individuals, such as their clothing, facial expressions, and actions, and visual properties of objects, such as color and shape. For example, "A smiling woman, wearing a loose white dress, types on a laptop in a softly lit room."

2. Objective: An objective description should report what is visible without adding personal opinions or assumptions. For instance, describe two people as "a woman and a man" without adding any relationship context unless it is mentioned. It should also avoid guessing personal attributes like racial or gender identities unless explicitly clear.

3. Accurate: An accurate description should aim for precision in describing visible elements without imagination. It should ensure that all details, such as colors and spatial arrangements, are reported correctly. For instance, "Blue sky with white clouds" instead of "White sky with blue clouds" if that is what appears. Additionally, it should avoid adding unnecessary details that do not contribute to a deeper understanding of the scene.

4. Clear: A clear description should present information in the videos in a way that is easy to follow for blind and low vision individuals. It should describe the object or character's properties before the actions or relationships with other objects or characters. For example, "A man wearing sunglasses plays the piano." Pronouns should only be used when it is clear who they refer to, and the description should include any on-screen text if it is central to understanding. For instance, "A man in a black polo shirt is speaking. He is in a studio setting with a red couch and a blue background featuring the text 'Talk of the Town'".

Input:

Visual information:
<visual_info>
VISUAL_INFO_GOES_HERE_IF_AVAILABLE
</visual_info>

Model-generated audio description:
<generated_desc>
GENERATED_DESC_GOES_HERE_IF_AVAILABLE
</generated_desc>

Output Format: Return the result as a string format dictionary with the following structure. Please output the fields in this exact order:

{
  "Summary": "Your brief summary of the video here",
  "Rationale": "Your rationale for the ratings here",
  "Descriptive": [Rating out of 5],
  "Objective": [Rating out of 5],
  "Accurate": [Rating out of 5],
  "Clear": [Rating out of 5]
}

"""

VIDEO_VISUAL_INFO_ASR_PROMPT = """
You are an expert evaluator with a deep understanding of audio description quality, particularly for making video content accessible to blind and low vision (BLV) individuals.

Your role is to assess and rate audio descriptions based on specific metrics. Audio descriptions should describe important visual events that are not already captured in the existing dialog or audio of the video. The provided ASR transcript represents the spoken content in the video.

Task: I have a video, visual information extracted from the video, an ASR transcript of the video, and a model-generated audio description. Additionally, I have four evaluation metrics: Descriptive, Objective, Accurate, and Clear.

Please evaluate the model-generated audio description using the above metrics by comparing it to the video, the provided visual information, and the ASR transcript.

Before providing the ratings, please provide a brief summary of the video and your rationale for the ratings.

Metrics Definition:

1. Descriptive: A descriptive description should provide vivid details about objects, people, and settings while maintaining a concise narrative flow. It should include essential information about the appearance of individuals, such as their clothing, facial expressions, and actions, and visual properties of objects, such as color and shape. For example, "A smiling woman, wearing a loose white dress, types on a laptop in a softly lit room."

2. Objective: An objective description should report what is visible without adding personal opinions or assumptions. For instance, describe two people as "a woman and a man" without adding any relationship context unless it is mentioned. It should also avoid guessing personal attributes like racial or gender identities unless explicitly clear.

3. Accurate: An accurate description should aim for precision in describing visible elements without imagination. It should ensure that all details, such as colors and spatial arrangements, are reported correctly. For instance, "Blue sky with white clouds" instead of "White sky with blue clouds" if that is what appears. Additionally, it should avoid adding unnecessary details that do not contribute to a deeper understanding of the scene.

4. Clear: A clear description should present information in the videos in a way that is easy to follow for blind and low vision individuals. It should describe the object or character's properties before the actions or relationships with other objects or characters. For example, "A man wearing sunglasses plays the piano." Pronouns should only be used when it is clear who they refer to, and the description should include any on-screen text if it is central to understanding. For instance, "A man in a black polo shirt is speaking. He is in a studio setting with a red couch and a blue background featuring the text 'Talk of the Town'".

Input:

Visual information:
<visual_info>
VISUAL_INFO_GOES_HERE_IF_AVAILABLE
</visual_info>

ASR transcript:
<asr_transcript>
ASR_TRANSCRIPT_GOES_HERE_IF_AVAILABLE
</asr_transcript>

Model-generated audio description:
<generated_desc>
GENERATED_DESC_GOES_HERE_IF_AVAILABLE
</generated_desc>

Output Format: Return the result as a string format dictionary with the following structure. Please output the fields in this exact order:

{
  "Summary": "Your brief summary of the video here",
  "Rationale": "Your rationale for the ratings here",
  "Descriptive": [Rating out of 5],
  "Objective": [Rating out of 5],
  "Accurate": [Rating out of 5],
  "Clear": [Rating out of 5]
}

"""

PROMPTS = {
    "video_asr": VIDEO_ASR_PROMPT,
    "video_only": VIDEO_ONLY_PROMPT,
    "video_visual_info": VIDEO_VISUAL_INFO_PROMPT,
    "video_visual_info_asr": VIDEO_VISUAL_INFO_ASR_PROMPT,
}

# Inputs each variant needs besides the video and the generated AD.
REQUIRED_INPUTS = {
    "video_asr": ("transcript",),
    "video_only": (),
    "video_visual_info": ("visual_info",),
    "video_visual_info_asr": ("visual_info", "transcript"),
}

_SCORE_KEYS = ("Descriptive", "Objective", "Accurate", "Clear")


def build_prompt(
    variant: str, generated_desc: str, transcript: str, visual_info: str
) -> str:
  """Fills a variant's prompt template.

  An empty transcript (silent video) or missing visual info is passed to the
  judge as NOT AVAILABLE.

  Args:
    variant: Key of PROMPTS.
    generated_desc: The generated AD.
    transcript: Timestamped ASR transcript ("" if none).
    visual_info: Dense visual information ("" if none).

  Returns:
    The filled prompt.
  """
  return (
      PROMPTS[variant]
      .replace(
          "VISUAL_INFO_GOES_HERE_IF_AVAILABLE", visual_info or NOT_AVAILABLE
      )
      .replace(
          "ASR_TRANSCRIPT_GOES_HERE_IF_AVAILABLE", transcript or NOT_AVAILABLE
      )
      .replace(
          "GENERATED_DESC_GOES_HERE_IF_AVAILABLE",
          generated_desc or NOT_AVAILABLE,
      )
  )


def parse_output(raw_output: str, variant: str) -> dict[str, Any] | None:
  """Parses the judge's JSON answer into prefixed metric keys, or None."""
  match = re.search(r"\{.*\}", raw_output or "", re.DOTALL)
  if not match:
    return None
  try:
    results = json.loads(match.group(0))
    scores = {f"{variant}_{k.lower()}": float(results[k]) for k in _SCORE_KEYS}
  except (json.JSONDecodeError, KeyError, TypeError, ValueError):
    return None
  return {
      f"{variant}_summary": results.get("Summary", ""),
      f"{variant}_rationale": results.get("Rationale", ""),
      **scores,
  }


def compute_reference_free_metric(
    read_or_generate_and_save: Callable[..., str | None],
    yt_id: str,
    variant: str,
    generated_desc: str,
    transcript: str,
    visual_info: str,
    metric_dir: str,
) -> dict[str, Any] | None:
  """Computes one Quality-RF variant for a video.

  Args:
    read_or_generate_and_save: Cached LLM-judge callable; called with
      no_url=False so the judge receives the video.
    yt_id: YouTube video ID.
    variant: Key of PROMPTS.
    generated_desc: The generated AD (canonical lines).
    transcript: Timestamped ASR transcript ("" for silent videos).
    visual_info: Dense visual information ("" if not needed/available).
    metric_dir: Cache directory for the judge's raw responses.

  Returns:
    The metric dict, or None if the judge call failed or could not be parsed.
  """
  prompt = build_prompt(variant, generated_desc, transcript, visual_info)
  raw_output = read_or_generate_and_save(
      yt_id=yt_id, prompt=prompt, dir=metric_dir, no_url=False
  )
  if raw_output is None:
    return None
  result = parse_output(raw_output, variant)
  if result is None:
    print(
        f"ERROR: Could not parse {variant} reference-free output for {yt_id}."
    )
  return result
