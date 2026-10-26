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

"""Evaluation metrics for inline audio descriptions."""

import json
import re

from common import ad_format
from evaluation import parsing_lib
import numpy as np

# Raw model output -> canonical AD lines (shared with TTS rendering).
normalize_ad_output = ad_format.normalize_ad_output


# Backward-compatibility alias.
def simple_ad_prompt_process(ad: str) -> str:
  """Alias for normalize_ad_output."""
  return normalize_ad_output(ad)


def compute_tts_overlap_metrics(tts_ad_ranges, transcript_ranges):
  """Computes overlap metrics using TTS-derived AD durations.

  Args:
    tts_ad_ranges: List of [start, end] ranges built from TTS durations.
    transcript_ranges: List of [start, end] ranges from the ASR transcript.

  Returns:
    Dict with tts_ad_self_overlap_seconds (consecutive AD bleed-over) and
    tts_ad_transcript_overlap_seconds (AD spoken during speech).
  """
  if not tts_ad_ranges:
    return {
        "tts_ad_self_overlap_seconds": 0.0,
        "tts_ad_transcript_overlap_seconds": 0.0,
    }
  # AD-AD self-overlap: time that a segment bleeds into the next segment's gap.
  self_overlap_seconds = 0.0
  for i in range(len(tts_ad_ranges) - 1):
    curr_end = tts_ad_ranges[i][1]
    next_start = tts_ad_ranges[i + 1][0]
    if curr_end > next_start:
      self_overlap_seconds += curr_end - next_start
  # AD-transcript overlap sampled at 0.1-second intervals.
  transcript_overlap_seconds = parsing_lib.total_overlap_discrete(
      tts_ad_ranges, transcript_ranges, interval=0.1
  )
  return {
      "tts_ad_self_overlap_seconds": self_overlap_seconds,
      "tts_ad_transcript_overlap_seconds": transcript_overlap_seconds,
  }


def calculate_avg_silence_duration(
    tts_ad_ranges, transcript_ranges, video_duration_seconds
):
  """Calculates the average silence left after placing AD and transcript.

  Merges all speech intervals (AD + transcript) and measures the gaps.

  Args:
    tts_ad_ranges: List of [start, end] AD ranges from TTS durations.
    transcript_ranges: List of [start, end] ranges from the ASR transcript.
    video_duration_seconds: Total video duration in seconds.

  Returns:
    Dict with total_silence_duration_w_ad_tts and avg_silence_duration_w_ad_tts,
    or None if there are no speech intervals.
  """
  speech_intervals_list = list(tts_ad_ranges) + list(transcript_ranges)
  if not speech_intervals_list:
    return None
  speech_intervals = parsing_lib.merge_intervals(speech_intervals_list)

  silence_durations = []
  last_end = 0.0
  for start, end in speech_intervals:
    if start > last_end:
      silence_durations.append(start - last_end)
    last_end = end
  if video_duration_seconds > last_end:
    silence_durations.append(video_duration_seconds - last_end)

  if not silence_durations:
    return {
        "total_silence_duration_w_ad_tts": 0.0,
        "avg_silence_duration_w_ad_tts": 0.0,
        "silence_durations_s": [],
    }
  total = sum(silence_durations)
  return {
      "total_silence_duration_w_ad_tts": total,
      "avg_silence_duration_w_ad_tts": total / len(silence_durations),
      "silence_durations_s": silence_durations,
  }


def interleave_transcript_and_ad(yt_id, ad_output_json, transcript_json):
  """Merges transcript and AD entries and sorts them by start time."""
  try:
    merged_list = ad_output_json + transcript_json
    sorted_list = sorted(merged_list, key=lambda x: x["start"])
  except (KeyError, TypeError) as e:
    print(f"Failed to interleave transcript and ad for {yt_id}: {e}")
    return None

  return sorted_list


def extract_text_on_screen(visual_info_text):
  """Extracts the lines under the 'Text on Screen' section of visual info."""
  if not visual_info_text:
    return []

  headers = [
      "Text on Screen and Appearance Timestamps",
      "Text on screen and when it appeared",
      "On-screen Text with Timestamps",
      "Text on Screen",
  ]
  header_regex = "|".join(re.escape(h) for h in headers)
  # The header must start its own line (optionally as a markdown heading or in
  # bold), so phrases like "...followed by a list of all text on screen..." in
  # the introduction are not mistaken for the section header.
  pattern = (
      rf"(?im)^[ \t]*(?:#+[ \t]*)?(?:\*\*)?[ \t]*(?:{header_regex})[ \t]*"
      r"(?:\([^)\n]*\))?[ \t]*(?::|\*\*|(?=\n)|$)[ \t]*(?:\*\*)?[ \t]*:?"
      r"([\s\S]*?)(?=\n[ \t]*(?:\*\*[A-Z]|#+[ \t]|---|\*\*\*)|\Z)"
  )
  match = re.search(pattern, visual_info_text)
  if not match:
    return []

  content = match.group(1).strip()
  lines = [
      line.strip().lstrip("*-").strip()
      for line in content.split("\n")
      if line.strip()
  ]

  parsed_list = []
  ts = r"\d{1,2}:\d{2}(?::\d{2,3})?(?:\.\d+)?"
  ts_prefix_re = re.compile(
      rf"^(?:\*\*)?\s*\[?{ts}\]?(?:\s*(?:--|[-–—]|to)\s*\[?{ts}\]?)?"
      r"\s*(?:\*\*)?\s*[:\-–—]?\s*(?:\*\*)?\s*"
  )
  for line in lines:
    cleaned = ts_prefix_re.sub("", line).strip()
    if not cleaned and "-" in line:
      cleaned = line.split("-", 1)[1].strip()
    cleaned = cleaned.strip("\"'")
    if not cleaned:
      continue
    lower_cleaned = cleaned.lower().rstrip(".")
    if lower_cleaned in (
        "none",
        "n/a",
        "no text on screen",
        "no on-screen text",
    ):
      continue
    parsed_list.append(cleaned)
  return parsed_list


def count_missing_onscreen_text_words(on_screen_text, ad_json):
  """Counts words in on-screen text not present in audio descriptions.

  Args:
    on_screen_text: A list of on-screen text strings.
    ad_json: A list of audio description/transcript dictionaries.

  Returns:
    The ratio of missed words to total words in on-screen text.
  """
  ad_words = set()
  if ad_json:
    for item in ad_json:
      text = item.get("text", "").lower()
      ad_words.update(re.findall(r"\b\w+\b", text))

  total_word_count = 0
  missed_word_count = 0

  for sentence in on_screen_text:
    words_in_sentence = re.findall(r"\b\w+\b", sentence)
    total_word_count += len(words_in_sentence)
    for word in words_in_sentence:
      if word.lower() not in ad_words:
        missed_word_count += 1

  print(
      "Missed words counted from on-screen text sentences not in"
      f" descriptions: {missed_word_count}"
  )
  print(
      f"Total words counted from on-screen text sentences: {total_word_count}"
  )
  return missed_word_count / (total_word_count + 0.01)


# Backward-compatibility alias.
count_words_if_sentence_not_found = count_missing_onscreen_text_words


def metric_1_2_narrates_onscreen_text(
    interleaved_transcript_ad_json, visual_info_str
):
  """Computes word-level recall of on-screen text in AD and dialogue."""
  on_screen_text_list = extract_text_on_screen(visual_info_str)
  missed_words_ratio = count_missing_onscreen_text_words(
      on_screen_text_list, interleaved_transcript_ad_json
  )
  metric_1_2_value = missed_words_ratio
  print(f"Missed words ratio: {missed_words_ratio}")

  return {
      "onscreen_text_present": 1 if on_screen_text_list else 0,
      "metric_1_2_narrates_onscreen_text": 1 - metric_1_2_value,
  }


def metric_compute_timeliness(
    read_or_generate_and_save,
    yt_id,
    visual_info,
    audio_descriptions,
    dir="",  # pylint: disable=redefined-builtin
):
  """Computes temporal alignment between AD segments and visual events."""

  def make_timeliness_prompt(visual_info, audio_descriptions):
    return f"""

    For each audio description item, identify the corresponding
    visual information if one exists, and report the time difference (in secs):

    [audio description time] audio description
    [visual information time] visual information
    **[audio description time - visual info time]**

    The response should not contain any thing else.

    For example:
    -----
    [00:03:01.183] Ants crawl on banana pieces.
    [00:03:12.257] Ants crawl on pieces of banana on rocks.
    **[11]**

    [199] A light brown cockroach hangs from a screen, shedding its exoskeleton.
    No audio description.
    **[N/A]**
    -----

    Here are the audio descriptions and visual information:

    Audio descriptions (timestamp isin hh:mm:ss.ms when the audio description starts in the video):
    {audio_descriptions}

    Visual information (timestamp isin hh:mm:ss.ms indicate when the visual appeared in the video):
    {visual_info}

    """

  timeliness_prompt = make_timeliness_prompt(visual_info, audio_descriptions)
  final_timeliness_report = read_or_generate_and_save(
      yt_id, prompt=timeliness_prompt, dir=dir, no_url=True
  )

  def parse_timeliness_report(timeliness_report):
    diff_lines = [
        l.split("**[")[1].split("]**")[0]
        for l in timeliness_report.split("\n")
        if "**" in l
    ]
    rel_diff_lines = [
        abs(float(re.findall(r"\d+", l)[0])) for l in diff_lines if l != "N/A"
    ]
    if not diff_lines:
      # No timeliness lines found in the LLM response at all.
      return None, None, None, None
    matched_visual_info_ratio = len(rel_diff_lines) / len(diff_lines)
    if not rel_diff_lines:
      # Every entry was N/A — no visual info could be matched to any AD.
      return matched_visual_info_ratio, None, 0, None
    average_diff_time = float(sum(rel_diff_lines)) / len(rel_diff_lines)
    diffs_over_ten = len([l for l in rel_diff_lines if l > 10])
    return (
        matched_visual_info_ratio,
        average_diff_time,
        diffs_over_ten,
        float(max(rel_diff_lines)),
    )

  try:
    matched_visual_info_ratio, average_diff_time, diffs_over_ten, max_diff = (
        parse_timeliness_report(final_timeliness_report)
    )
    if matched_visual_info_ratio is None:
      print(f"No timeliness data found in LLM response for {yt_id}")
      return None
  except (AttributeError, IndexError, TypeError, ValueError) as e:
    print(f"Error in parsing timeliness report: {e}")
    return None

  output_dict = {
      "metric_g_2_avg_diff_time": average_diff_time,
      "metric_g_2_max_diff_time": max_diff,
      "metric_g_2_matched_visual_info_ratio": matched_visual_info_ratio,
      "metric_g_2_diffs_over_ten": diffs_over_ten,
  }
  print(output_dict)
  return output_dict


def metric_2_1_does_not_restate(
    read_or_generate_and_save,
    yt_id,
    interleaved_transcript_ad_str,
    dir="",  # pylint: disable=redefined-builtin
):
  """Computes redundancy score measuring whether AD avoids repeating speech."""

  metric_2_1_prompt = """I will provide the video transcript [TRANSCRIPT] and audio description [AD]
  interleaved, such that they are in order as they would be played in the video.

  Then, for each audio description [AD], check if the
  audio description repeats the nearby transcript [TRANSCRIPT] or would otherwise be obvious
  from the audio alone (e.g., describes offscreen door slam). Score each AD between 0
  and 1 where no repeat is 0, slight repeat is 0.5, and egregious repeat is 1.0. The purpose is to remove
  unnecessary redundancy with the audio track from the audio description. Then, return a list of
  these assigned numbers e.g., [0, 0, 1.0, 0, 0, 0.5, 0.6]. ONLY RETURN THE LIST."""

  metric_2_1_prompt += (
      "\nHere is the video transcript and audio description"
      f" interleaved:\n{interleaved_transcript_ad_str}"
  )
  # print(interleaved_transcript_ad_str)
  metric_2_1_output = read_or_generate_and_save(
      yt_id=yt_id, prompt=metric_2_1_prompt, dir=dir, no_url=True
  )  # RESTATE_AD_DIR, no_url=True)
  # print(metric_2_1_output)

  if metric_2_1_output is None:
    return None
  # Use regex to robustly extract the list, even if the LLM adds preamble text
  # or wraps the output in a markdown code fence before the '[' character.
  match = re.search(r"\[([0-9., ]+)\]", metric_2_1_output)
  if not match:
    print(
        f"Warning: Could not parse list from metric_2_1 output for {yt_id}:"
        f" {metric_2_1_output!r}"
    )
    return None
  metric_2_1_value = [
      float(n.strip()) for n in match.group(1).split(",") if n.strip()
  ]
  if not metric_2_1_value:
    return None
  avg_metric_2_1_value = float(np.mean(metric_2_1_value))

  return {"metric_2_1_does_not_restate": 1 - avg_metric_2_1_value}


def compute_videoa11y_metric(
    read_or_generate_and_save,
    yt_id: str,
    ground_truth_desc: str,
    generated_desc: str,
    videoa11y_metric_dir: str,
):
  """Computes videoa11y metrics for audio descriptions."""
  prompt = """
    You are an expert evaluator with a deep understanding of video description quality, particularly for making content accessible to blind and low vision (BLV) individuals.

    Your role is to assess and rate video descriptions based on specific metrics.

    Task: I have two video descriptions: one is the ground truth, and the other is generated by a model. Additionally, I have four evaluation metrics: Descriptive, Objective, Accurate, and Clear.

    Please evaluate the model-generated description using the above metrics. Provide a rating from 1 to 5 for each metric, and briefly explain the reasons for each rating. Metrics Definition:

    1. Descriptive: A descriptive description should provide vivid details about objects, people, and settings while maintaining a concise narrative flow. It should include essential information about the appearance of individuals, such as their clothing, facial expressions, and actions, and visual properties of objects, such as color and shape. For example, "A smiling woman, wearing a loose white dress, types on a laptop in a softly lit room."

    2. Objective: An objective description should report what is visible without adding personal opinions or assumptions. For instance, describe two people as "a woman and a man" without adding any relationship context unless it is mentioned. It should also avoid guessing personal attributes like racial or gender identities unless explicitly clear.

    3. Accurate: An accurate description should aim for precision in describing visible elements without imagination. It should ensure that all details, such as colors and spatial arrangements, are reported correctly. For instance, "Blue sky with white clouds" instead of "White sky with blue clouds" if that is what appears. Additionally, it should avoid adding unnecessary details that do not contribute to a deeper understanding of the scene.

    4. Clear: A clear description should present information in the videos in a way that is easy to follow for blind and low vision individuals. It should describe the object or character's properties before the actions or relationships with other objects or characters. For example, "A man wearing sunglasses plays the piano." Pronouns should only be used when it is clear who they refer to, and the description should include any on-screen text if it is central to understanding. For instance, "A man in a black polo shirt is speaking. He is in a studio setting with a red couch and a blue background featuring the text 'Talk of the Town'".

    Input: -

    Ground truth video description:
    <ground_truth_desc>
    GROUND_TRUTH_DESC_GOES_HERE_IF_AVAILABLE
    </ground_truth_desc>

    Model-generated video description:
    <generated_desc>
    GENERATED_DESC_GOES_HERE_IF_AVAILABLE
    </generated_desc>

    Output Format: Return the result as a string format dictionary with the following structure:

    {"Descriptive": [Rating out of 5], "Objective": [Rating out of 5], "Accurate": [Rating out of 5], "Clear": [Rating out of 5], "Reason": "Your brief explanation here"}

  """
  prompt = prompt.replace(
      "GROUND_TRUTH_DESC_GOES_HERE_IF_AVAILABLE", ground_truth_desc
  )
  prompt = prompt.replace(
      "GENERATED_DESC_GOES_HERE_IF_AVAILABLE", generated_desc
  )

  videoa11y_metric_output = read_or_generate_and_save(
      yt_id=yt_id,
      prompt=prompt,
      dir=videoa11y_metric_dir,
      no_url=True,
  )

  if videoa11y_metric_output is None:
    return None

  try:
    # The output might be in a markdown block or have extra text.
    # Find the JSON object within the string.
    match = re.search(r"\{.*\}", videoa11y_metric_output, re.DOTALL)
    if not match:
      print(f"Could not find JSON in videoa11y metric output for {yt_id}")
      return None

    json_str = match.group(0)
    metric_results = json.loads(json_str)

    output_dict = {
        "videoa11y_descriptive": float(metric_results["Descriptive"]),
        "videoa11y_objective": float(metric_results["Objective"]),
        "videoa11y_accurate": float(metric_results["Accurate"]),
        "videoa11y_clear": float(metric_results["Clear"]),
        "videoa11y_reason": metric_results["Reason"],
    }
    return output_dict
  except (
      json.JSONDecodeError,
      KeyError,
      IndexError,
      TypeError,
      ValueError,
  ) as e:
    print(f"Error parsing videoa11y metric output for {yt_id}: {e}")
    return None


# Re-export compute_precision_visual_info so callers can access it via metrics.
compute_precision_visual_info = parsing_lib.compute_precision_visual_info
