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

"""Helpers for parsing timestamps, transcripts, and audio descriptions."""

import datetime
import json
import re
import string
import numpy as np


def parse_transcript_to_json_custom(transcript_string, speaker_id="TRANSCRIPT"):
  """Parses a timestamped transcript string into a list of dicts."""
  output_list = []
  lines = [
      line for line in transcript_string.strip().split("\n") if line.strip()
  ]

  for line in lines:
    # Match H:MM:SS.mmm or M:SS.mmm (Gemini sometimes omits the hours component)
    match = re.match(
        r"(\d{1,2}:\d{2}:\d{2}\.\d+|\d{1,2}:\d{2}\.\d+)--"
        r"(\d{1,2}:\d{2}:\d{2}\.\d+|\d{1,2}:\d{2}\.\d+)\s*(.*)",
        line,
    )
    if match:
      start_ts_str = match.group(1)
      end_ts_str = match.group(2)
      text = match.group(3).strip()

      try:
        start_time_seconds = convert_timestamp_to_seconds(start_ts_str)

        output_list.append({
            "endTimestamp": end_ts_str,
            "speaker": speaker_id,
            "startTimestamp": start_ts_str,
            "text": text,
            "start": start_time_seconds,
        })
      except (ValueError, IndexError) as e:
        print(
            f"Error parsing timestamp in line '{line}': {e}. Skipping this"
            " entry."
        )
    else:
      print(f"Warning: Skipping line with unexpected format: {line}")

  return json.dumps(output_list, indent=2)


def ad_string_to_json(s, speaker_id="AD"):
  """Parses a string of audio description lines into a clean JSON array."""
  # --- 1. Robustly parse the input string to get the list of ad lines ---
  # Clean up the markdown code block fences if they exist.
  clean_s = s.strip()
  if clean_s.startswith("```json"):
    clean_s = clean_s[7:].strip()
  if clean_s.endswith("```"):
    clean_s = clean_s[:-3].strip()

  lines = []
  try:
    # Attempt to load the string as a JSON object.
    data = json.loads(clean_s)
    lines = data.get("response", [])
    if not isinstance(lines, list):
      print("Warning: 'response' key exists but does not contain a list.")
      return json.dumps([], indent=2)
  except json.JSONDecodeError:
    # If JSON parsing fails, treat the input as a block of plain text lines.
    lines = [line for line in clean_s.split("\n") if line.strip()]

  # --- 2. Define helper functions for parsing and formatting time ---
  def parse_timestamp_to_timedelta(ts_str):
    """Parses timestamp strings into timedelta objects."""
    # Normalize ambiguous format "M:SS:ms" to "M:SS.ms".
    if ts_str.count(":") >= 2 and "." not in ts_str:
      parts = ts_str.rsplit(":", 1)
      ts_str = ".".join(parts)

    parts = ts_str.split(":")

    hours = 0
    minutes = 0
    seconds = 0
    milliseconds = 0

    if len(parts) > 3:
      raise ValueError(
          f"Unexpected timestamp format with too many colons: {ts_str}"
      )

    # Parse seconds and milliseconds from the last part of the timestamp.
    last_part = parts[-1]
    if "." in last_part:
      sec_ms = last_part.split(".")
      seconds = int(sec_ms[0])
      # Pad with zeros to handle cases like '5' -> 500ms
      ms_str = sec_ms[1].ljust(3, "0")
      milliseconds = int(ms_str[:3])  # Truncate to 3 digits just in case
    else:
      seconds = int(last_part)

    # Parse hours and minutes based on the number of parts.
    if len(parts) == 3:  # H:MM:SS format
      hours = int(parts[0])
      minutes = int(parts[1])
    elif len(parts) == 2:  # M:SS format
      minutes = int(parts[0])

    return datetime.timedelta(
        hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds
    )

  def format_timedelta_to_mm_ss_ms(td):
    """Formats a timedelta object into a MM:SS.ms string."""
    total_seconds = td.total_seconds()
    minutes, seconds_rem = divmod(total_seconds, 60)
    seconds = int(seconds_rem)
    milliseconds = int((seconds_rem - seconds) * 1000)
    return f"{int(minutes):02d}:{seconds:02d}.{milliseconds:03d}"

  # --- 3. Process each line to build a list of timed descriptions ---
  timestamps_and_descriptions = []
  for line in lines:
    parts = line.split("-", 1)
    if len(parts) == 2:
      timestamp_str = parts[0].strip().strip(string.punctuation)
      description = parts[1].strip()
      # Remove the "STANDARD " prefix if it exists.
      if description.startswith("STANDARD "):
        description = description[len("STANDARD ") :]

      try:
        timedelta_obj = parse_timestamp_to_timedelta(timestamp_str)
        timestamps_and_descriptions.append((timedelta_obj, description))
      except (ValueError, IndexError) as e:
        print(
            f"Error parsing timestamp '{timestamp_str}': {e}. Skipping this"
            " entry."
        )
    else:
      print(f"Warning: Skipping line with unexpected format: {line}")

  # Sort by timestamp to ensure correct order for calculating end times.
  timestamps_and_descriptions.sort(key=lambda x: x[0])

  # --- 4. Format the final output list ---
  output_list = []
  for i in range(len(timestamps_and_descriptions)):
    start_time_timedelta, description_text = timestamps_and_descriptions[i]
    start_timestamp_str = format_timedelta_to_mm_ss_ms(start_time_timedelta)

    # The end time of one description is the start time of the next.
    if i < len(timestamps_and_descriptions) - 1:
      end_time_timedelta = timestamps_and_descriptions[i + 1][0]
    else:
      # For the last description, add a default duration (e.g., 5 seconds).
      end_time_timedelta = start_time_timedelta + datetime.timedelta(seconds=5)

    end_timestamp_str = format_timedelta_to_mm_ss_ms(end_time_timedelta)
    start_time_seconds = start_time_timedelta.total_seconds()

    output_list.append({
        "endTimestamp": end_timestamp_str,
        "speaker": speaker_id,
        "startTimestamp": start_timestamp_str,
        "text": description_text,
        "start": start_time_seconds,
    })

  return json.dumps(output_list, indent=2)


def compute_precision_recall(
    read_or_generate_and_save, yt_id, ad_output, gt_output, trials=1
):
  """Computes precision and recall between generated AD and ground-truth AD."""

  ad_vs_gt_prompt = (
      """
  For each line in the first AD output,
  score if it is covered in the second AD output (0 - not covered, 100 - fully covered).

  Then, output only an array of all the scores like [0,50,100,0,0] """
      + "\n\nFirst AD Output:\n"
      + ad_output
      + "\n\nSecond AD Output:\n"
      + gt_output
  )

  gt_vs_ad_prompt = (
      """
  For each line in the first AD output,
  score if it is covered in the second AD output (0 - not covered, 100 - fully covered).

  Then, output only an array of all the scores like [0,50,100,0,0] """
      + "\n\nFirst AD Output:\n"
      + gt_output
      + "\n\nSecond AD Output:\n"
      + ad_output
  )

  all_ad_vs_gt_scores = []
  all_gt_vs_ad_scores = []
  for trial in range(trials):
    ad_vs_gt_output = read_or_generate_and_save(
        yt_id=yt_id, prompt=ad_vs_gt_prompt, dir=""
    )
    gt_vs_ad_output = read_or_generate_and_save(
        yt_id=yt_id, prompt=gt_vs_ad_prompt, dir=""
    )
    # Use regex to robustly extract the list from LLM output.
    ad_match = re.search(r"\[([0-9., ]+)\]", ad_vs_gt_output)
    gt_match = re.search(r"\[([0-9., ]+)\]", gt_vs_ad_output)
    if not ad_match or not gt_match:
      print(f"Warning: Could not parse scores for {yt_id} in trial {trial}")
      continue
    ad_vs_gt_scores = [
        float(n) for n in ad_match.group(1).split(",") if n.strip()
    ]
    gt_vs_ad_scores = [
        float(n) for n in gt_match.group(1).split(",") if n.strip()
    ]
    print(ad_vs_gt_scores)
    print(gt_vs_ad_scores)

    avg_ad_vs_gt_score = sum(ad_vs_gt_scores) / len(ad_vs_gt_scores)
    avg_gt_vs_ad_score = sum(gt_vs_ad_scores) / len(gt_vs_ad_scores)
    all_ad_vs_gt_scores.append(avg_ad_vs_gt_score)
    all_gt_vs_ad_scores.append(avg_gt_vs_ad_score)

    print(f"Trial {trial + 1}:")
    print(f"Average AD vs GT Score (Precision): {avg_ad_vs_gt_score}")
    print(f"Average GT vs AD Score (Recall): {avg_gt_vs_ad_score}")

  if not all_ad_vs_gt_scores:
    return None, None, None, None

  overall_avg_ad_vs_gt_score = sum(all_ad_vs_gt_scores) / len(
      all_ad_vs_gt_scores
  )
  overall_avg_gt_vs_ad_score = sum(all_gt_vs_ad_scores) / len(
      all_gt_vs_ad_scores
  )

  overall_std_ad_vs_gt_score = np.std(all_ad_vs_gt_scores)
  overall_std_gt_vs_ad_score = np.std(all_gt_vs_ad_scores)

  print(
      "Overall Average AD vs GT Score (Precision):"
      f" {overall_avg_ad_vs_gt_score} (sd={overall_std_ad_vs_gt_score})"
  )
  print(
      f"Overall Average GT vs AD Score (Recall): {overall_avg_gt_vs_ad_score}"
      f" (sd={overall_std_gt_vs_ad_score})"
  )

  return (
      overall_avg_ad_vs_gt_score,
      overall_std_ad_vs_gt_score,
      overall_avg_gt_vs_ad_score,
      overall_std_gt_vs_ad_score,
  )


def compute_precision_visual_info(
    read_or_generate_and_save,
    yt_id,
    ad_output,
    visual_info,
    dir="",  # pylint: disable=redefined-builtin
):
  """Computes Prioritization (G1.1): AD coverage of the extracted visual info.

  A single LLM-judge call scores each AD segment (0 = not covered, 1 = fully
  covered, partial values such as 0.5 allowed); the metric is the mean score.

  Args:
    read_or_generate_and_save: Cached LLM-judge callable.
    yt_id: Video ID.
    ad_output: AD text, one segment per line.
    visual_info: Dense visual information of the video.
    dir: Cache directory for the judge response.

  Returns:
    {"metric_G1.1_precision_avg_ad_in_visual_info": mean score or None}.
  """
  ad_in_visual_info_prompt = (
      """
  You are evaluating if audio descriptions (AD) are covering visual information
  necessary to understand the video.
  For each line in the first AD output,
  score if it is covered in the visual information (0 - not covered, 1 - fully covered).

  Then, output only an array of all the scores like [0,0.5,1.0,1.0,0] """
      + "\n\nFirst AD Output:\n"
      + ad_output
      + "\n\nSecond  visual information of the video:\n"
      + visual_info
  )
  failed = {"metric_G1.1_precision_avg_ad_in_visual_info": None}

  judge_output = read_or_generate_and_save(
      yt_id=yt_id, prompt=ad_in_visual_info_prompt, dir=dir
  )
  if judge_output is None:
    return failed
  # Use regex to robustly extract the list from LLM output.
  match = re.search(r"\[([0-9., ]+)\]", judge_output)
  if not match:
    print(
        "Warning: Could not parse scores from precision output for"
        f" {yt_id}: {judge_output!r}"
    )
    return failed
  scores = [float(n) for n in match.group(1).split(",") if n.strip()]
  print(scores)
  if not scores:
    return failed

  output_dict = {
      "metric_G1.1_precision_avg_ad_in_visual_info": sum(scores) / len(scores),
  }
  print(output_dict)
  return output_dict


def parse_timestamp(timestamp_str):
  """Parses an 'HH:MM.sss' or 'MM:SS.sss' timestamp string into seconds.

  Args:
    timestamp_str: The timestamp string.

  Returns:
    The timestamp in seconds as a float, or None if parsing fails.
  """
  try:
    if "." in timestamp_str:
      parts = timestamp_str.split(":")
      if len(parts) == 2:
        minutes = int(parts[0])
        seconds_parts = parts[1].split(".")
        seconds = int(seconds_parts[0])
        milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
        return float(minutes * 60 + seconds) + float(milliseconds) / 1000
      elif len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_parts = parts[2].split(".")
        seconds = int(seconds_parts[0])
        milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
        return (
            float(hours * 3600 + minutes * 60 + seconds)
            + float(milliseconds) / 1000
        )
    else:
      print(
          f"Warning: Timestamp string '{timestamp_str}' does not contain '.' or"
          " ':' for expected format."
      )
      return None
  except (ValueError, IndexError):
    print(f"Warning: Could not parse timestamp string: {timestamp_str}")
    return None


def count_words(text):
  """Counts the number of words in a given text string."""
  # Use regex to find sequences of word characters
  words = re.findall(r"\b\w+\b", text)
  return len(words)


def extract_timestamps(text):
  """Extracts a list of start and end timestamp ranges from a given text.

  Args:
      text (str): A string containing timestamps in the format
        HH:MM:SS.mmm--HH:MM:SS.mmm.

  Returns:
      list: A list of lists, where each inner list contains a start and end
      timestamp string.
  """
  pattern = (
      r"(\d{1,2}:\d{2}:\d{2}\.\d+|\d{1,2}:\d{2}\.\d+)--"
      r"(\d{1,2}:\d{2}:\d{2}\.\d+|\d{1,2}:\d{2}\.\d+)"
  )
  timestamps = re.findall(pattern, text)
  return [list(t) for t in timestamps]


def convert_timestamp_to_seconds(timestamp):
  """Converts a timestamp string to a floating-point value in seconds.

  Handles both H:MM:SS.mmm (or HH:MM:SS.mmm) and M:SS.mmm formats.

  Args:
      timestamp (str): The timestamp string to convert.

  Returns:
      float: The total time in seconds.
  """
  parts = re.split(r"[:\.]", timestamp)
  if len(parts) == 4:
    # H:MM:SS.mmm
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    milliseconds = int(parts[3])
  elif len(parts) == 3:
    # M:SS.mmm  (no hours)
    hours = 0
    minutes = int(parts[0])
    seconds = int(parts[1])
    milliseconds = int(parts[2])
  else:
    raise ValueError(f"Unrecognised timestamp format: {timestamp!r}")

  return (hours * 3600) + (minutes * 60) + seconds + (milliseconds / 1000)


def convert_timestamps_to_seconds_ranges(timestamp_ranges):
  """Converts a list of timestamp ranges from HH:MM:SS.mmm to seconds.ms.

  Args:
      timestamp_ranges (list): A list of lists, where each inner list contains a
        start and end timestamp string.

  Returns:
      list: A new list of lists with converted floating-point values.
  """
  converted_ranges = []
  for start, end in timestamp_ranges:
    start_secs = convert_timestamp_to_seconds(start)
    end_secs = convert_timestamp_to_seconds(end)
    converted_ranges.append([start_secs, end_secs])
  return converted_ranges


def get_time_ranges_from_transcript_timestamps(transcript_text):
  """Extracts start and end time ranges from a list of transcript lines."""
  extracted_timestamps = extract_timestamps(transcript_text)
  converted_timestamps = convert_timestamps_to_seconds_ranges(
      extracted_timestamps
  )
  return converted_timestamps


def get_time_ranges_from_list(json_list_data, wpm=None):
  """Extracts start and end time ranges from a list of dictionaries.

  Can use WPM to estimate end times if provided.

  Args:
    json_list_data: A list of dictionaries, where each dictionary is expected to
      have 'startTimestamp' and 'text' keys. 'endTimestamp' is used if wpm is
      not provided.
    wpm: Optional. Words Per Minute to estimate end times. If None or <= 0,
      'endTimestamp' from the data is used.

  Returns:
    A list of lists, where each inner list represents a start and end time range
    in seconds, like [[start1, end1], [start2, end2], ...]. Returns an empty
    list if the input data is not a list or is empty.
  """
  ranges = []
  if not isinstance(json_list_data, list):
    print("Warning: Input data is not a list.")
    return ranges

  for entry in json_list_data:
    if (
        isinstance(entry, dict)
        and "startTimestamp" in entry
        and "text" in entry
    ):
      start_time_str = entry["startTimestamp"]
      start_time = parse_timestamp(start_time_str)

      if start_time is not None:
        end_time = None
        if wpm is not None and wpm > 0:
          # Calculate estimated end time based on WPM
          text = entry.get("text", "")  # Get text, default to empty string
          num_words = count_words(text)
          # Time per word in seconds
          seconds_per_word = 60 / wpm if wpm > 0 else 0
          estimated_duration = num_words * seconds_per_word
          end_time = start_time + estimated_duration
        elif "endTimestamp" in entry:
          # Use the endTimestamp from the data if WPM is not used
          end_time_str = entry["endTimestamp"]
          end_time = parse_timestamp(end_time_str)

        if end_time is not None:
          ranges.append([start_time, end_time])
        else:
          print(f"Warning: Could not determine end time for entry: {entry}")
      else:
        print(
            "Warning: Skipping entry due to invalid start timestamp format:"
            f" {entry}"
        )
    else:
      print(f"Warning: Skipping invalid list entry: {entry}")

  return ranges


def is_overlapping(time, ranges):
  """Checks if a given time point falls within any of the ranges in a list.

  Args:
    time: The time point (in seconds) to check.
    ranges: A list of time ranges, e.g., [[start1, end1], ...].

  Returns:
    True if the time point is within any range, False otherwise.
  """
  for start, end in ranges:
    if (
        start <= time < end
    ):  # Use < for end time to avoid double counting at boundaries
      return True
  return False


def total_overlap_discrete(list1_ranges, list2_ranges, interval=0.1):
  """Calculates total overlap in seconds by sampling at discrete intervals.

  Args:
    list1_ranges: The first list of time ranges.
    list2_ranges: The second list of time ranges.
    interval: The size of the time interval to check for overlap (in seconds).

  Returns:
    The total duration of overlap in seconds as a float.
  """
  overlap_duration = 0.0

  # Find the overall minimum start time and maximum end time across both lists
  all_starts = [r[0] for r in list1_ranges + list2_ranges]
  all_ends = [r[1] for r in list1_ranges + list2_ranges]

  if not all_starts or not all_ends:
    return 0.0  # Handle empty input lists

  min_time = min(all_starts)
  max_time = max(all_ends)

  # Iterate through time points at the specified interval
  current_time = min_time
  while current_time < max_time:
    # Check if the current time point is within a range in both lists
    if is_overlapping(current_time, list1_ranges) and is_overlapping(
        current_time, list2_ranges
    ):
      overlap_duration += interval

    current_time += interval

  return overlap_duration


def total_overlap(list1_ranges, list2_ranges):
  """Analytically computes total overlap between two lists of time ranges.

  Merges each list of ranges first, then intersects the merged sets to find
  exact overlap intervals.

  Args:
    list1_ranges: A list of [start, end] time ranges (in seconds).
    list2_ranges: A list of [start, end] time ranges (in seconds).

  Returns:
    A dict with:
      'total_overlap_length': total seconds of overlap (float).
      'overlap_intervals': list of [start, end] overlap segments.
  """
  if not list1_ranges or not list2_ranges:
    return {"total_overlap_length": 0.0, "overlap_intervals": []}

  merged1 = merge_intervals(list(list1_ranges))
  merged2 = merge_intervals(list(list2_ranges))

  overlap_intervals = []
  i, j = 0, 0
  while i < len(merged1) and j < len(merged2):
    start = max(merged1[i][0], merged2[j][0])
    end = min(merged1[i][1], merged2[j][1])
    if start < end:
      overlap_intervals.append([start, end])
    # Advance the pointer whose interval ends first
    if merged1[i][1] < merged2[j][1]:
      i += 1
    else:
      j += 1

  total_length = sum(e - s for s, e in overlap_intervals)
  return {
      "total_overlap_length": total_length,
      "overlap_intervals": overlap_intervals,
  }


def parse_duration_to_seconds(duration_str):
  """Parses a duration string to total seconds.

  Handles the following formats:
    - SS[.SSS]       (e.g., "266" or "266.5"; plain seconds)
    - MM:SS          (e.g., "04:26")
    - MM:SS.SSS      (e.g., "04:26.547")
    - HH:MM:SS       (e.g., "1:04:26")
    - HH:MM:SS.SSS   (e.g., "1:04:26.547")

  Args:
    duration_str: The duration string to parse.

  Returns:
    float: The total duration in seconds, or None if the format is invalid.
  """
  if not duration_str:
    return None
  try:
    parts = str(duration_str).split(":")
    if len(parts) == 1:
      return float(parts[0])
    if len(parts) == 2:
      # MM:SS or MM:SS.SSS
      minutes = int(parts[0])
      seconds_parts = parts[1].split(".")
      seconds = int(seconds_parts[0])
      milliseconds = int(seconds_parts[1]) if len(seconds_parts) == 2 else 0
      return float(minutes * 60 + seconds) + milliseconds / 1000.0
    elif len(parts) == 3:
      # HH:MM:SS or HH:MM:SS.SSS
      hours = int(parts[0])
      minutes = int(parts[1])
      seconds_parts = parts[2].split(".")
      seconds = int(seconds_parts[0])
      milliseconds = int(seconds_parts[1]) if len(seconds_parts) == 2 else 0
      return (
          float(hours * 3600 + minutes * 60 + seconds) + milliseconds / 1000.0
      )
  except (ValueError, IndexError):
    pass
  return None  # Return None for any other invalid format


def merge_intervals(intervals):
  """Merges overlapping or adjacent intervals.

  Args:
    intervals: A list of [start, end] pairs (in seconds).

  Returns:
    A new sorted list of non-overlapping [start, end] pairs.
  """
  if not intervals:
    return []
  sorted_intervals = sorted(intervals, key=lambda x: x[0])
  merged = []
  for interval in sorted_intervals:
    if not merged or interval[0] > merged[-1][1]:
      merged.append(list(interval))
    else:
      merged[-1][1] = max(merged[-1][1], interval[1])
  return merged


def get_time_ranges_from_tts_json(tts_info):
  """Extracts [start, end] time ranges from TTS metadata.

  Accepts two formats:
    - A list of dicts with 'start' and 'duration_tts' keys, as written by
      generate_tts.py.
    - A dict mapping segment index to a dict with 'start_secs' and
      'duration_secs' keys.

  The end time is computed as start + duration_tts, the spoken duration of
  each AD segment.

  Args:
    tts_info: Either a list of segment dicts or a dict mapping segment index
      (str) to a segment dict.

  Returns:
    List of [start, end] ranges in seconds, sorted by start time. Segments
    without valid start or duration values are skipped.
  """
  ranges = []
  segments = tts_info if isinstance(tts_info, list) else tts_info.values()
  for seg in segments:
    # Use 'start'/'duration_tts', or 'start_secs'/'duration_secs' if absent.
    start = (
        seg.get("start")
        if seg.get("start") is not None
        else seg.get("start_secs")
    )
    duration = (
        seg.get("duration_tts")
        if seg.get("duration_tts") is not None
        else seg.get("duration_secs")
    )
    if start is not None and duration is not None:
      ranges.append([start, start + duration])
  ranges.sort(key=lambda x: x[0])
  return ranges


def format_seconds_to_timestamp(seconds):
  """Formats seconds as HH:MM:SS.mmm, or MM:SS.mmm if hours == 0."""
  if seconds is None or seconds < 0:
    seconds = 0.0
  hours = int(seconds // 3600)
  minutes = int((seconds % 3600) // 60)
  secs = int(seconds % 60)
  millis = int(round((seconds - int(seconds)) * 1000))
  if millis >= 1000:
    secs += 1
    millis -= 1000
    if secs >= 60:
      minutes += 1
      secs -= 60
      if minutes >= 60:
        hours += 1
        minutes -= 60
  if hours > 0:
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
  return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def enrich_ad_json_with_tts(ad_output_json, tts_info):
  """Adds TTS timing and file metadata to parsed AD JSON entries in place."""
  if not tts_info or not isinstance(tts_info, list):
    return ad_output_json
  for i, seg in enumerate(ad_output_json):
    if i < len(tts_info):
      t_info = tts_info[i]
      seg["planned_endTimestamp"] = seg.get("endTimestamp", "")
      if "end" in seg:
        seg["planned_end"] = seg["end"]
      start_val = t_info.get("start")
      if start_val is None:
        start_val = seg.get("start", 0.0)
      dur_val = t_info.get("duration_tts")
      if dur_val is None:
        dur_val = t_info.get("duration_secs")
      if dur_val is not None:
        end_val = start_val + dur_val
        seg["start"] = start_val
        seg["duration_tts"] = dur_val
        seg["end_tts"] = end_val
        seg["endTimestamp"] = format_seconds_to_timestamp(end_val)
      if "hashcode" in t_info:
        seg["hashcode"] = t_info["hashcode"]
      if "wav_path" in t_info:
        seg["wav_path"] = t_info["wav_path"]
      if "wav_filename" in t_info:
        seg["wav_filename"] = t_info["wav_filename"]
  return ad_output_json


def estimate_video_duration_seconds(
    explicit_duration=None,
    gt_output_json=None,
    transcript_ranges=None,
    ad_ranges=None,
    ad_output_json=None,
):
  """Returns the video duration in seconds.

  An explicit duration (seconds or a timestamp string, e.g. from the bundled
  metadata/video_durations.json) is used as-is. Otherwise the duration is
  estimated as the latest timestamp in the GT AD, transcript, or AD.
  """
  duration_secs = None
  if isinstance(explicit_duration, (int, float)) and not np.isnan(
      explicit_duration
  ):
    if explicit_duration > 0:
      duration_secs = float(explicit_duration)
  elif isinstance(explicit_duration, str) and explicit_duration.strip():
    duration_secs = parse_duration_to_seconds(explicit_duration.strip())

  if duration_secs is not None and duration_secs > 0:
    return duration_secs

  candidates = []
  if gt_output_json:
    last_gt_ts = gt_output_json[-1].get("endTimestamp") or gt_output_json[
        -1
    ].get("startTimestamp")
    gt_end = parse_timestamp(last_gt_ts) if last_gt_ts else None
    if gt_end is not None:
      candidates.append(gt_end)

  if transcript_ranges:
    candidates.append(max(r[1] for r in transcript_ranges))

  if ad_ranges:
    candidates.append(max(r[1] for r in ad_ranges))

  if ad_output_json:
    last_ad_ts = ad_output_json[-1].get("endTimestamp") or ad_output_json[
        -1
    ].get("startTimestamp")
    ad_end = parse_timestamp(last_ad_ts) if last_ad_ts else None
    if ad_end is not None:
      candidates.append(ad_end)

  return max(candidates) if candidates else 0.0
