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

"""Evaluates generated audio descriptions.

Library: compute_metrics() computes every metric for one video from in-memory
inputs. It is shared by run_ad_eval.py (through main() and
evaluate_video_from_dirs()) and run_end_to_end.py. make_gemini_judge() builds
the cached LLM-judge callable that compute_metrics() needs.

Metrics that cannot be computed for a video are left out of its metrics JSON
(never filled with 0), so means over videos are not skewed:
  * no parseable AD lines -> only "status": "no_parseable_ad" is written;
  * no TTS metadata -> TTS-based temporal metrics are skipped (durations are
    never estimated from word counts);
  * no visual info -> G1.1, G1.3 and G2 are skipped;
  * failed LLM-judge calls -> listed in "failed_judge_calls".
"""

import argparse
import json
import os
import sys
from typing import Any, Callable
import wave

from common import file_utils
from common import gemini_utils
from common import local_video
from common import transcripts as transcripts_lib
from common import tts
from common import usage
from evaluation import metrics
from evaluation import parsing_lib
from evaluation import reference_free
from evaluation import visual_info as visual_info_lib
from google import genai
import pandas as pd

# pylint: disable=g-import-not-at-top
try:
  import anthropic
except ImportError:
  anthropic = None

try:
  import dashscope
except ImportError:
  dashscope = None

try:
  import openai
except ImportError:
  openai = None
# pylint: enable=g-import-not-at-top

DEFAULT_GEMINI_MODEL = gemini_utils.DEFAULT_JUDGE_MODEL
DEFAULT_REFERENCE_FREE = (reference_free.DEFAULT_VARIANT,)

# Module state for the CLI's non-Gemini judge functions below.
API_KEY = ""
client = None

# read_or_generate_and_save(yt_id, prompt, dir, no_url=True) -> response text.
# Returns the cached response in {dir}/output_{yt_id}.txt if present; otherwise
# queries the LLM judge and caches the result. no_url=False asks the judge to
# also watch the video (needed by the reference-free metrics); judges that
# support this have a truthy `supports_video` attribute.
JudgeFn = Callable[..., str | None]

# Keys that depend on TTS-rendered AD durations.
TTS_METRIC_KEYS = (
    "total_overlap",
    "overlap_intervals",
    "num_overlap_intervals",
    "avg_overlap_ad_secs",
    "ad_total_time",
    "ad_total_time_tts",
    "ad_total_speaking_time_per_silent_second",
    "speech_density_w_ad_tts_dialog",
    "tts_ad_self_overlap_seconds",
    "tts_ad_transcript_overlap_seconds",
    "total_silence_duration_w_ad_tts",
    "avg_silence_duration_w_ad_tts",
    "silence_durations_s",
)


def metrics_path(output_dir: str, video_id: str) -> str:
  """Returns the path of a video's metrics JSON."""
  return os.path.join(output_dir, "metrics", f"output_{video_id}.json")


def _write_metrics(output_dir: str, video_id: str, data: dict[str, Any]):
  file_utils.write_text(
      metrics_path(output_dir, video_id), json.dumps(data, indent=2)
  )


def _enrich_ad_json_with_tts(ad_output_json, tts_info):
  """Adds TTS timing to parsed AD segments, matching segments by text."""
  by_hash = {seg.get("hashcode"): seg for seg in tts_info}
  for seg in ad_output_json:
    t_info = by_hash.get(tts.text_hash(seg["text"]))
    if t_info is None:
      continue
    seg["planned_endTimestamp"] = seg.get("endTimestamp", "")
    seg["start"] = t_info["start"]
    seg["duration_tts"] = t_info["duration_tts"]
    seg["end_tts"] = t_info["start"] + t_info["duration_tts"]
    seg["endTimestamp"] = parsing_lib.format_seconds_to_timestamp(
        seg["end_tts"]
    )
    for key in ("hashcode", "wav_path"):
      if key in t_info:
        seg[key] = t_info[key]


def compute_metrics(
    video_id: str,
    transcript: str,
    ad_text: str,
    visual_info: str,
    read_or_generate_and_save: JudgeFn,
    output_dir: str,
    gt_ad: str | None = None,
    tts_info: list[dict[str, Any]] | None = None,
    video_duration: float | str | None = None,
    reference_free_variants: tuple[str, ...] = DEFAULT_REFERENCE_FREE,
) -> dict[str, Any]:
  """Computes all metrics for one video and writes its metrics JSON.

  Args:
    video_id: YouTube video ID.
    transcript: Timestamped transcript text ("" for silent videos).
    ad_text: Generated AD; raw model output is normalized first.
    visual_info: Visual-information reference text ("" skips the metrics that
      need it).
    read_or_generate_and_save: Cached LLM-judge callable (see JudgeFn).
    output_dir: Directory for metrics/ and the per-metric LLM-judge caches.
    gt_ad: Optional ground-truth AD text, for reference-based metrics
      (VideoA11y-Ref and GT statistics).
    tts_info: TTS metadata (tts.synthesize_ad_file() output). Without it, the
      TTS-based temporal metrics are skipped.
    video_duration: Duration in seconds or as a timestamp string. If omitted, it
      is estimated from the latest timestamp seen.
    reference_free_variants: Quality-RF variants to compute (keys of
      reference_free.PROMPTS); () disables them.

  Returns:
    The metrics dict (also written to metrics_path(output_dir, video_id)).
    "status" is "ok", or "no_parseable_ad" when the AD has no timed lines (in
    which case no metric keys are present). "failed_judge_calls" lists the
    LLM-judge cache directories whose call returned no response; re-running
    fills them in, since successful judge responses are cached.
  """
  failed_judge_calls = set()

  def judge(*args, **kwargs):
    response = read_or_generate_and_save(*args, **kwargs)
    if response is None:
      judge_dir = kwargs.get("dir", args[2] if len(args) > 2 else "")
      failed_judge_calls.add(os.path.basename(os.path.normpath(judge_dir)))
    return response

  ad = metrics.normalize_ad_output(ad_text)
  ad_output_json = json.loads(parsing_lib.ad_string_to_json(ad)) if ad else []
  if not ad_output_json:
    print(
        f"ERROR: No timed AD lines could be parsed for {video_id}; writing a"
        " metrics file without metric values."
    )
    result = {
        "video_id": video_id,
        "status": "no_parseable_ad",
        "ad_raw": ad_text,
    }
    _write_metrics(output_dir, video_id, result)
    return result

  gt_output_json = None
  gt_ranges = []
  if gt_ad:
    try:
      gt_output_json = json.loads(parsing_lib.ad_string_to_json(gt_ad))
      gt_ranges = parsing_lib.get_time_ranges_from_list(gt_output_json, wpm=200)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
      print(f"Warning: Could not parse gt_ad for {video_id}: {e}")
      gt_output_json = None
      gt_ranges = []

  skipped = []
  ad_ranges = []
  if tts_info:
    _enrich_ad_json_with_tts(ad_output_json, tts_info)
    ad_ranges = parsing_lib.get_time_ranges_from_tts_json(tts_info)
  else:
    print(
        f"WARNING: No TTS metadata for {video_id}; skipping TTS-based temporal"
        " metrics. Run generate_tts.py and pass --input_tts_dir."
    )
    skipped.append("tts_temporal_metrics")

  transcript_json = json.loads(
      parsing_lib.parse_transcript_to_json_custom(
          transcript, speaker_id="TRANSCRIPT"
      )
  )
  transcript_ranges = parsing_lib.get_time_ranges_from_transcript_timestamps(
      transcript_text=transcript
  )

  video_duration_seconds = parsing_lib.estimate_video_duration_seconds(
      explicit_duration=video_duration,
      gt_output_json=gt_output_json,
      transcript_ranges=transcript_ranges,
      ad_ranges=ad_ranges,
      ad_output_json=ad_output_json,
  )
  if video_duration_seconds <= 0:
    video_duration_seconds = 1.0

  interleaved_transcript_ad_json = metrics.interleave_transcript_and_ad(
      yt_id=video_id,
      transcript_json=transcript_json,
      ad_output_json=ad_output_json,
  )

  # --- Word / dialogue statistics (no TTS needed) ---
  ad_total_words = sum(len(seg["text"].split()) for seg in ad_output_json)
  total_transcript_time = sum(t[1] - t[0] for t in transcript_ranges)
  total_silence_time = video_duration_seconds - total_transcript_time
  if total_silence_time <= 0:
    print(f"Warning: total silence time is not positive for {video_id}")
    total_silence_time = 1.0  # Avoid division by zero

  metrics_json = {
      "video_id": video_id,
      "status": "ok",
      "ad": ad,
      "ad_json": ad_output_json,
      "transcript": transcript,
      "ad_total_words": ad_total_words,
      "ad_total_words_per_silent_second": ad_total_words / total_silence_time,
      "total_silence_time": total_silence_time,
      "total_transcript_time": total_transcript_time,
      "dialog_density": total_transcript_time / video_duration_seconds,
      "video_duration_seconds": video_duration_seconds,
  }

  # --- TTS-based temporal metrics (G7, G8) ---
  if ad_ranges:
    overlap = parsing_lib.total_overlap(ad_ranges, transcript_ranges)
    overlap_intervals = overlap["overlap_intervals"]
    total_overlap_val = overlap["total_overlap_length"]
    ad_total_time_tts = sum(
        r[1] - r[0] for r in parsing_lib.merge_intervals(list(ad_ranges))
    )
    speech_merged = parsing_lib.merge_intervals(
        list(ad_ranges) + list(transcript_ranges)
    )
    metrics_json.update({
        "total_overlap": total_overlap_val,
        "overlap_intervals": overlap_intervals,
        # K_overlap in the paper; avg_overlap_ad_secs = T_overlap / K_overlap.
        "num_overlap_intervals": len(overlap_intervals),
        "avg_overlap_ad_secs": (
            total_overlap_val / len(overlap_intervals)
            if overlap_intervals
            else 0.0
        ),
        "ad_total_time": sum(r[1] - r[0] for r in ad_ranges),
        "ad_total_time_tts": ad_total_time_tts,
        "ad_total_speaking_time_per_silent_second": (
            ad_total_time_tts / total_silence_time
        ),
        "speech_density_w_ad_tts_dialog": (
            sum(r[1] - r[0] for r in speech_merged) / video_duration_seconds
        ),
    })
    metrics_json.update(
        metrics.compute_tts_overlap_metrics(ad_ranges, transcript_ranges)
    )
    silence_info = metrics.calculate_avg_silence_duration(
        ad_ranges, transcript_ranges, video_duration_seconds
    )
    if silence_info:
      metrics_json.update(silence_info)

  # --- Content metrics that need the visual information (G1.1, G1.3, G2) ---
  if visual_info:
    print("Computing Precision from Visual Info")
    try:
      precision = parsing_lib.compute_precision_visual_info(
          read_or_generate_and_save=judge,
          yt_id=video_id,
          ad_output=ad,
          visual_info=visual_info,
          dir=os.path.join(output_dir, "precision_visual_info") + "/",
      )
      metrics_json.update(
          {k: v for k, v in (precision or {}).items() if v is not None}
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"ERROR computing G1.1 precision for {video_id}: {e}")

    try:
      metric_1_2 = metrics.metric_1_2_narrates_onscreen_text(
          interleaved_transcript_ad_json=interleaved_transcript_ad_json,
          visual_info_str=visual_info,
      )
      file_utils.write_text(
          os.path.join(
              output_dir, "narrates_onscreen_text", f"output_{video_id}.txt"
          ),
          str(metric_1_2),
      )
      metrics_json.update(metric_1_2)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"ERROR computing G1.3 on-screen text for {video_id}: {e}")

    try:
      timeliness = metrics.metric_compute_timeliness(
          read_or_generate_and_save=judge,
          yt_id=video_id,
          visual_info=visual_info,
          audio_descriptions=ad,
          dir=os.path.join(output_dir, "timeliness") + "/",
      )
      metrics_json.update(
          {k: v for k, v in (timeliness or {}).items() if v is not None}
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"ERROR computing G2 timeliness for {video_id}: {e}")
  else:
    skipped.append("visual_info_metrics")

  # --- No redundancy (G1.2); needs dialogue to compare against ---
  if transcript:
    try:
      metric_2_1 = metrics.metric_2_1_does_not_restate(
          read_or_generate_and_save=judge,
          yt_id=video_id,
          interleaved_transcript_ad_str=interleaved_transcript_ad_json,
          dir=os.path.join(output_dir, "does_not_restate") + "/",
      )
      if metric_2_1 is not None:
        metrics_json.update(metric_2_1)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"ERROR computing G1.2 no-redundancy for {video_id}: {e}")

  # --- Qualitative alignment: reference-free (Quality-RF) ---
  if reference_free_variants and not getattr(
      read_or_generate_and_save, "supports_video", False
  ):
    print(
        "WARNING: The LLM judge cannot watch videos; skipping reference-free"
        " metrics (use the Gemini judge)."
    )
    skipped.append("reference_free_metrics")
  else:
    for variant in reference_free_variants:
      try:
        rf_metrics = reference_free.compute_reference_free_metric(
            read_or_generate_and_save=judge,
            yt_id=video_id,
            variant=variant,
            generated_desc=ad,
            transcript=transcript,
            visual_info=visual_info,
            metric_dir=os.path.join(output_dir, f"{variant}_metric") + "/",
        )
        if rf_metrics is not None:
          metrics_json.update(rf_metrics)
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"ERROR computing {variant} reference-free metric: {e}")

  # --- Qualitative alignment: reference-based (VideoA11y-Ref) + GT stats ---
  if gt_ad:
    try:
      videoa11y_metrics = metrics.compute_videoa11y_metric(
          read_or_generate_and_save=judge,
          yt_id=video_id,
          ground_truth_desc=gt_ad,
          generated_desc=ad,
          videoa11y_metric_dir=os.path.join(output_dir, "videoa11y_metric")
          + "/",
      )
      if videoa11y_metrics is not None:
        metrics_json.update(videoa11y_metrics)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"ERROR computing VideoA11y-Ref for {video_id}: {e}")

  if gt_ad and gt_output_json is not None:
    gt_total_time = sum(r[1] - r[0] for r in gt_ranges)
    gt_total_words = sum(len(seg["text"].split()) for seg in gt_output_json)
    metrics_json.update({
        "gt": gt_ad,
        "gt_json": gt_output_json,
        # GT AD has no TTS audio; its durations use a 200 WPM estimate.
        "total_overlap_gt": parsing_lib.total_overlap(
            gt_ranges, transcript_ranges
        )["total_overlap_length"],
        "gt_total_time": gt_total_time,
        "gt_total_words": gt_total_words,
        "gt_total_words_per_silent_second": gt_total_words / total_silence_time,
        "gt_total_speaking_time_per_silent_second": (
            gt_total_time / total_silence_time
        ),
    })

  metrics_json["skipped_metrics"] = skipped
  metrics_json["failed_judge_calls"] = sorted(failed_judge_calls)
  if failed_judge_calls:
    print(
        f"ERROR: LLM-judge calls failed for {video_id}"
        f" ({', '.join(sorted(failed_judge_calls))}); re-run to fill in the"
        " missing metrics."
    )

  _write_metrics(output_dir, video_id, metrics_json)
  return metrics_json


def _read_file(path: str) -> str | None:
  """Returns a file's text (possibly ""), or None if it does not exist."""
  if not os.path.exists(path):
    return None
  with open(path, "r", encoding="utf-8") as f:
    return f.read()


def load_or_generate_input(
    kind: str,
    yt_id: str,
    input_dir: str | None,
    output_dir: str,
    generator: Callable[[str], str | None] | None,
) -> str | None:
  """Returns a transcript or visual info, generating it if it is missing.

  Looks for {input_dir}/output_{yt_id}.txt first (an empty transcript file
  means a silent video), then for a previously generated copy in
  {output_dir}/generated_{kind}/. Otherwise calls `generator` and caches the
  result there.

  Args:
    kind: "transcript" or "visual_info".
    yt_id: YouTube video ID.
    input_dir: Directory of provided inputs, or None/"".
    output_dir: Evaluation output directory.
    generator: Callable(yt_id) -> text, or None to disable generation.

  Returns:
    The text, or None if it is unavailable.
  """
  file_name = f"output_{yt_id}.txt"
  if input_dir:
    text = _read_file(os.path.join(input_dir, file_name))
    if text is not None:
      return text
  generated_path = os.path.join(output_dir, f"generated_{kind}", file_name)
  text = _read_file(generated_path)
  if text is not None:
    return text
  if generator is None:
    return None
  print(f"  {kind} not found for {yt_id}; generating it with Gemini.")
  text = generator(yt_id)
  if text is not None:
    file_utils.write_text(generated_path, text)
  return text


def evaluate_video_from_dirs(
    yt_id,
    duration,
    output_dir,
    read_or_generate_and_save,
    transcript_dir="transcript",
    gt_ad_dir="gt_ad",
    visual_info_dir="visual_info",
    auto_ad_dir="auto_ad",
    tts_dir=None,
    input_generator=None,
    reference_free_variants=DEFAULT_REFERENCE_FREE,
):
  """Loads one video's inputs from per-step directories and evaluates it.

  Each directory holds output_{yt_id}.txt. A missing transcript or visual info
  is generated on the fly when `input_generator` is given. A missing GT AD
  only skips the reference-based metrics.

  Args:
    yt_id: YouTube video ID.
    duration: Optional video duration (seconds or timestamp string).
    output_dir: Evaluation output directory.
    read_or_generate_and_save: Cached LLM-judge callable (see JudgeFn).
    transcript_dir: Directory of transcripts.
    gt_ad_dir: Optional directory of ground-truth ADs.
    visual_info_dir: Directory of visual-information references.
    auto_ad_dir: Directory of generated ADs.
    tts_dir: Directory of TTS metadata JSONs or WAVs.
    input_generator: Optional dict {"transcript": fn, "visual_info": fn}, each
      fn(yt_id) -> text or None, used for missing inputs.
    reference_free_variants: Quality-RF variants to compute.

  Returns:
    The metrics dict.

  Raises:
    RuntimeError: If a required input is unavailable.
    tts.TtsError: If the TTS metadata is incomplete.
  """
  print("Processing: " + yt_id)
  file_name = f"output_{yt_id}.txt"
  generators = input_generator or {}

  ad = _read_file(os.path.join(auto_ad_dir, file_name))
  if ad is None:
    raise RuntimeError(f"generated AD not found: {auto_ad_dir}/{file_name}")

  transcript = load_or_generate_input(
      "transcript",
      yt_id,
      transcript_dir,
      output_dir,
      generators.get("transcript"),
  )
  if transcript is None:
    raise RuntimeError(
        f"transcript not found in {transcript_dir} and could not be generated"
    )

  visual_info = load_or_generate_input(
      "visual_info",
      yt_id,
      visual_info_dir,
      output_dir,
      generators.get("visual_info"),
  )
  if visual_info is None:
    print(f"WARNING: no visual info for {yt_id}; G1.1/G1.3/G2 are skipped.")
    visual_info = ""

  gt_ad = _read_file(os.path.join(gt_ad_dir, file_name)) if gt_ad_dir else None
  if gt_ad is None:
    print(f"Note: no GT AD for {yt_id}; reference-based metrics are skipped.")

  return compute_metrics(
      video_id=yt_id,
      transcript=transcript,
      ad_text=ad,
      visual_info=visual_info,
      read_or_generate_and_save=read_or_generate_and_save,
      output_dir=output_dir,
      gt_ad=gt_ad,
      tts_info=load_tts_info(yt_id, tts_dir, ad_text=ad),
      video_duration=duration,
      reference_free_variants=reference_free_variants,
  )


def make_gemini_judge(
    gemini_client: genai.Client,
    model: str,
    video_dir: str | None = None,
    video_parts: dict[str, Any] | None = None,
) -> JudgeFn:
  """Returns a cached read_or_generate_and_save backed by Gemini.

  Args:
    gemini_client: Gemini client.
    model: Gemini model name.
    video_dir: Optional directory of local MP4 files ({video_id}.mp4, or any MP4
      whose name maps to the ID, see common/local_video.py); for calls with
      no_url=False the judge watches the local MP4 if present, else the YouTube
      URL.
    video_parts: Optional {video_id: types.Part} cache of video references (e.g.
      an uploaded MP4), reused instead of uploading again. New references are
      added to it, so it can be shared with make_input_generators().

  Returns:
    The judge callable (see JudgeFn).
  """
  video_parts = {} if video_parts is None else video_parts

  def read_or_generate_and_save(
      yt_id, prompt, dir, no_url=True  # pylint: disable=redefined-builtin
  ):
    os.makedirs(dir, exist_ok=True)
    file_path = os.path.join(dir, f"output_{yt_id}.txt")
    cached = file_utils.read_text_if_nonempty(file_path)
    if cached:
      return cached
    if no_url:
      response = gemini_utils.generate_text(gemini_client, model, prompt)
    else:
      if yt_id not in video_parts:
        video_parts[yt_id] = gemini_utils.make_video_part(
            gemini_client, yt_id, _local_mp4(video_dir, yt_id)
        )
      response = gemini_utils.generate_with_video(
          gemini_client, model, video_parts[yt_id], prompt
      )
    if response:
      file_utils.write_text(file_path, response)
    else:
      print(f"ERROR: LLM-judge call failed for {yt_id} ({dir}).")
    return response

  read_or_generate_and_save.supports_video = True
  return read_or_generate_and_save


def _local_mp4(video_dir: str | None, yt_id: str) -> str | None:
  return local_video.find_local_video(video_dir, yt_id)


def make_input_generators(
    gemini_client: genai.Client,
    model: str,
    video_dir: str | None = None,
    video_parts: dict[str, Any] | None = None,
) -> dict[str, Callable[[str], str | None]]:
  """Returns {kind: fn(yt_id)} that generate missing inputs with Gemini.

  Args:
    gemini_client: Gemini client.
    model: Gemini model name.
    video_dir: Optional directory of local MP4 files (see make_gemini_judge).
    video_parts: Optional shared {video_id: types.Part} cache.
  """
  video_parts = {} if video_parts is None else video_parts

  def generate(kind: str, yt_id: str) -> str | None:
    if yt_id not in video_parts:
      video_parts[yt_id] = gemini_utils.make_video_part(
          gemini_client, yt_id, _local_mp4(video_dir, yt_id)
      )
    part = video_parts[yt_id]
    with usage.step(kind):
      if kind == "transcript":
        return transcripts_lib.generate_transcript(gemini_client, part, model)
      return visual_info_lib.generate_visual_info(gemini_client, part, model)

  return {
      "transcript": lambda yt_id: generate("transcript", yt_id),
      "visual_info": lambda yt_id: generate("visual_info", yt_id),
  }


def qwen_inference(prompt):
  """Calls Qwen VL model with a text prompt."""
  dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
  messages = [
      {"role": "system", "content": [{"text": "You are a helpful assistant."}]},
      {
          "role": "user",
          "content": [{"text": prompt}],
      },
  ]
  response = dashscope.MultiModalConversation.call(
      api_key=API_KEY,
      model="qwen-vl-max-latest",
      messages=messages,
  )
  return response


def qwen_read_or_generate_and_save(
    yt_id, prompt, dir, no_url=False  # pylint: disable=redefined-builtin
):
  """Reads cached Qwen response or generates and saves a new one."""
  del no_url
  os.makedirs(dir, exist_ok=True)
  file_path = os.path.join(dir, f"output_{yt_id}.txt")
  if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
      return f.read()

  try:
    response = qwen_inference(prompt)
    text = response["output"]["choices"][0]["message"].content[0]["text"]
    with open(file_path, "w", encoding="utf-8") as f:
      f.write(text)
    return text
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Error generating eval output for {yt_id}: {e}")
    return None


def claude_inference(prompt):
  """Calls Claude model with a text prompt."""
  with client.messages.stream(
      max_tokens=64000,
      messages=[
          {
              "role": "user",
              "content": [{
                  "type": "text",
                  "text": prompt,
              }],
          },
      ],
      model="claude-sonnet-4-5",
  ) as stream:
    final_message = stream.get_final_message()
    return final_message
  return None


def claude_read_or_generate_and_save(
    yt_id, prompt, dir, no_url=False  # pylint: disable=redefined-builtin
):
  """Reads cached Claude response or generates and saves a new one."""
  del no_url
  os.makedirs(dir, exist_ok=True)
  file_path = os.path.join(dir, f"output_{yt_id}.txt")
  if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
      return f.read()

  try:
    response = claude_inference(prompt)
    text = json.loads(response.to_json())["content"][0]["text"]
    with open(file_path, "w", encoding="utf-8") as f:
      f.write(text)
    return text
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Error generating eval output for {yt_id}: {e}")
    return None


def gpt_inference(prompt):
  """Calls OpenAI GPT model with a text prompt."""
  response = client.responses.create(
      model="gpt-4.1-mini",
      input=[{
          "role": "user",
          "content": [
              {"type": "input_text", "text": prompt},
          ],
      }],
  )
  return response


def gpt_read_or_generate_and_save(
    yt_id, prompt, dir, no_url=False  # pylint: disable=redefined-builtin
):
  """Reads cached GPT response or generates and saves a new one."""
  del no_url
  os.makedirs(dir, exist_ok=True)
  file_path = os.path.join(dir, f"output_{yt_id}.txt")
  if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
      return f.read()

  try:
    response = gpt_inference(prompt)
    with open(file_path, "w", encoding="utf-8") as f:
      f.write(response.output_text)
    return response.output_text
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Error generating eval output for {yt_id}: {e}")
    return None


def load_tts_info(video_id, tts_dir, ad_text=None):
  """Loads TTS metadata for a video, from a metadata JSON or WAV folder.

  Tries two sources in order:
    1. A metadata JSON produced by generate_tts.py (preferred):
       {tts_dir}/{video_id}_tts_metadata.json
    2. Pre-synthesized WAVs named {video_id}_tts_{md5(text)}.wav (see
       load_tts_from_wav_folder()).

  Args:
    video_id: YouTube video ID.
    tts_dir: Directory containing TTS outputs, or empty string / None.
    ad_text: Optional raw AD script text for the WAV folder source.

  Returns:
    List of segment dicts with 'start' and 'duration_tts', or None if tts_dir
    is not set or has no TTS output for this video.

  Raises:
    tts.TtsError: If the TTS output exists but is incomplete.
  """
  if not tts_dir:
    return None
  metadata = tts.load_metadata(video_id, tts_dir)
  if metadata is not None:
    return metadata
  if ad_text and os.path.isdir(tts_dir):
    wav_results = load_tts_from_wav_folder(video_id, tts_dir, ad_text)
    if wav_results:
      return wav_results
  print(
      f"ERROR: TTS output not found for {video_id}:"
      f" {tts.metadata_path(video_id, tts_dir)}"
  )
  return None


def load_tts_from_wav_folder(video_id, wav_folder, ad_text):
  """Loads TTS duration info from a folder of pre-synthesized WAV files.

  Uses a content-addressable hash to map each AD description to its WAV file:
    {wav_folder}/{video_id}_tts_{md5(text)}.wav
  so file ordering does not matter. This lets users bring their own TTS audio
  alongside the AD script (output_{video_id}.txt).

  Args:
    video_id: YouTube video ID.
    wav_folder: Path to folder containing the WAV files.
    ad_text: Raw AD script text (used to reconstruct expected filenames).

  Returns:
    List of segment dicts (matching generate_tts.py output format), or None if
    no WAV file for this video exists.

  Raises:
    tts.TtsError: If some, but not all, of the video's WAVs are missing or
      unreadable.
  """
  segments = tts.parse_ad_file(ad_text)
  if not segments:
    return None

  results = []
  missing = []
  for timestamp, text in segments:
    wav_path = os.path.join(wav_folder, tts.wav_filename(video_id, text))
    duration_secs = None
    if os.path.exists(wav_path):
      try:
        with wave.open(wav_path, "rb") as wf:
          duration_secs = wf.getnframes() / float(wf.getframerate())
      except (OSError, wave.Error, ZeroDivisionError) as e:
        print(f"  ERROR: Could not read WAV {wav_path}: {e}")
    if duration_secs is None:
      missing.append(text[:40])
      continue
    start_secs = tts.timestamp_to_seconds(timestamp)
    results.append({
        "timestamp": timestamp,
        "start": start_secs,
        "text": text,
        "hashcode": tts.text_hash(text),
        "duration_tts": duration_secs,
        "end_tts": start_secs + duration_secs,
        "wav_path": wav_path,
    })

  if not results:
    return None
  if missing:
    raise tts.TtsError(
        f"{len(missing)}/{len(segments)} WAVs missing or unreadable for"
        f" {video_id} in {wav_folder} (e.g. '{missing[0]}...')."
    )
  return results


def main():
  """Parses CLI arguments and evaluates AD outputs across all listed videos."""
  parser = argparse.ArgumentParser(
      description="A script to call GenAI APIs to evaluate audio AD."
  )
  parser.add_argument(
      "--genai_model",
      type=str,
      default="gemini",
      help=(
          "LLM-judge model family: gemini (default; the only judge that"
          " supports the reference-free metrics), qwen, gpt, or claude."
      ),
  )
  parser.add_argument(
      "--gemini_model",
      type=str,
      default=DEFAULT_GEMINI_MODEL,
      help=(
          "Gemini judge model when --genai_model=gemini. Also used to generate"
          " missing transcripts / visual info. Default:"
          f" {DEFAULT_GEMINI_MODEL}."
      ),
  )
  parser.add_argument(
      "--api_key_path", type=str, default="", help="API key path."
  )
  parser.add_argument(
      "--video_ids_file",
      type=str,
      default="",
      help=(
          "Video IDs file: one '<video_id> [<duration>]' per line. A duration"
          " here overrides --video_durations_file. May be omitted with"
          " --input_video_dir to evaluate every MP4 in it."
      ),
  )
  parser.add_argument(
      "--video_durations_file",
      type=str,
      default=file_utils.DEFAULT_VIDEO_DURATIONS_FILE,
      help=(
          "JSON map {video_id: duration} (seconds or M:SS). Defaults to the"
          " bundled durations of the 433 dataset videos. Videos not listed get"
          " their duration estimated from the latest timestamp."
      ),
  )
  parser.add_argument("--output_dir", type=str, default="", help="output_dir")
  parser.add_argument(
      "--input_transcript_dir",
      type=str,
      default="transcript",
      help=(
          "Directory containing transcript files named output_{VIDEO_ID}.txt"
          " (empty file = no speech). Missing transcripts are generated with"
          " Gemini unless --no_generate_missing_inputs."
      ),
  )
  parser.add_argument(
      "--input_gt_ad_dir",
      type=str,
      default="gt_ad",
      help=(
          "Optional directory containing ground truth AD files named"
          " output_{VIDEO_ID}.txt; enables VideoA11y-Ref and GT statistics."
      ),
  )
  parser.add_argument(
      "--input_visual_info_dir",
      type=str,
      default="visual_info",
      help=(
          "Directory containing visual info files named output_{VIDEO_ID}.txt."
          " Missing files are generated with Gemini unless"
          " --no_generate_missing_inputs."
      ),
  )
  parser.add_argument(
      "--input_auto_ad_dir",
      type=str,
      default="auto_ad",
      help=(
          "Directory containing generated AD files named output_{VIDEO_ID}.txt."
      ),
  )
  parser.add_argument(
      "--input_tts_dir",
      type=str,
      default="",
      help=(
          "Directory containing TTS metadata files"
          " ({video_id}_tts_metadata.json) or WAV files produced by"
          " generate_tts.py. Required for the temporal metrics (overlap,"
          " coverage, silence); without it they are skipped."
      ),
  )
  parser.add_argument(
      "--input_video_dir",
      type=str,
      default="",
      help=(
          "Optional directory of local MP4 files ({video_id}.mp4, or any MP4"
          " whose name maps to the ID, see common/local_video.py) for the"
          " judge, for generating missing inputs, and for measuring durations"
          " of videos not in --video_durations_file; otherwise YouTube URLs"
          " are used. Without --video_ids_file, every MP4 in it is evaluated."
      ),
  )
  parser.add_argument(
      "--reference_free_metrics",
      type=str,
      default=",".join(DEFAULT_REFERENCE_FREE),
      help=(
          "Comma-separated Quality-RF variants to compute:"
          f" {', '.join(reference_free.PROMPTS)}; or 'none'. Default:"
          f" {reference_free.DEFAULT_VARIANT}."
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
  parser.add_argument(
      "--no_generate_missing_inputs",
      action="store_true",
      help="Fail instead of generating missing transcripts / visual info.",
  )
  args = parser.parse_args()

  usage.set_pricing_file(args.pricing_file)
  rf_variants = tuple(
      v.strip()
      for v in args.reference_free_metrics.split(",")
      if v.strip() and v.strip() != "none"
  )
  unknown = [v for v in rf_variants if v not in reference_free.PROMPTS]
  if unknown:
    parser.error(f"unknown --reference_free_metrics variants: {unknown}")
  if not args.input_tts_dir:
    print(
        "WARNING: --input_tts_dir not set; TTS-based temporal metrics will be"
        " skipped for every video."
    )

  global API_KEY, client
  API_KEY = file_utils.get_api_key(args.api_key_path)
  if not API_KEY:
    sys.exit("ERROR: could not read an API key from --api_key_path.")

  input_generator = None
  match args.genai_model:
    case "qwen":
      if openai is None or dashscope is None:
        raise ImportError(
            "Please install 'openai' and 'dashscope' to use --genai_model=qwen."
        )
      client = openai.OpenAI(
          api_key=API_KEY,
          base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
      )
      judge = qwen_read_or_generate_and_save
    case "claude":
      if anthropic is None:
        raise ImportError(
            "Please install 'anthropic' to use --genai_model=claude."
        )
      client = anthropic.Anthropic(api_key=API_KEY)
      judge = claude_read_or_generate_and_save
    case "gpt":
      if openai is None:
        raise ImportError("Please install 'openai' to use --genai_model=gpt.")
      client = openai.OpenAI(api_key=API_KEY)
      judge = gpt_read_or_generate_and_save
    case "gemini":
      client = genai.Client(api_key=API_KEY)
      video_parts = {}  # Shared so each local MP4 is uploaded once.
      judge = make_gemini_judge(
          client, args.gemini_model, args.input_video_dir or None, video_parts
      )
      if not args.no_generate_missing_inputs:
        input_generator = make_input_generators(
            client, args.gemini_model, args.input_video_dir or None, video_parts
        )
    case _:
      sys.exit(f"ERROR: unsupported genai model {args.genai_model}.")

  durations = file_utils.load_video_durations(args.video_durations_file)
  if args.video_ids_file:
    entries = file_utils.read_video_ids_file(args.video_ids_file)
  elif args.input_video_dir:
    entries = [
        (vid, None)
        for vid, _ in local_video.list_local_videos(args.input_video_dir)
    ]
    print(f"Found {len(entries)} MP4 files in {args.input_video_dir}.")
  else:
    sys.exit("ERROR: pass --video_ids_file and/or --input_video_dir.")
  output_dir = args.output_dir
  os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

  failures = {}
  for video_id, duration in entries:
    duration = duration or durations.get(video_id)
    if not duration:
      # Local MP4 not in the durations file: measure it, rather than
      # estimating the duration from the latest timestamp.
      mp4_path = _local_mp4(args.input_video_dir, video_id)
      if mp4_path:
        duration = local_video.probe_duration_seconds(mp4_path)
    recorder = usage.UsageRecorder()
    try:
      with usage.recording(recorder), usage.step("evaluation"):
        result = evaluate_video_from_dirs(
            video_id,
            duration,
            output_dir,
            judge,
            transcript_dir=args.input_transcript_dir,
            gt_ad_dir=args.input_gt_ad_dir,
            visual_info_dir=args.input_visual_info_dir,
            auto_ad_dir=args.input_auto_ad_dir,
            tts_dir=args.input_tts_dir or None,
            input_generator=input_generator,
            reference_free_variants=rf_variants,
        )
    except gemini_utils.FatalApiError as e:
      sys.exit(f"FATAL: {e}")
    except (RuntimeError, tts.TtsError) as e:
      print(f"ERROR: {video_id}: {e}")
      failures[video_id] = str(e)
      continue
    finally:
      if args.genai_model == "gemini":
        # Gemini token usage and estimated cost of the judge (and generated
        # inputs), summed over runs.
        usage.save(
            os.path.join(output_dir, "usage", f"output_{video_id}.json"),
            recorder,
            extra={"video_id": video_id, "eval_model": args.gemini_model},
        )
    if result.get("status") != "ok":
      failures[video_id] = result.get("status")
    elif result.get("failed_judge_calls"):
      failures[video_id] = "failed judge calls: " + ", ".join(
          result["failed_judge_calls"]
      )
    elif args.input_tts_dir and "tts_temporal_metrics" in result.get(
        "skipped_metrics", []
    ):
      failures[video_id] = "no TTS output; temporal metrics skipped"

  file_utils.report_failures(failures, len(entries), "Evaluation")


if __name__ == "__main__":
  main()
