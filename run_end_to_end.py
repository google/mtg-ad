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

r"""End-to-end Audio Description generation and evaluation pipeline.

Given a YouTube video ID or a local MP4 file (and optionally a transcript),
this script runs the full pipeline in one go:

  Step 1: ASR transcript   — Gemini transcribes the video dialogue.
  Step 2: AD generation    — Gemini generates ADs (--prompt_type, default
                             timing_asr).
  Step 3: Visual info      — Gemini extracts key visual events (for eval).
  Step 4: TTS synthesis    — Gemini TTS (default) or Cloud TTS at 1.2x
                             (--cloud_tts) renders each AD segment to WAV.
  Step 5: Evaluation       — Computes all metrics (overlap, silence, redundancy,
                             timeliness, precision-from-visual-info, on-screen
                             text, Quality-RF, and VideoA11y-Ref when a GT AD
                             is given) with the judge model (--eval_model).

Existing transcripts / visual info can be supplied (--transcript_path,
--transcript_dir, --visual_info_dir); otherwise they are generated. Video
durations default to the bundled metadata/video_durations.json; local MP4s not
listed there are measured with ffprobe (or OpenCV).

Token usage and estimated cost (USD) of every Gemini call (input tokens by
modality, cached, output and thinking tokens; prices from
metadata/gemini_pricing.json or --pricing_file) are written per video and step
to a usage JSON, and summed over the batch in {output_dir}/usage_summary.csv.

Each step is a thin call into the common, generation and evaluation
libraries. All intermediate outputs are cached to --output_dir so the script is
safely re-runnable; completed steps are skipped automatically.

Usage (single video by YouTube URL; IDs starting with '-' also work, e.g.
--video_id --T4zFh82Ao or --video_id=--T4zFh82Ao):
  python run_end_to_end.py \
    --api_key_path /path/to/gemini_api_key.txt \
    --video_id dQw4w9WgXcQ \
    --output_dir results/dQw4w9WgXcQ/

Usage (batch of videos from a video IDs file):
  python run_end_to_end.py \
    --api_key_path /path/to/gemini_api_key.txt \
    --video_ids_file sample_video_ids.txt \
    --output_dir results/end_to_end/

Usage (single local MP4 instead of a YouTube video; the video ID used in
output file names is derived from the file name, e.g. "My video.mp4" ->
My_video, or set with --video_id):
  python run_end_to_end.py \
    --api_key_path /path/to/gemini_api_key.txt \
    --mp4_path /path/to/video.mp4 \
    --output_dir results/my_video/

Usage (batch of local MP4s: every MP4 in --video_dir, or with --video_ids_file
the listed IDs, looked up as {id}.mp4 or by derived ID):
  python run_end_to_end.py \
    --api_key_path /path/to/gemini_api_key.txt \
    --video_dir /path/to/videos/ \
    --output_dir results/my_videos/

Usage (single video with a pre-existing transcript, skipping the ASR step):
  python run_end_to_end.py \
    --api_key_path /path/to/gemini_api_key.txt \
    --video_id dQw4w9WgXcQ \
    --transcript_path /path/to/transcript.txt \
    --output_dir results/dQw4w9WgXcQ/

Output layouts (--layout):
  flat (default): all files of a video live in one directory — --output_dir for
    --video_id, or {output_dir}/vid_{video_id}/ for --video_ids_file:
      output_{id}_transcript.txt, output_{id}_ad_raw.txt, output_{id}.txt
      (normalized AD), output_{id}_visual_info.txt, tts/,
      metrics/output_{id}.json, usage/output_{id}.json
  per_step: one directory per step, shared by all videos and keyed by ID:
      transcript/, auto_ad_raw/, auto_ad/, visual_info/ (each output_{id}.txt),
      tts/, metrics/output_{id}.json, usage/output_{id}.json
    These directories can be passed straight to run_ad_eval.py and
    generate_tts.py.
"""

import argparse
import dataclasses
import functools
import inspect
import json
import os
import sys
from typing import Any, Callable

from common import file_utils
from common import gemini_utils
from common import local_video
from common import transcripts
from common import tts
from common import usage
from evaluation import evaluation
from evaluation import metrics
from evaluation import visual_info as visual_info_lib
from generation import gemini_prompt
from generation import generation
from google import genai

DEFAULT_GENERATION_MODEL = gemini_utils.DEFAULT_GEMINI_MODEL
DEFAULT_EVAL_MODEL = gemini_utils.DEFAULT_JUDGE_MODEL
DEFAULT_PROMPT_TYPE = "timing_asr"
LAYOUTS = ("flat", "per_step")

KEY_METRICS = (
    "metric_G1.1_precision_avg_ad_in_visual_info",
    "metric_2_1_does_not_restate",
    "metric_1_2_narrates_onscreen_text",
    "metric_g_2_avg_diff_time",
    "total_overlap",
    "ad_total_speaking_time_per_silent_second",
    "avg_silence_duration_w_ad_tts",
    "video_asr_descriptive",
    "video_asr_objective",
    "video_asr_accurate",
    "video_asr_clear",
)


@dataclasses.dataclass(frozen=True)
class OutputPaths:
  """Where one video's intermediate and final outputs are written."""

  transcript: str
  ad_raw: str
  ad: str
  visual_info: str
  tts_dir: str
  # Directory holding metrics/ and the LLM-judge caches.
  eval_dir: str
  # Token usage and cost JSON (see common/usage.py).
  usage: str


def make_output_paths(
    output_dir: str, video_id: str, layout: str = "flat"
) -> OutputPaths:
  """Returns the output paths for a video under the given layout.

  Args:
    output_dir: Directory for this video (flat) or for all videos (per_step).
    video_id: YouTube video ID.
    layout: "flat" or "per_step"; see the module docstring.

  Returns:
    The OutputPaths for the video.
  """
  if layout == "flat":
    prefix = os.path.join(output_dir, f"output_{video_id}")
    return OutputPaths(
        transcript=f"{prefix}_transcript.txt",
        ad_raw=f"{prefix}_ad_raw.txt",
        ad=f"{prefix}.txt",
        visual_info=f"{prefix}_visual_info.txt",
        tts_dir=os.path.join(output_dir, "tts"),
        eval_dir=output_dir,
        usage=os.path.join(output_dir, "usage", f"output_{video_id}.json"),
    )
  if layout == "per_step":
    name = f"output_{video_id}.txt"
    return OutputPaths(
        transcript=os.path.join(output_dir, "transcript", name),
        ad_raw=os.path.join(output_dir, "auto_ad_raw", name),
        ad=os.path.join(output_dir, "auto_ad", name),
        visual_info=os.path.join(output_dir, "visual_info", name),
        tts_dir=os.path.join(output_dir, "tts"),
        eval_dir=output_dir,
        usage=os.path.join(output_dir, "usage", f"output_{video_id}.json"),
    )
  raise ValueError(f"Unknown layout {layout!r}; expected one of {LAYOUTS}.")


def _in_step(name: str, fn: Callable[[], Any]) -> Callable[[], Any]:
  """Returns fn wrapped so its Gemini calls are recorded under step `name`."""

  def wrapped():
    with usage.step(name):
      return fn()

  return wrapped


def _cached_step(
    name: str,
    path: str,
    generate: Callable[[], str | None],
    allow_empty: bool = False,
) -> str | None:
  """Returns the cached text at `path`, or runs `generate` and caches it.

  Args:
    name: Step name for log messages.
    path: Cache file.
    generate: Callable producing the text, or None on failure.
    allow_empty: Treat "" as a valid result (e.g. a transcript of a video
      without speech) and cache it as an empty file.

  Returns:
    The text, or None if generation failed.
  """
  if allow_empty and os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
      print(f"  [{name}] Using cached output: {path}")
      return f.read()
  cached = file_utils.read_text_if_nonempty(path)
  if cached:
    print(f"  [{name}] Using cached output: {path}")
    return cached
  result = generate()
  if result or (allow_empty and result is not None):
    file_utils.write_text(path, result)
    print(f"  [{name}] Saved to {path}")
  else:
    print(f"  [{name}] ERROR: generation failed.")
  return result


def _records_usage(pipeline):
  """Records the Gemini token usage and cost of each run_pipeline() call.

  The usage is added to the video's usage JSON (OutputPaths.usage, summed over
  re-runs; see usage.save()) and printed, also when the pipeline stops early.
  """
  signature = inspect.signature(pipeline)

  @functools.wraps(pipeline)
  def wrapper(*args, **kwargs):
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    a = bound.arguments
    paths = make_output_paths(a["output_dir"], a["video_id"], a["layout"])
    with usage.recording() as recorder:
      try:
        return pipeline(*args, **kwargs)
      finally:
        data = usage.save(
            paths.usage,
            recorder,
            extra={
                "video_id": a["video_id"],
                "prompt_type": a["prompt_type"],
                "generation_model": a["generation_model"],
                "eval_model": a["eval_model"],
            },
        )
        print(
            "\n=== Gemini token usage and estimated cost (all runs of this"
            " video) ==="
        )
        print(usage.format_summary(data))
        print(f"  Saved to {paths.usage}")

  return wrapper


@_records_usage
def run_pipeline(
    client: genai.Client,
    video_id: str,
    output_dir: str,
    layout: str = "flat",
    prompt_type: str = DEFAULT_PROMPT_TYPE,
    mp4_path: str | None = None,
    transcript_path: str | None = None,
    visual_info_path: str | None = None,
    gt_ad_path: str | None = None,
    video_duration: float | str | None = None,
    generation_model: str = DEFAULT_GENERATION_MODEL,
    eval_model: str = DEFAULT_EVAL_MODEL,
    tts_voice: str = tts.DEFAULT_VOICE,
    tts_model: str = tts.TTS_MODEL,
    cloud_tts: bool = False,
    speaking_rate: float = tts.CLOUD_TTS_SPEAKING_RATE,
    skip_tts: bool = False,
    reference_free_variants: tuple[str, ...] = (
        evaluation.DEFAULT_REFERENCE_FREE
    ),
) -> dict[str, Any] | None:
  """Runs the full AD generation + evaluation pipeline for one video.

  Args:
    client: Gemini client, used for generation, TTS and LLM-judge calls.
    video_id: YouTube video ID, or the ID used in output file names for a local
      MP4 (see local_video.video_id_from_path()).
    output_dir: Output directory; see make_output_paths().
    layout: "flat" or "per_step".
    prompt_type: Key into gemini_prompt.PROMPTS.
    mp4_path: Optional local MP4 to upload instead of using the YouTube URL. It
      is uploaded once and reused by every Gemini call, including the LLM judge.
    transcript_path: Optional existing transcript; skips the ASR step.
    visual_info_path: Optional existing visual info; skips that step.
    gt_ad_path: Optional ground-truth AD file for reference-based metrics.
    video_duration: Optional duration (seconds or timestamp string).
    generation_model: Gemini model for transcript, AD and visual info.
    eval_model: Gemini model for LLM-judge metrics.
    tts_voice: Gemini TTS voice name.
    tts_model: Gemini TTS model name.
    cloud_tts: Use Google Cloud TTS (optional dependency) instead of Gemini.
    speaking_rate: Cloud TTS speaking rate.
    skip_tts: Skip TTS; the TTS-based temporal metrics are then skipped.
    reference_free_variants: Quality-RF variants to compute.

  Returns:
    The metrics dict, or None if AD generation failed. TTS failures are
    reported loudly and recorded in the metrics' "skipped_metrics" (the other
    metrics are still computed).
  """
  paths = make_output_paths(output_dir, video_id, layout)
  os.makedirs(paths.eval_dir, exist_ok=True)

  # Build the video reference once; it is reused by every Gemini call.
  print(f"\n=== Video: {video_id} ===")
  if mp4_path and not video_duration:
    video_duration = local_video.probe_duration_seconds(mp4_path)
  print("Building video reference...")
  video_part = gemini_utils.make_video_part(client, video_id, mp4_path)

  def provided_or_cached(
      name, provided_path, cache_path, generate, allow_empty=False
  ):
    if provided_path and os.path.exists(provided_path):
      with open(provided_path, "r", encoding="utf-8") as f:
        text = f.read()
      print(f"  Using provided {name}: {provided_path}")
      if not os.path.exists(cache_path):
        file_utils.write_text(cache_path, text)
      return text
    return _cached_step(name, cache_path, generate, allow_empty=allow_empty)

  print("\n[Step 1/5] ASR Transcript")
  transcript = provided_or_cached(
      "transcript",
      transcript_path,
      paths.transcript,
      _in_step(
          "transcript",
          lambda: transcripts.generate_transcript(
              client, video_part, model=generation_model
          ),
      ),
      allow_empty=True,
  )
  if transcript is None:
    # A failed call is not a silent video: continuing would cache ADs and
    # metrics generated without dialogue. Stop so a re-run retries it.
    print("ERROR: Could not obtain transcript; skipping this video.")
    return None

  print(f"\n[Step 2/5] AD Generation (prompt_type={prompt_type})")
  ad_raw = _cached_step(
      "ad_gen",
      paths.ad_raw,
      _in_step(
          "ad_generation",
          lambda: generation.generate_ad(
              client,
              video_part,
              gemini_prompt.PROMPTS[prompt_type],
              transcript=transcript,
              model=generation_model,
              video_duration=video_duration,
          ),
      ),
  )
  if not ad_raw:
    print("ERROR: Could not generate ADs; skipping this video.")
    return None

  # Normalize the raw model output into canonical AD lines and cache it.
  ad = metrics.normalize_ad_output(ad_raw)
  if ad:
    file_utils.write_text(paths.ad, ad)
  else:
    print("ERROR: No timed AD lines could be parsed from the model output.")

  print("\n[Step 3/5] Visual Info")
  visual_info = provided_or_cached(
      "visual_info",
      visual_info_path,
      paths.visual_info,
      _in_step(
          "visual_info",
          lambda: visual_info_lib.generate_visual_info(
              client, video_part, model=generation_model
          ),
      ),
  )
  if not visual_info:
    print(
        "WARNING: Could not generate visual info; G1.1/G1.3/G2 will be skipped."
    )

  tts_info = None
  if skip_tts:
    print(
        "\n[Step 4/5] TTS Synthesis — SKIPPED (--skip_tts); temporal metrics"
        " will be skipped."
    )
  elif ad:
    print("\n[Step 4/5] TTS Synthesis")
    try:
      tts_info = tts.load_metadata(video_id, paths.tts_dir)
      if tts_info is not None:
        print("  [tts] Using cached TTS metadata.")
      else:
        with usage.step("tts"):
          tts_info = tts.synthesize_ad_file(
              client,
              video_id,
              ad,
              paths.tts_dir,
              voice_name=tts_voice,
              use_cloud_tts=cloud_tts,
              tts_model=tts_model,
              speaking_rate=speaking_rate,
          )
    except tts.TtsError as e:
      print(f"ERROR: {e}")
      tts_info = None

  gt_ad = None
  if gt_ad_path and os.path.exists(gt_ad_path):
    with open(gt_ad_path, "r", encoding="utf-8") as f:
      gt_ad = f.read()

  print("\n[Step 5/5] Evaluation")
  metrics_file = evaluation.metrics_path(paths.eval_dir, video_id)
  if os.path.exists(metrics_file):
    with open(metrics_file, "r", encoding="utf-8") as f:
      cached = json.load(f)
    if (
        cached.get("status") == "ok"
        and not cached.get("failed_judge_calls")
        and not cached.get("skipped_metrics")
    ):
      print(f"  [eval] Using cached metrics: {metrics_file}")
      return cached
    print("  [eval] Cached metrics are incomplete; recomputing.")
  with usage.step("evaluation"):
    return evaluation.compute_metrics(
        video_id=video_id,
        transcript=transcript,
        ad_text=ad,
        visual_info=visual_info or "",
        read_or_generate_and_save=evaluation.make_gemini_judge(
            client, eval_model, video_parts={video_id: video_part}
        ),
        output_dir=paths.eval_dir,
        gt_ad=gt_ad,
        tts_info=tts_info,
        video_duration=video_duration,
        reference_free_variants=reference_free_variants,
    )


def _find_file(directory: str, video_id: str) -> str | None:
  """Returns {directory}/output_{video_id}.txt if it exists."""
  if not directory:
    return None
  candidate = os.path.join(directory, f"output_{video_id}.txt")
  return candidate if os.path.exists(candidate) else None


def _problems(result: dict[str, Any] | None) -> str | None:
  """Returns a description of what is missing from a result, or None."""
  if not result:
    return "pipeline failed"
  if result.get("status") != "ok":
    return str(result.get("status"))
  issues = list(result.get("skipped_metrics", []))
  if result.get("failed_judge_calls"):
    issues.append(
        "failed judge calls: " + ", ".join(result["failed_judge_calls"])
    )
  return "; ".join(issues) or None


def _print_key_metrics(result: dict[str, Any], metrics_file: str) -> None:
  print("\n=== Pipeline complete. Key metrics ===")
  for k in KEY_METRICS:
    if k in result:
      v = result[k]
      print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
  print(f"\nFull metrics: {metrics_file}")


def _parse_args() -> argparse.Namespace:
  """Parses and validates command-line arguments."""
  parser = argparse.ArgumentParser(
      description=(
          "End-to-end Audio Description generation and evaluation pipeline. "
          "Generates a transcript, ADs, visual info, TTS audio, then evaluates."
      )
  )
  parser.add_argument(
      "--api_key_path",
      type=str,
      required=True,
      help="Path to a file containing your Gemini API key.",
  )
  parser.add_argument(
      "--video_id",
      type=str,
      default="",
      help=(
          "Single YouTube video ID (e.g. dQw4w9WgXcQ). IDs starting with '-'"
          " are supported (--video_id --T4zFh82Ao). With --mp4_path, sets the"
          " ID used in output file names (default: derived from the file"
          " name)."
      ),
  )
  parser.add_argument(
      "--video_ids_file",
      type=str,
      default="",
      help="File with one '<video_id> [<duration>]' per line (batch mode).",
  )
  parser.add_argument(
      "--video_dir",
      type=str,
      default="",
      help=(
          "Directory of local MP4 files (batch mode). Alone, every MP4 in it"
          ' is processed, with IDs derived from the file names ("My video.mp4"'
          " -> My_video). With --video_ids_file, each listed ID uses its local"
          " MP4 ({id}.mp4 or matching derived ID) when present, else the"
          " YouTube URL."
      ),
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Output directory; see --layout.",
  )
  parser.add_argument(
      "--layout",
      choices=LAYOUTS,
      default="flat",
      help=(
          "flat: one directory per video; per_step: one directory per step"
          " (transcript/, auto_ad/, ...)."
      ),
  )
  parser.add_argument(
      "--prompt_type",
      type=str.lower,
      choices=sorted(gemini_prompt.PROMPTS),
      default=DEFAULT_PROMPT_TYPE,
      help=(
          "AD generation prompt (case-insensitive). Default:"
          f" {DEFAULT_PROMPT_TYPE}. agentic_gemini_3p7_up is for Gemini 3.7"
          " Flash and later and does not use the transcript."
      ),
  )
  parser.add_argument(
      "--mp4_path",
      type=str,
      default="",
      help=(
          "Path to a local MP4 file (single-video mode), uploaded to the"
          " Gemini Files API instead of using a YouTube URL. --video_id is"
          " optional with it."
      ),
  )
  parser.add_argument(
      "--transcript_path",
      type=str,
      default="",
      help=(
          "Optional pre-existing transcript file (single-video mode, "
          "H:MM:SS.mmm--H:MM:SS.mmm text format). Skips the ASR step."
      ),
  )
  parser.add_argument(
      "--transcript_dir",
      type=str,
      default="",
      help=(
          "Optional directory of existing transcripts (output_{id}.txt), e.g."
          " the dataset's raw/test/transcript/. Missing ones are generated."
      ),
  )
  parser.add_argument(
      "--visual_info_dir",
      type=str,
      default="",
      help=(
          "Optional directory of existing visual info (output_{id}.txt), e.g."
          " the dataset's raw/test/visual_info/. Missing ones are generated."
      ),
  )
  parser.add_argument(
      "--gt_ad_path",
      type=str,
      default="",
      help=(
          "Optional path to a ground-truth AD file (single-video mode) "
          "for computing reference-based metrics (VideoA11y-Ref, GT stats)."
      ),
  )
  parser.add_argument(
      "--gt_ad_dir",
      type=str,
      default="",
      help=(
          "Optional directory with ground-truth AD files (output_{id}.txt). "
          "Used in batch mode or when --gt_ad_path is not provided."
      ),
  )
  parser.add_argument(
      "--video_duration",
      type=str,
      default="",
      help=(
          "Optional explicit video duration (in seconds or MM:SS.mmm / "
          "HH:MM:SS.mmm format) for single-video mode. In batch mode, use the "
          "second column of --video_ids_file."
      ),
  )
  parser.add_argument(
      "--video_durations_file",
      type=str,
      default=file_utils.DEFAULT_VIDEO_DURATIONS_FILE,
      help=(
          "JSON map {video_id: duration}. Defaults to the bundled durations of"
          " the 433 dataset videos."
      ),
  )
  parser.add_argument(
      "--model",
      type=str,
      default="",
      help=(
          "Optional shorthand to set both --generation_model and --eval_model."
      ),
  )
  parser.add_argument(
      "--generation_model",
      type=str,
      default=DEFAULT_GENERATION_MODEL,
      help=(
          "Gemini model for transcript, AD, and visual info generation."
          f" Default: {DEFAULT_GENERATION_MODEL}."
      ),
  )
  parser.add_argument(
      "--eval_model",
      type=str,
      default=DEFAULT_EVAL_MODEL,
      help=(
          "Gemini model for LLM-judge evaluation calls. Default:"
          f" {DEFAULT_EVAL_MODEL}."
      ),
  )
  parser.add_argument(
      "--reference_free_metrics",
      type=str,
      default=",".join(evaluation.DEFAULT_REFERENCE_FREE),
      help=(
          "Comma-separated Quality-RF variants:"
          f" {', '.join(evaluation.reference_free.PROMPTS)}; or 'none'."
      ),
  )
  parser.add_argument(
      "--tts_voice",
      type=str,
      default=tts.DEFAULT_VOICE,
      help=f"Gemini TTS voice name. Default: {tts.DEFAULT_VOICE}.",
  )
  parser.add_argument(
      "--tts_model",
      type=str,
      default=tts.TTS_MODEL,
      help=f"Gemini TTS model. Default: {tts.TTS_MODEL}.",
  )
  parser.add_argument(
      "--cloud_tts",
      action="store_true",
      help=(
          "Use Google Cloud TTS at --speaking_rate (the paper's setting)"
          " instead of Gemini TTS. Needs the optional"
          " google-cloud-texttospeech package and Application Default"
          " Credentials."
      ),
  )
  parser.add_argument(
      "--speaking_rate",
      type=float,
      default=tts.CLOUD_TTS_SPEAKING_RATE,
      help=f"Cloud TTS speaking rate. Default: {tts.CLOUD_TTS_SPEAKING_RATE}.",
  )
  parser.add_argument(
      "--pricing_file",
      type=str,
      default=usage.DEFAULT_PRICING_FILE,
      help=(
          "JSON of Gemini prices per 1M tokens used for the cost estimates."
          " Default: metadata/gemini_pricing.json."
      ),
  )
  parser.add_argument(
      "--skip_tts",
      action="store_true",
      default=False,
      help="Skip TTS synthesis; the temporal metrics are then skipped.",
  )
  args = parser.parse_args(file_utils.fix_dash_prefixed_values(sys.argv[1:]))

  single = bool(args.video_id or args.mp4_path)
  batch = bool(args.video_ids_file or args.video_dir)
  if single == batch:
    parser.error(
        "pass either --video_id and/or --mp4_path (single video), or"
        " --video_ids_file and/or --video_dir (batch)."
    )
  if args.mp4_path and not os.path.isfile(args.mp4_path):
    parser.error(f"--mp4_path not found: {args.mp4_path}")
  if args.video_dir and not os.path.isdir(args.video_dir):
    parser.error(f"--video_dir not found: {args.video_dir}")
  if args.mp4_path and not args.video_id:
    args.video_id = local_video.video_id_from_path(args.mp4_path)

  if batch:
    single_video_flags = {
        "--mp4_path": args.mp4_path,
        "--transcript_path": args.transcript_path,
        "--gt_ad_path": args.gt_ad_path,
        "--video_duration": args.video_duration,
    }
    used = [flag for flag, value in single_video_flags.items() if value]
    if used:
      parser.error(
          f"{', '.join(used)} can only be used for a single video, not"
          " with --video_ids_file / --video_dir."
      )
  args.reference_free_variants = tuple(
      v.strip()
      for v in args.reference_free_metrics.split(",")
      if v.strip() and v.strip() != "none"
  )
  unknown = [
      v
      for v in args.reference_free_variants
      if v not in evaluation.reference_free.PROMPTS
  ]
  if unknown:
    parser.error(f"unknown --reference_free_metrics variants: {unknown}")
  if args.cloud_tts and not args.skip_tts:
    try:
      tts.require_cloud_tts()
    except tts.TtsError as e:
      parser.error(str(e))
  return args


def main():
  """Parses CLI arguments and runs the end-to-end AD pipeline."""
  args = _parse_args()

  api_key = file_utils.get_api_key(args.api_key_path)
  if not api_key:
    sys.exit(1)
  client = genai.Client(api_key=api_key)
  durations = file_utils.load_video_durations(args.video_durations_file)
  usage.set_pricing_file(args.pricing_file)

  common = dict(
      layout=args.layout,
      prompt_type=args.prompt_type,
      generation_model=args.model or args.generation_model,
      eval_model=args.model or args.eval_model,
      tts_voice=args.tts_voice,
      tts_model=args.tts_model,
      cloud_tts=args.cloud_tts,
      speaking_rate=args.speaking_rate,
      skip_tts=args.skip_tts,
      reference_free_variants=args.reference_free_variants,
  )

  if args.video_ids_file or args.video_dir:
    if args.video_ids_file:
      entries = file_utils.read_video_ids_file(args.video_ids_file)
    else:
      entries = [
          (vid, None)
          for vid, _ in local_video.list_local_videos(args.video_dir)
      ]
    print(
        f"Running end-to-end pipeline for {len(entries)} videos with"
        f" generation_model={common['generation_model']},"
        f" eval_model={common['eval_model']}, layout={args.layout}"
    )
    failures = {}
    usage_per_video = {}
    for video_id, duration in entries:
      if args.layout == "flat":
        out = os.path.join(args.output_dir, f"vid_{video_id}")
      else:
        out = args.output_dir
      mp4_path = local_video.find_local_video(args.video_dir, video_id)
      if args.video_dir and not mp4_path:
        print(f"No local MP4 for {video_id}; using the YouTube URL.")
      try:
        result = run_pipeline(
            client,
            video_id,
            out,
            mp4_path=mp4_path,
            transcript_path=_find_file(args.transcript_dir, video_id),
            visual_info_path=_find_file(args.visual_info_dir, video_id),
            gt_ad_path=_find_file(args.gt_ad_dir, video_id),
            video_duration=duration or durations.get(video_id),
            **common,
        )
      except gemini_utils.FatalApiError as e:
        sys.exit(f"FATAL: {e}")
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"ERROR: Pipeline failed for {video_id}: {e}")
        result = None
      problem = _problems(result)
      if problem:
        failures[video_id] = problem
      usage_file = make_output_paths(out, video_id, args.layout).usage
      if os.path.exists(usage_file):
        with open(usage_file, "r", encoding="utf-8") as f:
          usage_per_video[video_id] = json.load(f)
    if usage_per_video:
      summary_csv = os.path.join(args.output_dir, "usage_summary.csv")
      usage.write_summary_csv(summary_csv, usage_per_video)
      costs = [d["total"]["cost_usd"] for d in usage_per_video.values()]
      known = [c for c in costs if c is not None]
      print(
          f"\n=== Estimated Gemini cost: {usage.format_cost(sum(known))} for"
          f" {len(known)} videos"
          + (
              f" ({len(costs) - len(known)} without a price)"
              if len(known) < len(costs)
              else ""
          )
          + f"; per video and step: {summary_csv} ==="
      )
    file_utils.report_failures(failures, len(entries), "End-to-end")
    return

  video_id = args.video_id
  try:
    result = run_pipeline(
        client,
        video_id,
        args.output_dir,
        mp4_path=args.mp4_path or None,
        transcript_path=args.transcript_path
        or _find_file(args.transcript_dir, video_id),
        visual_info_path=_find_file(args.visual_info_dir, video_id),
        gt_ad_path=args.gt_ad_path or _find_file(args.gt_ad_dir, video_id),
        video_duration=args.video_duration or durations.get(video_id),
        **common,
    )
  except gemini_utils.FatalApiError as e:
    sys.exit(f"FATAL: {e}")
  if not result:
    sys.exit("\nPipeline failed — see errors above.")
  eval_dir = make_output_paths(args.output_dir, video_id, args.layout).eval_dir
  _print_key_metrics(result, evaluation.metrics_path(eval_dir, video_id))
  problem = _problems(result)
  if problem:
    sys.exit(f"\nIncomplete metrics for {video_id}: {problem}")


if __name__ == "__main__":
  main()
