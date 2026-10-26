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

"""Token usage and cost (USD) accounting for Gemini API calls.

Every Gemini response carries usage metadata: input (prompt) tokens, split by
modality (TEXT / VIDEO / AUDIO / IMAGE), implicitly cached input tokens, output
tokens, and thinking tokens. gemini_utils and tts call record_response() after
each API call; the counts are added to the active UsageRecorder (if any) under
the current step name:

  with usage.recording() as recorder:
    with usage.step("ad_generation"):
      ...  # Gemini calls
  summary = recorder.summary()

Costs use the per-model prices in metadata/gemini_pricing.json (Standard paid
tier, see that file for its source and date), evaluated at the date of each
call. A model without a price gets token counts but cost_usd = None. Calls that
raise an exception have no usage metadata and are not counted. Cost estimates
cover tokens only (not context-cache storage, grounding, or free-tier usage).
"""

import contextlib
import contextvars
import csv
import datetime
import json
import os
from typing import Any, Iterator

DEFAULT_PRICING_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metadata",
    "gemini_pricing.json",
)

_COUNT_KEYS = (
    "calls",
    "input_tokens",
    "input_audio_tokens",
    "input_video_tokens",
    "input_text_tokens",
    "input_image_tokens",
    "cached_input_tokens",
    "cached_audio_tokens",
    "output_tokens",
    "thinking_tokens",
    "total_tokens",
)

_active = contextvars.ContextVar("usage_recorder", default=None)
_step = contextvars.ContextVar("usage_step", default="other")
_pricing: dict[str, Any] | None = None
_pricing_file = DEFAULT_PRICING_FILE
_warned_models: set[str] = set()


def set_pricing_file(path: str | None) -> None:
  """Uses `path` (a JSON in the format of gemini_pricing.json) for costs."""
  global _pricing, _pricing_file
  _pricing_file = path or DEFAULT_PRICING_FILE
  _pricing = None


def _load_pricing() -> dict[str, Any]:
  global _pricing
  if _pricing is None:
    try:
      with open(_pricing_file, "r", encoding="utf-8") as f:
        _pricing = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
      print(f"WARNING: could not read pricing file {_pricing_file}: {e}")
      _pricing = {"models": {}}
  return _pricing


def pricing_source() -> str | None:
  return _load_pricing().get("source")


def _prices(model: str, day: datetime.date) -> dict[str, float] | None:
  """Returns the price period of `model` that contains `day`, or None."""
  name = model.removeprefix("models/")
  periods = _load_pricing().get("models", {}).get(name)
  if not periods:
    if name not in _warned_models:
      _warned_models.add(name)
      print(
          f"  Note: no price for model '{name}' in {_pricing_file}; its token"
          " counts are recorded without a cost."
      )
    return None
  iso = day.isoformat()
  for period in periods:
    if period.get("from", "0000") <= iso <= period.get("until", "9999"):
      return period
  return None


def _n(value) -> int:
  """Returns a token count, or 0 if it is missing (None or not an int)."""
  return value if isinstance(value, int) else 0


def _modality_counts(details) -> dict[str, int]:
  counts = {}
  if not isinstance(details, list):
    return counts
  for d in details:
    modality = str(getattr(d, "modality", "") or "").rsplit(".", 1)[-1].upper()
    counts[modality] = counts.get(modality, 0) + _n(d.token_count)
  return counts


def usage_counts(usage_metadata) -> dict[str, int]:
  """Converts a response's usage_metadata into the recorded token counts."""
  if usage_metadata is None:
    return {}
  prompt = _n(usage_metadata.prompt_token_count)
  prompt += _n(usage_metadata.tool_use_prompt_token_count)
  by_modality = _modality_counts(usage_metadata.prompt_tokens_details)
  cached_by_modality = _modality_counts(usage_metadata.cache_tokens_details)
  return {
      "calls": 1,
      "input_tokens": prompt,
      "input_audio_tokens": by_modality.get("AUDIO", 0),
      "input_video_tokens": by_modality.get("VIDEO", 0),
      "input_text_tokens": by_modality.get("TEXT", 0),
      "input_image_tokens": by_modality.get("IMAGE", 0),
      "cached_input_tokens": _n(usage_metadata.cached_content_token_count),
      "cached_audio_tokens": cached_by_modality.get("AUDIO", 0),
      "output_tokens": _n(usage_metadata.candidates_token_count),
      "thinking_tokens": _n(usage_metadata.thoughts_token_count),
      "total_tokens": _n(usage_metadata.total_token_count),
  }


def cost_usd(counts: dict[str, int], prices: dict[str, float]) -> float:
  """Returns the cost of one call's token counts under a price period."""
  cached = counts.get("cached_input_tokens", 0)
  cached_audio = counts.get("cached_audio_tokens", 0)
  audio = counts.get("input_audio_tokens", 0)
  uncached_audio = max(audio - cached_audio, 0)
  uncached_other = max(
      counts.get("input_tokens", 0) - audio - cached + cached_audio, 0
  )
  in_price = prices["input"]
  in_audio = prices.get("input_audio", in_price)
  cached_price = prices.get("cached_input", in_price)
  cached_audio_price = prices.get(
      "cached_input_audio",
      cached_price if "input_audio" not in prices else in_audio,
  )
  output = counts.get("output_tokens", 0) + counts.get("thinking_tokens", 0)
  return (
      uncached_other * in_price
      + uncached_audio * in_audio
      + (cached - cached_audio) * cached_price
      + cached_audio * cached_audio_price
      + output * prices["output"]
  ) / 1e6


def _empty() -> dict[str, Any]:
  entry = {k: 0 for k in _COUNT_KEYS}
  entry["cost_usd"] = 0.0
  return entry


def _add(entry: dict[str, Any], counts: dict[str, Any]) -> None:
  for k in _COUNT_KEYS:
    entry[k] = entry.get(k, 0) + counts.get(k, 0)
  if entry.get("cost_usd") is None or counts.get("cost_usd") is None:
    entry["cost_usd"] = None
  else:
    entry["cost_usd"] += counts["cost_usd"]


class UsageRecorder:
  """Accumulates token counts and costs per (step, model)."""

  def __init__(self):
    # {step: {model: entry}}
    self.steps: dict[str, dict[str, dict[str, Any]]] = {}

  def add(self, step_name: str, model: str, counts: dict[str, Any]) -> None:
    entry = self.steps.setdefault(step_name, {}).setdefault(model, _empty())
    _add(entry, counts)

  def merge(self, steps: dict[str, Any]) -> None:
    """Adds the per-step entries of a previous summary()["steps"]."""
    for step_name, step_data in steps.items():
      for model, entry in step_data.get("models", {}).items():
        self.add(step_name, model, entry)

  def summary(self) -> dict[str, Any]:
    """Returns the per-step and overall usage totals.

    The result has the form
    {"steps": {step: {"models": {...}, totals...}}, "total": {...}}.
    """
    steps = {}
    total = _empty()
    for step_name, models in self.steps.items():
      step_total = _empty()
      for entry in models.values():
        _add(step_total, entry)
      _add(total, step_total)
      steps[step_name] = {"models": models, **step_total}
    return {"steps": steps, "total": total}


@contextlib.contextmanager
def recording(recorder: UsageRecorder | None = None) -> Iterator[UsageRecorder]:
  """Records the usage of Gemini calls made inside the block."""
  recorder = recorder or UsageRecorder()
  token = _active.set(recorder)
  try:
    yield recorder
  finally:
    _active.reset(token)


@contextlib.contextmanager
def step(name: str) -> Iterator[None]:
  """Labels Gemini calls made inside the block with step `name`."""
  token = _step.set(name)
  try:
    yield
  finally:
    _step.reset(token)


def record_response(model: str, response) -> None:
  """Adds a Gemini response's usage to the active recorder (if any)."""
  recorder = _active.get()
  if recorder is None:
    return
  counts = usage_counts(getattr(response, "usage_metadata", None))
  if not counts:
    return
  prices = _prices(model, datetime.datetime.now(datetime.timezone.utc).date())
  counts["cost_usd"] = cost_usd(counts, prices) if prices else None
  recorder.add(_step.get(), model.removeprefix("models/"), counts)


def save(
    path: str,
    recorder: UsageRecorder,
    extra: dict[str, Any] | None = None,
    accumulate: bool = True,
) -> dict[str, Any]:
  """Writes the recorder's summary to `path` and returns it.

  Args:
    path: Output JSON path.
    recorder: Usage of this run.
    extra: Extra top-level fields (e.g. video_id, prompt_type).
    accumulate: Add the counts already in `path` (from earlier runs of the same
      video), so the file holds the total spend over all runs; steps served from
      cache on a re-run cost nothing and keep their earlier counts.

  Returns:
    The written summary.
  """
  combined = UsageRecorder()
  if accumulate and os.path.exists(path):
    try:
      with open(path, "r", encoding="utf-8") as f:
        combined.merge(json.load(f).get("steps", {}))
    except (OSError, json.JSONDecodeError) as e:
      print(f"WARNING: ignoring unreadable usage file {path}: {e}")
  combined.merge(recorder.summary()["steps"])
  data = {**(extra or {}), "pricing_source": pricing_source()}
  data.update(combined.summary())
  parent = os.path.dirname(path)
  if parent:
    os.makedirs(parent, exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
  return data


def format_cost(value: float | None) -> str:
  return "n/a" if value is None else f"${value:.4f}"


def format_summary(data: dict[str, Any]) -> str:
  """Returns a short per-step table of tokens and cost."""
  lines = [
      f"  {'step':<16}{'calls':>6}{'input':>10}{'output':>9}{'thinking':>10}"
      f"{'cost':>11}"
  ]
  rows = list(data["steps"].items()) + [("TOTAL", data["total"])]
  for name, e in rows:
    lines.append(
        f"  {name:<16}{e['calls']:>6}{e['input_tokens']:>10}"
        f"{e['output_tokens']:>9}{e['thinking_tokens']:>10}"
        f"{format_cost(e['cost_usd']):>11}"
    )
  return "\n".join(lines)


SUMMARY_STEPS = (
    "transcript",
    "ad_generation",
    "visual_info",
    "tts",
    "evaluation",
)
_SUMMARY_FIELDS = (
    "cost_usd",
    "input_tokens",
    "output_tokens",
    "thinking_tokens",
)


def write_summary_csv(path: str, per_video: dict[str, dict[str, Any]]) -> None:
  """Writes one row per video (plus a TOTAL row) of per-step tokens and cost.

  Args:
    path: Output CSV path.
    per_video: {video_id: data returned by save()}.
  """
  steps = list(SUMMARY_STEPS) + sorted(
      {s for d in per_video.values() for s in d["steps"]} - set(SUMMARY_STEPS)
  )
  header = ["video_id"] + [
      f"{s}_{f}" for s in steps + ["total"] for f in _SUMMARY_FIELDS
  ]
  rows = []
  totals = {}
  for video_id, data in per_video.items():
    row = [video_id]
    for s in steps + ["total"]:
      entry = data["total"] if s == "total" else data["steps"].get(s)
      for f in _SUMMARY_FIELDS:
        value = (entry or {}).get(f, 0 if entry is None else None)
        row.append("" if value is None else value)
        key = (s, f)
        if value is None or totals.get(key, 0) is None:
          totals[key] = None
        else:
          totals[key] = totals.get(key, 0) + value
    rows.append(row)
  rows.append(
      ["TOTAL"]
      + [
          "" if totals.get((s, f)) is None else totals[(s, f)]
          for s in steps + ["total"]
          for f in _SUMMARY_FIELDS
      ]
  )
  parent = os.path.dirname(path)
  if parent:
    os.makedirs(parent, exist_ok=True)
  with open(path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in rows:
      writer.writerow([f"{v:.6f}" if isinstance(v, float) else v for v in row])
