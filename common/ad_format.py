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

"""Parsing of raw model AD output into canonical timed AD lines.

The canonical AD line format, used by TTS and every evaluation metric, is:
  H:MM:SS.mmm-STANDARD description
e.g. "0:01:05.250-STANDARD A woman walks into the room."

normalize_ad_output() accepts the response shapes produced by the models in the
paper, including:
  * {"response": "line\\nline"} or {"response": ["line", ...]}, optionally in a
    ```json fence and/or after a reasoning preamble;
  * {"response": [{"timestamp": ..., "text": ...}, ...]};
  * plain-text lines;
  * timestamps as H:MM:SS.mmm, M:SS.mmm, M:SS:mmm, H:MM:SS, M:SS, and ranges
    such as "0:00:51.360-0:00:57.039 text" (the start time is kept).

A shared parser keeps TTS rendering and evaluation in agreement about which AD
lines exist and when they start.
"""

import json
import re

_TS = r"\d{1,2}:\d{2}(?::\d{2,3})?(?:[.,]\d+)?"
# "<ts>-STANDARD text" (optionally "<ts>-<end_ts>-STANDARD text"). Other
# upper-case type tags (e.g. EXTENDED, SCENE) are accepted too; SKIP lines are
# placeholders and are dropped.
_TAG = r"(?!SKIP\b)(?:STANDARD|[A-Z0-9][A-Z0-9_]{2,})"
_STRICT_LINE = re.compile(
    rf"^\W*?(?P<ts>{_TS})(?:\s*-\s*{_TS})?\s*-\s*{_TAG}\b[\s:]*(?P<text>.+)$"
)
# "<ts> - text", "[<ts>] text", "<ts>-<end_ts> text", ... (no STANDARD tag).
_LOOSE_LINE = re.compile(
    rf"^\W*?\[?(?P<ts>{_TS})\]?(?:\s*(?:-|–|to)\s*\[?{_TS}\]?)?"
    r"\s*[-–:]?\s*(?P<text>[^\d\s].*)$"
)
_TEXT_KEYS = ("text", "description", "ad", "audio_description")
_TS_KEYS = ("timestamp", "start", "start_time", "time")


def timestamp_to_seconds(ts: str) -> float | None:
  """Converts an AD timestamp to seconds.

  Supported: H:MM:SS.mmm, M:SS.mmm, M:SS, and the colon-separated variants
  H:MM:SS / M:SS:mmm. A three-part timestamp without a decimal point is read as
  M:SS:mmm (e.g. "0:17:460" -> 17.46 s), matching the paper's pipeline.

  Args:
    ts: Timestamp string.

  Returns:
    Seconds as a float, or None if the string is not a timestamp.
  """
  ts = ts.strip().replace(",", ".")
  if ts.count(":") == 2 and "." not in ts:
    head, ms = ts.rsplit(":", 1)
    ts = f"{head}.{ms}"
  parts = ts.split(":")
  try:
    if len(parts) == 3:
      hours, minutes, sec = int(parts[0]), int(parts[1]), parts[2]
    elif len(parts) == 2:
      hours, minutes, sec = 0, int(parts[0]), parts[1]
    else:
      return None
    if "." in sec:
      whole, frac = sec.split(".", 1)
      seconds = int(whole) + int(frac.ljust(3, "0")[:3]) / 1000
    else:
      seconds = int(sec)
  except ValueError:
    return None
  return hours * 3600 + minutes * 60 + seconds


def format_timestamp(seconds: float) -> str:
  """Formats seconds as the canonical H:MM:SS.mmm."""
  millis = int(round(seconds * 1000))
  hours, rem = divmod(millis, 3_600_000)
  minutes, rem = divmod(rem, 60_000)
  secs, millis = divmod(rem, 1000)
  return f"{hours}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _load_json(text: str):
  """Parses JSON from text, allowing a code fence or a preamble."""
  fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
  candidates = [fence.group(1)] if fence else []
  candidates.append(text)
  brace = re.search(r"\{[\s\S]*\}", text)
  if brace:
    candidates.append(brace.group(0))
  for candidate in candidates:
    try:
      return json.loads(candidate.strip())
    except (json.JSONDecodeError, ValueError):
      continue
  return None


def _item_to_line(item) -> str:
  """Converts one element of a JSON response list to a text line."""
  if isinstance(item, dict):
    ts = next((str(item[k]) for k in _TS_KEYS if item.get(k) is not None), "")
    text = next((str(item[k]) for k in _TEXT_KEYS if item.get(k)), "")
    return f"{ts}-STANDARD {text}" if ts and text else ""
  return str(item)


def _unwrap(ad: str) -> str:
  """Returns the AD text inside a {"response": ...} wrapper, if any."""
  parsed = _load_json(ad.strip())
  if isinstance(parsed, dict) and "response" in parsed:
    parsed = parsed["response"]
  if isinstance(parsed, list):
    return "\n".join(_item_to_line(item) for item in parsed)
  if isinstance(parsed, str):
    return parsed
  # Not valid JSON (e.g. unescaped quotes inside the response string): strip a
  # code fence and the {"response": "..."} wrapper by hand.
  fence = re.search(r"```(?:json|text)?\s*([\s\S]*?)```", ad)
  text = fence.group(1) if fence else ad
  wrapped = re.search(r'"response"\s*:\s*"([\s\S]*)"\s*\}?\s*$', text.strip())
  if wrapped:
    text = wrapped.group(1).replace("\\n", "\n").replace('\\"', '"')
  return text


def _split_run_on_lines(text: str) -> str:
  """Starts a new line at every '<ts>-<TAG>' that is not at a line start."""
  return re.sub(
      rf"[ \t]+(?=(?:{_TS})(?:\s*-\s*{_TS})?\s*-\s*[A-Z0-9][A-Z0-9_]{{2,}}\b)",
      "\n",
      text,
  )


def parse_ad_lines(ad: str) -> list[tuple[float, str]]:
  """Parses raw model AD output into (start_seconds, text) pairs.

  Lines tagged "-STANDARD" are preferred. Only if there are none, untagged
  "<timestamp> text" lines are accepted, which avoids picking up timestamps in
  a model's reasoning preamble.

  Args:
    ad: Raw model output or canonical AD text.

  Returns:
    (start_seconds, description) pairs in output order.
  """
  if not ad or not ad.strip():
    return []
  # The dataset's ground-truth AD files separate "<ts>-STANDARD" from the text
  # with a Unicode line separator (U+2028), which str.splitlines() would treat
  # as a line break; keep each segment on one line.
  ad = ad.replace("\u2028", " ").replace("\u2029", " ")
  text = _split_run_on_lines(_unwrap(ad))
  lines = [l.strip() for l in text.splitlines() if l.strip()]
  for pattern in (_STRICT_LINE, _LOOSE_LINE):
    segments = []
    for line in lines:
      m = pattern.match(line)
      if not m:
        continue
      start = timestamp_to_seconds(m.group("ts"))
      text = m.group("text").strip().rstrip(",").strip().strip('"').strip()
      if start is not None and text:
        segments.append((start, text))
    if segments:
      return segments
  return []


def normalize_ad_output(ad: str) -> str:
  """Returns canonical "H:MM:SS.mmm-STANDARD text" lines ("" if none parse)."""
  return "\n".join(
      f"{format_timestamp(start)}-STANDARD {text}"
      for start, text in parse_ad_lines(ad)
  )
