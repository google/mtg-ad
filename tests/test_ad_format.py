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

"""Tests for parsing raw model AD output (common/ad_format.py)."""

import json
import unittest

from common import ad_format
from common import tts
from tests import support


class TimestampTest(support.QuietTestCase):

  def test_timestamp_formats(self):
    cases = {
        "0:01:05.250": 65.25,  # H:MM:SS.mmm
        "1:05.250": 65.25,  # M:SS.mmm
        "01:05": 65.0,  # M:SS
        "0:17:460": 17.46,  # M:SS:mmm (paper pipeline convention)
        "0:00:05,5": 5.5,  # comma decimal separator
        "1:00:00.000": 3600.0,
    }
    for ts, expected in cases.items():
      with self.subTest(ts=ts):
        self.assertAlmostEqual(ad_format.timestamp_to_seconds(ts), expected)

  def test_invalid_timestamp(self):
    for ts in ("h:00:00.000", "abc", "12"):
      with self.subTest(ts=ts):
        self.assertIsNone(ad_format.timestamp_to_seconds(ts))

  def test_format_timestamp(self):
    self.assertEqual(ad_format.format_timestamp(65.25), "0:01:05.250")
    self.assertEqual(ad_format.format_timestamp(3725.0), "1:02:05.000")


class ParseAdLinesTest(support.QuietTestCase):

  def assert_segments(self, raw, expected):
    got = ad_format.parse_ad_lines(raw)
    self.assertEqual(len(got), len(expected), got)
    for (start, text), (exp_start, exp_text) in zip(got, expected):
      self.assertAlmostEqual(start, exp_start)
      self.assertEqual(text, exp_text)

  def test_canonical_lines(self):
    self.assert_segments(
        "0:00:01.000-STANDARD A cat.\n0:00:05.500-STANDARD A dog.",
        [(1.0, "A cat."), (5.5, "A dog.")],
    )

  def test_json_string_response_in_fence(self):
    raw = (
        '```json\n{"response": "0:00:01.000-STANDARD A'
        ' cat.\\n0:00:02.000-STANDARD A dog."}\n```'
    )
    self.assert_segments(raw, [(1.0, "A cat."), (2.0, "A dog.")])

  def test_json_list_response(self):
    # Gemini 3.7 / 3.8 Flash often return the lines as a list.
    raw = json.dumps({
        "response": [
            "0:00:01.000-STANDARD A cat.",
            "0:00:02.000-STANDARD A dog.",
        ]
    })
    self.assert_segments(raw, [(1.0, "A cat."), (2.0, "A dog.")])

  def test_json_list_of_objects(self):
    raw = json.dumps({
        "response": [
            {"timestamp": "0:00:03.000", "type": "STANDARD", "text": "A bird."}
        ]
    })
    self.assert_segments(raw, [(3.0, "A bird.")])

  def test_reasoning_preamble_before_json(self):
    raw = (
        "Let me find the gaps.\n1. Gap 1: 11.340 to 13.440 (2.1"
        ' seconds)\n{"response": "0:11.340-STANDARD Tom'
        ' mixes.\\n1:19.920-STANDARD Berries."}'
    )
    self.assert_segments(raw, [(11.34, "Tom mixes."), (79.92, "Berries.")])

  def test_invalid_json_with_unescaped_quotes(self):
    raw = (
        '```json\n{\n  "response": "0:00:14.200-STANDARD A cougar. \\n'
        '0:05:11.080-STANDARD The screen shows "BIG CAT DEREK"."\n}\n```'
    )
    self.assert_segments(
        raw,
        [(14.2, "A cougar."), (311.08, 'The screen shows "BIG CAT DEREK".')],
    )

  def test_colon_milliseconds(self):
    raw = json.dumps({"response": ["0:17:460-STANDARD A title appears."]})
    self.assert_segments(raw, [(17.46, "A title appears.")])

  def test_time_ranges_keep_start(self):
    self.assert_segments(
        "0:00:51.360-0:00:57.039 A title card.", [(51.36, "A title card.")]
    )

  def test_run_on_single_line(self):
    raw = (
        "0:00:01.319- STANDARD Pink packages. "
        "0:00:09.599-0:02:52.960- STANDARD A white tree. "
        "0:15:12.000-0:15:13.000- SKIP"
    )
    self.assert_segments(
        raw, [(1.319, "Pink packages."), (9.599, "A white tree.")]
    )

  def test_other_type_tags_kept_skip_dropped(self):
    raw = (
        "0:00:01.000-STANDARD A.\n0:00:02.000-SCENE B.\n"
        "0:00:03.000-SKIP\n0:00:04.000-EXTENDED C."
    )
    self.assert_segments(raw, [(1.0, "A."), (2.0, "B."), (4.0, "C.")])

  def test_untagged_lines_used_only_without_tagged_lines(self):
    raw = (
        "Here are the inline audio descriptions:\n\n"
        "00:14:120 The man places his mug.\n01:04:780 He rubs his eyes."
    )
    self.assert_segments(
        raw, [(14.12, "The man places his mug."), (64.78, "He rubs his eyes.")]
    )
    # With tagged lines present, untagged timestamps (e.g. reasoning) are
    # ignored.
    mixed = "0:00:05 - gap of 3 seconds\n0:00:06.000-STANDARD A cat."
    self.assert_segments(mixed, [(6.0, "A cat.")])

  def test_no_parseable_lines(self):
    for raw in (
        "",
        "   ",
        "Sorry, I cannot help with that.",
        "<400> InternalError.Algo.DataInspectionFailed: ...",
    ):
      with self.subTest(raw=raw):
        self.assertEqual(ad_format.parse_ad_lines(raw), [])
        self.assertEqual(ad_format.normalize_ad_output(raw), "")

  def test_normalize_is_canonical_and_idempotent(self):
    raw = json.dumps({"response": ["1:05.250-STANDARD A cat."]})
    once = ad_format.normalize_ad_output(raw)
    self.assertEqual(once, "0:01:05.250-STANDARD A cat.")
    self.assertEqual(ad_format.normalize_ad_output(once), once)

  def test_tts_parser_agrees_with_evaluation_parser(self):
    raw = json.dumps({
        "response": ["0:00:01.000-STANDARD A cat.", "0:17:460-STANDARD A dog."]
    })
    self.assertEqual(
        tts.parse_ad_file(raw),
        [
            (ad_format.format_timestamp(s), t)
            for s, t in ad_format.parse_ad_lines(raw)
        ],
    )

  def test_ground_truth_line_separator(self):
    # Dataset GT AD files use U+2028 between "<ts>-STANDARD" and the text.
    raw = (
        "00:00:10.043-STANDARD\u2028She closes her eyes.\n"
        "00:00:21.671-STANDARD\u2028She models her outfit.\n"
    )
    self.assertEqual(
        ad_format.parse_ad_lines(raw),
        [(10.043, "She closes her eyes."), (21.671, "She models her outfit.")],
    )


if __name__ == "__main__":
  unittest.main()
