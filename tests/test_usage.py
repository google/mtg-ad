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

"""Offline tests for token usage and cost accounting (no API calls)."""

import csv
import datetime
import json
import os
import tempfile
from unittest import mock

from common import gemini_utils
from common import usage
from google.genai import types
from tests import support


def _response(
    prompt=1000, video=700, audio=200, cached=0, output=50, thoughts=100
):
  return types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=types.Content(parts=[types.Part(text="ok")]),
              finish_reason=types.FinishReason.STOP,
          )
      ],
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=prompt,
          prompt_tokens_details=[
              types.ModalityTokenCount(modality="VIDEO", token_count=video),
              types.ModalityTokenCount(modality="AUDIO", token_count=audio),
              types.ModalityTokenCount(
                  modality="TEXT", token_count=prompt - video - audio
              ),
          ],
          cached_content_token_count=cached,
          candidates_token_count=output,
          thoughts_token_count=thoughts,
          total_token_count=prompt + output + thoughts,
      ),
  )


class CostTest(support.QuietTestCase):

  def test_flat_input_price(self):
    counts = usage.usage_counts(_response().usage_metadata)
    prices = {"input": 0.75, "cached_input": 0.075, "output": 3.75}
    # 1000 input tokens + (50 output + 100 thinking) tokens.
    self.assertAlmostEqual(
        usage.cost_usd(counts, prices), (1000 * 0.75 + 150 * 3.75) / 1e6
    )

  def test_cached_and_audio_prices(self):
    counts = usage.usage_counts(_response(cached=400).usage_metadata)
    prices = {
        "input": 0.30,
        "input_audio": 1.00,
        "cached_input": 0.03,
        "output": 2.50,
    }
    expected = (
        (1000 - 200 - 400) * 0.30 + 200 * 1.00 + 400 * 0.03 + 150 * 2.50
    ) / 1e6
    self.assertAlmostEqual(usage.cost_usd(counts, prices), expected)

  def test_bundled_pricing_periods(self):
    usage.set_pricing_file(None)
    dec = usage._prices("gemini-3.8-flash", datetime.date(2026, 12, 31))
    jan = usage._prices("models/gemini-3.8-flash", datetime.date(2027, 1, 1))
    self.assertEqual((dec["input"], jan["input"]), (0.75, 1.50))
    self.assertIsNone(usage._prices("unknown-model", datetime.date.today()))
    self.assert_printed("no price for model 'unknown-model'")


class RecorderTest(support.QuietTestCase):

  def test_records_steps_through_gemini_utils(self):
    client = mock.Mock()
    client.models.generate_content.return_value = _response()
    with usage.recording() as recorder:
      with usage.step("ad_generation"):
        self.assertEqual(
            gemini_utils.generate_text(client, "gemini-3.8-flash", "p"), "ok"
        )
      with usage.step("evaluation"):
        gemini_utils.generate_text(client, "gemini-3.8-flash", "p")
        gemini_utils.generate_text(client, "no-price-model", "p")
    # Outside recording(): not counted.
    gemini_utils.generate_text(client, "gemini-3.8-flash", "p")
    summary = recorder.summary()
    self.assertEqual(set(summary["steps"]), {"ad_generation", "evaluation"})
    ad = summary["steps"]["ad_generation"]
    self.assertEqual(
        (
            ad["calls"],
            ad["input_tokens"],
            ad["input_video_tokens"],
            ad["output_tokens"],
            ad["thinking_tokens"],
        ),
        (1, 1000, 700, 50, 100),
    )
    self.assertGreater(ad["cost_usd"], 0)
    # One unpriced call makes the step and total cost unknown.
    self.assertIsNone(summary["steps"]["evaluation"]["cost_usd"])
    self.assertIsNone(summary["total"]["cost_usd"])
    self.assertEqual(summary["total"]["calls"], 3)

  def test_save_accumulates_and_csv(self):
    path = os.path.join(tempfile.mkdtemp(), "usage", "output_v.json")
    for _ in range(2):
      recorder = usage.UsageRecorder()
      recorder.add(
          "tts", "m", {"calls": 1, "input_tokens": 10, "cost_usd": 0.5}
      )
      data = usage.save(path, recorder, extra={"video_id": "v"})
    self.assertEqual(data["total"]["calls"], 2)
    self.assertAlmostEqual(data["total"]["cost_usd"], 1.0)
    with open(path, encoding="utf-8") as f:
      self.assertEqual(json.load(f)["video_id"], "v")
    csv_path = os.path.join(os.path.dirname(path), "summary.csv")
    usage.write_summary_csv(csv_path, {"v": data, "w": data})
    with open(csv_path, encoding="utf-8") as f:
      rows = list(csv.DictReader(f))
    self.assertEqual([r["video_id"] for r in rows], ["v", "w", "TOTAL"])
    self.assertAlmostEqual(float(rows[-1]["tts_cost_usd"]), 2.0)
    # Steps that did not run for a video cost 0.
    self.assertEqual(rows[0]["transcript_cost_usd"], "0")


if __name__ == "__main__":
  import unittest  # pylint: disable=g-import-not-at-top

  unittest.main()
