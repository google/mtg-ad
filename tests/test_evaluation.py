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

"""Offline tests for evaluation: metrics, TTS handling and errors.

No API calls are made.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from common import file_utils
from common import gemini_utils
from common import tts
from evaluation import evaluation
from evaluation import metrics
from evaluation import reference_free
from google.genai import errors
from tests import support

TRANSCRIPT = (
    "0:00:01.000--0:00:03.000 hello there\n0:00:12.000--0:00:14.000 bye"
)
AD = json.dumps({
    "response": [
        "0:00:02.000-STANDARD A woman waves.",
        "0:00:06.000-STANDARD A sign reads OPEN TODAY.",
    ]
})
VISUAL_INFO = """Here is a list of visual events, followed by text on screen.

**Important Visual Events:**
00:00:02.000 - A woman waves.

**Text on Screen:**
*   00:00:06.000 - OPEN TODAY

**People in the video:**
*   A woman.
"""

# Canned judge answers, keyed by the metric's cache directory.
JUDGE_ANSWERS = {
    "precision_visual_info": "[1, 0.5]",
    "timeliness": "[0:02] a\n[0:02] b\n**[0]**\n[0:06] c\n[0:01] d\n**[5]**",
    "does_not_restate": "[0, 0.5]",
    "videoa11y_metric": (
        '{"Descriptive": 3, "Objective": 4, "Accurate": 2, '
        '"Clear": 3, "Reason": "ok"}'
    ),
    "video_asr_metric": (
        '```json\n{"Summary": "s", "Rationale": "r", '
        '"Descriptive": 4, "Objective": 5, "Accurate": 3, '
        '"Clear": 4}\n```'
    ),
}


def fake_judge(
    yt_id, prompt, dir, no_url=True  # pylint: disable=redefined-builtin
):
  del yt_id, prompt, no_url
  return JUDGE_ANSWERS[os.path.basename(os.path.normpath(dir))]


fake_judge.supports_video = True


def tts_info_for(ad_text, durations):
  return [
      {
          "start": tts.timestamp_to_seconds(ts),
          "duration_tts": d,
          "end_tts": tts.timestamp_to_seconds(ts) + d,
          "text": text,
          "hashcode": tts.text_hash(text),
      }
      for (ts, text), d in zip(tts.parse_ad_file(ad_text), durations)
  ]


class ComputeMetricsTest(support.QuietTestCase):

  def setUp(self):
    super().setUp()
    self.out = tempfile.mkdtemp()

  def compute(self, **kwargs):
    args = dict(
        video_id="vid",
        transcript=TRANSCRIPT,
        ad_text=AD,
        visual_info=VISUAL_INFO,
        read_or_generate_and_save=fake_judge,
        output_dir=self.out,
        video_duration="0:20",
        tts_info=tts_info_for(AD, [2.0, 1.0]),
    )
    args.update(kwargs)
    return evaluation.compute_metrics(**args)

  def test_all_metrics(self):
    m = self.compute(gt_ad="0:00:05.000-STANDARD A woman waves at the camera.")
    self.assertEqual(m["status"], "ok")
    self.assertEqual(m["skipped_metrics"], [])
    self.assertEqual(m["failed_judge_calls"], [])
    self.assertEqual(m["video_duration_seconds"], 20.0)  # explicit duration
    # AD 1 spans [2, 4] and overlaps dialogue [1, 3] by 1 s; AD 2 [6, 7]
    # does not.
    self.assertAlmostEqual(m["total_overlap"], 1.0)
    self.assertEqual(m["num_overlap_intervals"], 1)
    self.assertAlmostEqual(m["avg_overlap_ad_secs"], 1.0)
    self.assertAlmostEqual(m["ad_total_time_tts"], 3.0)
    # Silence = 20 s - 4 s of dialogue = 16 s.
    self.assertAlmostEqual(
        m["ad_total_speaking_time_per_silent_second"], 3 / 16
    )
    self.assertAlmostEqual(
        m["metric_G1.1_precision_avg_ad_in_visual_info"], 0.75
    )
    self.assertAlmostEqual(m["metric_2_1_does_not_restate"], 0.75)
    self.assertEqual(m["metric_g_2_avg_diff_time"], 2.5)
    self.assertEqual(m["metric_g_2_max_diff_time"], 5.0)
    self.assertEqual(m["onscreen_text_present"], 1)
    self.assertAlmostEqual(
        m["metric_1_2_narrates_onscreen_text"], 1.0, places=3
    )
    self.assertEqual(m["video_asr_descriptive"], 4.0)
    self.assertEqual(m["videoa11y_accurate"], 2.0)
    # TTS timings are attached to the matching AD segments.
    self.assertEqual(m["ad_json"][0]["duration_tts"], 2.0)
    self.assertTrue(os.path.exists(evaluation.metrics_path(self.out, "vid")))

  def test_without_tts_temporal_metrics_are_omitted(self):
    m = self.compute(tts_info=None)
    self.assertIn("tts_temporal_metrics", m["skipped_metrics"])
    for key in evaluation.TTS_METRIC_KEYS:
      self.assertNotIn(key, m)
    self.assertIn("ad_total_words", m)  # word statistics need no TTS
    self.assert_printed("WARNING: No TTS metadata for vid")

  def test_unparseable_ad_has_no_metric_values(self):
    m = self.compute(ad_text="I cannot describe this video.")
    self.assertEqual(m["status"], "no_parseable_ad")
    self.assertNotIn("total_overlap", m)
    self.assertNotIn("metric_G1.1_precision_avg_ad_in_visual_info", m)
    with open(evaluation.metrics_path(self.out, "vid")) as f:
      self.assertEqual(json.load(f)["status"], "no_parseable_ad")
    self.assert_printed("ERROR: No timed AD lines could be parsed for vid")

  def test_failed_judge_call_is_recorded_not_zeroed(self):
    def failing_judge(
        yt_id, prompt, dir, no_url=True  # pylint: disable=redefined-builtin
    ):
      if "does_not_restate" in dir:
        return None
      return fake_judge(yt_id, prompt, dir, no_url)

    failing_judge.supports_video = True
    m = self.compute(read_or_generate_and_save=failing_judge)
    self.assertEqual(m["failed_judge_calls"], ["does_not_restate"])
    self.assertNotIn("metric_2_1_does_not_restate", m)
    self.assert_printed("ERROR: LLM-judge calls failed for vid")

  def test_reference_free_needs_a_video_capable_judge(self):
    def text_judge(
        yt_id, prompt, dir, no_url=True  # pylint: disable=redefined-builtin
    ):
      return fake_judge(yt_id, prompt, dir, no_url)

    m = self.compute(read_or_generate_and_save=text_judge)
    self.assertIn("reference_free_metrics", m["skipped_metrics"])
    self.assertNotIn("video_asr_descriptive", m)
    self.assert_printed("WARNING: The LLM judge cannot watch videos")


class OnScreenTextTest(support.QuietTestCase):

  def test_header_in_intro_sentence_is_ignored(self):
    text = (
        "Here is a list of all text on screen, information about people.\n"
        "\n**Important Visual Events:**\n00:00:01.000 - A blue screen.\n"
        "\n**Text on Screen:**\n\n*   **00:00:00.000 - 00:00:01.373:** NEVER"
        " TOO SMALL\n\n**People in the video:**\n*   None\n"
    )
    self.assertEqual(metrics.extract_text_on_screen(text), ["NEVER TOO SMALL"])

  def test_markdown_heading_and_colon_milliseconds(self):
    text = (
        "### Visual Events\n00:00:00.000 - Stairs.\n\n### Text on Screen\n\n"
        "*   01:44:400 - LEARNING THE ROUTINE\n\n### People\n- A dancer.\n"
    )
    self.assertEqual(
        metrics.extract_text_on_screen(text), ["LEARNING THE ROUTINE"]
    )

  def test_stops_at_next_section_and_skips_none(self):
    text = (
        "### **Text on screen:**\n- **00:00:01 - 00:09:30**: Bayoona\n\n"
        "### **People in the video:**\n- **Woman / Host**: cooks dinner\n"
    )
    self.assertEqual(metrics.extract_text_on_screen(text), ["Bayoona"])
    self.assertEqual(
        metrics.extract_text_on_screen("**Text on Screen:**\nNone\n"), []
    )


class ReferenceFreeTest(support.QuietTestCase):

  def test_prompt_fills_inputs_and_marks_missing_ones(self):
    prompt = reference_free.build_prompt("video_asr", "AD TEXT", "", "")
    self.assertIn("AD TEXT", prompt)
    self.assertIn(reference_free.NOT_AVAILABLE, prompt)
    self.assertNotIn("_GOES_HERE_IF_AVAILABLE", prompt)

  def test_unparseable_output(self):
    self.assertIsNone(reference_free.parse_output("no json", "video_asr"))
    self.assertIsNone(
        reference_free.parse_output('{"Descriptive": 4}', "video_asr")
    )


class TtsTest(support.QuietTestCase):

  def test_incomplete_metadata_raises(self):
    d = tempfile.mkdtemp()
    with open(tts.metadata_path("v", d), "w") as f:
      json.dump([{"start": 1.0, "duration_tts": None}], f)
    with self.assertRaises(tts.TtsError):
      tts.load_metadata("v", d)

  def test_failed_segment_writes_no_metadata(self):
    d = tempfile.mkdtemp()
    with mock.patch.object(tts, "synthesize_text", return_value=None):
      with self.assertRaises(tts.TtsError):
        tts.synthesize_ad_file(mock.Mock(), "v", AD, d)
    self.assertFalse(os.path.exists(tts.metadata_path("v", d)))
    self.assert_printed("ERROR: TTS failed for segment 0 of v")

  def test_no_segments_raises(self):
    with self.assertRaises(tts.TtsError):
      tts.synthesize_ad_file(
          mock.Mock(), "v", "nothing here", tempfile.mkdtemp()
      )


def client_error(code, message):
  return errors.ClientError(
      code, {"error": {"code": code, "message": message, "status": "X"}}
  )


class GeminiErrorHandlingTest(support.QuietTestCase):

  def make_client(self, side_effect):
    client = mock.Mock()
    client.models.generate_content.side_effect = side_effect
    return client

  def test_unknown_model_is_fatal(self):
    client = self.make_client(client_error(404, "models/foo is not found"))
    with self.assertRaises(gemini_utils.FatalApiError):
      gemini_utils.generate_text(client, "foo", "hi", retries=3, delay=0)
    self.assertEqual(client.models.generate_content.call_count, 1)

  def test_invalid_api_key_is_fatal(self):
    client = self.make_client(client_error(400, "API key not valid"))
    with self.assertRaises(gemini_utils.FatalApiError):
      gemini_utils.generate_text(client, "m", "hi", retries=3, delay=0)

  def test_per_request_client_error_is_not_retried(self):
    client = self.make_client(client_error(400, "video is unavailable"))
    self.assertIsNone(
        gemini_utils.generate_text(client, "m", "hi", retries=3, delay=0)
    )
    self.assertEqual(client.models.generate_content.call_count, 1)
    self.assert_printed("ERROR: Gemini request rejected (400), not retrying")

  def test_rate_limit_is_retried(self):
    ok = mock.Mock(text="done")
    client = self.make_client([client_error(429, "quota"), ok])
    self.assertEqual(
        gemini_utils.generate_text(client, "m", "hi", retries=3, delay=0),
        "done",
    )
    self.assert_printed("Gemini attempt 1/3 failed: 429")


class FileUtilsTest(support.QuietTestCase):

  def test_dash_prefixed_video_id(self):
    self.assertEqual(
        file_utils.fix_dash_prefixed_values(
            ["--video_id", "--T4zFh82Ao", "--output_dir", "out"]
        ),
        ["--video_id=--T4zFh82Ao", "--output_dir", "out"],
    )

  def test_bundled_durations(self):
    durations = file_utils.load_video_durations(
        file_utils.DEFAULT_VIDEO_DURATIONS_FILE
    )
    self.assertEqual(len(durations), 433)
    self.assertEqual(durations["--T4zFh82Ao"], "2:43")
    self.assertIn("NtUjRbnMziM", durations)  # key had a stray space upstream


if __name__ == "__main__":
  unittest.main()
