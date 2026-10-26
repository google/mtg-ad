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

"""Tests for analysis/aggregate.py (tables from per-video metric JSONs)."""

import json
import math
import os
import tempfile
import unittest

from analysis import aggregate as agg
from tests import support

PRIO = "metric_G1.1_precision_avg_ad_in_visual_info"
ONSCREEN = "metric_1_2_narrates_onscreen_text"
TIMELINESS = "metric_g_2_avg_diff_time"


def write_metrics(root, config, video_id, **values):
  d = os.path.join(root, config)
  os.makedirs(d, exist_ok=True)
  with open(os.path.join(d, f"output_{video_id}.json"), "w") as f:
    json.dump(values, f)
  return d


class AggregateTest(support.QuietTestCase):

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    r = self.root
    # Config A has videos v1-v3, B has v1, v2, v4.
    write_metrics(
        r,
        "A",
        "v1",
        **{
            PRIO: 1.0,
            ONSCREEN: 0.5,
            "onscreen_text_present": 1,
            TIMELINESS: 2.0,
            "video_duration_seconds": 60,
        },
    )
    write_metrics(
        r,
        "A",
        "v2",
        **{
            PRIO: 0.5,
            ONSCREEN: 1.0,
            "onscreen_text_present": 0,
            TIMELINESS: 90.0,
            "video_duration_seconds": 60,
        },
    )
    write_metrics(r, "A", "v3", **{PRIO: 0.0})
    write_metrics(r, "B", "v1", **{PRIO: 0.2})
    write_metrics(r, "B", "v2", **{PRIO: 0.4})
    write_metrics(r, "B", "v4", **{PRIO: 0.9})
    with open(os.path.join(r, "B", "output_v5.json"), "w") as f:
      f.write("")  # unreadable file is skipped
    self.dirs = {c: os.path.join(r, c) for c in ("A", "B")}

  def test_per_video_filters(self):
    df = agg.load_metrics_dir(self.dirs["A"])
    v2 = df.set_index("video_id").loc["v2"]
    self.assertTrue(math.isnan(v2[ONSCREEN]))  # no on-screen text in video
    self.assertTrue(math.isnan(v2[TIMELINESS]))  # longer than the video
    self.assertEqual(len(agg.load_metrics_dir(self.dirs["B"])), 3)  # v5 skipped
    self.assert_printed("Skipping unreadable")

  def test_summary_uses_common_videos_only(self):
    df = agg.load_configs(self.dirs)
    s = agg.summarize(df, ["A", "B"])
    self.assertEqual(s.attrs["num_common_videos"], 2)  # v1, v2
    row = s[(s.metric == PRIO) & (s.config == "A")].iloc[0]
    self.assertAlmostEqual(row["mean"], 0.75)
    self.assertEqual(row["n"], 2)
    # 1.96 * std(ddof=1) / sqrt(n) for values [1.0, 0.5].
    self.assertAlmostEqual(row["ci95"], 1.96 * (0.5**2 / 2) ** 0.5 / 2**0.5)
    # Metrics without any value (e.g. Quality-RF here) are left out.
    self.assertNotIn("video_asr_descriptive", set(s.metric))

  def test_summary_all_videos_per_config(self):
    df = agg.load_configs(self.dirs)
    s = agg.summarize(df, ["A", "B"], metrics=[PRIO], common_videos=False)
    self.assertEqual(s.attrs["num_videos_per_config"], {"A": 3, "B": 3})
    row_a = s[s.config == "A"].iloc[0]
    self.assertAlmostEqual(row_a["mean"], 0.5)  # v1-v3: 1.0, 0.5, 0.0
    row_b = s[s.config == "B"].iloc[0]
    self.assertAlmostEqual(row_b["mean"], 0.5)  # v1, v2, v4: 0.2, 0.4, 0.9

  def test_exclusions(self):
    exclusions_file = os.path.join(self.root, "scoring_exclusions.json")
    with open(exclusions_file, "w") as f:
      json.dump({"exclusions": {"B": ["v2"]}}, f)
    df = agg.load_configs(
        self.dirs, exclusions=agg.load_exclusions(exclusions_file)
    )
    self.assertNotIn("v2", set(df[df.config == "B"].video_id))
    s = agg.summarize(df, ["A", "B"])
    self.assertEqual(s.attrs["num_common_videos"], 1)  # only v1 left
    self.assertEqual(agg.load_exclusions(os.path.join(self.root, "nope")), {})

  def test_latex_table_shades_best_and_second(self):
    df = agg.load_configs(self.dirs)
    s = agg.summarize(df, ["A", "B"], metrics=[PRIO])
    latex = agg.latex_table(s, {"A": "System A", "B": "System B"})
    prio_row = next(
        l for l in latex.splitlines() if l.startswith("Prioritization")
    )
    a_cell, b_cell = prio_row.split(" & ")[1:]
    self.assertIn("E2E2E2", a_cell)  # A (0.75) beats B (0.30) on a ↑ metric
    self.assertIn("0.75", a_cell)
    self.assertIn("F2F2F2", b_cell)


if __name__ == "__main__":
  unittest.main()
