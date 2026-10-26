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

"""Offline tests for local MP4 inputs and prompt filling (no API calls)."""

import os
import tempfile
from unittest import mock

from common import ad_format
from common import local_video
from evaluation import evaluation
from generation import gemini_prompt
from generation import generation
from tests import support


def _touch(directory: str, name: str) -> str:
  path = os.path.join(directory, name)
  with open(path, "wb") as f:
    f.write(b"\x00")
  return path


class VideoIdTest(support.QuietTestCase):

  def test_youtube_ids_are_unchanged(self):
    for vid in ("dQw4w9WgXcQ", "--T4zFh82Ao", "0jK0ytvjv-E", "a_b"):
      self.assertEqual(local_video.video_id_from_path(f"v/{vid}.mp4"), vid)

  def test_titles_become_safe_ids(self):
    self.assertEqual(
        local_video.video_id_from_path("How to build a catio ｜ DIY.mp4"),
        "How_to_build_a_catio_DIY",
    )
    self.assertEqual(
        local_video.video_id_from_path("[No Music] Frappuccino？.mp4"),
        "No_Music_Frappuccino",
    )
    self.assertEqual(local_video.video_id_from_path("｜｜.mp4"), "video")


class LocalVideoLookupTest(support.QuietTestCase):

  def setUp(self):
    super().setUp()
    self.dir = tempfile.mkdtemp()
    _touch(self.dir, "dQw4w9WgXcQ.mp4")
    _touch(self.dir, "World’s Grumpiest Cat.mp4")
    _touch(self.dir, "notes.txt")

  def test_list_local_videos(self):
    self.assertEqual(
        [vid for vid, _ in local_video.list_local_videos(self.dir)],
        ["World_s_Grumpiest_Cat", "dQw4w9WgXcQ"],
    )

  def test_find_by_exact_and_derived_id(self):
    self.assertEqual(
        local_video.find_local_video(self.dir, "dQw4w9WgXcQ"),
        os.path.join(self.dir, "dQw4w9WgXcQ.mp4"),
    )
    self.assertEqual(
        local_video.find_local_video(self.dir, "World_s_Grumpiest_Cat"),
        os.path.join(self.dir, "World’s Grumpiest Cat.mp4"),
    )
    self.assertIsNone(local_video.find_local_video(self.dir, "missing"))
    self.assertIsNone(local_video.find_local_video("", "dQw4w9WgXcQ"))

  def test_colliding_ids_are_rejected(self):
    _touch(self.dir, "World's Grumpiest Cat.mp4")
    with self.assertRaises(ValueError):
      local_video.list_local_videos(self.dir)

  def test_judge_uses_derived_id_mp4(self):
    with mock.patch.object(
        evaluation.gemini_utils, "make_video_part", return_value="part"
    ) as make_part, mock.patch.object(
        evaluation.gemini_utils, "generate_with_video", return_value="ok"
    ):
      judge = evaluation.make_gemini_judge(None, "m", self.dir)
      judge("World_s_Grumpiest_Cat", "p", self.dir + "/cache", no_url=False)
    make_part.assert_called_once_with(
        None,
        "World_s_Grumpiest_Cat",
        os.path.join(self.dir, "World’s Grumpiest Cat.mp4"),
    )

  def test_judge_reuses_given_video_part(self):
    with mock.patch.object(
        evaluation.gemini_utils, "make_video_part"
    ) as make_part, mock.patch.object(
        evaluation.gemini_utils, "generate_with_video", return_value="ok"
    ) as generate:
      judge = evaluation.make_gemini_judge(
          None, "m", video_parts={"vid": "uploaded"}
      )
      judge("vid", "p", self.dir + "/cache", no_url=False)
    make_part.assert_not_called()
    self.assertEqual(generate.call_args[0][2], "uploaded")


class AgenticPromptTest(support.QuietTestCase):

  def test_registered_and_transcript_free(self):
    prompt = gemini_prompt.PROMPTS["agentic_gemini_3p7_up"]
    self.assertIs(prompt, gemini_prompt.AGENTIC_GEMINI_3P7_UP)
    self.assertNotIn(generation.TRANSCRIPT_PLACEHOLDER, prompt)
    self.assertIn('"response"', prompt)
    self.assertNotIn("-->", prompt)  # No WebVTT output.

  def test_duration_is_filled(self):
    prompt = gemini_prompt.AGENTIC_GEMINI_3P7_UP
    for duration, text in (
        (287.04, "287.0"),
        ("4:47", "287.0"),
        (None, "unknown"),
    ):
      filled = generation.fill_prompt(prompt, "", duration)
      self.assertNotIn(generation.DURATION_PLACEHOLDER, filled)
      self.assertIn(f"Total Duration: {text} seconds", filled)

  def test_other_prompts_unchanged(self):
    for name in ("timing_asr", "timing_asr_json_example"):
      prompt = gemini_prompt.PROMPTS[name]
      self.assertEqual(
          generation.fill_prompt(prompt, "T", 10),
          generation.fill_transcript(prompt, "T"),
      )

  def test_prompt_example_lines_parse(self):
    example = (
        '{"response": "0:00:10.000-STANDARD The fox walks.\\n'
        '0:01:21.000-STANDARD The fox approaches."}'
    )
    self.assertEqual(
        ad_format.parse_ad_lines(example),
        [(10.0, "The fox walks."), (81.0, "The fox approaches.")],
    )


class GeminiInferenceTest(support.QuietTestCase):
  """generation.gemini_inference() follows run_end_to_end's generation."""

  def setUp(self):
    super().setUp()
    self.dir = tempfile.mkdtemp()
    self.mp4 = _touch(self.dir, "My video.mp4")
    patcher = mock.patch.object(generation, "client", "client")
    patcher.start()
    self.addCleanup(patcher.stop)

  def _run(self, prompt_type, **kwargs):
    with mock.patch.object(
        generation.gemini_utils, "make_video_part", return_value="part"
    ) as make_part, mock.patch.object(
        generation.transcripts_lib,
        "generate_transcript",
        return_value="0:00:01.000--0:00:02.000 hi",
    ) as transcribe, mock.patch.object(
        generation.gemini_utils, "generate_with_video", return_value="AD"
    ) as generate:
      out = generation.gemini_inference(
          gemini_prompt.PROMPTS[prompt_type],
          "My_video",
          None,
          "",
          "",
          mp4_path=self.mp4,
          **kwargs,
      )
    return out, make_part, transcribe, generate

  def test_missing_transcript_is_generated_and_cached(self):
    cache = os.path.join(self.dir, "generated_transcript")
    out, make_part, transcribe, generate = self._run(
        "timing_asr", generated_transcript_dir=cache
    )
    self.assertEqual(out, "AD")
    make_part.assert_called_once_with("client", "My_video", self.mp4)
    transcribe.assert_called_once()
    self.assertIn("0:00:01.000--0:00:02.000 hi", generate.call_args[0][3])
    with open(os.path.join(cache, "output_My_video.txt")) as f:
      self.assertIn("hi", f.read())
    # Second run reuses the cached transcript.
    _, _, transcribe, _ = self._run(
        "timing_asr", generated_transcript_dir=cache
    )
    transcribe.assert_not_called()

  def test_missing_transcript_without_generation_fails(self):
    out, _, transcribe, generate = self._run("timing_asr")
    self.assertIsNone(out)
    transcribe.assert_not_called()
    generate.assert_not_called()

  def test_agentic_prompt_needs_no_transcript(self):
    with mock.patch.object(
        local_video, "probe_duration_seconds", return_value=84.3
    ):
      out, _, transcribe, generate = self._run(
          "agentic_gemini_3p7_up",
          generated_transcript_dir=os.path.join(self.dir, "t"),
      )
    self.assertEqual(out, "AD")
    transcribe.assert_not_called()
    self.assertIn("Total Duration: 84.3 seconds", generate.call_args[0][3])


if __name__ == "__main__":
  import unittest  # pylint: disable=g-import-not-at-top

  unittest.main()
