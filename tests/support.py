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

"""Shared test helpers."""

import contextlib
import io
import unittest


class QuietTestCase(unittest.TestCase):
  """Captures what the code under test prints, so passing runs stay quiet.

  The pipeline prints progress and loud ERROR / WARNING messages by design.
  Tests check the important ones with assert_printed(), which shows the
  captured output when it fails.
  """

  def setUp(self):
    super().setUp()
    self._stdout = io.StringIO()
    redirect = contextlib.redirect_stdout(self._stdout)
    redirect.__enter__()
    self.addCleanup(redirect.__exit__, None, None, None)

  @property
  def printed(self) -> str:
    return self._stdout.getvalue()

  def assert_printed(self, text: str) -> None:
    self.assertIn(text, self.printed, msg=f"Output was:\n{self.printed}")
