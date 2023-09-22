# Copyright 2023 Google LLC
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

"""Tests for main.py."""

import unittest
from unittest import mock
import main

_FAKE_SUMMARY = 'fake summary'


class MainTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    self.mock_from_pretrained = mock.patch.object(
        main.TextGenerationModel,
        'from_pretrained',
        autospec=True).start()
    self.mock_from_pretrained.return_value.predict.return_value = (
        _FAKE_SUMMARY
    )

  def test_post_job_to_summarize(self):
    expected_context = (
        'This is a fake summary. It is rain in Tokyo, cloudy in Osaka.'
    )
    expected_calls = [mock.call(main._MODEL),
                      mock.call(main._MODEL).predict(
                          expected_context, **main._PARAMETERS),
                      ]

    actual_response = main.post_job_to_summarize(expected_context)

    self.mock_from_pretrained.assert_has_calls(expected_calls)
    self.assertEqual(_FAKE_SUMMARY, actual_response)


if __name__ == '__main__':
  unittest.main()
