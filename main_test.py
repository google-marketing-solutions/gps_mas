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

  @mock.patch.object(main, 'post_job_to_summarize')
  def test_emmbed_article_to_prompt_ja(self, mock_post_job_to_summarize):
    expected_lang = 'JA'
    expected_article = 'これはテストです。'

    _ = main.emmbed_article_to_prompt(
        expected_lang,
        expected_article,
    )

    mock_post_job_to_summarize.assert_called_once_with(
        main._TEMPLATE_PROMPTS[expected_lang].format(article=expected_article)
    )

  def test_emmbed_article_to_prompt_failure_with_wrong_lang(self):
    expected_lang = 'FAKE_LANG'
    expected_article = 'This is a test.'

    with self.assertRaisesRegex(
        ValueError,
        'The input LANG is not supported.'
    ):
      main.emmbed_article_to_prompt(
          expected_lang,
          expected_article,
      )

if __name__ == '__main__':
  unittest.main()
