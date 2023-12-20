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

import os
import unittest
from unittest import mock

from absl.testing import parameterized
import main
import numpy as np


_FAKE_SUMMARY = 'fake summary'
_FAKE_NP_LOADTXT_INPUT_2_LINES = np.array(
    [(1, 'JP', 'text-bison', 'fake article 01-02'),
     (2, 'JP', 'text-bison', 'fake article 02-02')],
    dtype=main._CSV_INPUT_DTYPES,
)
_FAKE_NP_LOADTXT_INPUT_1_LINE = np.array(
    [(1, 'JP', 'text-bison', 'fake article 01-01')],
    dtype=main._CSV_INPUT_DTYPES,
)
_FAKE_COMMON_PATH = '/path/to'
_FAKE_INPUT_FILEPATH = os.path.join(_FAKE_COMMON_PATH, 'input.csv')
_FAKE_OUTPUT_FILEPATH = os.path.join(_FAKE_COMMON_PATH, 'output.csv')


class MainTest(parameterized.TestCase):
  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    self.mock_from_pretrained = mock.patch.object(
        main.TextGenerationModel,
        'from_pretrained',
        autospec=True,
    ).start()
    self.mock_from_pretrained.return_value.predict.return_value = (
        _FAKE_SUMMARY
    )
    self.mock_generative_model = mock.patch.object(
        main,
        'GenerativeModel',
        autospec=True,
    ).start()
    self.mock_generative_model.return_value.generate_content.return_value = (
        _FAKE_SUMMARY
    )
    mock.patch(
        'main._INPUT_CSV_FILE',
        _FAKE_INPUT_FILEPATH,
    ).start()
    mock.patch(
        'main._OUTPUT_CSV_FILE',
        _FAKE_OUTPUT_FILEPATH,
    ).start()

  def test_post_job_to_summarize(self):
    expected_context = (
        'This is a fake summary. It is rain in Tokyo, cloudy in Osaka.'
    )
    expected_model = 'text-bison'
    expected_calls = [mock.call(main._DEFAULT_MODEL),
                      mock.call(main._DEFAULT_MODEL).predict(
                          expected_context, **main._PARAMETERS),
                      ]

    actual_response = main.post_job_to_summarize(
        expected_context,
        expected_model
    )

    self.mock_from_pretrained.assert_has_calls(expected_calls)
    self.assertEqual(_FAKE_SUMMARY, actual_response)

  def test_post_job_to_summarize_with_gemini_pro(self):
    expected_context = (
        'This is a fake summary. It is rain in Tokyo, cloudy in Osaka.'
    )
    expected_model = 'gemini-pro'
    expected_calls = [mock.call(expected_model),
                      mock.call(expected_model).generate_content(
                          expected_context
                      ),
                      ]

    actual_response = main.post_job_to_summarize(
        expected_context,
        expected_model
    )

    self.mock_generative_model.assert_has_calls(expected_calls)
    self.assertEqual(_FAKE_SUMMARY, actual_response)

  def test_post_job_to_summarize_with_not_exist_llm_model(self):
    expected_context = (
        'This is a fake summary. It is rain in Tokyo, cloudy in Osaka.'
    )
    expected_model = 'non-exist-model'

    with self.assertRaisesRegex(
        ValueError,
        'The input llm model name, {model_name}, is not available.'.format(
            model_name=expected_model
        ),
    ):
      _ = main.post_job_to_summarize(
          expected_context,
          expected_model
      )

  @mock.patch.object(main, 'post_job_to_summarize')
  def test_emmbed_article_to_prompt_ja(self, mock_post_job_to_summarize):
    expected_lang = 'JP'
    expected_article = 'これはテストです。'
    expected_model = 'text-bison'

    main.emmbed_article_to_prompt(
        expected_lang,
        expected_article,
        expected_model,
    )

    mock_post_job_to_summarize.assert_called_once_with(
        main._TEMPLATE_PROMPTS[expected_lang].format(article=expected_article),
        expected_model,
    )

  def test_emmbed_article_to_prompt_failure_with_wrong_lang(self):
    expected_lang = 'FAKE_LANG'
    expected_article = 'This is a test.'
    expected_model = 'text-bison'

    with self.assertRaisesRegex(
        ValueError,
        'The input LANG is not supported.',
    ):
      main.emmbed_article_to_prompt(
          expected_lang,
          expected_article,
          expected_model,
      )

  @parameterized.named_parameters([
      {
          'testcase_name': '2_rows_file_success',
          'loadtxt_input_data': _FAKE_NP_LOADTXT_INPUT_2_LINES,
          'expected_output_data': (
              np.array([(1, 'fake summary'), (2, 'fake summary')],
                       dtype=[('ID', '<i4'), ('Output', 'O')]),
          ),
      },
      {
          'testcase_name': '1_row_file_success',
          'loadtxt_input_data': _FAKE_NP_LOADTXT_INPUT_1_LINE,
          'expected_output_data': (
              np.array([(1, 'fake summary')],
                       dtype=[('ID', '<i4'), ('Output', 'O')]),
          ),
      }
  ])
  @mock.patch.object(main.np, 'savetxt')
  @mock.patch.object(main.os.path, 'exists', return_value=True)
  def test_execute_summarization_from_csv_file_with(
      self,
      _,
      mock_np_savetxt,
      loadtxt_input_data,
      expected_output_data,
  ):
    mock_np_loadtxt = mock.patch.object(
        main.np,
        'loadtxt',
        return_value=loadtxt_input_data,
        autospec=True
    ).start()

    main.execute_summarization_from_csv_file()

    mock_np_loadtxt.assert_called_with(
        _FAKE_INPUT_FILEPATH,
        delimiter=main._DELIMITER,
        skiprows=main._SKIPROWS,
        dtype=np.dtype([('ID', '<i4'),
                        ('Lang', 'O'),
                        ('Model', 'O'),
                        ('Article', 'O')]
                       )
    )
    np.testing.assert_array_equal(
        expected_output_data[0],
        mock_np_savetxt.call_args[0][1],
    )
    mock_np_savetxt.assert_called_with(
        _FAKE_OUTPUT_FILEPATH,
        mock.ANY,
        delimiter=main._DELIMITER,
        newline=main._NEWLINE,
        fmt=main._OUTPUT_FMT,
        encoding=main._ENCODING,
    )

  @mock.patch.object(main.os.path, 'exists', return_value=False)
  def test_execute_summarization_from_csv_file_with_failure_no_file(
      self,
      _
  ):
    with self.assertRaisesRegex(
        IOError,
        'The input file dose not exist.'
    ):
      main.execute_summarization_from_csv_file()


if __name__ == '__main__':
  unittest.main()
