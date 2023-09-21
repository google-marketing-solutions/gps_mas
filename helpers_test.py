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

"""Tests for helpers."""

import types
import unittest
from unittest import mock

import helpers


class UtilsTest(unittest.TestCase):

  def test_get_configs_returns(self):
    test_configs = """
      project_id: 'source_project_id'
    """
    mock_open = mock.mock_open(read_data=test_configs)
    with mock.patch('builtins.open', mock_open, create=True):
      config = helpers.get_configs('configs.yaml')

    self.assertIsInstance(config, types.SimpleNamespace)

    for attr in ['project_id']:
      self.assertTrue(hasattr(config, attr))
