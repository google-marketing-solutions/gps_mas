# Copyright 2023-2024 Google LLC
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

"""Defines all error messages that are used across MAS."""

from typing import Final

NOT_EXISTS_LANG_INFO: Final[str] = (
    'The input LANG is not supported.'
)

NOT_EXISTS_INPUT_FILE: Final[str] = (
    'The input file dose not exist.'
)

NOT_AVAILABLE_LLM_MODEL: Final[str] = (
    'The input llm model name, {model_name}, is not available.'
)
