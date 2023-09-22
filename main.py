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

"""Functions for organizing posing jobs for Vertex AI for summarization.

Run from project's root directory.
"""

import immutabledict
from vertexai.language_models import TextGenerationModel

_TEMPERATURE = 0.7
_MAX_OUTPUT_TOKENS = 300
_TOP_P = 0.8
_TOP_K = 40
_MODEL = 'text-bison@001'

_PARAMETERS = immutabledict.immutabledict({
    'temperature': _TEMPERATURE,
    'max_output_tokens': _MAX_OUTPUT_TOKENS,
    'top_p': _TOP_P,
    'top_k': _TOP_K,
})


def post_job_to_summarize(context: str) -> str:
  model = TextGenerationModel.from_pretrained(
      _MODEL,
  )
  response = model.predict(
      context,
      **_PARAMETERS,
  )

  return response
