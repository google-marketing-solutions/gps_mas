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

import error_messages
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

_TEMPLATE_PROMPTS = immutabledict.immutabledict({
    'JA': """ あなたは記事の編集担当です。次の記事の要約を考えてください。
    ただし次の条件を満たしてください。
    条件:
      - 要約だけ返してください。
      - ユーザーがもっと読みたくなるような要約を作成してください。
    記事:
      {article}
    """
})


def post_job_to_summarize(context: str) -> str:
  """Posts job to Vertex AI API for summarization.

  Posts job for summarization task for Vertext AI API and gets a result of
  summarization from posting prompts.

  Args:
    context: A language of the text you would like to summarize.

  Returns:
    A string summarization results from Vertex AI API.

    'This is a sampple summary. It is rain in Tokyo.'
  """
  model = TextGenerationModel.from_pretrained(
      _MODEL,
  )
  response = model.predict(
      context,
      **_PARAMETERS,
  )

  return response


def emmbed_article_to_prompt(lang: str, article: str) -> None:
  """Embeds article to template prompt for summarization.

  Embeds article to template promt with specified language calls
  post_job_to_summarize function with prompts embedded artcile for the
  summarization.

  Args:
    lang: A language of the text you would like to summarize.
    article: A context that user would like to summaize.

  Returns:
    A string summarization results from Vertex AI API via post_job_to_summze
    function.

  Raises:
    ValueError: An error occurred if it enter a lang that does not exist in
    templates.
  """

  if lang == 'JA':
    context = _TEMPLATE_PROMPTS['JA'].format(article=article)
    response = post_job_to_summarize(context)
    return response
  else:
    raise ValueError(error_messages.NOT_EXISTS_LANG_INFO)
