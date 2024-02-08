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

"""Functions for organizing posing jobs for Vertex AI for summarization.

Run from project's root directory.
"""

import enum
import os

import error_messages
import helpers
import immutabledict
import numpy as np
import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.preview.generative_models import GenerativeModel

_DEFAULT_CONFIGS = 'configs.yaml'
_TEMPERATURE = 0.7
_MAX_OUTPUT_TOKENS = 300
_TOP_P = 0.8
_TOP_K = 40

_REGION = 'us-central1'
_OUTPUT_FMT = '%s'
_NEWLINE = '\n'
_DELIMITER = ','
_ENCODING = 'utf8'
_SKIPROWS = 1

_PARAMETERS = immutabledict.immutabledict({
    'temperature': _TEMPERATURE,
    'max_output_tokens': _MAX_OUTPUT_TOKENS,
    'top_p': _TOP_P,
    'top_k': _TOP_K,
})

_TEMPLATE_PROMPTS = immutabledict.immutabledict({
    'JP': """ あなたは記事の編集担当です。次の記事の要約を考えてください。
    ただし次の条件を満たしてください。
    条件:
      - 要約だけ返してください。
      - ユーザーがもっと読みたくなるような要約を作成してください。
    記事:
      {article}
    """,
    'EN': """ As an article editor, please provide me with a summary of article
    that meets the following criteria.
    Criteria:
      - Summary only
      - Written to entice readers to read the full article
    Article:
      {article}
    """
})

_ID = 'ID'
_LANG = 'Lang'
_ARTICLE = 'Article'
_OUTPUT = 'Output'
_MODEL = 'Model'
_CSV_INPUT_DTYPES = np.dtype([
    (_ID, 'i4'),
    (_LANG, 'O'),
    (_MODEL, 'O'),
    (_ARTICLE, 'O'),
])

_CSV_OUTPUT_DTYPES = np.dtype([
    (_ID, 'i4'),
    (_OUTPUT, 'O'),
])

config_data = helpers.get_configs(_DEFAULT_CONFIGS)
_PROJECT_ID = config_data.project_id
_INPUT_CSV_FILE = config_data.input_csv_file
_OUTPUT_CSV_FILE = config_data.output_csv_file


@enum.unique
class ModelName(enum.Enum):
  """LLM Models that we can use in this solution."""
  TEXT_BISON = 'text-bison'
  TEXT_UNICORN = 'text-unicorn'
  GEMINI_PRO = 'gemini-pro'


@enum.unique
class AvailableLang(enum.Enum):
  """Available Language in this solution."""
  JAPANESE = 'JP'
  ENGLISH = 'EN'


def post_job_to_summarize(
    context: str,
    model: str = ModelName.GEMINI_PRO.value,
) -> str:
  """Posts job to Vertex AI API for summarization.

  Posts job for summarization task for Vertext AI API and gets a result of
  summarization from posting prompts.

  Args:
    context: A language of the text you would like to summarize.
    model: A model that you want to use.

  Returns:
    A string summarization results from Vertex AI API.

    'This is a sample summary. It is rainy in Tokyo.'
  """

  if model in [ModelName.TEXT_BISON.value, ModelName.TEXT_UNICORN.value]:
    llm_model = TextGenerationModel.from_pretrained(
        model,
    )
    response = llm_model.predict(
        context,
        **_PARAMETERS,
    ).text
  elif model in [ModelName.GEMINI_PRO.value]:
    llm_model = GenerativeModel(model)
    response = llm_model.generate_content(
        context,
    ).text
  else:
    raise ValueError(
        error_messages.NOT_AVAILABLE_LLM_MODEL.format(
            model_name=model
        )
    )

  return response


def emmbed_article_to_prompt(lang: str, article: str, model: str) -> None:
  """Embeds article to template prompt for summarization.

  Embeds article to template promt with specified language calls
  post_job_to_summarize function with prompts embedded artcile for the
  summarization.

  Args:
    lang: A language of the text you would like to summarize.
    article: A context that user would like to summaize.
    model: A model that you want to use.

  Returns:
    A string summarization results from Vertex AI API via post_job_to_summze
    function.

  Raises:
    ValueError: An error occurred if it enter a lang that does not exist in
    templates.
  """

  if lang in [availablelang.value for availablelang in AvailableLang]:
    context = _TEMPLATE_PROMPTS[lang].format(article=article)
    response = post_job_to_summarize(context, model)
    return response
  else:
    raise ValueError(error_messages.NOT_EXISTS_LANG_INFO)


def execute_summarization_from_csv_file() -> None:
  """Executes summarization process from csv file.

  Loads csv files with id, language and article. Calls emmbed_article_to_prompt
  in each line of loaded array with lang and article data. After compliting
  process of calling summarization process, they are saved to csv file.

  Raises:
    IOError: An error occurred if input file dose not exist.
  """

  if os.path.exists(_INPUT_CSV_FILE):
    media_data = np.loadtxt(
        _INPUT_CSV_FILE,
        delimiter=_DELIMITER,
        skiprows=_SKIPROWS,
        dtype=_CSV_INPUT_DTYPES,
    )
  else:
    raise IOError(error_messages.NOT_EXISTS_INPUT_FILE)

  output_array = np.zeros((1, 2))
  if media_data.size > 1.0:
    for row_no, (row_id, lang, model, article) in enumerate(media_data):
      output = str(emmbed_article_to_prompt(lang, article, model))
      if row_no == 0:
        output_array = np.array(
            [(row_id, output)],
            dtype=_CSV_OUTPUT_DTYPES,
        )
      else:
        output_array = np.concatenate(
            (
                output_array,
                np.array([(row_id, output)], dtype=_CSV_OUTPUT_DTYPES)
            ),
            axis=0,
        )
  else:
    output = emmbed_article_to_prompt(
        media_data[_LANG],
        media_data[_ARTICLE],
        media_data[_MODEL]
    )
    output_array = np.array([(1, output)], dtype=_CSV_OUTPUT_DTYPES)

  np.savetxt(
      _OUTPUT_CSV_FILE,
      output_array,
      delimiter=_DELIMITER,
      newline=_NEWLINE,
      fmt=_OUTPUT_FMT,
      encoding=_ENCODING
  )


def main() -> None:
  """Executes Media Atricle Summaizaer all steps."""
  vertexai.init(project=_PROJECT_ID, location=_REGION)
  execute_summarization_from_csv_file()


if __name__ == '__main__':
  main()
