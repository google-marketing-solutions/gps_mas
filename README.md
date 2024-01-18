<!--
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Media Article Summarizer (MAS)

**Disclaimer: This is not an official Google product.**

## Overview
Media Article Summarizer is an open-source tool which utilizes Google Cloud's
state-of-the-art Large Language Models (LLMs) to generate article summaries.
It helps publishers who want to provide a summarization of their article to
improve user experience.

## Requirements
- Python 3.11.4+
- Google Cloud

## Get Started
1. If you don't already have one, create a Google Cloud project and enable
Vertex AI API following this guide
[link](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).
2. In you local or server, authenticate with one of following methods:
  - Using the gcloud CLI: [link](https://cloud.google.com/docs/authentication/gcloud)
  - Using service account impersonation:
[link](https://cloud.google.com/docs/authentication/use-service-account-impersonation)
  - Using API keys: link
[link](https://cloud.google.com/docs/authentication/api-keys)
3. Install Python libraries.
```
pip install requirements.txt
```
4. Prepare input files following **example below**.
5. Fill in your `project id` in `configs.yaml`.
6. Execute main.py and get the results in output.csv.

## Input File Format (CSV)

| Columns | Explanation |
| ---- | ---- |
| ID | Used to link with the output. Please use unique numbers. |
| Lang | Language of your prompt. As of 2023 Jan 15th, only JP is supported. |
| Model | Model that you would like to use. As of 2023 Jan 15, `text-bison`, `text-unicorn`, `gemini-pro` are supported. |
| Article | The article content to summarize. |

### Example

| ID | Lang | Model | Article |
| ---- | ---- | ---- | ---- |
| 1 | JP | gemini-pro | 田中さんは、東京でサラリーマンをしている30歳の男性です。彼は、幼い頃から世界中を旅してみたいと思っていました。しかし、仕事が忙しく、なかなか実現できませんでした。ある日、田中さんは会社を辞めて、世界一周をすることを決意しました。彼は、貯金を取り崩しながら、一人で旅を始めました。|
| 2 | JP | text-bison | 佐藤さんは、東京でサラリーマンをしている30歳の男性です。彼は、幼い頃から世界中を旅してみたいと思っていました。しかし、仕事が忙しく、なかなか実現できませんでした。ある日、佐藤さんは会社を辞めて、世界一周をすることを決意しました。彼は、貯金を取り崩しながら、一人で旅を始めました。|
