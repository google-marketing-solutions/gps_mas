<!--
Copyright 2023-2024 Google LLC

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

> [!CAUTION]
> Note that this solution has been archived as of June 2025.
> No further development or updates will be made to the solution on the Github.

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
pip install -r requirements.txt
```
4. Prepare input files following **example below**.
5. Fill in your `project id` in `configs.yaml`.
6. Execute main.py and get the results in output.csv.

## Input File Format (CSV)

| Columns | Explanation |
| ---- | ---- |
| ID | Used to link with the output. Please use unique numbers. |
| Lang | Language of your prompt. As of 2024 Feb 8th, only JP and EN are supported. |
| Model | Model that you would like to use. As of 2024 Jun 26, `text-bison`, `text-unicorn`, `gemini-1.0-pro`, `gemini-1.5-pro` and `gemini-1.5-flash` are supported. |
| Article | The article content to summarize. |

### Example

| ID | Lang | Model | Article |
| ---- | ---- | ---- | ---- |
| 1 | JP | gemini-pro | 田中さんは、東京でサラリーマンをしている30歳の男性です。彼は、幼い頃から世界中を旅してみたいと思っていました。しかし、仕事が忙しく、なかなか実現できませんでした。ある日、田中さんは会社を辞めて、世界一周をすることを決意しました。彼は、貯金を取り崩しながら、一人で旅を始めました。|
| 2 | JP | text-bison | 佐藤さんは、東京でサラリーマンをしている30歳の男性です。彼は、幼い頃から世界中を旅してみたいと思っていました。しかし、仕事が忙しく、なかなか実現できませんでした。ある日、佐藤さんは会社を辞めて、世界一周をすることを決意しました。彼は、貯金を取り崩しながら、一人で旅を始めました。|
| 3 | EN | text-unicorn | Tom works in e-commerce company located in Tokyo when he is 30 years old. Tom has always wanted to travel around the world. However, Tom was busy with work and couldn't make it happen. One day, Tom decided to quit his job and travel around the world. He started the journey alone, dipping into his savings.|
| 4 | EN | gemini-1.5-pro | Tom works in e-commerce company located in Tokyo when he is 30 years old. Tom has always wanted to travel around the world. However, Tom was busy with work and couldn't make it happen. One day, Tom decided to quit his job and travel around the world. He started the journey alone, dipping into his savings.|
| 5 | EN | gemini-1.5-flash | Tom works in e-commerce company located in Tokyo when he is 30 years old. Tom has always wanted to travel around the world. However, Tom was busy with work and couldn't make it happen. One day, Tom decided to quit his job and travel around the world. He started the journey alone, dipping into his savings.|
