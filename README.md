# Mentorea Platform
**Mentorea Platform** is a Platform designed to streamline mentor-mentee matching by leveraging AI-driven sentiment analysis and CV evaluation. Built as a graduation project, Mentorea consists of core components: *Sentiment Analysis* API for evaluating text inputs in Arabic and English, and a *CV Evaluation* System for assessing candidates' suitability as mentors based on their professional qualifications. The platform aims to support mentorship programs by providing automated, data-driven insights into candidates' skills and sentiments.

## Requirements

- Python 3.10
#### Install Python using MiniConda

1) Download and install MiniConda from [here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)
2) Create a new environment using the following command:
```bash
$ conda create -n mentorea_app python=3.10
```
3) Activate the environment:
```bash
$ conda activate mentorea_app
```

## Installation

### Install the required packages

```bash
$ pip install -r requirements.txt
```

### Setup the environment variables

```bash
$ cp .env.example .env
```

The environment variables is for config and secret variable Like `API_KEY`.

### Sentiment Analysis API
- Supports Arabic and English with BERT-based models.
- Fine Tunning Models on a real data.[Fine_Tunning_Code](https://github.com/abdullah1812/Fine-Tunning-BERT-for-Sentiment-Analysis)
- Utilizes `MARBERT` for Arabic and `distilbert` for English.
- Automatically detects input language using `langdetect`.
- Maps sentiment to a 0–100 scale for standardized evaluation.


### CV Evaluation System
- Extracts text from PDF CVs using PyPDF2 and langchain_community.
- Evaluates mentor eligibility with Groq’s openai/gpt-oss-safeguard-20b.
- Returns JSON with summary, score (0–100), strengths, gaps, and recommendation.
- Supports up to 5MB PDF files with secure handling.


___
## **Mentorea Team**
