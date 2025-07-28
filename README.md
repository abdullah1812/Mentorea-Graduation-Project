# Mentorea Platform
The **Mentorea Platform** is a Platform designed to streamline mentor-mentee matching by leveraging AI-driven sentiment analysis and CV evaluation. Built as a graduation project, Mentorea consists of three core components: Recmmendation System, Sentiment Analysis API for evaluating text inputs in Arabic and English, and a CV Evaluation System for assessing candidates' suitability as mentors based on their professional qualifications. The platform aims to support mentorship programs by providing automated, data-driven insights into candidates' skills and sentiments.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)

## Project Overview
Mentorea is a platform designed to streamline mentor-mentee matching and evaluation for mentorship programs. It addresses the challenge of inefficient manual matching by leveraging AI-driven recommendations, sentiment analysis, and CV evaluation. The platform consists of three Flask-based APIs:
- **Mentor Recommendation System**: Matches mentees with mentors using a hybrid of collaborative and content-based filtering.
- **Sentiment Analysis API**: Analyzes Arabic and English text inputs to assess communication tone.
- **CV Evaluation System**: Evaluates mentor candidates based on professional qualifications via PDF CV uploads.

## Features
### Mentor Recommendation System
- Personalized mentor-mentee matching using LightFM and Content_Base(Sentence Transformers).
- Supports new mentees via content-based filtering.
- Automated model retraining every 24 hours.
- Robust error handling and logging.

### Sentiment Analysis API
- Supports Arabic and English with BERT-based models.
- Utilizes `CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment` for Arabic and `distilbert-base-uncased-finetuned-sst-2-english` for English.
- Automatically detects input language using `langdetect`.
- Maps sentiment to a 0.5–10 scale for standardized evaluation.
- Handles up to 512-character inputs with language detection.

### CV Evaluation System
- Extracts text from PDF CVs using PyPDF2.
- Evaluates mentor eligibility with Groq’s Llama3 model.
- Returns JSON with summary, score (0–100), strengths, gaps, and recommendation.
- Supports up to 5MB PDF files with secure handling.

## Technologies Used
- **Machine Learning**:
  - LightFM: Hybrid recommendation modeling
  - Sentence Transformers (`all-MiniLM-L6-v2`): Skill embeddings
  - Transformers: Sentiment analysis (BERT models)
- **Web Development**:
  - Flask: API framework
  - Pyngrok: Local tunneling
- **Database**:
  - SQLAlchemy, pyodbc: SQL Server connectivity
  - Microsoft ODBC Driver 17: Database access
- **Utilities**:
  - PyPDF2: PDF text extraction
  - APScheduler: Periodic updates
  - Pandas, NumPy, SciPy, scikit-learn: Data processing
  - Langdetect: Language detection
  - Logging: Monitoring
  - dotenv: Environment management
  - Docker: Deployment
___
## **Mentorea Team**
