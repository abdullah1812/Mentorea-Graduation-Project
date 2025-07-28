# Mentorea Platform

The **Mentorea Platform** is a platform designed to streamline mentor-mentee matching and evaluation by leveraging AI-driven recommendation systems, sentiment analysis, and CV evaluation. Built as a graduation project, Mentorea consists of three core components: a Mentor Recommendation System for personalized mentor-mentee matching, a Sentiment Analysis API for evaluating text inputs in Arabic and English, and a CV Evaluation System for assessing candidates' suitability as mentors based on their professional qualifications. The platform aims to support mentorship programs by providing automated, data-driven insights into candidates' skills, sentiments, and compatibility.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)

## Project Overview
The Mentorea Platform integrates three Flask-based APIs to support mentorship evaluation and matching:
1. **Mentor Recommendation System**: Matches mentees with mentors based on skills, experience, and preferences using a hybrid approach combining collaborative filtering (LightFM) and content-based filtering (Sentence Transformers). It connects to a SQL Server database to fetch mentee, mentor, and interaction data, with periodic model updates via APScheduler.
2. **Sentiment Analysis API**: Analyzes text inputs in Arabic and English to determine sentiment (positive, negative, or neutral), providing insights into communication tone. It uses BERT-based models and custom scoring to map sentiments to a 1–10 scale.
3. **CV Evaluation System**: Processes uploaded PDF CVs to evaluate candidates for mentor roles based on criteria like professional experience, leadership, and soft skills. It leverages the Groq AI platform for detailed analysis and generates structured JSON reports.
The platform demonstrates proficiency in machine learning, natural language processing, API development, and database integration, offering a robust solution for mentorship programs.

## Features
### Mentor Recommendation System
- Provides personalized mentor recommendations for mentees using a hybrid model combining collaborative filtering (LightFM) and content-based filtering (Sentence Transformers).
- Supports both existing mentees (using historical interaction data) and new mentees (using profile-based similarity).
- Fetches data from a SQL Server database, including mentee profiles, mentor qualifications, and session ratings.
- Uses Sentence Transformers (`all-MiniLM-L6-v2`) for embedding mentor and mentee skills, locations, and specializations.
- Implements automated model retraining every 24 hours using APScheduler.
- Served via a Flask API endpoint (`/recommend`) with ngrok for public access.
- Includes robust error handling and logging for database connectivity and model inference.

### Sentiment Analysis API
- Supports sentiment analysis for Arabic and English texts.
- Utilizes `CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment` for Arabic and `distilbert-base-uncased-finetuned-sst-2-english` for English.
- Automatically detects input language using `langdetect`.
- Maps sentiment and confidence to a custom value range (1–10).
- Handles text inputs up to 512 characters with validation and error handling.

### CV Evaluation System
- Accepts PDF CV uploads and extracts text using `PyPDF2`.
- Evaluates mentor eligibility using the Groq API (`llama3-70b-8192` model).
- Assesses criteria including ≥3 years of professional experience, leadership, communication skills, education, and soft skills.
- Returns a JSON response with a summary, score (0–100), strengths, gaps, and a recommendation.
- Implements secure file handling, logging, and a 5MB file size limit.

## Technologies Used
- **Python**: Core programming language.
- **Flask**: Web framework for API development.
- **LightFM**: For hybrid recommendation modeling.
- **Sentence Transformers**: For embedding skills and profiles in the recommendation system.
- **Transformers**: For sentiment analysis with BERT-based models.
- **PyPDF2**: For PDF text extraction.
- **Groq API**: For AI-powered CV analysis.
- **SQLAlchemy** and **pyodbc**: For SQL Server database connectivity.
- **Langdetect**: For language detection.
- **Pyngrok**: For local development tunneling.
- **APScheduler**: For scheduling periodic model updates.
- **Pandas**, **NumPy**, **SciPy**, **scikit-learn**: For data processing and machine learning.
- **Logging**: For application monitoring and debugging.
- **JSON**: For structured API responses.
- **dotenv**: For environment variable management.
- **Docker**: For containerized deployment.
- **Microsoft ODBC Driver 17**: For SQL Server database access.
___
## **Mentorea Team**


