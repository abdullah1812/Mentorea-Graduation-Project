import os 
os.environ["TRANSFORMERS_NO_TF"] = "1"
from transformers import pipeline
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
arabic_path = os.path.join(BASE_DIR, "mentorea_sentiment_model")
english_path = os.path.join(BASE_DIR, "mentorea_english_model_final")

id2label = {
    'LABEL_0': "neutral", 'LABEL_1': "frustrated",
    'LABEL_2': 'positive', 'LABEL_3': 'negative', 'LABEL_4': 'grateful'
}

class SentimentAnalysis:
    def __init__(self):
        self.mentorea_analyzer_Ar = pipeline("text-classification", model=arabic_path, tokenizer=arabic_path, local_files_only=True)
        self.mentorea_analyzer_En = pipeline("text-classification", model=english_path, tokenizer=english_path, local_files_only=True)

    def analyze_mentee_review(self, review_text):
        contains_arabic = re.search(r'[\u0600-\u06FF]', review_text)
    
        if contains_arabic:
            model_used = "MARBERT (Arabic/Mix)"
            result = self.mentorea_analyzer_Ar(review_text)[0]
        else:
            model_used = "DistilBERT (English)"
            result = self.mentorea_analyzer_En(review_text)[0]
        current_label = result['label']
        if current_label in id2label:
            result['label'] = id2label[current_label]

        return result, model_used

<<<<<<< HEAD
sentiment_service = SentimentAnalysis()
=======
sentiment_service = SentimentAnalysis()
>>>>>>> eec3e76f0a1621ebdf6a192ee1b53f3e149e5f10
