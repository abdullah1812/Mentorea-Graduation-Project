from .Base import BaseController
from services.sentiment_analysis import sentiment_service
class SentimentAnalysisController(BaseController):

    def __init__(self):
        super().__init__()
    def map_sentiment_to_value(self, sentiment_result):
        label = sentiment_result['label']
        confidence = sentiment_result['score'] 

        if label == 'frustrated':
            final_score = 20 - (confidence * 20)
        elif label == 'negative':
            final_score = 40 - (confidence * 19)
        elif label == 'neutral':
            final_score = 41 + (confidence * 19)
        elif label == 'positive':
            final_score = 61 + (confidence * 19)
        elif label == 'grateful':
            final_score = 81 + (confidence * 19)
        else:
            final_score = 50

        return int(final_score)
    
    def analyze_sentiment(self, text:str):
        try:
            sentiment_values, model_used = sentiment_service.analyze_mentee_review(text) 
        except Exception as e:
            print(e)
        value = self.map_sentiment_to_value(sentiment_values)
        
        return value, model_used, sentiment_values
    
