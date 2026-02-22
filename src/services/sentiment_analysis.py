# import os 
# os.environ["TRANSFORMERS_NO_TF"] = "1"
# from transformers import pipeline
# import re

# id2lable = {'LABEL_0':"neutral", 'LABEL_1':"frustrated",
#             'LABEL_2':'positive', 'LABEL_3':'negative', 'LABEL_4':'grateful'}
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# arabic_path = os.path.join(
#     BASE_DIR,
#     "mentorea_sentiment_model"
# )
# english_path = os.path.join(
#     BASE_DIR,
#     "mentorea_english_model_final"
# )
# class SentimentAnalysis:
#     def __init__(self):
#         self.mentorea_analyzer_Ar = pipeline("text-classification", model=arabic_path, tokenizer=arabic_path)
#         self.mentorea_analyzer_En = pipeline("text-classification", model=english_path, tokenizer=english_path)


#     def analyze_mentee_review(self, review_text, id2lable=id2lable):
#         # def analyze_mentee_review(text):
#         contains_arabic = re.search(r'[\u0600-\u06FF]', review_text)
    
#         if contains_arabic:
#             # لو فيه عربي -> نستخدم موديل MARBERT
#             model_used = "MARBERT (Arabic/Mix)"
#             result = self.mentorea_analyzer_Ar(review_text)
#         else:
#             # لو مفيش ولا حرف عربي -> نستخدم موديل DistilBERT
#             model_used = "DistilBERT (English)"
#             result = self.mentorea_analyzer_En(review_text)

#         return result, model_used


import os 
os.environ["TRANSFORMERS_NO_TF"] = "1"
from transformers import pipeline
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
arabic_path = os.path.join(BASE_DIR, "mentorea_sentiment_model")
english_path = os.path.join(BASE_DIR, "mentorea_english_model_final")

# تصحيح اسم المتغير واستخدامه للتحويل
id2label = {
    'LABEL_0': "neutral", 'LABEL_1': "frustrated",
    'LABEL_2': 'positive', 'LABEL_3': 'negative', 'LABEL_4': 'grateful'
}

class SentimentAnalysis:
    def __init__(self):
        print("🚀 جاري تحميل موديلات الـ AI في الذاكرة (مرة واحدة فقط)...")
        self.mentorea_analyzer_Ar = pipeline("text-classification", model=arabic_path, tokenizer=arabic_path, local_files_only=True)
        self.mentorea_analyzer_En = pipeline("text-classification", model=english_path, tokenizer=english_path, local_files_only=True)

    def analyze_mentee_review(self, review_text):
        contains_arabic = re.search(r'[\u0600-\u06FF]', review_text)
    
        if contains_arabic:
            model_used = "MARBERT (Arabic/Mix)"
            # إضافة [0] لاستخراج الـ Dictionary من الـ List
            result = self.mentorea_analyzer_Ar(review_text)[0]
        else:
            model_used = "DistilBERT (English)"
            # إضافة [0] لاستخراج الـ Dictionary من الـ List
            result = self.mentorea_analyzer_En(review_text)[0]

        # التأكد من تحويل LABEL_X إلى الاسم النصي لو الموديل رجعها كـ Label
        current_label = result['label']
        if current_label in id2label:
            result['label'] = id2label[current_label]

        return result, model_used

# 🔥 التعديل الأهم: إنشاء نسخة واحدة (Singleton) لتستخدمها كل التطبيقات
# ده هيخلي الموديلات تتحمل مرة واحدة بس لما السيرفر يشتغل
sentiment_service = SentimentAnalysis()
# def analyze_sentiment( text:str):
#         # نستخدم الـ service الجاهزة مباشرة (بدون أقواس للكلاس)
#         try:
#             sentiment_values, model_used = sentiment_service.analyze_mentee_review(text) 
#         except Exception as e:
#             print(e)
#         # value = self.map_sentiment_to_value(sentiment_values)
        
#         return model_used, sentiment_values
    


# print(analyze_sentiment("السيشن كانت كويسة بس المينتور كان time management بتاعه سيء جدا واتاخرنا"))
    
    

