import re
from transformers import pipeline

# ==========================================
# 1. تحميل الموديلات (Pipelines)
# ==========================================
print("جاري تحميل الموديلات... (قد يستغرق بضع ثوانٍ)")

# مسارات الموديلات اللي إنت حفظتها (تأكد من تعديل الأسماء لو حفظتها باسم مختلف)
arabic_model_path = "./mentorea_arabic_model"       # مسار موديل MARBERT
english_model_path = "./mentorea_english_model_final" # مسار موديل DistilBERT

# تحميل الـ Pipelines
arabic_analyzer = pipeline("text-classification", model=arabic_model_path, tokenizer=arabic_model_path)
english_analyzer = pipeline("text-classification", model=english_model_path, tokenizer=english_model_path)

# ==========================================
# 2. دالة حساب الـ Score (من 0 لـ 100)
# ==========================================
def calculate_feedback_score(sentiment_result):
    label = sentiment_result['label']
    confidence = sentiment_result['score']
    
    # تحويل الليبلز لو كانت LABEL_0, LABEL_1 للأسماء الحقيقية
    # (لو الموديل بيرجع الاسم مباشرة زي frustrated، السطور دي مش هتأثر)
    label_mapping = {
        'LABEL_0': 'neutral', 'LABEL_1': 'frustrated',
        'LABEL_2': 'positive', 'LABEL_3': 'negative', 'LABEL_4': 'grateful'
    }
    if label in label_mapping:
        label = label_mapping[label]

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

    return int(final_score), label, confidence

# ==========================================
# 3. الـ Router الذكي (اكتشاف اللغة والتوجيه)
# ==========================================
def analyze_mentee_review(text):
    # البحث عن أي حرف عربي في النص باستخدام الـ RegEx
    contains_arabic = re.search(r'[\u0600-\u06FF]', text)
    
    if contains_arabic:
        # لو فيه عربي -> نستخدم موديل MARBERT
        model_used = "MARBERT (Arabic/Mix)"
        result = arabic_analyzer(text)[0]
    else:
        # لو مفيش ولا حرف عربي -> نستخدم موديل DistilBERT
        model_used = "DistilBERT (English)"
        result = english_analyzer(text)[0]
    return result
    # حساب السكور النهائي
    score, final_label, confidence = calculate_feedback_score(result)
    
    return {
        "text": text,
        "routed_to": model_used,
        "sentiment": final_label,
        "confidence": round(confidence, 2),
        "database_score": score
    }

# ==========================================
# 4. تجربة السيستم (Testing)
# ==========================================
reviews_to_test = [
    "السيشن كانت كويسة جدا والمحاضر كان بيشرح بضمير بس ال time management كان وحش", # ميكس (عربي + إنجليزي)
    "The mentor was extremely professional and the content was amazing!", # إنجليزي صافي
    "سيء جدا ومفيش أي استفادة", # عربي صافي
    "The session was boring and not helpful at all" # إنجليزي صافي سلبي
]

print("\n--- نتائج تحليل منصة Mentorea ---")
for review in reviews_to_test:
    analysis = analyze_mentee_review(review)
    print(f"\nReview: {analysis['text']}")
    print(f"Model:  {analysis['routed_to']}")
    print(f"Result: {analysis['sentiment']} (Score: {analysis['database_score']}/100)")



###########################################################################

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI()

# ==========================================
# 1. تعريف شكل البيانات اللي جاية من الـ Frontend
# ==========================================
class ReviewRequest(BaseModel):
    mentor_id: int
    mentee_id: int
    review_text: str

# ==========================================
# 2. دالة محاكاة الداتابيز (للتوضيح فقط)
# ==========================================
def save_score_to_db(mentor_id: int, score: int, sentiment: str):
    # هنا بتكتب كود الـ SQLAlchemy أو الـ SQL العادي بتاعك
    # عشان تعمل Update لتقييم المينتور في الداتابيز
    print(f"✅ [Database] تم تحديث تقييم المينتور {mentor_id} | السكور الجديد: {score}/100 | التصنيف: {sentiment}")

# ==========================================
# 3. الدالة اللي هتشتغل في الخلفية (Background Task)
# ==========================================
def process_sentiment_in_background(mentor_id: int, review_text: str):
    print(f"⏳ [Background] جاري تحليل تعليق للمينتور {mentor_id}...")
    
    # 1. بننادي على دالة الـ Router اللي عملناها قبل كده (اللي بتحدد عربي ولا إنجليزي وتطلع السكور)
    # هفترض إنك عامل import للدالة analyze_mentee_review
    analysis_result = analyze_mentee_review(review_text) 
    
    # 2. بنستخرج السكور والتصنيف
    final_score = analysis_result["database_score"]
    sentiment_label = analysis_result["sentiment"]
    
    # 3. بنحفظ النتيجة في الداتابيز
    save_score_to_db(mentor_id, final_score, sentiment_label)

# ==========================================
# 4. الـ Endpoint السريعة (تستقبل وترد فوراً)
# ==========================================
@app.post("/api/reviews/submit")
async def submit_mentor_review(review: ReviewRequest, background_tasks: BackgroundTasks):
    
    # الخطوة 1: (اختياري) ممكن تحفظ النص نفسه في الداتابيز هنا بسرعة
    
    # الخطوة 2: رمي عملية التحليل التقيلة للخلفية
    background_tasks.add_task(
        process_sentiment_in_background, 
        mentor_id=review.mentor_id, 
        review_text=review.review_text
    )
    
    # الخطوة 3: الرد الفوري على اليوزر (مش هيستنى الموديل يخلص)
    return {
        "status": "success",
        "message": "شكراً لتقييمك! تم استلام الفيدباك بنجاح."
    }