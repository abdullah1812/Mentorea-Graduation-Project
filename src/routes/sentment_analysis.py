# from fastapi import APIRouter, status, Body
# from controllers.SAnalysisController import SentimentAnalysisController
# from fastapi.responses import JSONResponse
# from Models.ResponsesEnum import ResponseSignal

# sent_rout =APIRouter(prefix="/sentiment")

# @sent_rout.post("/")
# async def analyze_text(text: str = Body(..., embed=True)):
#     print(text)
    
#     try: 
#         s_values, model_used, s_score=SentimentAnalysisController().analyze_sentiment(text)
    
#     except Exception as e:
#         return JSONResponse(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             content={
#                 "signal":ResponseSignal.SENTIMENT_TEXT_FAILD.value
#             }
#         )
        
     
#     return JSONResponse(
#             status_code=status.HTTP_202_ACCEPTED,
#             content={
#                 "signal":ResponseSignal.FILE_UPLOADED_SUCCES.value,
#                 "TEXT": text,
#                 "Sentiment_type":s_values,
#                 "Sentiment_confidence":model_used,
#                 "Sentiment_score":s_score
#             }
#         )


from fastapi import APIRouter, status, Body
from controllers.SAnalysisController import SentimentAnalysisController
from fastapi.responses import JSONResponse
# import BackgroundTasks # لو هتشغلها في الخلفية زي ما اتفقنا قبل كده
from Models.ResponsesEnum import ResponseSignal

sent_rout = APIRouter(prefix="/sentiment")

# عملنا instance من الـ Controller بره عشان نستخدمها على طول
sentiment_controller = SentimentAnalysisController()

@sent_rout.post("/")
async def analyze_text(text: str = Body(..., embed=True)):
    print(f"Received Text: {text}")
    
    try: 
        # الترتيب الصح: (الرقم من 100, اسم الموديل, الـ Dictionary بتاع النتيجة)
        db_score, model_used, raw_result = sentiment_controller.analyze_sentiment(text)
        print(db_score, model_used, raw_result)
    
    except Exception as e:
        print(f"Error during sentiment analysis: {e}") # عشان لو حصل إيرور تشوفه في الـ Terminal
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.SENTIMENT_TEXT_FAILD.value
            }
        )
        
    return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED, # أو 200 OK
            content={
                # تأكد إن الـ Enum ده موجود عندك، أو غيره لـ SENTIMENT_SUCCESS مثلاً
                
                "TEXT": text,
                "Sentiment_Label": raw_result['label'],       # هيطلعلك: grateful, frustrated, etc
                "Sentiment_Confidence": raw_result['score'],  # هيطلعلك: 0.95 (نسبة التأكد)
                "Model_Used": model_used,                     # هيطلعلك: MARBERT أو DistilBERT
                "Database_Score": db_score                    # هيطلعلك السكور من 0 لـ 100
            }
        )