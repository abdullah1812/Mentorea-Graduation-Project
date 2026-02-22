from fastapi import APIRouter, status, Body
from controllers.SAnalysisController import SentimentAnalysisController
from fastapi.responses import JSONResponse
from Models.ResponsesEnum import ResponseSignal

sent_rout = APIRouter(prefix="/sentiment")

sentiment_controller = SentimentAnalysisController()

@sent_rout.post("/")
async def analyze_text(text: str = Body(..., embed=True)):
    try: 
        db_score, model_used, raw_result = sentiment_controller.analyze_sentiment(text)
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.SENTIMENT_TEXT_FAILD.value
            }
        )
        
    return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "TEXT": text,
                "Sentiment_Label": raw_result['label'],       
                "Sentiment_Confidence": raw_result['score'],  
                "Model_Used": model_used,                     
                "Database_Score": db_score 
            }
        )