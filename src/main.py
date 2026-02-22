from fastapi import FastAPI
from routes import cv_checker, sentment_analysis

app= FastAPI()

app.include_router(cv_checker.cv_rout)
app.include_router(sentment_analysis.sent_rout)

@app.get("/")
def welcom():
    return{"message": "welcom"}