from flask import Flask, request, jsonify
from transformers import pipeline
from langdetect import detect

app = Flask(__name__)

def map_sentiment_to_value(sentiment, confidence):
    """Map sentiment and confidence to a value in the specified range."""
    if sentiment == "positive" or sentiment == "POSITIVE":
        # Map confidence [0,1] to [7.1, 10]
        value = 7.1 + (10 - 7.1) * confidence
    elif sentiment == "negative" or sentiment == "NEGATIVE":
        # Map confidence [0,1] to [4, 1] (high confidence -> 1, low confidence -> 4)
        value = 4 - (4 - 0.5) * confidence
    else:  # neutral
        # Map confidence [0,1] to [4.1, 7]
        value = 4.1 + (7 - 4.1) * confidence
    return round(value, 2)


###############

# Initialize the sentiment analysis pipeline
sentiment_pipeline_ar = pipeline(
    "sentiment-analysis",
    model="CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
)

# Initialize the sentiment analysis pipeline with an English model
sentiment_pipeline_en = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

############# Analyze function
def analyze_sentiment(text, lang):

    if lang in ['ar', 'fa']:
        # Run the model
        response = sentiment_pipeline_ar(text)
    else :
        # Run the model
        response = sentiment_pipeline_en(text)
    sentiment = response[0]['label']
    confidence = response[0]['score']

    # Map sentiment and confidence to the specified value range
    value = map_sentiment_to_value(sentiment, confidence)

    return sentiment, confidence, value


###############################

@app.route('/analyze', methods=['POST'])
def analyze_text():

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text = data['text'].strip()
    
    # Validate input
    if not text:
        return jsonify({"error": "Empty text input"}), 400
    
    if len(text) > 512:  
        text = text[:512]

     # Detect the language of the user's question
    language = detect(text)
    # print(f"\n\n {language} \n\n")

    try:
      
        sentiment, confidence, value = analyze_sentiment(text, language)
        return jsonify({
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "value": value
        })
      
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)