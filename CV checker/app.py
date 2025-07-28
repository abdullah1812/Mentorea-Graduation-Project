import PyPDF2
import os
import json
import uuid
import logging
from groq import Groq
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from pyngrok import ngrok

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cv_checker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Loading environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    logger.error("GROQ_API_KEY not found in .env file")
    raise ValueError("GROQ_API_KEY not found")

# Initializing Flask app
app = Flask(__name__)

# Configuring upload directory and port
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploaded_pdfs")
PORT = int(os.getenv("PORT", 5000))

# Ensuring upload directory exists
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    logger.info(f"Created upload directory: {UPLOAD_DIR}")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            if not text.strip():
                logger.warning(f"No text extracted from {pdf_path}")
                return None
            logger.info(f"Successfully extracted text from {pdf_path}")
            return text
    except (PyPDF2.errors.PdfReadError, IOError) as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        return None

# Initializing Groq client
try:
    client = Groq(api_key=api_key)
    logger.info("Groq client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    raise

# Function to analyze CV with Groq API
def analyze_cv(cv_text):
    if not cv_text:
        logger.warning("No CV text provided for analysis")
        return {"error": "No text extracted from PDF"}

    prompt = """
You are an AI assistant evaluating a CV to determine if the candidate is suitable to be a mentor. A mentor must have:
- At least 3 years of professional experience in software engineering or a related field.
- If they built projects, include them in experience.
- Demonstrated leadership or mentoring experience (e.g., leading teams, training others).
- Strong communication skills (e.g., presentations, workshops).
- Relevant education or certifications (e.g., Bachelor's in Computer Science, coaching certifications).
- Evidence of soft skills like empathy or problem-solving.
- Compute the experience from projects also.

CV Text: {cv_text}

Provide a JSON response with:
1. A summary of the candidate's relevant qualifications.
2. A score (0-100) for mentor eligibility based on the criteria.
3. If score >= 75, the candidate can be a mentor.
4. Specific strengths (list of qualifications that match the criteria).
5. Specific gaps (list of missing or weak criteria).
6. A recommendation: Can this candidate be a mentor? Why or why not?

{{"summary": "", "score": 0, "strengths": [], "gaps": [], "recommendation": ""}}
""".format(cv_text=cv_text)

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.6,
            max_tokens=2000
        )
        response = chat_completion.choices[0].message.content
        try:
            result = json.loads(response)
            logger.info("Successfully analyzed CV with Groq API")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Groq API: {e}")
            return {"error": "Invalid response format from Groq API"}
    except Exception as e:
        logger.error(f"Groq API processing failed: {e}")
        return {"error": f"Groq API processing failed: {e}"}

# Endpoint to receive PDF files via POST
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    try:
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.endswith('.pdf'):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({"error": "File must be a PDF"}), 400

        # Checking file size (5MB limit)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        max_size = 5 * 1024 * 1024  # 5MB
        if file_size > max_size:
            logger.warning(f"File too large: {file_size} bytes")
            return jsonify({"error": "File size exceeds 5MB limit"}), 400
        file.seek(0)  # Reset file pointer

        # Generating secure filename
        secure_filename = f"{uuid.uuid4()}.pdf"
        file_path = os.path.join(UPLOAD_DIR, secure_filename)
        file.save(file_path)
        logger.info(f"Saved file: {file_path}")

        text = extract_text_from_pdf(file_path)
        if not text:
            logger.warning(f"Failed to extract text from {file_path}")
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
            return jsonify({"error": "Failed to extract text from PDF"}), 400

        result = analyze_cv(text)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")

        return jsonify({"analysis": result}), 200

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    try:
        
        app.run(host="0.0.0.0", port=PORT)
    except Exception as e:
        logger.error(f"Failed to start ngrok or Flask: {e}")
        raise