import json
from groq import Groq

class CVAIService:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    async def analyze(self, cv_text: str):

        prompt = f"""
        You are evaluating a CV for mentor eligibility. 

        Criteria:
        - Minimum 3 years professional experience (include projects).
        - Leadership or mentoring experience.
        - Strong communication and soft skills.
        - Relevant education or certifications.

        CV Text:
        {cv_text}

        Return JSON with:
        {{
            "summary": "Brief summary of relevant qualifications",
            "score": 0,  # 0-100
            "strengths": [],  # list relevant qualifications
            "gaps": [],  # list missing/weak criteria
            "recommendation": ""  # "Mentor" or "Not Mentor"
        }}
        """

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a strict evaluator. Return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                model="openai/gpt-oss-safeguard-20b",
                # model="llama3-70b-8192",
                temperature=0.6,
                max_tokens=1500
            )

            response_text = chat_completion.choices[0].message.content

            if response_text is None:
                return {"error": "Empty response from model"}

            try:
                parsed = json.loads(response_text)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON returned from model"}

            return parsed
        except Exception as e:
            return {"error": f"Groq API processing failed: {e}"}

