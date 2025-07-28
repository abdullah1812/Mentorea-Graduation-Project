FROM python:3.11.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY . .
CMD exec gunicorn --bind 0.0.0.0:$PORT app:app