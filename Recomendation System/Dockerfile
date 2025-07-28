```
# Use a Python 3.9 slim base image for a smaller footprint
FROM python:3.9-slim

# Install system dependencies required for ODBC and SQL Server connectivity
RUN apt-get update && apt-get install -y \
    unixodbc \
    unixodbc-dev \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC Driver 17 for SQL Server
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and app files
COPY requirements.txt .
COPY app.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Flask
EXPOSE 5000

# Set environment variable for ngrok auth token (replace with your token or use environment variable)
ENV NGROK_AUTH_TOKEN="your_ngrok_auth_token"

# Run the Flask app
CMD ["python", "app.py"]
```