FROM python:3.12-slim

# Install system deps for pdfplumber
RUN apt-get update && \
    apt-get install -y build-essential python3-dev poppler-utils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV OPENAI_API_KEY=""
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]