FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/papers data/chroma_db data/software_docs data/chunks

EXPOSE 8000 8501

ENV PYTHONUNBUFFERED=1

# Hugging Face Docker Spaces and a default local image both serve Streamlit.
# docker-compose.yml overrides this command for the API service.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]