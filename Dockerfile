FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-space.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements-space.txt

COPY . .

RUN mkdir -p data/papers data/chroma_db data/software_docs data/chunks \
    && chmod +x entrypoint.sh

EXPOSE 8000 8501

ENV PYTHONUNBUFFERED=1

# Hugging Face Docker Spaces: fetch gitignored Chroma index, then Streamlit.
# docker-compose.yml overrides this command for the API service.
CMD ["./entrypoint.sh"]