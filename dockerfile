# Use AMD64 platform explicitly as required
FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/input /app/output /app/models

# Set environment variables for offline operation
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV SENTENCE_TRANSFORMERS_HOME=/app/models

# Download the embedding model (all-mpnet-base-v2) during build time
RUN python -c "\
import os; \
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/app/models'; \
from sentence_transformers import SentenceTransformer; \
print('Downloading all-mpnet-base-v2 model...'); \
model = SentenceTransformer('all-mpnet-base-v2', cache_folder='/app/models'); \
print('Model downloaded successfully'); \
print(f'Model dimension: {model.get_sentence_embedding_dimension()}'); \
"

# Copy all source files
COPY main_1b.py .
COPY persona_document_analyzer.py .
COPY embedding_engine.py .
COPY persona_processor.py .
COPY section_ranker.py .
COPY subsection_analyzer.py .
COPY config_1b.py .

# Copy Part 1A dependencies (assuming they exist)
COPY main.py .
COPY pdf_processor.py .

# Copy any additional modules that might be needed
COPY *.py .

# Set proper permissions
RUN chmod +x main_1b.py

# Verify model is accessible and working
RUN python -c "\
from embedding_engine import EmbeddingEngine; \
print('Testing embedding engine...'); \
engine = EmbeddingEngine(); \
test_embedding = engine.encode_text('test'); \
print(f'Test embedding shape: {test_embedding.shape}'); \
print('Embedding engine working correctly'); \
"

# Set the entry point
ENTRYPOINT ["python", "main_1b.py", "--input-dir", "/app/input", "--output-dir", "/app/output"]
