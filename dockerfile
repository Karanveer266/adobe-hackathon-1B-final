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

# Copy the pre-downloaded model from local models folder
COPY models/ /app/models/

# Copy all source files  
COPY main_1b.py .
COPY persona_document_analyzer.py .
COPY embedding_engine.py .
COPY persona_processor.py .
COPY section_ranker.py .
COPY subsection_analyzer.py .
COPY config_1b.py .

COPY main.py .
COPY pdf_processor.py .

# Copy any additional modules that might be needed
COPY *.py .

# Set proper permissions
RUN chmod +x main_1b.py

# Set the entry point
ENTRYPOINT ["python", "main_1b.py", "--input-dir", "/app/input", "--output-dir", "/app/output"]
