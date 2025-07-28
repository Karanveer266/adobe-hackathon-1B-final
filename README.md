# Persona-Driven Document Intelligence  
Adobe Hackathon – Part 1B

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Build](https://img.shields.io/badge/Status-Prototype-orange)

> A lightweight *offline-capable* system that analyzes PDF documents through the lens of a user-defined persona and “job-to-be-done” (JTBD).  
> It reuses Part 1A’s heading-detection engine, ranks sections with all-mpnet-base-v2 semantic embeddings, and returns the most relevant sections & subsections as structured JSON.

---

## Table of Contents
1. [Features](#features)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Project Layout](#project-layout)
7. [Docker](#docker)
8. [Contributing](#contributing)
9. [License](#license)

---

## Features
• Persona / JTBD parsing into a rich profile  
• PDF heading & section extraction (reuses Part 1A)  
• Semantic ranking (CPU-only, offline cache)  
• Fine-grained subsection analysis with relevance scores & key concepts  
• JSON output for downstream workflows  
• Fully configurable via config_1b.py

---

## Quick Start
git clone https://github.com/<your-handle>/persona-driven-document-intelligence.git
cd persona-driven-document-intelligence
python -m venv venv && source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main_1b.py --input-dir ./input --output-dir ./output --debug



---

## Installation

1. *Clone & create virtual env*


git clone https://github.com/<your-handle>/persona-driven-document-intelligence.git
cd persona-driven-document-intelligence
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate



2. *Install dependencies*
pip install -r requirements.txt
Default requirements.txt:

sentence-transformers==2.2.2
torch==2.0.1 # CPU build
numpy==1.24.3
pypdf==3.17.4

3. *Download model (one-time)*
python download_models.py # optional helper
or manually copy all-mpnet-base-v2 into ./models (local) or /app/models (Docker).

---

## Usage

### 1 – Prepare input
Create input/input_spec.json:

{
"persona": "HR professional specializing in employee onboarding",
"job_to_be_done": "Create fillable forms for compliance and e-signatures",
"documents": ["employee_handbook.pdf", "form_guide.pdf"]
}
Put the PDFs inside input/.

### 2 – Run

python main_1b.py
--input-dir ./input
--output-dir ./output
--debug

### 3 – Result
output/analysis_result.json (excerpt):

{
"metadata": {
"persona": "HR professional specializing in employee onboarding",
"processing_time_seconds": 12.34,
"total_sections_analyzed": 18
},
"extracted_sections": [
{
"document": "form_guide.pdf",
"page_number": 7,
"section_title": "Creating Fillable Forms",
"importance_rank": 1,
"relevance_score": 0.94
}

],
"subsection_analysis": [
{
"document": "form_guide.pdf",
"page_number": 8,
"parent_section": "Creating Fillable Forms",
"refined_text": "Use the Prepare Form tool in Acrobat …",
"relevance_score": 0.88,
"key_concepts": ["Prepare Form", "Text Fields", "Signature"]
}
]




---

## Configuration
All knobs live in config_1b.py:

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MAX_SECTIONS_TO_ANALYZE = 20
SEMANTIC_SIMILARITY_WEIGHT = 0.50
DOMAIN_RELEVANCE_WEIGHT = 0.20
KEYWORD_MATCH_WEIGHT = 0.15
SECTION_TYPE_WEIGHT = 0.10
CONTENT_QUALITY_WEIGHT = 0.05


---

## Project Layout

.
├─ main_1b.py # CLI entry point
├─ persona_document_analyzer.py
├─ embedding_engine.py
├─ persona_processor.py
├─ section_ranker.py
├─ subsection_analyzer.py
├─ config_1b.py
├─ models/ # local HF model cache (git-ignored)
├─ input/ # PDFs + input_spec.json
└─ output/ # analysis_result.json


(Part 1A files like main.py, pdf_processor.py, etc., are reused internally.)

---

## Docker
docker build -t persona-analyzer .
docker run
-v $(pwd)/input:/app/input
-v $(pwd)/output:/app/output
persona-analyzer


The image sets HF_HUB_OFFLINE=1 so it never calls the internet.

---

## Contributing
1. Fork → git checkout -b feature/awesome  
2. Commit → git commit -m "Add awesome feature"  
3. Push  → git push origin feature/awesome  
4. Open a PR

Please follow PEP 8 and add tests where possible.

---

## License
Distributed under the *MIT License*.  
See [LICENSE](LICENSE) for full text.

---

## Acknowledgments
• Adobe Hackathon 2023 – Problem 1B  
• Sentence-Transformers team for all-mpnet-base-v2  
• The open-source community ❤





# Persona-Driven Document Intelligence System (1B)

## Executive Summary

This project (Part 1B) builds directly upon the heading detection system from Part 1A to create a Persona-Driven Document Intelligence engine. The primary challenge this system addresses is information overload; professionals often need to find specific, relevant information scattered across multiple long and complex documents. Sifting through this content manually is inefficient and time-consuming.

Our solution is an intelligent pipeline that understands a user's persona (who they are) and their job-to-be-done (what they need to accomplish). By combining the structured output from Part 1A with a powerful semantic search and a hybrid ranking model, the system analyzes a collection of documents and automatically identifies, ranks, and extracts the most relevant sections and subsections. The final output is a concise, actionable summary tailored specifically to the user's needs, dramatically accelerating their workflow.

## System Architecture & Core Approach

The system extends the Part 1A pipeline with a sophisticated analysis layer. It processes a user query and a set of documents to deliver a targeted response.

### Architectural Flow:

[Input Spec: Persona, Job, Docs] → (1) Persona Processor & Part 1A Heading Detector → [Structured Persona] & [Document Structures] → (2) Embedding Engine → (3) Section Ranker → [Ranked Sections] → (4) Subsection Analyzer → [Final JSON Result]


### Core Principles:

1. *Foundation on Part 1A*: The system first uses the PDFHeadingDetectionSystem (from Part 1A) to parse every document, transforming unstructured PDFs into a structured list of sections with associated content.

2. *Persona Understanding*: A dedicated PersonaProcessor analyzes the natural language descriptions of the user's role and task, converting them into a structured profile with keywords, expertise areas, and a primary domain.

3. *Semantic Search Core*: The EmbeddingEngine uses a state-of-the-art sentence-transformer model (all-mpnet-base-v2) to convert the persona query and all document sections into numerical vectors (embeddings). This allows the system to match content based on semantic meaning, not just keywords.

4. *Hybrid Relevance Ranking*: The SectionRanker is the system's brain. It calculates a relevance score for every section using a hybrid, weighted formula. This prevents over-reliance on any single metric and produces a more robust ranking. The formula is:
   - Relevance = (w_s * Semantic) + (w_d * Domain) + (w_k * Keywords) + (w_t * Type) + (w_q * Quality)

5. *Progressive Summarization*: The system first identifies the most relevant sections. It then performs a deeper analysis on only these top-ranked sections, breaking them down into concise, refined subsections and extracting key concepts. This ensures the final output is both relevant and highly digestible.

## Detailed Component Breakdown

### main_1b.py:
- Handles the overall execution, parsing command-line arguments, and managing I/O
- Reads an input_spec.json file containing the persona, job, and document list
- Writes the final analysis_result.json

### persona_document_analyzer.py: 
- Central class that manages the entire analysis pipeline
- Initializes all other components and calls them in the correct sequence: process persona, extract document structures, rank sections, analyze subsections, and build the final output

### persona_processor.py:
- Uses rule-based logic and keyword extraction to parse the user's persona and job_to_be_done
- Identifies the user's role (e.g., "HR Professional"), domain ("HR"), expertise areas, and key concepts from their task description to create a comprehensive profile

### embedding_engine.py: 
- Critical component that loads and manages the all-mpnet-base-v2 sentence-transformer model
- Optimized for offline, CPU-only execution to meet platform constraints
- Primary functions: encode text into 768-dimension vectors and calculate cosine similarity between them, enabling powerful semantic matching

### section_ranker.py: 
- Iterates through every section from every document and scores them against the persona profile
- Calculates the hybrid relevance score by combining:
  - Semantic similarity
  - Domain keyword matches
  - Task keyword matches
  - Section type (e.g., "Introduction" vs. "How to")
  - Content quality

### subsection_analyzer.py:
- Takes the top-ranked sections and performs a deeper dive
- Splits the section's raw content into smaller, logical subsections (e.g., paragraphs or groups of sentences)
- Scores these subsections for relevance, refines the text for clarity, and extracts key concepts
- Provides the final layer of summarization

### config_1b.py: 
- Contains all key parameters for the system, including:
  - Embedding model name
  - Text length limits
  - Weights for the hybrid scoring model in the SectionRanker
- Allows for easy tuning of the ranking algorithm

## Dependencies

- *Part 1A*: PDF Heading Detection System (required foundation)
- *all-mpnet-base-v2*: Sentence transformer model for semantic embeddings
- *CPU-optimized execution*: Designed for offline processing without GPU requirements

## Technical Specifications

- *Embedding Model*: all-mpnet-base-v2 sentence transformer
- *Vector Dimensions*: 768-dimension vectors
- *Execution Environment*: CPU-only, offline capable
- *Input Format*: JSON specification with persona, job, and document list
- *Output Format*: Structured JSON with analysis results

## Key Features

- *Semantic Understanding*: Goes beyond keyword matching to understand meaning
- *Hybrid Ranking*: Multi-factor scoring system for optimal relevance
- *Progressive Analysis*: Focuses computational resources on most relevant content
- *Persona-Driven*: Tailored results based on user role and objectives
- *Document Structure Awareness*: Leverages Part 1A's heading detection for better organization

*Note*: This system requires the PDF Heading Detection System (Part 1A) to be properly installed and configured as a foundation component.
```
