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

## Running Instructions

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t adobe-hackathon-1b -f dockerfile .
   ```
2. Run the following commands for the respective collection folder you want to run
```bash
# For Collection 1
docker run -v "$(pwd)/input/Collection 1:/app/input" -v "$(pwd)/output:/app/output" adobe-hackathon-1b

# For Collection 2
docker run -v "$(pwd)/input/Collection 2:/app/input" -v "$(pwd)/output:/app/output" adobe-hackathon-1b

# For Collection 3
docker run -v "$(pwd)/input/Collection 3:/app/input" -v "$(pwd)/output:/app/output" adobe-hackathon-1b
```
The output will be generated in the output folder.

### Without docker

```bash
pip install -r requirements.txt
```

```bash
python main_1b.py --input-dir "input/Collection 1" --output-dir "output"
```
