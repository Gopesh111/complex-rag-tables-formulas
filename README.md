# Table-Aware RAG Pipeline for Complex Documents

## Overview
This repository implements an advanced Retrieval-Augmented Generation (RAG) pipeline designed specifically to handle complex PDF documents containing nested tables, multi-column layouts, and mathematical formulas (LaTeX). 

## The Problem: Context Loss in Structural Data
Standard chunking algorithms process documents purely based on token or character limits. When applied to technical documents, they frequently split tables across chunks or break mathematical formulas. This destroys the structural integrity of the data (e.g., separating headers from their respective rows), which directly causes the LLM to hallucinate or fabricate numerical values during the generation phase.

## The Solution & Architecture
This system solves the structural context-loss problem using a robust, four-tier approach:

1. Layout-Aware Parsing: Replaces traditional parsers (like PyPDF) with Docling. This converts complex PDFs directly into Markdown, preserving tables in a structured format (| col1 | col2 |) and maintaining equations in LaTeX.
2. Intelligent Fallback Mechanism: Incorporates pdfplumber to automatically handle "borderless" tables or edge cases where standard layout-aware parsers fail to detect grid structures.
3. Semantic Element Chunking: Utilizes LlamaIndex's MarkdownElementNodeParser to identify tables and text as distinct structural elements. Tables are extracted whole, preventing mid-row splits.
4. Natural Language Context Enrichment: Extracts raw markdown tables and dynamically converts them into descriptive, row-wise English sentences before vectorization. This dual-representation drastically reduces LLM hallucinations and boosts semantic similarity search accuracy.

## Tech Stack
* Framework: LlamaIndex
* Parsing: Docling, pdfplumber (Fallback)
* LLM: Groq (Llama-3-70b)
* Embeddings: HuggingFace (BAAI/bge-small-en-v1.5)
* Vector Database: ChromaDB (Persistent)

## Repository Structure

```text
├── src/
│   ├── __init__.py
│   ├── parser.py               # Markdown and fallback parsing logic
│   ├── chunker.py              # Semantic element node extraction
│   ├── table_formatter.py      # Converts Markdown tables to natural language context
│   ├── pipeline.py             # Ingestion and indexing orchestrator
│   └── retriever.py            # Multi-vector recursive retrieval logic
├── .env.example                # Environment variables template
├── .gitignore
├── requirements.txt            # Pinned dependencies for reproducible environments
└── main.py                     # Entry point for execution
## Getting Started

### 1. Installation
Clone the repository and install the required dependencies:

git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt

### 2. Environment Setup
Create a .env file in the root directory based on the provided template and add your GROQ_API_KEY.

### 3. Execution
Place a test document (e.g., sample_complex.pdf) inside the data/ directory and run:

python main.py

## Future Enhancements
* Reranking Integration: Implement a reranker (e.g., BGE-Reranker) in the retriever.py post-processing step.
* OCR Fallback: Integrate Tesseract for scanned image-based tables.

## Acknowledgments
The architectural decisions implemented here—specifically the natural language table contextualization—were refined based on peer reviews and feedback from industry Engineering Leads regarding the limitations of standard text splitters on enterprise tabular data.
