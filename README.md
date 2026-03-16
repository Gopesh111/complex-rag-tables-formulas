# Table-Aware RAG Pipeline for Complex Documents

## Overview
This repository implements an advanced Retrieval-Augmented Generation (RAG) pipeline designed specifically to handle complex PDF documents containing nested tables, multi-column layouts, and mathematical formulas (LaTeX). 

## The Problem: Context Loss in Structural Data
Standard chunking algorithms process documents purely based on token or character limits. When applied to technical documents, they frequently split tables across chunks or break mathematical formulas. This destroys the structural integrity of the data (e.g., separating headers from their respective rows), which directly causes the LLM to hallucinate or fabricate numerical values during the generation phase.

## The Solution & Architecture
This system solves the structural context-loss problem using a three-tier approach:

1. **Layout-Aware Parsing:** Replaces traditional parsers (like PyPDF) with `Docling`. This converts complex PDFs directly into Markdown, preserving tables in a structured format (`| col1 | col2 |`) and maintaining equations in LaTeX.
2. **Semantic Element Chunking:** Utilizes LlamaIndex's `MarkdownElementNodeParser` to identify tables and text as distinct structural elements. Tables are extracted whole, preventing mid-row splits.
3. **Multi-Vector Retrieval (Parent-Child):** Implements a recursive retrieval strategy. The vector database searches against lightweight "summary nodes" of the tables, but retrieves the full, raw Markdown table (the parent node) to provide complete context to the LLM.

## Tech Stack
* **Framework:** LlamaIndex
* **Parsing:** Docling, pdfplumber (Fallback)
* **LLM:** Groq (Llama-3-70b)
* **Embeddings:** HuggingFace (`BAAI/bge-small-en-v1.5`)
* **Vector Database:** ChromaDB (Persistent)

## Repository Structure
.
├── data/                       # Directory for input PDF documents
├── notebooks/                  # R&D and parser evaluation experiments
├── src/
│   ├── __init__.py
│   ├── parser.py               # Markdown and fallback parsing logic
│   ├── chunker.py              # Semantic element node extraction
│   ├── pipeline.py             # Ingestion and indexing orchestrator
│   └── retriever.py            # Multi-vector recursive retrieval logic
├── .env.example                # Environment variables template
├── .gitignore
├── requirements.txt
└── main.py                     # Entry point for execution

## Getting Started

### 1. Installation
Clone the repository and install the required dependencies:

git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt

### 2. Environment Setup
Create a `.env` file in the root directory based on the provided template:

cp .env.example .env

Add your `GROQ_API_KEY` to the `.env` file to enable the LLM for element summarization and query synthesis.

### 3. Execution
Place a test document (e.g., `sample_complex.pdf`) inside the `data/` directory. Run the main orchestrator:

python main.py

## Future Enhancements
* **Reranking Integration:** Implement a reranker (e.g., CohereRerank or BGE-Reranker) in the `retriever.py` post-processing step to filter out noise from dense tabular searches.
* **OCR Fallback:** Integrate `unstructured.io` or `Tesseract` for documents where tables are embedded as scanned images rather than raw text.

## Acknowledgments
The parsing strategies implemented here were refined based on discussions and feedback from the AI engineering community regarding the limitations of standard text splitters on tabular data.
