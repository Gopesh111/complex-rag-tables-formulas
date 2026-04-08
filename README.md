# Table-Aware RAG Pipeline for Complex Documents

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LlamaIndex](https://img.shields.io/badge/Framework-LlamaIndex-orange)
![Docker](https://img.shields.io/badge/Deployed-Docker-blue)

## Overview
This repository implements a Retrieval-Augmented Generation (RAG) pipeline designed to handle complex PDF documents containing nested tables, multi-column layouts, and mathematical formulas (LaTeX). 

## The Problem: Context Loss in Structural Data
Standard chunking algorithms process documents purely based on token limits. When applied to technical documents, they frequently split tables across chunks or break mathematical formulas. This destroys structural integrity (e.g., separating headers from rows), causing the LLM to hallucinate numerical values during the generation phase.

## The Solution & Architecture
This system solves the structural context-loss problem using a four-tier approach:

1. **Layout-Aware Parsing:** Replaces traditional parsers with Docling. This converts complex PDFs directly into Markdown, preserving tables in a structured format (`| col1 | col2 |`) and maintaining equations in LaTeX.
2. **Intelligent Fallback Mechanism:** Incorporates `pdfplumber` to automatically handle "borderless" tables or edge cases where standard parsers fail to detect grid structures.
3. **Semantic Element Chunking:** Utilizes LlamaIndex's `MarkdownElementNodeParser` to identify tables and text as distinct structural elements. Tables are extracted whole, preventing mid-row splits.
4. **Natural Language Context Enrichment:** Extracts raw markdown tables and converts them into descriptive, row-wise English sentences before vectorization. This dual-representation reduces LLM hallucinations and boosts semantic search accuracy.

## Tech Stack
* **Language:** Python  
* **Framework:** LlamaIndex  
* **Parsing:** Docling, pdfplumber (Fallback)  
* **LLM:** Groq (LLaMA-3-70B)  
* **Embeddings:** HuggingFace (BAAI/bge-small-en-v1.5)  
* **Vector DB:** FAISS (primary), ChromaDB (persistent storage)  
* **Deployment:** Docker  

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
```

## Getting Started

### 1. Installation

**Option A: Using Docker (Recommended)**
```bash
docker build -t rag-pipeline .
docker run -d rag-pipeline
```

**Option B: Local Setup**
```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file in the root directory based on the `.env.example` template and add your `GROQ_API_KEY`.

### 3. Execution
Place a test document (e.g., `sample_complex.pdf`) inside the `data/` directory and run:
```bash
python main.py
```

### Example Output
```text
> Query: What is the Q3 revenue listed in the financial summary table?
> Context Retrieved: Table 2 (Row 3): Q3 Revenue is $45.2M.
> Answer: Based on the financial summary, the Q3 revenue is $45.2M.
```

## Future Enhancements
* **Reranking Integration:** Implement a reranker (e.g., BGE-Reranker) in the post-processing step.
* **OCR Fallback:** Integrate Tesseract for scanned image-based tables.

## Acknowledgments
Built to solve real-world tabular data extraction issues encountered when standard text splitters fail on enterprise data.
