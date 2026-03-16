import os
import logging
from dotenv import load_dotenv

# Clean imports thanks to your updated src/__init__.py!
from src import RAGPipeline, AdvancedTableRetriever

# Setup minimal logging for the main execution script
logging.basicConfig(level=logging.INFO, format='[*] %(message)s')

def main():
    print("\n" + "="*50)
    print("🚀 Complex RAG System: Table & Math Handling")
    print("="*50 + "\n")

    # 1. Load Environment Variables (Ensure GROQ_API_KEY is set)
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        logging.error("GROQ_API_KEY missing in .env file. Please add it.")
        return

    # 2. Check for the test PDF
    test_pdf = "data/sample_complex.pdf"
    if not os.path.exists(test_pdf):
        logging.warning(f"Test PDF not found at: {test_pdf}")
        logging.info("Creating 'data' directory. Please place your messy PDF there.")
        os.makedirs("data", exist_ok=True)
        return

    # 3. Initialize the Engine
    logging.info("Initializing RAG Pipeline Orchestrator...")
    pipeline = RAGPipeline(db_path="./chroma_db")

    # 4. Run Ingestion (Docling Parse -> Element Chunking -> ChromaDB)
    logging.info("Starting document ingestion workflow...")
    success = pipeline.ingest_pdf(test_pdf)
    
    if not success or not pipeline.index:
        logging.error("Pipeline ingestion failed. Exiting application.")
        return

    # 5. Initialize Advanced Retriever (The Multi-Vector / Parent-Child logic)
    logging.info("Setting up Multi-Vector Retriever for precise table extraction...")
    retriever = AdvancedTableRetriever(index=pipeline.index)
    
    # We build the query engine from our custom retriever instead of the basic pipeline
    query_engine = retriever.build_query_engine()

    # 6. Execute a Complex Test Query
    print("\n" + "-"*50)
    test_query = "Extract the specific numerical data from the efficiency tables and explain the formulas used."
    print(f"User Query: {test_query}")
    print("-"*50)
    
    logging.info("Executing query against Vector Database...")
    try:
        response = query_engine.query(test_query)
        print(f"\n[AI Response]:\n{response}\n")
    except Exception as e:
        logging.error(f"Query execution failed: {e}")

if __name__ == "__main__":
    main()
