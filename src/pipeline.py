import os
import logging
import chromadb
from typing import Optional

# LlamaIndex core and vector store components
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Import our custom modules
from .parser import DocumentParser
from .chunker import TableAwareChunker

# Setup production-style logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)

class RAGPipeline:
    """
    Main Orchestrator for the Table-Aware RAG system.
    
    This pipeline integrates the Docling parser, semantic chunking, and 
    ChromaDB vector storage to solve the context-loss problem in complex PDFs.
    It uses local HuggingFace embeddings for cost-efficiency and privacy.
    """

    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "complex_rag_tables"):
        logging.info("Initializing the RAG Pipeline Orchestrator...")
        
        # 1. Initialize custom components
        self.parser = DocumentParser()
        self.chunker = TableAwareChunker()
        
        # 2. Configure Global Settings (Embeddings & LLM)
        # BGE-small is highly optimized for semantic search without heavy compute overhead
        logging.info("Loading HuggingFace Embeddings (BAAI/bge-small-en-v1.5)...")
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Share the LLM from the chunker for synthesis
        Settings.llm = self.chunker.llm 
        
        # 3. Setup Persistent Vector Database
        # Using persistent ChromaDB so we don't re-ingest the same PDF every run
        try:
            self.db = chromadb.PersistentClient(path=db_path)
            self.chroma_collection = self.db.get_or_create_collection(collection_name)
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            logging.info(f"Connected to persistent ChromaDB at {db_path}")
        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB: {e}")
            raise

        self.index: Optional[VectorStoreIndex] = None

    def ingest_pdf(self, file_path: str) -> bool:
        """
        Full ingestion flow: Parse (Markdown) -> Chunk (Elements) -> Index -> Store.
        """
        logging.info(f"Starting ingestion workflow for: {file_path}")
        
        # Step 1: Parse to Markdown (Preserves tables & math)
        markdown_text = self.parser.parse_to_markdown(file_path)
        if not markdown_text:
            logging.error("Ingestion aborted: Parser returned empty content.")
            return False
            
        # Passing metadata is crucial for tracing context during retrieval
        doc = Document(text=markdown_text, metadata={"source_file": os.path.basename(file_path)})
        
        # Step 2: Table-aware chunking
        nodes = self.chunker.process_documents([doc])
        if not nodes:
            logging.error("Ingestion aborted: Chunker failed to generate nodes.")
            return False
            
        # Step 3: Indexing and Storage
        try:
            logging.info("Building Vector Index and storing embeddings in ChromaDB...")
            self.index = VectorStoreIndex(
                nodes, 
                storage_context=self.storage_context
            )
            logging.info("Ingestion successfully completed.")
            return True
        except Exception as e:
            logging.error(f"Failed to build Vector Index: {e}")
            return False

    def query_system(self, user_query: str) -> str:
        """
        Query the index with a focus on retrieving structured table data accurately.
        """
        if not self.index:
            logging.info("Index not in memory. Attempting to load from existing ChromaDB...")
            try:
                # Load existing index if DB exists but memory is cleared
                self.index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                    storage_context=self.storage_context
                )
            except Exception as e:
                logging.error(f"Could not load index from DB: {e}")
                return "System Error: Index not initialized. Please run ingestion first."
                
        try:
            # TODO: Implement a Reranker (e.g., Cohere/BGE-Reranker) here to filter 
            # noise from table search results for even higher accuracy.
            query_engine = self.index.as_query_engine(
                similarity_top_k=5,
                response_mode="compact" # 'compact' works well for dense tabular contexts
            )
            
            logging.info(f"Executing query: '{user_query}'")
            response = query_engine.query(user_query)
            return str(response)
            
        except Exception as e:
            logging.error(f"Query execution failed: {e}")
            return f"Error processing query: {str(e)}"

# -------------------------------------------------------------------
# Quick test execution block
# -------------------------------------------------------------------
if __name__ == "__main__":
    logging.info("Pipeline module loaded successfully.")
