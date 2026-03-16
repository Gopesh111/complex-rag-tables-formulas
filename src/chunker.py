import os
import logging
from typing import List

# LlamaIndex core components
from llama_index.core.schema import Document, BaseNode
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.groq import Groq

# Setup logging for production-style tracing
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)

class TableAwareChunker:
    """
    Advanced chunking strategy to prevent context loss in structured data.
    
    Standard text splitters (like RecursiveCharacterTextSplitter) blindly cut 
    tables and LaTeX formulas in half based on character count. This causes 
    the retrieval step to fetch broken rows, leading to LLM hallucinations.
    
    This class uses MarkdownElementNodeParser to identify Tables and Text as 
    distinct 'Elements'. It extracts tables whole and creates summary nodes 
    to route queries accurately without butchering the raw data.
    """
    
    def __init__(self):
        logging.info("Initializing TableAwareChunker...")
        
        # We need an LLM to generate summaries for the extracted table nodes.
        # Using Groq (Llama3) here for fast, cost-effective local testing.
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logging.error("GROQ_API_KEY is missing from environment variables.")
            raise ValueError("Please set GROQ_API_KEY to use the Element Node Parser.")
            
        try:
            self.llm = Groq(model="llama3-70b-8192", api_key=api_key)
            
            # num_workers=4 enables parallel processing for large research papers
            self.node_parser = MarkdownElementNodeParser(
                llm=self.llm, 
                num_workers=4
            )
            logging.info("MarkdownElementNodeParser successfully loaded with Groq LLM backend.")
        except Exception as e:
            logging.error(f"Failed to initialize LLM or Node Parser: {e}")
            raise

    def process_documents(self, documents: List[Document]) -> List[BaseNode]:
        """
        Parses a list of Documents into semantic nodes. 
        It isolates tables into distinct IndexNodes, ensuring structural integrity 
        is maintained for the VectorStore.
        """
        if not documents:
            logging.warning("Empty document list provided for chunking.")
            return []

        logging.info(f"Starting semantic chunking for {len(documents)} document(s)...")
        
        try:
            # The node parser will automatically detect markdown tables (e.g., | col | col |)
            # and prevent them from being split mid-row.
            nodes = self.node_parser.get_nodes_from_documents(documents)
            
            # Trace extraction metrics to verify the parser is actually catching tables
            table_nodes = [n for n in nodes if hasattr(n, 'metadata') and n.metadata.get('table_df') is not None]
            
            logging.info(f"Chunking complete. Generated {len(nodes)} total nodes.")
            logging.info(f"Successfully isolated {len(table_nodes)} complex table elements intact.")
            
            return nodes
            
        except Exception as e:
            logging.error(f"Error during node parsing and chunking: {e}")
            return []

# -------------------------------------------------------------------
# Quick test execution block (Will not run when imported as a module)
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Mock test to ensure the class initializes correctly
    try:
        chunker = TableAwareChunker()
        logging.info("Chunker module is ready for integration.")
    except Exception as err:
        logging.error(f"Module test failed: {err}")
