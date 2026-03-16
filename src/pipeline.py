import os
from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.groq import Groq  # Fast inference for testing
from .parser import DocumentParser

class RAGPipeline:
    """
    Main Orchestrator for the Table-Aware RAG system.
    Integrates parsing, element-aware chunking, and vector indexing.
    """

    def __init__(self, api_key: str):
        # Using Groq for high-speed open-source model inference (Llama3)
        self.llm = Groq(model="llama3-70b-8192", api_key=api_key)
        self.parser = DocumentParser()
        
        # This parser is key: it identifies tables as distinct elements 
        # so they aren't butchered during standard character splitting.
        self.node_parser = MarkdownElementNodeParser(llm=self.llm, num_workers=4)
        print("RAG Pipeline initialized with Table-Aware Node Parser.")

    def run_ingestion(self, file_path: str):
        """
        Full ingestion flow: Parse -> Element Extraction -> Indexing.
        """
        print(f"Ingesting: {file_path}")
        
        # Step 1: Extract Markdown content (Docling handles the tables/math)
        markdown_text = self.parser.get_full_context(file_path)
        doc = Document(text=markdown_text, metadata={"source": os.path.basename(file_path)})

        # Step 2: Extract base nodes and 'Element' nodes (Tables/Images)
        # This prevents the 'guesswork' by the LLM mentioned in the LinkedIn post.
        nodes = self.node_parser.get_nodes_from_documents([doc])
        
        # Step 3: Create Vector Index
        # TODO: Replace with Persistent ChromaDB for production use.
        self.index = VectorStoreIndex(nodes)
        print(f"Successfully indexed {len(nodes)} nodes (including table elements).")

    def query(self, question: str) -> str:
        """
        Query the index with a focus on retrieving structured table data.
        """
        if not self.index:
            return "Index not initialized. Please run ingestion first."
        
        # Customizing synthesis to ensure LLM respects the Markdown table structure
        query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact" 
        )
        
        response = query_engine.query(question)
        return str(response)

# Example logic for internal testing
# if __name__ == "__main__":
#     pipeline = RAGPipeline(api_key="your_key_here")
#     pipeline.run_ingestion("data/technical_specs.pdf")
#     print(pipeline.query("Compare the values in the efficiency table."))
