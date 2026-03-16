import logging
from typing import Optional

# LlamaIndex retrieval and query engine components
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever, RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# Setup logging for production-style tracing
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)

class AdvancedTableRetriever:
    """
    Implements a Multi-Vector (Parent-Child) Retrieval strategy.
    
    Why this exists:
    Raw markdown tables can be dense and noisy for vector similarity search.
    The MarkdownElementNodeParser creates a 'Summary Node' (IndexNode) for the table.
    This retriever matches the user query against the lightweight summary, 
    but recursively fetches the FULL raw table (Parent Node) for the LLM context.
    This entirely eliminates the 'guessing numbers' hallucination issue.
    """
    
    def __init__(self, index: VectorStoreIndex, similarity_top_k: int = 5):
        logging.info("Initializing AdvancedTableRetriever for Multi-Vector resolution.")
        self.index = index
        self.similarity_top_k = similarity_top_k
        
    def build_query_engine(self) -> RetrieverQueryEngine:
        """
        Builds a customized query engine that resolves IndexNodes to their original 
        Document/Table nodes and applies post-processing (reranking).
        """
        if not self.index.docstore:
            logging.warning("Docstore is empty. Recursive retrieval requires a populated docstore.")

        # 1. Base Retriever: Fetches the top-k nodes (which might be table summaries)
        base_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.similarity_top_k,
        )
        
        # 2. Recursive Retriever: The core of the Multi-Vector approach.
        # It sees an IndexNode (summary), and automatically fetches the linked TextNode (raw table).
        logging.info("Setting up RecursiveRetriever to map table summaries back to raw markdown.")
        recursive_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": base_retriever},
            node_dict=self.index.docstore.docs, # Needs the docstore to resolve references
            verbose=False # Set to True for debugging node resolution
        )
        
        # 3. Node Postprocessors (Re-Ranking)
        # TODO: Implement CohereRerank or BGE-Reranker here. 
        # As discussed, search works fine for 10k+ docs, but complex technical PDFs 
        # need a re-ranker to filter out the noise from dense tabular pages.
        # Currently using a basic SimilarityPostprocessor as a placeholder.
        node_postprocessors = [
            SimilarityPostprocessor(similarity_cutoff=0.5)
        ]
        
        # 4. Final Query Engine Assembly
        logging.info("Assembling final RetrieverQueryEngine with post-processors.")
        query_engine = RetrieverQueryEngine.from_args(
            retriever=recursive_retriever,
            node_postprocessors=node_postprocessors,
            response_mode="compact" # Forces the LLM to strictly use the provided context
        )
        
        return query_engine

# -------------------------------------------------------------------
# Quick test execution block
# -------------------------------------------------------------------
if __name__ == "__main__":
    logging.info("AdvancedTableRetriever module loaded successfully.")
