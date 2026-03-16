"""
Core Source Module for Table-Aware RAG Pipeline.
Exposing key classes for cleaner imports in main.py.
"""

from .parser import DocumentParser
from .pipeline import RAGPipeline

# Optional: You can also expose sub-modules if needed
# from .chunker import TableAwareChunker 

__all__ = [
    "DocumentParser",
    "RAGPipeline",
]

# This ensures that when someone does 'from src import *', 
# only these classes are exposed.
