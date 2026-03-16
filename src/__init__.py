"""
Core Source Module for Table-Aware RAG Pipeline.
Exposes key classes for clean and professional imports.
"""

from .parser import DocumentParser
from .chunker import TableAwareChunker
from .pipeline import RAGPipeline
from .retriever import AdvancedTableRetriever

__all__ = [
    "DocumentParser",
    "TableAwareChunker",
    "RAGPipeline",
    "AdvancedTableRetriever"
]

