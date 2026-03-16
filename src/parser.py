import os
from typing import List, Optional
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
import pdfplumber

class DocumentParser:
    """
    Handles PDF parsing with a focus on structural integrity of tables 
    and mathematical formulas.
    """
    
    def __init__(self):
        # Initializing Docling for complex layout analysis
        self.converter = DocumentConverter()
        print("Parser initialized with Docling and pdfplumber fallback.")

    def parse_with_docling(self, file_path: str) -> str:
        """
        Converts PDF to Markdown to preserve table structures and LaTeX.
        Recommended by community experts for production RAG.
        """
        try:
            # Converting to markdown to keep | col | structure intact
            result = self.converter.convert(file_path)
            markdown_output = result.document.export_to_markdown()
            return markdown_output
        except Exception as e:
            print(f"Error parsing with Docling: {e}")
            return ""

    def extract_tables_pdfplumber(self, file_path: str):
        """
        Fallback method for clean, non-scanned tabular data.
        Implementation based on Alampally's suggestion for high-accuracy row extraction.
        """
        tables = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # Extracting raw table data to maintain row/column alignment
                extracted_table = page.extract_table()
                if extracted_table:
                    tables.append(extracted_table)
        return tables

    def get_full_context(self, file_path: str) -> str:
        """
        Main entry point for parsing. 
        TODO: Implement hybrid approach to merge pdfplumber accuracy with Docling layout.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
            
        print(f"Starting extraction for: {os.path.basename(file_path)}")
        return self.parse_with_docling(file_path)

# Example usage (commented out for production)
# if __name__ == "__main__":
#     parser = DocumentParser()
#     content = parser.get_full_context("data/sample.pdf")
#     print(content[:500]) # Preview first 500 chars
