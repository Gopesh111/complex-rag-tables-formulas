import os
import logging
from typing import Optional, List
import pdfplumber
from docling.document_converter import DocumentConverter

# Setup basic logging to simulate a production-ready environment
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)

class DocumentParser:
    """
    Advanced document parser designed to handle complex PDFs containing 
    nested tables and mathematical formulas.
    
    Standard parsers (PyPDF, PDFMiner) strip out structural boundaries, 
    causing the LLM to hallucinate numbers during RAG retrieval. 
    This implementation primarily uses Docling to convert documents into Markdown, 
    preserving the | col1 | col2 | format and LaTeX formulas.
    """
    
    def __init__(self):
        # Vishal from LinkedIn suggested Docling for better table normalization.
        # Initializing the converter here. In the future, this can be accelerated 
        # with GPU backends if document volume scales up.
        logging.info("Initializing DocumentParser with Docling backend...")
        try:
            self.converter = DocumentConverter()
        except Exception as e:
            logging.error(f"Failed to initialize Docling converter: {e}")
            raise

    def parse_to_markdown(self, file_path: str) -> Optional[str]:
        """
        Primary extraction method. Converts the PDF into a Markdown string.
        This ensures that downstream chunkers (like MarkdownElementNodeParser)
        can accurately identify table boundaries and avoid context loss.
        """
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Missing input file at {file_path}")
            
        try:
            logging.info(f"Parsing {os.path.basename(file_path)} via Docling...")
            result = self.converter.convert(file_path)
            
            # Exporting to markdown prevents the 'context loss' across chunks.
            # It keeps equations as $latex$ and tables as structured markdown tables.
            markdown_text = result.document.export_to_markdown()
            logging.info("Successfully converted document to Markdown format.")
            
            return markdown_text
            
        except Exception as e:
            logging.error(f"Docling parsing failed on {file_path}. Error: {e}")
            return None

    def extract_clean_tables_fallback(self, file_path: str) -> List[List[List[str]]]:
        """
        Fallback method using pdfplumber.
        As suggested by Alampally, this works exceptionally well for clean, 
        non-scanned tabular data where Docling's layout analysis might be overkill.
        
        Returns a list of tables, where each table is a list of rows (which are lists of strings).
        """
        extracted_tables = []
        try:
            logging.info(f"Running pdfplumber fallback for explicit table extraction on {file_path}")
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # extract_tables() gets all tables on the page
                    tables = page.extract_tables()
                    for idx, table in enumerate(tables):
                        if table:
                            extracted_tables.append(table)
                            logging.debug(f"Extracted table {idx} from page {page_num + 1}")
                            
            logging.info(f"Extracted {len(extracted_tables)} raw tables via pdfplumber fallback.")
            return extracted_tables
            
        except Exception as e:
            logging.error(f"pdfplumber fallback extraction failed: {e}")
            return []

# -------------------------------------------------------------------
# Quick test execution block (Will not run when imported as a module)
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate a quick local test
    parser = DocumentParser()
    test_file = "../data/sample_complex.pdf"
    
    if os.path.exists(test_file):
        md_output = parser.parse_to_markdown(test_file)
        if md_output:
            print(f"\n[Preview of extracted markdown]:\n{md_output[:500]}...\n")
    else:
        logging.warning("No test file found. Add a PDF to data/sample_complex.pdf to test locally.")
