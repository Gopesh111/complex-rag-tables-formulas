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
        
        logging.info("Initializing DocumentParser with Docling backend...")
        try:
            self.converter = DocumentConverter()
        except Exception as e:
            logging.error(f"Failed to initialize Docling converter: {e}")
            raise

    def parse_to_markdown(self, file_path: str) -> Optional[str]:
        """
        Primary extraction method. Converts the PDF into a Markdown string.
        Includes a fallback mechanism to pdfplumber for borderless tables.
        """
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Missing input file at {file_path}")
            
        try:
            logging.info(f"Parsing {os.path.basename(file_path)} via Docling...")
            result = self.converter.convert(file_path)
            markdown_text = result.document.export_to_markdown()
            
            # Heuristic: If Docling fails to extract meaningful content, trigger fallback
            if not markdown_text or len(markdown_text.strip()) < 10:
                raise ValueError("Docling extracted empty or negligible content.")

            logging.info("Successfully converted document to Markdown format.")
            return markdown_text
            
        except Exception as e:
            
            logging.warning(f"Docling parsing failed or returned empty: {e}. Triggering pdfplumber fallback...")
            return self._execute_fallback(file_path)

    def extract_clean_tables_fallback(self, file_path: str) -> List[List[List[str]]]:
        """
        Fallback method using pdfplumber.
        Works exceptionally well for clean, non-scanned tabular data.
        """
        extracted_tables = []
        try:
            logging.info(f"Running pdfplumber fallback for explicit table extraction on {file_path}")
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
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

    def _execute_fallback(self, file_path: str) -> str:
        """
        Executes the fallback extraction and formats text + tables into Markdown
        so downstream chunkers don't break.
        """
        fallback_md = "## Fallback Document Extraction\n\n"
        
        # 1. Extract basic text
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text: fallback_md += text + "\n\n"
        except Exception as e:
            logging.error(f"Text extraction failed in fallback: {e}")

        # 2. Extract tables and convert them to Markdown
        tables = self.extract_clean_tables_fallback(file_path)
        if tables:
            fallback_md += "### Extracted Tables\n\n"
            fallback_md += self._plumber_to_markdown(tables)

        return fallback_md

    def _plumber_to_markdown(self, tables: List[List[List[str]]]) -> str:
        """Helper function to convert pdfplumber's List of Lists into a Markdown string."""
        md_strings = []
        for table in tables:
            # Need at least a header and one row
            if not table or len(table) < 2: 
                continue 
            
            # Clean None values
            cleaned_table = [[str(cell).replace('\n', ' ').strip() if cell else "" for cell in row] for row in table]
            
            header = cleaned_table[0]
            md_table = f"| {' | '.join(header)} |\n"
            md_table += f"| {' | '.join(['---'] * len(header))} |\n"
            
            for row in cleaned_table[1:]:
                # Pad row if it has fewer columns than header
                row = row + [''] * (len(header) - len(row))
                md_table += f"| {' | '.join(row[:len(header)])} |\n"
                
            md_strings.append(md_table)
            
        return "\n\n".join(md_strings)

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
