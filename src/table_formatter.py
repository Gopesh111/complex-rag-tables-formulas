import logging
from typing import List

logger = logging.getLogger(__name__)

class TableFormatter:
    """
    Utility class to transform Markdown tables into natural language context.
    
    This approach enhances semantic search capabilities in vector databases by
    converting rigid tabular structures into row-wise descriptive sentences,
    significantly reducing LLM hallucinations during retrieval.
    """

    @staticmethod
    def markdown_to_natural_language(markdown_table: str) -> str:
        """
        Parses a Markdown formatted table and converts it into a natural language paragraph.
        
        Args:
            markdown_table (str): The raw markdown table string extracted from the document.
            
        Returns:
            str: A concatenated string of natural language sentences representing each row,
                 or the original string if parsing fails or input is not a valid table.
        """
        if not markdown_table or "|" not in markdown_table:
            return markdown_table
        
        # Split into lines and filter out empty strings or non-table lines
        lines = [line.strip() for line in markdown_table.strip().split('\n') if "|" in line]
        
        # A valid markdown table needs at least a header, a separator, and one data row (3 lines)
        if len(lines) < 3:
            return markdown_table

        try:
            # Extract headers (removing empty strings caused by leading/trailing '|')
            header_line = lines[0]
            headers = [col.strip() for col in header_line.split('|')][1:-1]
            
            # The second line is the separator (e.g., |---|---|), so we skip it
            data_lines = lines[2:]
            
            sentences = []
            for row_idx, line in enumerate(data_lines):
                # Extract cells for the current row
                cells = [col.strip() for col in line.split('|')][1:-1]
                
                # Structural validation: Ensure cell count matches header count
                if len(cells) != len(headers):
                    logger.debug(f"Row anomaly detected at index {row_idx}. Skipping row contextualization.")
                    continue
                    
                row_context = []
                for header, cell in zip(headers, cells):
                    # Ignore empty cells or basic markdown placeholders
                    if cell and cell not in ('-', 'N/A', ''):
                        row_context.append(f"the {header} is {cell}")
                
                # Construct the sentence for the row
                if row_context:
                    sentence = f"In this record, {', '.join(row_context)}."
                    sentences.append(sentence)

            if not sentences:
                return markdown_table
                
            return " ".join(sentences)
            
        except Exception as e:
            logger.warning(f"Failed to format table to natural language: {e}. Returning raw markdown.")
            return markdown_table

if __name__ == "__main__":
    # Local testing execution block
    sample_table = """
    | Region | Quarter | Revenue | Growth |
    |--------|---------|---------|--------|
    | North  | Q1      | 450.5M  | 12.4%  |
    | South  | Q1      | 380.2M  | 10.1%  |
    """
    
    formatted_text = TableFormatter.markdown_to_natural_language(sample_table)
    print("Original Table:\n", sample_table)
    print("\nFormatted Natural Language:\n", formatted_text)
