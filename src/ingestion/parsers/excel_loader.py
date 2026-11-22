# src/ingestion/parsers/excel_loader.py
"""
Excel Document Loader.
Extracts tabular data from Excel (.xlsx, .xls) files using Pandas.
Handles multiple sheets, converting each row in each sheet into a ParsedResult.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from src.ingestion.models import ParsedResult
from src.ingestion.parsers.base import BaseDocumentLoader
from src.ingestion.parsers.csv_loader import CSVLoader  # Reuse the conversion logic

logger = logging.getLogger(__name__)

class ExcelLoader(BaseDocumentLoader):
    """Loader for Excel (.xlsx, .xls) files."""
    
    def __init__(self):
        # We reuse the CSVLoader's conversion logic to avoid duplicating code
        self._csv_loader = CSVLoader()

    def load(self, file_path: Path) -> List[ParsedResult]:
        logger.info(f"Starting Excel load for: {file_path.name}")
        
        try:
            # Load the Excel file to inspect sheets
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            if not sheet_names:
                logger.warning(f"Excel file {file_path.name} contains no sheets.")
                return []
            
            all_results = []
            
            # Process each sheet independently
            for sheet_name in sheet_names:
                logger.info(f"Processing sheet: {sheet_name}")
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                if df.empty:
                    logger.warning(f"Sheet '{sheet_name}' in {file_path.name} is empty. Skipping.")
                    continue
                
                # Convert rows to ParsedResults
                sheet_results = self._csv_loader._convert_dataframe_to_results(
                    df, file_path, source_format="excel"
                )
                
                # Inject sheet-specific metadata into each result
                for result in sheet_results:
                    result.metadata["sheet_name"] = sheet_name
                
                all_results.extend(sheet_results)
            
            logger.info(f"Successfully extracted {len(all_results)} rows from {file_path.name}")
            return all_results
            
        except Exception as e:
            logger.error(f"Failed to load Excel {file_path.name}: {e}")
            raise ValueError(f"Failed to parse Excel: {e}")