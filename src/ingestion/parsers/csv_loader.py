
"""
CSV Document Loader
===================

Loads CSV files and converts rows into ParsedResult objects.

Features:
    1. Encoding fallback for Arabic CSV files.
    2. SHA-256 file hashing.
    3. Semantic column classification.
    4. Legal text extraction.
    5. Metadata extraction.
    6. Original row preservation.
    7. Schema validation.
    8. Row-level traceability.

Architecture:

    CSV File
       |
       v
    Pandas DataFrame
       |
       v
    Column Semantic Classifier
       |
       v
    Validated CSV Schema
       |
       v
    Row Extraction
       |
       +---- Legal Text
       |
       +---- Metadata
       |
       +---- Original Row
       |
       v
    ParsedResult Objects
"""

import hashlib
import logging

from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from src.ingestion.models import ParsedResult
from src.ingestion.parsers.base import BaseDocumentLoader

from src.ingestion.parsers.csv_column_classifier import (
    classify_csv_columns,
    CSVColumnSchema,
)

logger = logging.getLogger(__name__)


class CSVLoader(BaseDocumentLoader):
    """Loader for CSV (.csv) files containing Egyptian legal data."""

    def __init__(
        self,
        encoding_fallbacks: Optional[List[str]] = None,
        llm_provider=None,
        use_llm: bool = True,
        reject_schema_needs_review: bool = False,
        preserve_raw_row: bool = True,
    ):
        """
        Initialize CSV loader.

        Args:
            encoding_fallbacks:
                Encodings to try if UTF-8 fails.

            llm_provider:
                Optional LLM provider used for ambiguous columns.

            use_llm:
                Whether semantic classification can use an LLM.

            reject_schema_needs_review:
                If True, reject CSV files whenever the classifier
                marks their schema as requiring review.

                If False, allow ingestion when a reliable text column
                exists, even if some metadata columns are uncertain.

            preserve_raw_row:
                Whether to preserve the original CSV row in metadata.
        """

        self.encoding_fallbacks = encoding_fallbacks or [
            "utf-8-sig",
            "cp1256",
            "latin-1",
        ]

        self.llm_provider = llm_provider

        self.use_llm = use_llm

        self.reject_schema_needs_review = (
            reject_schema_needs_review
        )

        self.preserve_raw_row = preserve_raw_row

    # ========================================================
    # 1. MAIN ENTRY POINT
    # ========================================================

    def load(self, file_path: Path) -> List[ParsedResult]:
        """
        Load CSV and convert its rows into ParsedResult objects.

        Each CSV row becomes one ParsedResult.

        The content field contains only the detected legal text.

        Metadata contains:
            - File information
            - Law information
            - Article information
            - Other classified metadata
            - Original row data (optional)
        """

        file_path = Path(file_path)

        logger.info(
            "Starting CSV load: %s",
            file_path.name,
        )

        try:

            # ------------------------------------------------
            # Read CSV
            # ------------------------------------------------

            df = self._read_csv_with_fallback(file_path)

            if df.empty:

                logger.warning(
                    "CSV file %s is empty.",
                    file_path.name,
                )

                return []

            # ------------------------------------------------
            # Calculate file hash
            # ------------------------------------------------

            file_hash = self._calculate_file_hash(
                file_path
            )

            # ------------------------------------------------
            # Classify columns
            # ------------------------------------------------

            schema = self._classify_columns(
                df=df,
                file_hash=file_hash,
            )

            # ------------------------------------------------
            # Validate schema
            # ------------------------------------------------

            self._validate_schema(
                schema=schema,
                file_path=file_path,
            )

            # ------------------------------------------------
            # Convert rows
            # ------------------------------------------------

            results = self._convert_dataframe_to_results(
                df=df,
                file_path=file_path,
                source_format="csv",
                schema=schema,
            )

            logger.info(
                "Successfully loaded CSV: %s | "
                "Rows extracted: %d | "
                "Text column: %s | "
                "Metadata columns: %d",
                file_path.name,
                len(results),
                schema.text_column,
                len(schema.metadata_columns),
            )

            return results

        except Exception as error:

            logger.exception(
                "Failed to load CSV %s: %s",
                file_path.name,
                error,
            )

            raise ValueError(
                f"Failed to parse CSV '{file_path.name}': {error}"
            ) from error

    # ========================================================
    # 2. CSV READING
    # ========================================================

    def _read_csv_with_fallback(
        self,
        file_path: Path,
    ) -> pd.DataFrame:
        """
        Read CSV using multiple encodings.

        Supports common Arabic CSV encodings:
            - UTF-8
            - UTF-8 with BOM
            - Windows Arabic (CP1256)
            - Latin-1
        """

        encodings = [
            "utf-8",
            *self.encoding_fallbacks,
        ]

        # Remove duplicate encodings while preserving order.
        encodings = list(dict.fromkeys(encodings))

        last_error = None

        for encoding in encodings:

            try:

                logger.debug(
                    "Reading %s using encoding=%s",
                    file_path.name,
                    encoding,
                )

                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    low_memory=False,
                )

                # Remove accidental whitespace from headers.
                df.columns = [
                    str(col).strip()
                    for col in df.columns
                ]

                logger.info(
                    "CSV successfully decoded using %s",
                    encoding,
                )

                return df

            except UnicodeDecodeError as error:

                last_error = error

                logger.warning(
                    "Encoding %s failed for %s.",
                    encoding,
                    file_path.name,
                )

        raise ValueError(
            f"Could not decode {file_path.name} "
            f"using supported encodings: {encodings}. "
            f"Last error: {last_error}"
        )

    # ========================================================
    # 3. FILE HASHING
    # ========================================================

    def _calculate_file_hash(
        self,
        file_path: Path,
    ) -> str:
        """
        Calculate SHA-256 hash of the CSV file.

        Used by the semantic classifier to cache
        the column classification result.

        Streaming is used to avoid loading the entire
        file into memory a second time.
        """

        sha256 = hashlib.sha256()

        with open(file_path, "rb") as file:

            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):

                sha256.update(chunk)

        return sha256.hexdigest()

    # ========================================================
    # 4. COLUMN CLASSIFICATION
    # ========================================================

    def _classify_columns(
        self,
        df: pd.DataFrame,
        file_hash: str,
    ) -> CSVColumnSchema:
        """
        Classify CSV columns using the semantic classifier.

        The classifier runs once per unique file hash.
        """

        logger.info(
            "Classifying CSV columns: %s",
            list(df.columns),
        )

        schema = classify_csv_columns(
            df=df,
            file_hash=file_hash,
            llm_provider=self.llm_provider,
            use_llm=self.use_llm,
        )

        # ----------------------------------------------------
        # Log classification results
        # ----------------------------------------------------

        for classification in schema.classifications:

            logger.info(
                "Column: '%s' | "
                "Role: %s | "
                "Confidence: %.2f | "
                "Method: %s | "
                "Review: %s",
                classification.name,
                classification.role.value,
                classification.confidence,
                classification.method,
                classification.needs_review,
            )

        if schema.needs_review:

            logger.warning(
                "CSV schema requires review: %s",
                schema.issues,
            )

        return schema

    # ========================================================
    # 5. SCHEMA VALIDATION
    # ========================================================

    def _validate_schema(
        self,
        schema: CSVColumnSchema,
        file_path: Path,
    ) -> None:
        """
        Validate the schema before row extraction.

        Critical rule:
            Never generate legal chunks without
            identifying a reliable text column.

        Unknown metadata columns do not necessarily
        prevent ingestion.
        """

        # ----------------------------------------------------
        # No text column
        # ----------------------------------------------------

        if schema.text_column is None:

            raise ValueError(
                "No reliable legal text column detected. "
                f"File: {file_path.name}. "
                f"Candidates: {schema.text_candidates}. "
                f"Issues: {schema.issues}"
            )

        # ----------------------------------------------------
        # Schema requires manual review
        # ----------------------------------------------------

        if (
            schema.needs_review
            and self.reject_schema_needs_review
        ):

            raise ValueError(
                "CSV schema requires manual review. "
                f"File: {file_path.name}. "
                f"Issues: {schema.issues}"
            )

        # ----------------------------------------------------
        # Warning
        # ----------------------------------------------------

        if schema.needs_review:

            logger.warning(
                "Proceeding with CSV ingestion despite "
                "schema warnings. Text column='%s'. "
                "File='%s'.",
                schema.text_column,
                file_path.name,
            )

    # ========================================================
    # 6. ROW CONVERSION
    # ========================================================

    def _convert_dataframe_to_results(
        self,
        df: pd.DataFrame,
        file_path: Path,
        source_format: str,
        schema: CSVColumnSchema,
    ) -> List[ParsedResult]:
        """
        Convert each DataFrame row into ParsedResult.

        IMPORTANT:

            content:
                Only the legal text.

            metadata:
                Classified metadata + file information.

            raw_row_data:
                Original row values for debugging and
                future processing.
        """

        results: List[ParsedResult] = []

        columns = list(df.columns)

        total_rows = len(df)

        # ----------------------------------------------------
        # File-level metadata
        # ----------------------------------------------------

        file_metadata = self._extract_file_metadata(
            file_path=file_path,
            source_format=source_format,
            total_rows=total_rows,
            columns=columns,
            schema=schema,
        )

        text_column = schema.text_column

        logger.info(
            "Extracting legal text from column: '%s'",
            text_column,
        )

        # ----------------------------------------------------
        # Process rows
        # ----------------------------------------------------

        for position, (idx, row) in enumerate(
            df.iterrows()
        ):

            # ------------------------------------------------
            # Clean row values
            # ------------------------------------------------

            raw_row_data = self._clean_row(row)

            # ------------------------------------------------
            # Extract legal text
            # ------------------------------------------------

            content = schema.get_text_for_row(
                raw_row_data
            )

            if not content:

                logger.debug(
                    "Skipping empty text at row %s in %s.",
                    idx,
                    file_path.name,
                )

                continue

            # ------------------------------------------------
            # Extract semantic metadata
            # ------------------------------------------------

            semantic_metadata = (
                schema.get_metadata_for_row(
                    raw_row_data
                )
            )

            # ------------------------------------------------
            # Build row metadata
            # ------------------------------------------------

            row_metadata = {

                **file_metadata,

                **semantic_metadata,

                "row_index": self._normalize_row_index(
                    idx,
                    position,
                ),

                "text_column": text_column,

                "schema_needs_review": schema.needs_review,

            }

            # ------------------------------------------------
            # Preserve original row
            # ------------------------------------------------

            if self.preserve_raw_row:

                row_metadata["raw_row_data"] = (
                    raw_row_data
                )

            # ------------------------------------------------
            # Preserve row
            # ------------------------------------------------

            parsed_result = ParsedResult(

                content=content,

                metadata=row_metadata,

            )

            results.append(parsed_result)

        # ----------------------------------------------------
        # Final statistics
        # ----------------------------------------------------

        skipped_rows = total_rows - len(results)

        logger.info(
            "CSV row extraction completed: "
            "Total=%d | Extracted=%d | Skipped=%d",
            total_rows,
            len(results),
            skipped_rows,
        )

        return results

    # ========================================================
    # 7. ROW CLEANING
    # ========================================================

    def _clean_row(
        self,
        row: pd.Series,
    ) -> Dict[str, str]:
        """
        Convert a pandas row into a clean dictionary.

        Handles:
            - NaN
            - None
            - Pandas missing values
            - Whitespace
        """

        cleaned = {}

        for col, value in row.items():

            if self._is_missing(value):

                cleaned[str(col)] = ""

            else:

                cleaned[str(col)] = str(value).strip()

        return cleaned

    @staticmethod
    def _is_missing(value: Any) -> bool:
        """Safely check whether a pandas value is missing."""

        if value is None:
            return True

        try:

            result = pd.isna(value)

            if isinstance(result, bool):
                return result

        except (TypeError, ValueError):

            pass

        return False

    @staticmethod
    def _normalize_row_index(
        original_index: Any,
        position: int,
    ) -> Any:
        """
        Preserve the original row index when possible.

        Position is used as a fallback.
        """

        try:

            return int(original_index)

        except (TypeError, ValueError):

            return position

    # ========================================================
    # 8. FILE METADATA
    # ========================================================

    def _extract_file_metadata(
        self,
        file_path: Path,
        source_format: str,
        total_rows: int,
        columns: List[str],
        schema: CSVColumnSchema,
    ) -> Dict[str, Any]:
        """
        Extract file-level metadata.

        This metadata is copied into every ParsedResult.
        """

        classifications = {

            classification.name: {

                "role": classification.role.value,

                "confidence": classification.confidence,

                "method": classification.method,

            }

            for classification in schema.classifications

        }

        return {

            "file_name": file_path.name,

            "file_path": str(
                file_path.absolute()
            ),

            "file_size_bytes": file_path.stat().st_size,

            "source_format": source_format,

            "total_rows": total_rows,

            "columns": columns,

            "csv_text_column": schema.text_column,

            "csv_text_candidates": schema.text_candidates,

            "csv_column_classifications": classifications,

            "csv_schema_needs_review": schema.needs_review,

            "csv_schema_issues": schema.issues,

        }
