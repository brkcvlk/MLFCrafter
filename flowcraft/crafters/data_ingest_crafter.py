import pandas as pd
from pathlib import Path
from typing import Optional
import logging

# Setup logger for this crafter
logger = logging.getLogger("flowcraft.DataIngestCrafter")

class DataIngestCrafter:
    """
    Data Ingestion Crafter for loading data from various file formats.
    
    This crafter handles the first step of ML pipeline by loading data from
    supported file formats (CSV, Excel, JSON) with automatic format detection
    or explicit format specification.
    
    Supported Formats:
        - CSV files (.csv)
        - Excel files (.xls, .xlsx)  
        - JSON files (.json)
    
    Parameters:
        data_path (Optional[str]): Path to the data file. Can be None if provided in context.
        source_type (str): File format type. Options:
            - "auto": Automatically detect format from file extension (default)
            - "csv": Force CSV reading
            - "excel": Force Excel reading  
            - "json": Force JSON reading
    
    Context Input:
        - data_path (Optional[str]): Alternative way to provide file path
        
    Context Output:
        - data (pd.DataFrame): Loaded dataset
        - original_shape (tuple): Shape of original data (rows, columns)
        
    Example Usage:
        # Auto-detect format
        crafter = DataIngestCrafter(data_path="data/sales.csv")
        
        # Explicit format
        crafter = DataIngestCrafter(data_path="data/sales.xlsx", source_type="excel")
        
        # Path from context
        crafter = DataIngestCrafter()  # data_path provided in context
        
    Workflow:
        1. Determine data path (from constructor or context)
        2. Detect or validate file format
        3. Load data using appropriate pandas reader
        4. Add loaded data and metadata to context
    """
    
    def __init__(self, data_path: Optional[str] = None, source_type: str = "auto"):
        self.data_path = data_path
        self.source_type = source_type.lower()

    def run(self, context: dict) -> dict:
        """
        Load data from file and add to context
        Args:
            context: Pipeline context dict
        Returns:
            Updated context with 'data' key containing loaded DataFrame
        """
        logger.info("Starting data ingestion...")
        
        # Use data_path from context if not provided in constructor
        data_path = self.data_path or context.get('data_path')
        if not data_path:
            logger.error("No data path provided")
            raise ValueError("data_path must be provided either in constructor or context")
            
        logger.info(f"Loading data from: {data_path}")
        logger.debug(f"Source type: {self.source_type}")
            
        file = Path(str(data_path))
        suffix = file.suffix.lower()

        if self.source_type == "auto":
            logger.debug(f"Auto-detecting format from extension: {suffix}")
            data = self._read_auto(file, suffix)
        else:
            expected_ext = self._expected_extension(self.source_type)
            if suffix not in expected_ext:
                logger.error(f"File extension mismatch: {suffix} not in {expected_ext}")
                raise ValueError(f"File extension and source type dont match:\n"
                                 f" - File extension: {suffix}\n - Expected extension: {expected_ext}")
            data = self._read_by_type(file, self.source_type)

        # Update context with loaded data
        context["data"] = data
        context["original_shape"] = data.shape
        
        logger.info(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
        logger.info("Data ingestion completed")
        
        return context

    def _expected_extension(self, source_type: str):
        return {
            "csv": [".csv"],
            "excel": [".xls", ".xlsx"],
            "json": [".json"]
        }.get(source_type, [])

    def _read_by_type(self, file: Path, source_type: str):
        if source_type == "csv":
            logger.debug("Reading CSV file")
            return pd.read_csv(file)
        elif source_type == "excel":
            logger.debug("Reading Excel file")
            return pd.read_excel(file)
        elif source_type == "json":
            logger.debug("Reading JSON file")
            return pd.read_json(file)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    def _read_auto(self, file: Path, suffix: str):
        if suffix == ".csv":
            logger.debug("Auto-detected CSV format")
            return pd.read_csv(file)
        elif suffix in [".xls", ".xlsx"]:
            logger.debug("Auto-detected Excel format")
            return pd.read_excel(file)
        elif suffix == ".json":
            logger.debug("Auto-detected JSON format")
            return pd.read_json(file)
        else:
            raise ValueError(f"Unsupported source type: {suffix}")
