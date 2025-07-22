import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import List, Optional
import logging

# Setup logger for this crafter
logger = logging.getLogger("flowcraft.ScalerCrafter")

class ScalerCrafter:
    """
    Data Scaling Crafter for numerical feature normalization.
    
    This crafter applies various scaling techniques to numerical columns in the dataset.
    It's essential for machine learning algorithms that are sensitive to feature scales,
    such as logistic regression, SVM, and neural networks.
    
    Parameters:
        scaler_type (str): Type of scaling technique. Options:
            - "minmax": MinMaxScaler - scales features to [0,1] range
            - "standard": StandardScaler - standardizes features (mean=0, std=1) 
            - "robust": RobustScaler - uses median and IQR, robust to outliers
            
        columns (Optional[List[str]]): Specific columns to scale.
            - None: Auto-select all numerical columns (default)
            - List of column names: Scale only specified columns
            
    Context Input:
        - data (pd.DataFrame): Dataset to scale (required)
        - target_column (Optional[str]): Target column to exclude from scaling
        
    Context Output:
        - data (pd.DataFrame): Dataset with scaled numerical features
        - scaler (sklearn transformer): Fitted scaler object for future use
        - scaled_columns (list): Names of columns that were scaled
        - scaler_type (str): Type of scaler used
        
    Example Usage:
        # Scale all numerical columns with MinMax
        scaler = ScalerCrafter(scaler_type="minmax")
        
        # Standard scaling for specific columns
        scaler = ScalerCrafter(
            scaler_type="standard", 
            columns=["feature1", "feature2"]
        )
        
        # Robust scaling (good for data with outliers)
        scaler = ScalerCrafter(scaler_type="robust")
        
    Workflow:
        1. Identify numerical columns (excluding target if specified)
        2. Initialize appropriate scaler based on scaler_type
        3. Fit scaler on training data and transform all data
        4. Update context with scaled data and fitted scaler
        5. Preserve original column order and non-numerical columns
        
    Important Notes:
        - Only numerical columns are scaled
        - Target column is automatically excluded from scaling
        - Scaler object is saved for applying same scaling to new data
        - Categorical columns remain unchanged
    """
    
    def __init__(
        self,
        scaler_type: str = "minmax",
        columns: Optional[List[str]] = None
    ):
        self.scaler_type = scaler_type.lower()
        self.columns = columns

    def run(self, context: dict) -> dict:
        """
        Scale numerical features in the dataset
        Args:
            context: Pipeline context dict with 'data' key
        Returns:
            Updated context with scaled data and scaler object
        """
        logger.info("Starting data scaling...")
        logger.info(f"Scaler type: {self.scaler_type}")
        
        if "data" not in context:
            logger.error("No data found in context")
            raise ValueError("No data found in context. Run DataIngestCrafter first.")
            
        df = context["data"] 
        df_scaled = df.copy()
        
        logger.info(f"Input data shape: {df_scaled.shape}")

        # Determine columns to scale
        columns_to_scale = self.columns
        if columns_to_scale is None:
            # Auto-select numerical columns, but exclude target column if specified in context
            target_col = context.get("target_column")
            numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
            if target_col and target_col in numerical_cols:
                numerical_cols.remove(target_col)  # Don't scale target column
                logger.debug(f"Excluding target column '{target_col}' from scaling")
            columns_to_scale = numerical_cols
            logger.info(f"Auto-selected {len(columns_to_scale)} numerical columns for scaling")
        else:
            logger.info(f"Scaling specified columns: {len(columns_to_scale)} columns")
        
        if not columns_to_scale:
            logger.warning("No columns to scale found")
            context["scaled_columns"] = []
            context["scaler"] = None
            context["scaler_type"] = self.scaler_type
            return context
        
        logger.debug(f"Columns to scale: {columns_to_scale}")

        # Validate columns exist
        missing_cols = [col for col in columns_to_scale if col not in df_scaled.columns]
        if missing_cols:
            logger.error(f"Columns not found in data: {missing_cols}")
            raise ValueError(f"Columns {missing_cols} not found in data")

        # Initialize scaler based on type
        scaler = self._get_scaler()
        logger.info(f"Initialized {type(scaler).__name__}")

        # Fit and transform the selected columns
        logger.info("Fitting scaler on selected columns...")
        df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])
        
        logger.info(f"Scaling completed for {len(columns_to_scale)} columns")
        
        # Log scaling statistics for verification
        for col in columns_to_scale:
            col_stats = df_scaled[col]
            logger.debug(f"Column '{col}': mean={col_stats.mean():.4f}, std={col_stats.std():.4f}, "
                        f"min={col_stats.min():.4f}, max={col_stats.max():.4f}")

        # Update context
        context["data"] = df_scaled
        context["scaler"] = scaler
        context["scaled_columns"] = columns_to_scale
        context["scaler_type"] = self.scaler_type
        
        logger.info("Data scaling completed successfully")
        return context

    def _get_scaler(self):
        """Initialize and return the appropriate scaler"""
        if self.scaler_type == "minmax":
            logger.debug("Creating MinMaxScaler")
            return MinMaxScaler()
        elif self.scaler_type == "standard":
            logger.debug("Creating StandardScaler")
            return StandardScaler()
        elif self.scaler_type == "robust":
            logger.debug("Creating RobustScaler")
            return RobustScaler()
        else:
            logger.error(f"Unsupported scaler type: {self.scaler_type}")
            raise ValueError(f"Unsupported scaler type: {self.scaler_type}")
