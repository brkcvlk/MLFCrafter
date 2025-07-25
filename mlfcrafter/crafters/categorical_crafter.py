import logging 

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from typing import List, Optional
import pandas as pd
# Setup logger for this crafter
logger = logging.getLogger("mlfcrafter.CategoricalCrafter")
class CategoricalCrafter:
    """
    Data Categorical Crafter for encoding categorical features.

    This crafter applies various encoding techniques to categorical columns in the dataset.
    It's essential for machine learning algorithms that require numerical input.

    Parameters:
        encoder_type (str): Type of encoding technique. Options:
            - "onehot": OneHotEncoder - creates binary columns for each category
            - "label": LabelEncoder - converts categories to integer labels

        columns (Optional[List[str]]): Specific columns to encode.
            - None: Auto-select all categorical columns (default)
            - List of column names: Encode only specified columns

    Context Input:
        - data (pd.DataFrame): Dataset to encode (required)
        - target_column (Optional[str]): Target column to exclude from encoding

    Context Output:
        - data (pd.DataFrame): Dataset with encoded categorical features
        - encoder (sklearn transformer): Fitted encoder object for future use
        - encoded_columns (list): Names of columns that were encoded
        - encoder_type (str): Type of encoding used

    Example Usage:
        # One-hot encode all categorical columns
        encoder = Categorical(encoder_type="onehot")

        # Label encoding for specific columns
        encoder = Categorical(
            encoder_type="label",
            columns=["category1", "category2"]
        )

    Workflow:
        1. Identify categorical columns (excluding target if specified)
        2. Initialize appropriate encoder based on encoder_type
        3. Fit encoder on training data and transform all data
        4. Update context with encoded data and fitted encoder
        5. Preserve original column order and non-categorical columns

    Important Notes:
        - Only categorical columns are encoded
    """
    
    def __init__(self, encoder_type: str = "onehot", columns: Optional[List[str]] = None):
        self.encoder_type = encoder_type.lower()
        self.columns = columns if columns is not None else []
    
    def run(self, context: dict) -> dict:
        """
        Run the categorical encoding process and update the context.
        
        Args:
            context (dict): Pipeline context containing 'data' key with DataFrame.
        
        Returns:
            dict: Updated context with encoded data and encoder metadata.
        """
        logger.info("Starting categorical encoding...")
        
        # Extract data from context
        if "data" not in context:
            logger.error("No data found in context")
            raise ValueError("No data found in context. Run DataIngestCrafter first.")
        # Identify categorical columns
        df = context["data"]
        df_encoded = df.copy()

        if not self.columns:
            # Auto-select categorical columns if none specified
            self.columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        logger.info(f"Columns to encode: {self.columns}")
        if not self.columns:
            # If no columns to encode, skip encoding
            logger.warning("No columns to encoding found")
            context["encoded_columns"] = []
            context["encoder"] = None
            context["encoder_type"] = self.encoder_type
            return context
        # Initialize the appropriate encoder
        encoder = self._get_encoder()
        logger.info(f"Initialized {type(encoder).__name__}")

        # Fit and transform the selected columns
        if self.encoder_type == "label":
            logger.info("Fitting LabelEncoder on selected columns...")
            for col in self.columns:
                df_encoded[col] = encoder.fit_transform(df_encoded[col])
        elif self.encoder_type == "onehot":
            logger.info("Fitting OneHotEncoder on selected columns...")
            encoded_array = encoder.fit_transform(df_encoded[self.columns])
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=encoder.get_feature_names_out(self.columns),
                index=df_encoded.index
            )
            df_encoded = df_encoded.drop(columns=self.columns)
            # Concatenate the original DataFrame with the encoded DataFrame
            df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

        logger.info(f"Encoding completed for {len(self.columns)} columns")
        # Update context with encoded data and encoder
        context["data"] = df_encoded
        context["encoder"] = encoder
        context["encoded_columns"] = self.columns
        context["encoder_type"] = self.encoder_type


        logger.info("Data encoding completed successfully")
        return context
    
    def _get_encoder(self):
        """Initialize the appropriate encoder based on encoder_type."""

        if self.encoder_type == "onehot":
            # OneHotEncoder with sparse output disabled for easier DataFrame handling
            return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        elif self.encoder_type == "label":
            # LabelEncoder does not require fitting on the entire DataFrame
            return LabelEncoder()
        else:
            raise ValueError(f"Unsupported encoding type: {self.encoder_type}")
        
        
    
        