import logging

import numpy as np

# Setup logger for this crafter
logger = logging.getLogger("flowcraft.CleanerCrafter")


class CleanerCrafter:
    """
    Data Cleaning Crafter for handling missing values in datasets.

    This crafter provides multiple strategies for handling missing values in both
    numerical and categorical columns. It supports automatic detection of data types
    and applies appropriate cleaning strategies accordingly.

    Parameters:
        strategy (str): Missing value handling strategy. Options:
            - "auto": Automatically choose strategy based on data type
                     * Numerical columns: filled with int_fill value
                     * Categorical columns: filled with str_fill value
            - "mean": Fill numerical columns with column mean (categorical unchanged)
            - "median": Fill numerical columns with column median (categorical unchanged)
            - "mode": Fill all columns with most frequent value
            - "drop": Drop rows containing any missing values
            - "constant": Fill with constant values (str_fill for strings, int_fill for numbers)

        str_fill (str): Fill value for categorical/string columns (default: "missing")
        int_fill (float): Fill value for numerical columns (default: 0.0)

    Context Input:
        - data (pd.DataFrame): Dataset to clean (required)

    Context Output:
        - data (pd.DataFrame): Cleaned dataset
        - cleaned_shape (tuple): Shape after cleaning
        - missing_values_handled (bool): Flag indicating cleaning was performed

    Example Usage:
        # Automatic cleaning
        cleaner = CleanerCrafter(strategy="auto")

        # Mean imputation for numerical columns
        cleaner = CleanerCrafter(strategy="mean")

        # Custom fill values
        cleaner = CleanerCrafter(
            strategy="constant",
            str_fill="Unknown",
            int_fill=-1
        )

    Workflow:
        1. Check for missing values in each column
        2. Apply selected strategy based on data type
        3. Update context with cleaned data and metadata
        4. Preserve original data types where possible
    """

    def __init__(
        self, strategy: str = "auto", str_fill: str = "missing", int_fill: float = 0.0
    ):
        self.strategy = strategy
        self.str_fill = str_fill
        self.int_fill = int_fill

    def run(self, context: dict) -> dict:
        """
        Clean data in context
        Args:
            context: Pipeline context dict with 'data' key
        Returns:
            Updated context with cleaned data
        """
        logger.info("Starting data cleaning...")
        logger.debug(f"Cleaning strategy: {self.strategy}")

        if "data" not in context:
            logger.error("No data found in context")
            raise ValueError("No data found in context. Run DataIngestCrafter first.")

        df = context["data"]
        logger.info(f"Input data shape: {df.shape}")

        # Check for missing values
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        logger.info(f"Total missing values found: {total_missing}")

        if total_missing == 0:
            logger.info("No missing values found, skipping cleaning")
            context["missing_values_handled"] = False
            return context

        # Log missing values per column
        for col, count in missing_counts[missing_counts > 0].items():
            logger.debug(f"Column '{col}': {count} missing values")

        df_cleaned = df.copy()

        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() == 0:
                continue
            else:
                if self.strategy == "auto":
                    if df_cleaned[col].dtype == "object":
                        logger.debug(
                            f"Auto-filling categorical column '{col}' with '{self.str_fill}'"
                        )
                        df_cleaned[col] = df_cleaned[col].fillna(self.str_fill)
                    else:
                        logger.debug(
                            f"Auto-filling numerical column '{col}' with {self.int_fill}"
                        )
                        df_cleaned[col] = df_cleaned[col].fillna(self.int_fill)

                elif self.strategy == "mean":
                    if np.issubdtype(df_cleaned[col].dtype, np.number):
                        mean_val = df_cleaned[col].mean()
                        logger.debug(
                            f"Filling column '{col}' with mean value: {mean_val:.4f}"
                        )
                        df_cleaned[col] = df_cleaned[col].fillna(mean_val)

                elif self.strategy == "median":
                    if np.issubdtype(df_cleaned[col].dtype, np.number):
                        median_val = df_cleaned[col].median()
                        logger.debug(
                            f"Filling column '{col}' with median value: {median_val:.4f}"
                        )
                        df_cleaned[col] = df_cleaned[col].fillna(median_val)

                elif self.strategy == "mode":
                    mode_val = df_cleaned[col].mode().iloc[0]
                    logger.debug(f"Filling column '{col}' with mode value: {mode_val}")
                    df_cleaned[col] = df_cleaned[col].fillna(mode_val)

                elif self.strategy == "drop":
                    original_rows = len(df_cleaned)
                    df_cleaned = df_cleaned.dropna()
                    dropped_rows = original_rows - len(df_cleaned)
                    logger.debug(
                        f"Dropped {dropped_rows} rows containing missing values"
                    )

                elif self.strategy == "constant":
                    if df_cleaned[col].dtype == "object":
                        logger.debug(
                            f"Filling categorical column '{col}' with constant '{self.str_fill}'"
                        )
                        df_cleaned[col] = df_cleaned[col].fillna(self.str_fill)
                    else:
                        logger.debug(
                            f"Filling numerical column '{col}' with constant {self.int_fill}"
                        )
                        df_cleaned[col] = df_cleaned[col].fillna(self.int_fill)

                else:
                    logger.error(f"Unsupported cleaning strategy: {self.strategy}")
                    raise ValueError(f"Unsupported cleaning strategy: {self.strategy}")

        # Update context
        context["data"] = df_cleaned
        context["cleaned_shape"] = df_cleaned.shape
        context["missing_values_handled"] = True

        # Final verification
        remaining_missing = df_cleaned.isnull().sum().sum()
        logger.info(
            f"Cleaning completed. Remaining missing values: {remaining_missing}"
        )
        logger.info(f"Final data shape: {df_cleaned.shape}")

        return context
