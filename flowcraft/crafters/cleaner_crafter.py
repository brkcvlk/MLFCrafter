import pandas as pd
import numpy as np

class CleanerCrafter:
    def __init__(
        self,
        strategy: str = "auto",
        str_fill: str = "missing",
        int_fill: float = 0.0
    ):
        self.strategy = strategy
        self.str_fill = str_fill
        self.int_fill = int_fill

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()

        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() == 0:
                continue
            else: 
                if self.strategy == "auto":
                    if df_cleaned[col].dtype == "object":
                        df_cleaned[col] = df_cleaned[col].fillna(self.str_fill)
                    else:
                        df_cleaned[col] = df_cleaned[col].fillna(self.int_fill)

                elif self.strategy == "mean":
                    if np.issubdtype(df_cleaned[col].dtype, np.number):
                        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())

                elif self.strategy == "median":
                    if np.issubdtype(df_cleaned[col].dtype, np.number):
                        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

                elif self.strategy == "mode":
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode().iloc[0])

                elif self.strategy == "drop":
                    df_cleaned = df_cleaned.dropna()

                elif self.strategy == "constant":
                    if df_cleaned[col].dtype == "object":
                        df_cleaned[col] = df_cleaned[col].fillna(self.str_fill)
                    else:
                        df_cleaned[col] = df_cleaned[col].fillna(self.int_fill)

                else:
                    raise ValueError(f"Unsupported cleaning strategy: {self.strategy}")

        return df_cleaned
