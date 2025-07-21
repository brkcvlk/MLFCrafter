from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler
import numpy as np
import pandas as pd

class ScalerCrafter:
    def __init__(self, strategy=None, columns=None):
        self.strategy = strategy
        self.columns = columns
        self.scaler = None

        if self.strategy == "minmax":
            self.scaler = MinMaxScaler()
        elif self.strategy == "standard":
            self.scaler = StandardScaler()
        elif self.strategy == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaling strategy: {self.strategy}")

    def run(self, df: pd.DataFrame):
        if self.strategy is None:
            return df  

        df_scaled = df.copy()
        if self.columns is None:
            self.columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()

        df_scaled[self.columns] = self.scaler.fit_transform(df_scaled[self.columns])
        return df_scaled
