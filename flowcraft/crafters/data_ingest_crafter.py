import pandas as pd
from pathlib import Path

class DataIngestCrafter:
    def __init__(self, data_path: str, source_type: str = "auto"):
        self.data_path = data_path
        self.source_type = source_type.lower()

    def run(self):
        file = Path(self.data_path)
        suffix = file.suffix.lower()

        if self.source_type == "auto":
            return self._read_auto(file, suffix)
        else:
            expected_ext = self._expected_extension(self.source_type)
            if suffix not in expected_ext:
                raise ValueError(f"File extension and source type dont match:\n"
                                 f" - File extension: {suffix}\n - Expected extension: {expected_ext}")
            return self._read_by_type(file, self.source_type)

    def _expected_extension(self, source_type: str):
        return {
            "csv": [".csv"],
            "excel": [".xls", ".xlsx"],
            "json": [".json"]
        }.get(source_type, [])

    def _read_by_type(self, file: Path, source_type: str):
        if source_type == "csv":
            return pd.read_csv(file)
        elif source_type == "excel":
            return pd.read_excel(file)
        elif source_type == "json":
            return pd.read_json(file)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    def _read_auto(self, file: Path, suffix: str):
        if suffix == ".csv":
            return pd.read_csv(file)
        elif suffix in [".xls", ".xlsx"]:
            return pd.read_excel(file)
        elif suffix == ".json":
            return pd.read_json(file)
        else:
            raise ValueError(f"Unsupported source type: {suffix}")
