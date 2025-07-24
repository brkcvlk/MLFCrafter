# DataIngestCrafter

The **DataIngestCrafter** is responsible for loading data from various file formats into your mlfcrafter pipeline. It serves as the entry point for your data, supporting multiple formats with automatic detection capabilities.

## Overview

```python
from mlfcrafter import DataIngestCrafter

crafter = DataIngestCrafter(
    data_path="data/sales.csv",
    source_type="auto"
)
```

## Purpose & Function

DataIngestCrafter handles the first critical step of any ML pipeline by:

- ✅ Loading data from **multiple file formats**
- 🔍 **Auto-detecting** file format from extension
- 🛡️ **Validating** file format consistency
- 📊 Adding **metadata** about the loaded dataset

## Parameters

### `data_path` *(Optional[str])*

**Purpose**: Path to the data file to be loaded.

**Options**:
- **File path string**: `"data/my_file.csv"`
- **`None`**: Path will be provided via context

**Examples**:
```python
# Direct file path
DataIngestCrafter(data_path="data/sales.csv")

# Path from context
DataIngestCrafter()  # data_path provided in context
```

!!! warning "Required Either Here or Context"
    If `data_path` is `None`, it must be provided in the pipeline context.

---

### `source_type` *(str)*

**Purpose**: Specifies the file format type for reading the data.

**Default**: `"auto"`

**Available Options**:

| Option | Description | Use Case |
|--------|-------------|----------|
| `"auto"` | **Auto-detect** format from file extension | Most common - let mlfcrafter decide |
| `"csv"` | Force **CSV** reading | When extension doesn't match or manual control |
| `"excel"` | Force **Excel** reading | For .xls/.xlsx files |  
| `"json"` | Force **JSON** reading | For structured JSON data |

**Examples**:
```python
# Auto-detection (recommended)
DataIngestCrafter(data_path="data.csv", source_type="auto")

# Force specific format
DataIngestCrafter(data_path="data.txt", source_type="csv")  # CSV in .txt file
DataIngestCrafter(data_path="data.xlsx", source_type="excel")
DataIngestCrafter(data_path="data.json", source_type="json")
```

---

## Context Flow

### Context Input
The crafter accepts these optional context keys:

```python
context = {
    "data_path": "alternative/path/to/data.csv"  # Alternative to constructor parameter
}
```

### Context Output
Returns updated context with:

```python
context = {
    "data": pd.DataFrame,           # The loaded dataset
    "original_shape": (rows, cols), # Shape tuple (1000, 15)
    # ... other existing context keys preserved
}
```

---

## Supported File Formats

### CSV Files (.csv)
```python
# Auto-detected
DataIngestCrafter(data_path="sales_data.csv")

# Explicit
DataIngestCrafter(data_path="sales_data.csv", source_type="csv")
```

**Supported Features**:
- Headers automatically detected
- Various separators (comma, semicolon)
- Encoding handling

### Excel Files (.xls, .xlsx)
```python
DataIngestCrafter(data_path="financial_data.xlsx", source_type="excel")
```

**Supported Features**:
- First sheet loaded by default
- Headers from first row
- Date parsing

### JSON Files (.json)
```python  
DataIngestCrafter(data_path="api_response.json", source_type="json")
```

**Supported Features**:
- Structured JSON to DataFrame
- Nested object flattening
- Array handling

---

## Usage Examples

### Basic Usage
```python
from mlfcrafter import MLFChain, DataIngestCrafter

# Simple CSV loading
chain = MLFChain(
    DataIngestCrafter(data_path="data/customers.csv")
)

result = chain.run()
print(f"Loaded {result['original_shape'][0]} rows")
```

### Multiple Format Support
```python
# Auto-detect different formats
csv_chain = MLFChain(DataIngestCrafter(data_path="data.csv"))
excel_chain = MLFChain(DataIngestCrafter(data_path="data.xlsx"))
json_chain = MLFChain(DataIngestCrafter(data_path="data.json"))
```

### Force Format Override
```python
# CSV data in .txt file
chain = MLFChain(
    DataIngestCrafter(
        data_path="exported_data.txt",
        source_type="csv"  # Force CSV parsing
    )
)
```

### Path from Context
```python
chain = MLFChain(DataIngestCrafter())  # No path specified

# Path provided at runtime
result = chain.run(data_path="runtime_data.csv")
```

---

## Complete Pipeline Example

```python
from mlfcrafter import MLFChain, DataIngestCrafter, CleanerCrafter, ModelCrafter

# Real-world pipeline
chain = MLFChain(
    DataIngestCrafter(
        data_path="data/customer_churn.csv",
        source_type="auto"  # Let mlfcrafter auto-detect
    ),
    CleanerCrafter(strategy="auto"),
    ModelCrafter(model_name="random_forest")
)

# Run pipeline
results = chain.run(target_column="churn")

print(f"📊 Dataset: {results['original_shape']}")
print(f"🎯 Accuracy: {results['test_score']:.4f}")
```

---

## Error Handling

### File Not Found
```python
# ❌ This will raise FileNotFoundError
DataIngestCrafter(data_path="nonexistent.csv")
```

### Format Mismatch
```python
# ❌ This will raise ValueError - Excel file with CSV source_type
DataIngestCrafter(
    data_path="data.xlsx", 
    source_type="csv"  # Mismatch!
)
```

### Missing Path
```python
# ❌ This will raise ValueError
crafter = DataIngestCrafter()  # No path
crafter.run({})  # No path in context either
```

---

## Best Practices

!!! tip "Use Auto-Detection"
    Use `source_type="auto"` for most cases - mlfcrafter handles format detection reliably.

!!! success "Validate Your Data"
    Always check `original_shape` in results to confirm data loaded correctly.

!!! warning "Large Files"
    For very large files (>1GB), consider chunked processing or data sampling.

!!! info "File Paths"
    Use absolute paths or ensure your working directory is correct for relative paths.

---

## Advanced Usage

### Conditional File Loading
```python
import os

def create_data_pipeline(file_path):
    """Create pipeline based on available data."""
    if not os.path.exists(file_path):
        file_path = "backup_data.csv"
    
    return MLFChain(
        DataIngestCrafter(data_path=file_path),
        # ... other crafters
    )
```

### Format-Specific Optimization
```python
# Optimize for known Excel files
excel_crafter = DataIngestCrafter(
    data_path="quarterly_reports.xlsx",
    source_type="excel"  # Skip auto-detection
)
```

---

## Integration with Other Crafters

DataIngestCrafter is typically the **first crafter** in any pipeline:

```python
# Typical pipeline flow
MLFChain(
    DataIngestCrafter(data_path="raw_data.csv"),      # 1️⃣ Load data
    CleanerCrafter(strategy="mean"),                   # 2️⃣ Clean data  
    ScalerCrafter(scaler_type="standard"),            # 3️⃣ Scale features
    ModelCrafter(model_name="xgboost"),               # 4️⃣ Train model
    ScorerCrafter(),                                   # 5️⃣ Evaluate
    DeployCrafter(model_path="model.joblib")          # 6️⃣ Save model
)
```

---

## Troubleshooting

**Q: "File extension and source type don't match"**  
A: Either use `source_type="auto"` or ensure the source_type matches your file format.

**Q: "Empty DataFrame loaded"**  
A: Check your file format, headers, and ensure the file contains data.

**Q: "Encoding issues with CSV"**  
A: mlfcrafter uses UTF-8 by default. Ensure your CSV file is properly encoded.

**Q: "Excel file not loading"**  
A: Ensure you have the required dependencies installed: `pip install openpyxl`. 
