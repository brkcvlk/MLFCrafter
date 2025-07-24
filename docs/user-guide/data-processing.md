# Data Processing

Learn how to effectively clean, transform, and prepare your data using MLFCrafter's data processing crafters.

## Data Ingestion

The `DataIngestCrafter` loads data from various file formats into your pipeline.

### Basic Usage

```python
from mlfcrafter import MLFChain, DataIngestCrafter

# Load CSV file
pipeline = MLFChain(
    DataIngestCrafter(data_path="data.csv")
)
result = pipeline.run()

# Load Excel file
pipeline = MLFChain(
    DataIngestCrafter(data_path="data.xlsx", source_type="excel")
)
result = pipeline.run()

# Auto-detect format
pipeline = MLFChain(
    DataIngestCrafter(data_path="data.json")
)
result = pipeline.run()
```

### Supported Formats

- **CSV files** (.csv) - Auto-detected
- **Excel files** (.xls, .xlsx) - Specify `source_type="excel"`
- **JSON files** (.json) - Specify `source_type="json"`

### Parameters

- `data_path`: Path to the data file
- `source_type`: Format type ("auto", "csv", "excel", "json")

## Data Cleaning

The `CleanerCrafter` handles missing values in your dataset.

### Missing Value Strategies

```python
from mlfcrafter import CleanerCrafter

# Auto strategy - automatically choose based on data type
cleaner = CleanerCrafter(strategy="auto")

# Statistical imputation
cleaner = CleanerCrafter(strategy="mean")      # Fill with column mean
cleaner = CleanerCrafter(strategy="median")    # Fill with column median
cleaner = CleanerCrafter(strategy="mode")      # Fill with most frequent value

# Drop rows with missing values
cleaner = CleanerCrafter(strategy="drop")

# Constant fill values
cleaner = CleanerCrafter(
    strategy="constant",
    str_fill="Unknown",  # For categorical columns
    int_fill=-1          # For numerical columns
)
```

### Complete Example

```python
# Clean dataset with auto strategy
pipeline = MLFChain(
    DataIngestCrafter("messy_data.csv"),
    CleanerCrafter(strategy="auto")
)
result = pipeline.run()

print(f"Cleaned data shape: {result['cleaned_shape']}")
print(f"Missing values handled: {result['missing_values_handled']}")
```

## Feature Scaling

The `ScalerCrafter` normalizes numerical features.

### Scaling Methods

```python
from mlfcrafter import ScalerCrafter

# MinMax scaling - scales to [0,1] range
scaler = ScalerCrafter(scaler_type="minmax")

# Standard scaling - mean=0, std=1
scaler = ScalerCrafter(scaler_type="standard")

# Robust scaling - uses median and IQR, good for outliers
scaler = ScalerCrafter(scaler_type="robust")
```

### Column Selection

```python
# Scale all numerical columns (default)
scaler = ScalerCrafter(scaler_type="standard")

# Scale specific columns only
scaler = ScalerCrafter(
    scaler_type="minmax",
    columns=["age", "income", "score"]
)
```

### Complete Example

```python
# Full data processing pipeline
pipeline = MLFChain(
    DataIngestCrafter("customer_data.csv"),
    CleanerCrafter(strategy="auto"),
    ScalerCrafter(scaler_type="standard")
)
result = pipeline.run()

print(f"Scaled columns: {result['scaled_columns']}")
print(f"Scaler type: {result['scaler_type']}")
```

## Processing Pipeline

Combine all data processing steps:

```python
from mlfcrafter import MLFChain, DataIngestCrafter, CleanerCrafter, ScalerCrafter, ModelCrafter

# Complete preprocessing pipeline
pipeline = MLFChain(
    # 1. Load data
    DataIngestCrafter(
        data_path="raw_data.csv",
        source_type="auto"
    ),
    
    # 2. Clean missing values
    CleanerCrafter(
        strategy="auto",
        str_fill="Unknown",
        int_fill=0
    ),
    
    # 3. Scale numerical features
    ScalerCrafter(
        scaler_type="robust"  # Good for real-world data with outliers
    ),
    
    # 4. Train model
    ModelCrafter(
        model_name="random_forest",
        test_size=0.2
    )
)

# Run the complete pipeline
result = pipeline.run()

print(f"Original shape: {result['original_shape']}")
print(f"Cleaned shape: {result['cleaned_shape']}")
print(f"Test accuracy: {result['test_score']:.4f}")
```

## Best Practices

### Data Quality Checks

Always check your data after processing:

```python
result = pipeline.run()

# Check data shape and quality
print(f"Dataset shape: {result['data'].shape}")
print(f"Missing values: {result['data'].isnull().sum().sum()}")
print(f"Data types:\n{result['data'].dtypes}")
```

### Strategy Selection

- **Auto strategy**: Good starting point for mixed datasets
- **Mean/Median**: For numerical data (median better with outliers)
- **Mode**: For categorical data
- **Drop**: When you have plenty of data and few missing values
- **Robust scaling**: When data has outliers
- **Standard scaling**: For normally distributed data
- **MinMax scaling**: When you need bounded [0,1] output

### Pipeline Order

Always process data in this order:
1. **Data Ingestion** - Load your data
2. **Data Cleaning** - Handle missing values first
3. **Feature Scaling** - Scale clean data
4. **Model Training** - Train on processed data 
