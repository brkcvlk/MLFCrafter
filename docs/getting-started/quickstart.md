# Quick Start

Get up and running with MLFCrafter in just a few minutes! This guide will walk you through creating your first ML pipeline.

## Basic Example

Here's a simple example that demonstrates MLFCrafter's core functionality:

```python
import pandas as pd
from mlfcrafter import MLFChain
from mlfcrafter.crafters import (
    DataIngestCrafter, 
    CleanerCrafter, 
    ScalerCrafter, 
    ModelCrafter,
    ScorerCrafter
)
import tempfile

# Create sample data and save to CSV
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, None],
    'feature2': [10, 20, 30, 40, 50, 60],
    'target': [0, 1, 0, 1, 0, 1]
})
temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
data.to_csv(temp_file.name, index=False)
temp_file.close()

# Create pipeline with crafters
pipeline = MLFChain(
    DataIngestCrafter(data_path=temp_file.name),
    CleanerCrafter(strategy='drop'),
    ScalerCrafter(scaler_type='standard'),
    ModelCrafter(model_name='random_forest'),
    ScorerCrafter()
)

# Run the pipeline
results = pipeline.run(target_column='target')

# Print results
print(f"Test Accuracy: {results['test_score']:.4f}")
print(f"Model: {results['model_name']}")
```

## Step-by-Step Breakdown

### 1. Import Required Components

```python
from mlfcrafter import MLFChain, DataIngestCrafter, CleanerCrafter, ModelCrafter
```

### 2. Create a MLFChain

The `MLFChain` is the main orchestrator that manages your pipeline. You can initialize it with crafters or add them later:

```python
# Option 1: Initialize with crafters
pipeline = MLFChain(
    DataIngestCrafter(data_path="data.csv"),
    CleanerCrafter(strategy='auto'),
    ModelCrafter(model_name='random_forest')
)

# Option 2: Initialize empty and add crafters
pipeline = MLFChain()
pipeline.add_crafter(DataIngestCrafter(data_path="data.csv"))
pipeline.add_crafter(CleanerCrafter(strategy='auto'))
```

### 3. Run the Pipeline

Execute the entire pipeline with target column specified:

```python
results = pipeline.run(target_column='target')
```

## Available Crafters

MLFCrafter provides the following built-in crafters:

- **DataIngestCrafter**: Load data from CSV, Excel, or JSON files
- **CleanerCrafter**: Handle missing values with various strategies
- **ScalerCrafter**: Normalize features using standard, minmax, or robust scaling
- **ModelCrafter**: Train models (Random Forest, XGBoost, Logistic Regression)
- **ScorerCrafter**: Evaluate model performance with multiple metrics
- **DeployCrafter**: Save trained models and artifacts

## Configuration Options

Each crafter accepts various configuration parameters:

```python
# Data ingestion from different file types
ingester = DataIngestCrafter(
    data_path="data.xlsx", 
    source_type="excel"  # or "csv", "json", "auto"
)

# Cleaner with specific strategy
cleaner = CleanerCrafter(
    strategy='mean',  # "auto", "mean", "median", "mode", "drop", "constant"
    str_fill='unknown',
    int_fill=-1
)

# Scaler with different methods
scaler = ScalerCrafter(
    scaler_type='robust',  # "minmax", "standard", "robust"
    columns=['feature1', 'feature2']  # specific columns or None for all
)

# Model with custom parameters
model = ModelCrafter(
    model_name='xgboost',  # "random_forest", "logistic_regression", "xgboost"
    model_params={
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    },
    test_size=0.3,
    random_state=42
)
```

## Complete Pipeline Example

Here's a full pipeline with all crafters:

```python
from mlfcrafter import MLFChain
from mlfcrafter.crafters import *

# Complete ML pipeline
pipeline = MLFChain(
    DataIngestCrafter(data_path="dataset.csv"),
    CleanerCrafter(strategy="median"),
    ScalerCrafter(scaler_type="standard"),
    ModelCrafter(model_name="xgboost", test_size=0.2),
    ScorerCrafter(metrics=["accuracy", "f1", "precision"]),
    DeployCrafter(model_path="trained_model.joblib")
)

# Run pipeline
results = pipeline.run(target_column="target")

# Access results
print(f"Model: {results['model_name']}")
print(f"Test Accuracy: {results['test_score']:.4f}")
print(f"F1 Score: {results['scores']['f1']:.4f}")
print(f"Model saved: {results['deployment_successful']}")
```

## What's Next?

- Learn about [Pipeline Basics](../user-guide/pipeline-basics.md)
- Explore [Data Processing](../user-guide/data-processing.md) techniques
- Check out more [Examples](../examples/basic-usage.md)
- Build [Your First Complete Pipeline](first-pipeline.md) 