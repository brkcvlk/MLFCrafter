# FlowChain API Reference

The `FlowChain` class is the core orchestrator for building and executing ML pipelines in FlowCraft.

## Class Definition

```python
class FlowChain:
    """
    Initialize FlowChain with multiple crafters
    Usage: FlowChain(DataIngestCrafter(...), CleanerCrafter(...), ...)
    """
```

## Constructor

### `__init__(self, *crafters)`

Creates a new FlowChain instance with the specified crafters.

**Parameters:**

- `*crafters`: Variable number of crafter instances to add to the pipeline

**Example:**

```python
from flowcraft import FlowChain
from flowcraft.crafters import DataIngestCrafter, CleanerCrafter, ModelCrafter

# Initialize with crafters
pipeline = FlowChain(
    DataIngestCrafter(data_path="data.csv"),
    CleanerCrafter(strategy="auto"),
    ModelCrafter(model_name="random_forest")
)

# Or initialize empty and add crafters later
pipeline = FlowChain()
```

## Methods

### `add_crafter(crafter)`

Adds a single crafter to the pipeline chain.

**Parameters:**

- `crafter`: The crafter instance to add to the pipeline

**Example:**

```python
from flowcraft.crafters import DataIngestCrafter, CleanerCrafter

pipeline = FlowChain()
pipeline.add_crafter(DataIngestCrafter(data_path="data.csv"))
pipeline.add_crafter(CleanerCrafter(strategy="auto"))
```

### `run(target_column=None, **kwargs)`

Runs the entire pipeline chain.

**Parameters:**

- `target_column` (str, optional): Target column name for ML tasks
- `**kwargs`: Additional parameters to pass to the first crafter

**Returns:** `dict` - Context dictionary containing all pipeline results

**Raises:**

- `RuntimeError`: If any crafter in the pipeline fails
- `TypeError`: If a crafter doesn't return a dict (context)
- `ValueError`: If required data or parameters are missing

**Example:**

```python
# Run pipeline with target column
results = pipeline.run(target_column="target")

# Access results
print(f"Model accuracy: {results['test_score']:.4f}")
print(f"Dataset shape: {results['original_shape']}")
```

## Properties

### `crafters`

List of all crafters in the pipeline.

**Type:** `list`

**Example:**

```python
print(f"Pipeline has {len(pipeline.crafters)} crafters")
for i, crafter in enumerate(pipeline.crafters, 1):
    print(f"Crafter {i}: {type(crafter).__name__}")
```

## Context Dictionary

The `run()` method returns a context dictionary containing all pipeline results. Key elements include:

### Data Processing Results
- `data` (pd.DataFrame): Current state of the data
- `original_shape` (tuple): Shape of original data (rows, columns)
- `cleaned_shape` (tuple): Shape after cleaning (if cleaning was performed)
- `missing_values_handled` (bool): Whether missing values were processed

### Model Training Results
- `model`: Trained model object
- `X_train, X_test` (pd.DataFrame): Feature splits
- `y_train, y_test` (pd.Series): Target splits
- `y_pred` (np.array): Predictions on test set
- `train_score` (float): Training accuracy
- `test_score` (float): Test accuracy
- `model_name` (str): Name of algorithm used
- `features` (list): List of feature column names

### Scaling Results
- `scaler`: Fitted scaler object for future use
- `scaled_columns` (list): Names of columns that were scaled
- `scaler_type` (str): Type of scaler used

### Scoring Results
- `scores` (dict): Dictionary containing calculated metrics

### Deployment Results
- `deployment_path` (str): Path where model was saved
- `artifacts_saved` (list): List of artifact keys that were saved
- `deployment_successful` (bool): Whether deployment completed successfully

**Example:**

```python
results = pipeline.run(target_column="target")

# Access various results
print(f"Original data shape: {results['original_shape']}")
print(f"Model used: {results['model_name']}")
print(f"Test accuracy: {results['test_score']:.4f}")
print(f"Model saved to: {results.get('deployment_path', 'Not saved')}")
```

## Error Handling

FlowChain handles errors that occur during pipeline execution:

**Common Exceptions:**

- `RuntimeError`: Raised when a crafter fails during execution
- `TypeError`: Raised when a crafter doesn't return a dict (context)
- `ValueError`: Raised when required data or parameters are missing

**Example:**

```python
try:
    results = pipeline.run(target_column="target")
    print("Pipeline completed successfully!")
except RuntimeError as e:
    print(f"Pipeline failed: {e}")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Examples

### Basic Pipeline

```python
from flowcraft import FlowChain
from flowcraft.crafters import *

# Create pipeline with all crafters
pipeline = FlowChain(
    DataIngestCrafter(data_path="data.csv"),
    CleanerCrafter(strategy="auto"),
    ScalerCrafter(scaler_type="standard"),
    ModelCrafter(model_name="random_forest"),
    ScorerCrafter(),
    DeployCrafter()
)

# Run pipeline
results = pipeline.run(target_column="target")
print(f"Model accuracy: {results['test_score']:.4f}")
```

### Adding Crafters Individually

```python
# Create empty pipeline
pipeline = FlowChain()

# Add crafters one by one
pipeline.add_crafter(DataIngestCrafter(data_path="data.csv"))
pipeline.add_crafter(CleanerCrafter(strategy="median"))
pipeline.add_crafter(ModelCrafter(model_name="xgboost"))

# Run pipeline
results = pipeline.run(target_column="target")
```

### Error Handling

```python
try:
    results = pipeline.run(target_column="target")
    print(f"Success! Model accuracy: {results['test_score']:.4f}")
    print(f"Model saved to: {results.get('deployment_path', 'Not saved')}")
except RuntimeError as e:
    print(f"Pipeline failed: {e}")
except ValueError as e:
    print(f"Configuration error: {e}")
``` 