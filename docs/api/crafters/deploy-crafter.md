# DeployCrafter

The **DeployCrafter** handles model deployment by saving trained models and associated artifacts to disk. It supports joblib and pickle serialization formats.

## Overview

```python
from mlfcrafter import DeployCrafter

crafter = DeployCrafter(
    model_path="my_model.joblib",
    save_format="joblib",
    include_scaler=True,
    include_metadata=True
)
```

## Parameters

### `model_path` *(Optional[str])*

**Default**: `None` (auto-generates timestamped filename)

Path where model should be saved. If None, auto-generates filename like `random_forest_20250122_143022.joblib`.

### `save_format` *(str)*

**Default**: `"joblib"`

**Options**:
- `"joblib"`: Use joblib (recommended for sklearn models)
- `"pickle"`: Use Python's pickle module

### `include_scaler` *(bool)*

**Default**: `True`

Whether to include fitted scaler in saved artifacts.

### `include_metadata` *(bool)*

**Default**: `True`

Whether to include training metadata (model name, features, scores, etc.).

## Context Input

- `model`: Trained model (required)
- `scaler`: Fitted scaler (optional)
- `model_name`: Name of algorithm used (optional)
- `features`: Feature column names (optional)
- `train_score`, `test_score`: Performance scores (optional)
- `target_column`: Target variable name (optional)

## Context Output

- `deployment_path`: Absolute path where model was saved
- `artifacts_saved`: List of artifact keys that were saved
- `deployment_successful`: Whether deployment completed successfully

## Example Usage

```python
from mlfcrafter import MLFChain
from mlfcrafter.crafters import *

# Basic usage - auto-generate filename
pipeline = MLFChain()
pipeline.add_crafter(DataIngestCrafter("data.csv"))
pipeline.add_crafter(CleanerCrafter())
pipeline.add_crafter(ModelCrafter())
pipeline.add_crafter(DeployCrafter())
result = pipeline.run()

# Custom path and format
pipeline = MLFChain()
pipeline.add_crafter(DataIngestCrafter("data.csv"))
pipeline.add_crafter(ModelCrafter())
pipeline.add_crafter(DeployCrafter(
    model_path="production_model.pkl",
    save_format="pickle"
))
result = pipeline.run()
```

## Static Methods

### `load_model(model_path, load_format="auto")`

Loads saved model and artifacts from file.

```python
artifacts = DeployCrafter.load_model("my_model.joblib")
model = artifacts["model"]
scaler = artifacts.get("scaler")
metadata = artifacts.get("metadata")
``` 
