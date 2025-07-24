# ScalerCrafter

The **ScalerCrafter** normalizes numerical features using various scaling techniques.

## Overview

```python
from mlfcrafter import ScalerCrafter

crafter = ScalerCrafter(
    scaler_type="minmax",
    columns=None  # Auto-select numerical columns
)
```

## Parameters

### `scaler_type` *(str)*

**Default**: `"minmax"`

**Available Types**:
- `"minmax"`: MinMaxScaler - scales features to [0,1] range
- `"standard"`: StandardScaler - standardizes features (mean=0, std=1) 
- `"robust"`: RobustScaler - uses median and IQR, robust to outliers

### `columns` *(Optional[List[str]])*

**Default**: `None` (auto-select all numerical columns)

Specific columns to scale. If None, automatically selects all numerical columns except target column.

## Context Input

- `data`: Dataset to scale (required)
- `target_column`: Target column to exclude from scaling (optional)

## Context Output

- `data`: Dataset with scaled numerical features
- `scaler`: Fitted scaler object for future use
- `scaled_columns`: Names of columns that were scaled
- `scaler_type`: Type of scaler used

## Example Usage

```python
from mlfcrafter import MLFChain
from mlfcrafter.crafters import *

# Scale all numerical columns with MinMax
pipeline = MLFChain()
pipeline.add_crafter(DataIngestCrafter("data.csv"))
pipeline.add_crafter(ScalerCrafter(scaler_type="minmax"))
pipeline.add_crafter(ModelCrafter())
result = pipeline.run()

# Standard scaling for specific columns
pipeline = MLFChain()
pipeline.add_crafter(DataIngestCrafter("data.csv"))
pipeline.add_crafter(ScalerCrafter(
    scaler_type="standard",
    columns=["feature1", "feature2"]
))
result = pipeline.run()

# Robust scaling (good for data with outliers)
pipeline = MLFChain()
pipeline.add_crafter(DataIngestCrafter("data.csv"))
pipeline.add_crafter(ScalerCrafter(scaler_type="robust"))
result = pipeline.run()
``` 
