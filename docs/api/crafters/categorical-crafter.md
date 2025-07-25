# CategoricalCrafter

The **CategoricalCrafter** converts categorical variables into numerical data.

## Overview

```python
from mlfcrafter import CategoricalCrafter

crafter = CategoricalCrafter(
    encoder_type="minmax",
    columns=None  # Auto-select categorical columns
)
```

## Parameters

### `encoder_type` *(str)*

**Default**: `"onehot"`

**Available Types**:
- `"onehot"`: One-Hot Encoding - Converts each categorical value into a new binary column (0 or 1) representing the presence of that category.
- `"standard"`: Label Encoding - Assigns each unique categorical value an integer label, converting categories into numeric codes.

### `columns` *(Optional[List[str]])*

**Default**: `None` (auto-select all categorical columns)

Specific columns to encode . If None, automatically selects all categoricals columns.

## Context Input

- `data`: Dataset to scale (required)
- `target_column`: Target column to exclude from scaling (optional)

## Context Output

- `data`: Dataset with encoded categorical features
- `encoder`: Fitted encoder object for future use
- `encoded_columns`: Names of columns that were encoded
- `encoder_type`: Type of encoder used

## Example Usage

```python
from mlfcrafter import MLFChain
from mlfcrafter.crafters import *

# Encode all categorical columns with One-Hot Encoding
pipeline = MLFChain()
pipeline.add_crafter(DataIngestCrafter("data.csv"))
pipeline.add_crafter(CategoricalCrafter(encoder_type="onehot"))
pipeline.add_crafter(ModelCrafter())
result = pipeline.run()

# Standard encoding for specific columns
pipeline = MLFChain()
pipeline.add_crafter(DataIngestCrafter("data.csv"))
pipeline.add_crafter(CategoricalCrafter(
    encoder_type="onehot",
    columns=["feature1", "feature2"]
))
result = pipeline.run()
