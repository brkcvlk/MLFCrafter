# CleanerCrafter

The **CleanerCrafter** handles missing values in datasets using multiple strategies.

## Overview

```python
from flowcraft import CleanerCrafter

crafter = CleanerCrafter(
    strategy="auto",
    str_fill="missing",
    int_fill=0.0
)
```

## Parameters

### `strategy` *(str)*

**Default**: `"auto"`

**Available Strategies**:
- `"auto"`: Automatically choose strategy based on data type (numerical: uses int_fill, categorical: uses str_fill)
- `"mean"`: Fill numerical columns with column mean (categorical columns unchanged)
- `"median"`: Fill numerical columns with column median (categorical columns unchanged)
- `"mode"`: Fill all columns with most frequent value
- `"drop"`: Drop rows containing any missing values
- `"constant"`: Fill with constant values (str_fill for strings, int_fill for numbers)

### `str_fill` *(str)*

**Default**: `"missing"`

Fill value for categorical/string columns.

### `int_fill` *(float)*

**Default**: `0.0`

Fill value for numerical columns.

## Context Input

- `data`: Dataset to clean (required)

## Context Output

- `data`: Cleaned dataset
- `cleaned_shape`: Shape after cleaning
- `missing_values_handled`: Flag indicating cleaning was performed

## Example Usage

```python
from flowcraft import FlowChain
from flowcraft.crafters import *

# Automatic cleaning
pipeline = FlowChain()
pipeline.add_crafter(DataIngestCrafter("data.csv"))
pipeline.add_crafter(CleanerCrafter(strategy="auto"))
result = pipeline.run()

# Mean imputation for numerical columns
pipeline = FlowChain()
pipeline.add_crafter(DataIngestCrafter("data.csv"))
pipeline.add_crafter(CleanerCrafter(strategy="mean"))
result = pipeline.run()

# Custom fill values
pipeline = FlowChain()
pipeline.add_crafter(DataIngestCrafter("data.csv"))
pipeline.add_crafter(CleanerCrafter(
    strategy="constant",
    str_fill="Unknown",
    int_fill=-1
))
result = pipeline.run()
``` 