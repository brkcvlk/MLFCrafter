# ScorerCrafter

The **ScorerCrafter** calculates performance metrics for trained models. It automatically handles both binary and multi-class classification scenarios.

## Overview

```python
from flowcraft import ScorerCrafter

crafter = ScorerCrafter(
    metrics=["accuracy", "precision", "recall", "f1"]
)
```

## Parameters

### `metrics` *(Optional[List[str]])*

**Default**: `["accuracy", "precision", "recall", "f1"]` (all metrics)

List of metrics to calculate:
- `"accuracy"`: Overall accuracy (correct predictions / total predictions)
- `"precision"`: Positive predictive value (TP / (TP + FP))
- `"recall"`: Sensitivity or true positive rate (TP / (TP + FN))
- `"f1"`: Harmonic mean of precision and recall

## Context Input

- `y_test`: True labels from test set (required)
- `y_pred`: Predicted labels from model (required)

## Context Output

- `scores`: Dictionary containing calculated metrics (keys: metric names, values: calculated scores)

## Example Usage

```python
from flowcraft import FlowChain
from flowcraft.crafters import *

# Calculate all metrics
pipeline = FlowChain()
pipeline.add_crafter(DataIngestCrafter("data.csv"))
pipeline.add_crafter(ModelCrafter())
pipeline.add_crafter(ScorerCrafter())
result = pipeline.run()

# Calculate specific metrics only
pipeline = FlowChain()
pipeline.add_crafter(DataIngestCrafter("data.csv"))
pipeline.add_crafter(ModelCrafter())
pipeline.add_crafter(ScorerCrafter(metrics=["accuracy", "f1"]))
result = pipeline.run()

# Just accuracy
pipeline = FlowChain()
pipeline.add_crafter(DataIngestCrafter("data.csv"))
pipeline.add_crafter(ModelCrafter())
pipeline.add_crafter(ScorerCrafter(metrics=["accuracy"]))
result = pipeline.run()
```

## Notes

- Automatically detects binary vs multi-class classification
- Uses "binary" averaging for binary classification
- Uses "macro" averaging for multi-class classification
- Handles edge cases and zero division warnings 