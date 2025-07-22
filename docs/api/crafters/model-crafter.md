# ModelCrafter

The **ModelCrafter** handles model selection, training, and evaluation. It supports multiple algorithms and provides train/test splitting.

## Overview

```python
from flowcraft import ModelCrafter

crafter = ModelCrafter(
    model_name="random_forest",
    model_params={"n_estimators": 100},
    test_size=0.2,
    random_state=61,
    stratify=True
)
```

## Parameters

### `model_name` *(str)*

**Default**: `"random_forest"`

**Available Models**:
- `"random_forest"`: RandomForestClassifier
- `"logistic_regression"`: LogisticRegression  
- `"xgboost"`: XGBClassifier

### `model_params` *(Optional[Dict])*

**Default**: `{}` (use sklearn defaults)

Hyperparameters for the selected model:

```python
# Random Forest examples
{"n_estimators": 100, "max_depth": 10}

# XGBoost examples  
{"learning_rate": 0.1, "max_depth": 6}

# Logistic Regression examples
{"C": 1.0, "max_iter": 1000}
```

### `test_size` *(float)*

**Default**: `0.2`

Proportion of data for testing (0.0 to 1.0).

### `random_state` *(int)*

**Default**: `61`

Seed for reproducible results.

### `stratify` *(bool)*

**Default**: `True`

Whether to maintain class proportions in train/test split.

## Context Input

- `data`: Dataset (required)
- `target_column`: Name of target variable column (required)

## Context Output

- `model`: Trained model object
- `X_train`, `X_test`: Feature splits
- `y_train`, `y_test`: Target splits  
- `y_pred`: Predictions on test set
- `train_score`: Training accuracy
- `test_score`: Test accuracy
- `model_name`: Algorithm name used
- `features`: List of feature column names

## Example Usage

```python
from flowcraft import FlowChain
from flowcraft.crafters import *

# Basic Random Forest
pipeline = FlowChain()
pipeline.add_crafter(DataIngestCrafter("data.csv"))
pipeline.add_crafter(ModelCrafter())
result = pipeline.run()

# Tuned XGBoost
pipeline = FlowChain()
pipeline.add_crafter(DataIngestCrafter("data.csv"))
pipeline.add_crafter(ModelCrafter(
    model_name="xgboost",
    model_params={
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 6
    },
    test_size=0.25
))
result = pipeline.run()

# Logistic Regression
pipeline = FlowChain()
pipeline.add_crafter(DataIngestCrafter("data.csv"))
pipeline.add_crafter(ModelCrafter(
    model_name="logistic_regression",
    model_params={"C": 0.5, "max_iter": 2000},
    stratify=False
))
result = pipeline.run()
``` 