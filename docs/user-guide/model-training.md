# Model Training

Learn how to train and evaluate machine learning models using MLFCrafter's ModelCrafter.

## ModelCrafter Overview

The `ModelCrafter` handles model selection, training, and evaluation. It supports multiple algorithms and provides train/test splitting with evaluation.

## Supported Algorithms

MLFCrafter supports three main algorithms:

```python
from mlfcrafter import ModelCrafter

# Random Forest (default)
model = ModelCrafter(model_name="random_forest")

# Logistic Regression
model = ModelCrafter(model_name="logistic_regression")

# XGBoost
model = ModelCrafter(model_name="xgboost")
```

## Basic Training

### Simple Model Training

```python
from mlfcrafter import MLFChain, DataIngestCrafter, CleanerCrafter, ModelCrafter

# Train a Random Forest model
pipeline = MLFChain(
    DataIngestCrafter("data.csv"),
    CleanerCrafter(strategy="auto"),
    ModelCrafter(model_name="random_forest")
)
result = pipeline.run()

print(f"Training accuracy: {result['train_score']:.4f}")
print(f"Test accuracy: {result['test_score']:.4f}")
```

## Model Parameters

### Random Forest Parameters

```python
# Customize Random Forest
model = ModelCrafter(
    model_name="random_forest",
    model_params={
        "n_estimators": 100,     # Number of trees
        "max_depth": 10,         # Maximum depth
        "min_samples_split": 5,  # Min samples to split
        "random_state": 42       # For reproducibility
    }
)
```

### XGBoost Parameters

```python
# Customize XGBoost
model = ModelCrafter(
    model_name="xgboost",
    model_params={
        "n_estimators": 200,     # Number of boosting rounds
        "learning_rate": 0.1,    # Learning rate
        "max_depth": 6,          # Tree depth
        "random_state": 42
    }
)
```

### Logistic Regression Parameters

```python
# Customize Logistic Regression
model = ModelCrafter(
    model_name="logistic_regression",
    model_params={
        "C": 1.0,                # Regularization strength
        "max_iter": 1000,        # Maximum iterations
        "random_state": 42
    }
)
```

## Training Configuration

### Data Splitting

```python
model = ModelCrafter(
    model_name="random_forest",
    test_size=0.2,        # 20% for testing
    random_state=61,      # Reproducible splits
    stratify=True         # Maintain class balance
)
```

### Complete Training Pipeline

```python
# Full training pipeline
pipeline = MLFChain(
    # Data processing
    DataIngestCrafter("customer_data.csv"),
    CleanerCrafter(strategy="median"),
    ScalerCrafter(scaler_type="standard"),

    # Model training
    ModelCrafter(
        model_name="xgboost",
        model_params={
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 6
        },
        test_size=0.25,
        stratify=True
    ),

    # Model evaluation
    ScorerCrafter(
        metrics=["accuracy", "precision", "recall", "f1"]
    )
)

result = pipeline.run()

# Print results
print(f"Model: {result['model_name']}")
print(f"Features: {len(result['features'])}")
print(f"Test accuracy: {result['test_score']:.4f}")
print("\nDetailed metrics:")
for metric, score in result['scores'].items():
    print(f"  {metric}: {score:.4f}")
```

## Model Comparison

Compare different algorithms:

```python
models = {
    "Random Forest": ModelCrafter(
        model_name="random_forest",
        model_params={"n_estimators": 100}
    ),
    "XGBoost": ModelCrafter(
        model_name="xgboost",
        model_params={"learning_rate": 0.1}
    ),
    "Logistic Regression": ModelCrafter(
        model_name="logistic_regression",
        model_params={"C": 1.0}
    )
}

results = {}
for name, model in models.items():
    pipeline = MLFChain(
        DataIngestCrafter("data.csv"),
        CleanerCrafter(strategy="auto"),
        ScalerCrafter(scaler_type="standard"),
        model,
        ScorerCrafter()
    )
    
    result = pipeline.run()
    results[name] = result['scores']

# Compare results
print("Model Comparison:")
for name, scores in results.items():
    print(f"{name}: Accuracy={scores['accuracy']:.4f}, F1={scores['f1']:.4f}")
```

## Model Evaluation

### Using ScorerCrafter

```python
# Evaluate with multiple metrics
pipeline = MLFChain(
    # ... other crafters ...
    ScorerCrafter(
        metrics=["accuracy", "precision", "recall", "f1"]
    )
)

result = pipeline.run()

# Access detailed scores
scores = result['scores']
print(f"Accuracy: {scores['accuracy']:.4f}")
print(f"Precision: {scores['precision']:.4f}")
print(f"Recall: {scores['recall']:.4f}")
print(f"F1-Score: {scores['f1']:.4f}")
```

### Custom Evaluation

```python
# Just specific metrics
scorer = ScorerCrafter(metrics=["accuracy", "f1"])

# Or just accuracy
scorer = ScorerCrafter(metrics=["accuracy"])
```

## Algorithm Selection Guide

### Choose Random Forest when:
- Need good baseline performance
- Want feature importance
- Working with mixed data types
- Don't want to tune many parameters

### Choose XGBoost when:
- Want maximum performance
- Have time for parameter tuning
- Working with structured data
- Need to handle missing values automatically

### Choose Logistic Regression when:
- Need interpretable results
- Want fast training/prediction
- Working with linear relationships
- Building baseline models

## Complete Example

```python
from mlfcrafter import MLFChain, DataIngestCrafter, CleanerCrafter, ScalerCrafter, ModelCrafter, ScorerCrafter, DeployCrafter

# E-commerce customer classification
pipeline = MLFChain(
    # Load and process data
    DataIngestCrafter("customers.csv"),
    CleanerCrafter(
        strategy="auto",
        str_fill="Unknown",
        int_fill=0
    ),
    ScalerCrafter(scaler_type="robust"),

    # Train XGBoost model
    ModelCrafter(
        model_name="xgboost",
        model_params={
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "random_state": 42
        },
        test_size=0.2,
        stratify=True
    ),

    # Evaluate performance
    ScorerCrafter(),

    # Deploy model
    DeployCrafter(
        model_path="customer_model.joblib",
        save_format="joblib"
    )
)

# Run complete pipeline
result = pipeline.run()

# Results
print(f"✅ Model trained successfully!")
print(f"📊 Test F1-Score: {result['scores']['f1']:.4f}")
print(f"💾 Model saved to: {result['deployment_path']}")
```

## Best Practices

### Start Simple
Begin with Random Forest using default parameters, then optimize.

### Use Proper Scaling
Always use ScalerCrafter before ModelCrafter, especially for logistic regression.

### Validate Performance
Use multiple metrics (accuracy, precision, recall, F1) to evaluate models.

### Set Random Seeds
Use consistent `random_state` values for reproducible results.

### Monitor Overfitting
Compare `train_score` vs `test_score` - large gaps indicate overfitting. 
