# Model Deployment

Learn how to save and deploy your trained MLFCrafter models using the DeployCrafter.

## DeployCrafter Overview

The `DeployCrafter` handles model deployment by saving trained models and associated artifacts to disk. It supports multiple serialization formats.

## Basic Deployment

### Save Model with Default Settings

```python
from mlfcrafter import MLFChain, DataIngestCrafter, CleanerCrafter, ModelCrafter, DeployCrafter

# Train and save model
pipeline = MLFChain(
    DataIngestCrafter("data.csv"),
    CleanerCrafter(strategy="auto"),
    ModelCrafter(),
    DeployCrafter()  # Auto-generates filename
)
result = pipeline.run()

print(f"Model saved to: {result['deployment_path']}")
print(f"Artifacts saved: {result['artifacts_saved']}")
```

## Deployment Configuration

### Custom Model Path

```python
# Save to specific location
deployer = DeployCrafter(
    model_path="models/customer_model.joblib",
    save_format="joblib"
)
```

### Serialization Formats

```python
# Joblib format (recommended for sklearn models)
deployer = DeployCrafter(
    model_path="model.joblib",
    save_format="joblib"
)

# Pickle format (for compatibility)
deployer = DeployCrafter(
    model_path="model.pkl", 
    save_format="pickle"
)
```

### Artifact Control

```python
# Include all artifacts (default)
deployer = DeployCrafter(
    model_path="full_model.joblib",
    include_scaler=True,      # Include fitted scaler
    include_metadata=True     # Include training metadata
)

# Model only
deployer = DeployCrafter(
    model_path="model_only.joblib",
    include_scaler=False,
    include_metadata=False
)
```

## Complete Deployment Pipeline

```python
from mlfcrafter import MLFChain, DataIngestCrafter, CleanerCrafter, ScalerCrafter, ModelCrafter, ScorerCrafter, DeployCrafter

# Full pipeline with deployment
pipeline = MLFChain(
    # Data processing
    DataIngestCrafter("sales_data.csv"),
    CleanerCrafter(
        strategy="auto",
        str_fill="Unknown"
    ),
    ScalerCrafter(scaler_type="standard"),

    # Model training
    ModelCrafter(
        model_name="xgboost",
        model_params={
            "n_estimators": 200,
            "learning_rate": 0.05
        },
        test_size=0.2
    ),

    # Model evaluation
    ScorerCrafter(),

    # Model deployment
    DeployCrafter(
        model_path="production/sales_model_v1.joblib",
        save_format="joblib",
        include_scaler=True,     # Essential for production
        include_metadata=True    # Track model details
    )
)

result = pipeline.run()

# Deployment results
if result['deployment_successful']:
    print("✅ Model deployed successfully!")
    print(f"📍 Location: {result['deployment_path']}")
    print(f"📊 Test F1: {result['scores']['f1']:.4f}")
    print(f"🔧 Artifacts: {', '.join(result['artifacts_saved'])}")
else:
    print("❌ Deployment failed")
```

## Loading Saved Models

### Load Model for Inference

```python
# Load saved model and artifacts
artifacts = DeployCrafter.load_model("sales_model_v1.joblib")

# Access components
model = artifacts["model"]
scaler = artifacts.get("scaler")      # May be None if not saved
metadata = artifacts.get("metadata")  # May be None if not saved

print(f"Model type: {type(model).__name__}")
if scaler:
    print(f"Scaler type: {type(scaler).__name__}")
if metadata:
    print(f"Training date: {metadata['timestamp']}")
    print(f"Features: {metadata['features']}")
```

### Make Predictions

```python
import pandas as pd

# Load model artifacts
artifacts = DeployCrafter.load_model("customer_model.joblib")
model = artifacts["model"]
scaler = artifacts.get("scaler")

# Load new data
new_data = pd.read_csv("new_customers.csv")

# Apply same preprocessing if scaler was saved
if scaler:
    # Get feature columns from metadata
    feature_cols = artifacts["metadata"]["features"]
    new_data_scaled = scaler.transform(new_data[feature_cols])
    predictions = model.predict(new_data_scaled)
else:
    predictions = model.predict(new_data)

print(f"Predictions: {predictions}")
```

## Deployment Strategies

### Versioned Models

```python
from datetime import datetime

# Create versioned model name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"models/customer_churn_v{timestamp}.joblib"

deployer = DeployCrafter(
    model_path=model_path,
    include_metadata=True  # Track version info
)
```

### Environment-Specific Deployment

```python
import os

# Development environment
if os.getenv('ENV') == 'dev':
    deployer = DeployCrafter(
        model_path="dev/models/test_model.joblib",
        include_metadata=True
    )
# Production environment
else:
    deployer = DeployCrafter(
        model_path="prod/models/production_model.joblib",
        save_format="joblib",
        include_scaler=True,
        include_metadata=True
    )
```

## Model Artifacts

### What Gets Saved

When `include_metadata=True`, the following information is saved:

- Model algorithm name and parameters
- Feature names used for training
- Target column name
- Training and test scores
- Dataset shape information
- Scaler type (if scaler included)
- Training timestamp

### Accessing Metadata

```python
artifacts = DeployCrafter.load_model("model.joblib")
metadata = artifacts["metadata"]

print(f"Model trained on: {metadata['timestamp']}")
print(f"Algorithm: {metadata['model_name']}")
print(f"Test accuracy: {metadata['test_score']:.4f}")
print(f"Features used: {metadata['features']}")
```

## Best Practices

### Always Include Scaler
Set `include_scaler=True` for production models to ensure consistent preprocessing.

### Use Metadata for Tracking
Enable `include_metadata=True` to track model versions and performance.

### Check Deployment Success
Always verify the `deployment_successful` flag:

```python
result = pipeline.run()
if not result['deployment_successful']:
    print(f"Deployment failed: {result.get('deployment_error', 'Unknown error')}")
```

### Choose Appropriate Format
- Use `"joblib"` for scikit-learn models (faster, optimized)
- Use `"pickle"` for custom objects or compatibility needs


## Example: Production Deployment

```python
# Complete production deployment example
pipeline = MLFChain(
    # Data pipeline
    DataIngestCrafter("production_data.csv"),
    CleanerCrafter(strategy="auto"),
    ScalerCrafter(scaler_type="robust"),

    # Model training
    ModelCrafter(
        model_name="random_forest",
        model_params={"n_estimators": 200},
        test_size=0.2,
        stratify=True
    ),

    # Evaluation
    ScorerCrafter(),

    # Production deployment
    DeployCrafter(
        model_path="production/fraud_detection_v2.1.joblib",
        save_format="joblib",
        include_scaler=True,
        include_metadata=True
    )
)

result = pipeline.run()

# Validate deployment
if result['deployment_successful']:
    print("🚀 Production model deployed!")
    
    # Test loading
    artifacts = DeployCrafter.load_model(result['deployment_path'])
    print(f"✅ Model loading test successful")
    print(f"📊 F1 Score: {result['scores']['f1']:.4f}")
else:
    print("❌ Deployment failed - check logs")
``` 
