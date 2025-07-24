# Basic Usage Examples

Learn MLFCrafter through practical examples that demonstrate core functionality and common use cases.

## Simple Classification Pipeline

Let's start with a basic classification example using sample data:

```python
import pandas as pd
import numpy as np
import tempfile
from sklearn.datasets import make_classification
from mlfcrafter import MLFChain, DataIngestCrafter, CleanerCrafter, ScalerCrafter, ModelCrafter, ScorerCrafter

# Generate sample classification data
X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
data['target'] = y

# Save to temporary CSV file
temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
data.to_csv(temp_file.name, index=False)
temp_file.close()

# Create and run pipeline
pipeline = MLFChain(
    DataIngestCrafter(data_path=temp_file.name),
    CleanerCrafter(strategy='drop'),
    ScalerCrafter(scaler_type='standard'),
    ModelCrafter(model_name='random_forest'),
    ScorerCrafter(metrics=['accuracy', 'precision', 'recall', 'f1'])
)

results = pipeline.run(target_column='target')

# Display results
print(f"Model: {results['model_name']}")
print(f"Test accuracy: {results['test_score']:.4f}")
print(f"Precision: {results['scores']['precision']:.4f}")
print(f"F1 Score: {results['scores']['f1']:.4f}")
```

## Working with Different File Formats

MLFCrafter supports CSV, Excel, and JSON files:

```python
from mlfcrafter import DataIngestCrafter

# CSV files (auto-detected)
csv_ingester = DataIngestCrafter(data_path='data.csv', source_type='auto')

# Excel files  
excel_ingester = DataIngestCrafter(data_path='data.xlsx', source_type='excel')

# JSON files
json_ingester = DataIngestCrafter(data_path='data.json', source_type='json')

# Force CSV format even if extension is different
csv_forced = DataIngestCrafter(data_path='data.txt', source_type='csv')
```

## Handling Missing Values

Different strategies for handling missing data:

```python
# Add some missing values to demonstrate
data_with_missing = data.copy()
data_with_missing.loc[0:50, 'feature_0'] = np.nan
data_with_missing.loc[100:120, 'feature_1'] = np.nan

temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
data_with_missing.to_csv(temp_file.name, index=False)
temp_file.close()

# Strategy 1: Drop rows with missing values
pipeline_drop = MLFChain(
    DataIngestCrafter(data_path=temp_file.name),
    CleanerCrafter(strategy='drop')
)
results_drop = pipeline_drop.run(target_column='target')
print(f"After dropping: {results_drop['data'].shape}")

# Strategy 2: Fill with mean for numerical columns
pipeline_mean = MLFChain(
    DataIngestCrafter(data_path=temp_file.name),
    CleanerCrafter(strategy='mean')
)
results_mean = pipeline_mean.run(target_column='target')

# Strategy 3: Fill with median
pipeline_median = MLFChain(
    DataIngestCrafter(data_path=temp_file.name),
    CleanerCrafter(strategy='median')
)
results_median = pipeline_median.run(target_column='target')

# Strategy 4: Fill with constant values
pipeline_constant = MLFChain(
    DataIngestCrafter(data_path=temp_file.name),
    CleanerCrafter(strategy='constant', int_fill=-999, str_fill='MISSING')
)
results_constant = pipeline_constant.run(target_column='target')
```

## Different Scaling Methods

Compare different feature scaling approaches:

```python
# Create sample data
sample_data = pd.DataFrame({
    'small_values': np.random.normal(0, 1, 1000),
    'large_values': np.random.normal(1000, 100, 1000),
    'target': np.random.randint(0, 2, 1000)
})

temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
sample_data.to_csv(temp_file.name, index=False)
temp_file.close()

# MinMax scaling (scales to 0-1 range)
pipeline_minmax = MLFChain(
    DataIngestCrafter(data_path=temp_file.name),
    ScalerCrafter(scaler_type='minmax')
)
results_minmax = pipeline_minmax.run(target_column='target')

# Standard scaling (mean=0, std=1) 
pipeline_standard = MLFChain(
    DataIngestCrafter(data_path=temp_file.name),
    ScalerCrafter(scaler_type='standard')
)
results_standard = pipeline_standard.run(target_column='target')

# Robust scaling (uses median and IQR, good for outliers)
pipeline_robust = MLFChain(
    DataIngestCrafter(data_path=temp_file.name),
    ScalerCrafter(scaler_type='robust')
)
results_robust = pipeline_robust.run(target_column='target')

# Scale only specific columns
pipeline_selective = MLFChain(
    DataIngestCrafter(data_path=temp_file.name),
    ScalerCrafter(scaler_type='standard', columns=['large_values'])
)
results_selective = pipeline_selective.run(target_column='target')
```

## Model Comparison

Compare different algorithms on the same dataset:

```python
# Algorithms available in MLFCrafter
algorithms = ['random_forest', 'logistic_regression', 'xgboost']
results = {}

for algorithm in algorithms:
    pipeline = MLFChain(
        DataIngestCrafter(data_path=temp_file.name),
        CleanerCrafter(strategy='drop'),
        ScalerCrafter(scaler_type='standard'),
        ModelCrafter(model_name=algorithm, random_state=42),
        ScorerCrafter(metrics=['accuracy', 'f1'])
    )
    
    result = pipeline.run(target_column='target')
    results[algorithm] = {
        'accuracy': result['test_score'],
        'f1': result['scores']['f1']
    }

# Display comparison
print("Algorithm Comparison:")
print("-" * 40)
for algo, metrics in results.items():
    print(f"{algo:20s}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
```

## Custom Model Parameters

Configure model hyperparameters:

```python
# Random Forest with custom parameters
rf_pipeline = MLFChain(
    DataIngestCrafter(data_path=temp_file.name),
    CleanerCrafter(strategy='median'),
    ScalerCrafter(scaler_type='robust'),
    ModelCrafter(
        model_name='random_forest',
        model_params={
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42
        },
        test_size=0.3
    ),
    ScorerCrafter(metrics=['accuracy', 'precision', 'recall'])
)

# XGBoost with custom parameters
xgb_pipeline = MLFChain(
    DataIngestCrafter(data_path=temp_file.name),
    CleanerCrafter(strategy='mean'),
    ScalerCrafter(scaler_type='standard'),
    ModelCrafter(
        model_name='xgboost',
        model_params={
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8
        }
    ),
    ScorerCrafter()
)

# Logistic Regression with custom parameters
lr_pipeline = MLFChain(
    DataIngestCrafter(data_path=temp_file.name),
    CleanerCrafter(strategy='drop'),
    ScalerCrafter(scaler_type='standard'),
    ModelCrafter(
        model_name='logistic_regression',
        model_params={
            'C': 0.1,
            'max_iter': 1000,
            'penalty': 'l2'
        },
        stratify=True
    )
)
```

## Saving and Loading Models

Save trained models for later use:

```python
# Train and save model
save_pipeline = MLFChain(
    DataIngestCrafter(data_path=temp_file.name),
    CleanerCrafter(strategy='median'),
    ScalerCrafter(scaler_type='standard'),
    ModelCrafter(model_name='random_forest'),
    ScorerCrafter(),
    DeployCrafter(
        model_path='my_model.joblib',
        save_format='joblib',
        include_scaler=True,
        include_metadata=True
    )
)

results = save_pipeline.run(target_column='target')
print(f"Model saved: {results.get('deployment_successful', False)}")
print(f"Saved to: {results.get('deployment_path', 'N/A')}")

# Load the saved model
if results.get('deployment_successful'):
    artifacts = DeployCrafter.load_model('my_model.joblib')
    
    print(f"Loaded artifacts: {list(artifacts.keys())}")
    if 'metadata' in artifacts:
        metadata = artifacts['metadata']
        print(f"Model info: {metadata.get('model_name')}")
        print(f"Features: {metadata.get('features')}")
        print(f"Training score: {metadata.get('train_score'):.4f}")
```

## Error Handling

Handle pipeline errors gracefully:

```python
try:
    # This might fail if file doesn't exist
    pipeline = MLFChain(
        DataIngestCrafter(data_path='nonexistent_file.csv'),
        CleanerCrafter(strategy='drop'),
        ModelCrafter(model_name='random_forest')
    )
    
    results = pipeline.run(target_column='target')
    print("Pipeline completed successfully!")
    
except FileNotFoundError as e:
    print(f"File not found: {e}")
    
except ValueError as e:
    print(f"Configuration error: {e}")
    
except RuntimeError as e:
    print(f"Pipeline execution failed: {e}")
    
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Pipeline with Logging

Enable detailed logging to see what's happening:

```python
from mlfcrafter import setup_logger

# Enable DEBUG logging to see detailed progress
setup_logger(level='DEBUG')

# Run pipeline with detailed logging
pipeline = MLFChain(
    DataIngestCrafter(data_path=temp_file.name),
    CleanerCrafter(strategy='auto'),
    ScalerCrafter(scaler_type='standard'),
    ModelCrafter(model_name='xgboost'),
    ScorerCrafter(),
    DeployCrafter(model_path='debug_model.joblib')
)

results = pipeline.run(target_column='target')

# The logs will show detailed information about each step:
# - Data loading and shape
# - Missing values found and handled
# - Scaling applied to which columns
# - Model training progress
# - Evaluation metrics
# - Model saving status
```

## Adding Crafters Individually

You can build pipelines step by step:

```python
# Start with empty pipeline
pipeline = MLFChain()

# Add crafters one by one
pipeline.add_crafter(DataIngestCrafter(data_path=temp_file.name))
pipeline.add_crafter(CleanerCrafter(strategy='median'))

# Conditionally add scaling
if data.select_dtypes(include=[np.number]).shape[1] > 0:
    pipeline.add_crafter(ScalerCrafter(scaler_type='standard'))

# Add model and scoring
pipeline.add_crafter(ModelCrafter(model_name='random_forest'))
pipeline.add_crafter(ScorerCrafter(metrics=['accuracy', 'f1']))

# Run the pipeline
results = pipeline.run(target_column='target')
```

## Complete Workflow Example

Here's a complete example showing all features:

```python
import os
from mlfcrafter import MLFChain, setup_logger
from mlfcrafter.crafters import *

# Setup logging
setup_logger(level='INFO')

# Create sample data with missing values and mixed types
sample_data = pd.DataFrame({
    'numeric_1': np.random.normal(10, 2, 1000),
    'numeric_2': np.random.normal(50, 10, 1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'target': np.random.randint(0, 3, 1000)
})

# Add some missing values
sample_data.loc[0:50, 'numeric_1'] = np.nan
sample_data.loc[100:120, 'category'] = np.nan

# Save to file
temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
sample_data.to_csv(temp_file.name, index=False)
temp_file.close()

try:
    # Complete pipeline
    pipeline = MLFChain(
        DataIngestCrafter(data_path=temp_file.name, source_type='auto'),
        CleanerCrafter(strategy='auto', str_fill='UNKNOWN', int_fill=0),
        ScalerCrafter(scaler_type='robust'),
        ModelCrafter(
            model_name='xgboost',
            model_params={
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1
            },
            test_size=0.25,
            random_state=42
        ),
        ScorerCrafter(metrics=['accuracy', 'precision', 'recall', 'f1']),
        DeployCrafter(
            model_path='final_model.joblib',
            include_metadata=True
        )
    )
    
    # Execute pipeline
    results = pipeline.run(target_column='target')
    
    # Print comprehensive results
    print("\n" + "="*50)
    print("PIPELINE RESULTS")
    print("="*50)
    print(f"Original data shape: {results['original_shape']}")
    print(f"Final data shape: {results['data'].shape}")
    print(f"Missing values handled: {results['missing_values_handled']}")
    print(f"Scaled columns: {results['scaled_columns']}")
    print(f"Model used: {results['model_name']}")
    print(f"Features: {len(results['features'])}")
    print(f"Training accuracy: {results['train_score']:.4f}")
    print(f"Test accuracy: {results['test_score']:.4f}")
    
    print("\nDetailed Metrics:")
    for metric, value in results['scores'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nModel saved: {results.get('deployment_successful', False)}")
    if results.get('deployment_path'):
        print(f"Model path: {results['deployment_path']}")

finally:
    # Cleanup
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)
```

## Key Points to Remember

1. **File-based input**: DataIngestCrafter requires data to be in a file (CSV, Excel, JSON)
2. **Available strategies**: CleanerCrafter supports 'auto', 'mean', 'median', 'mode', 'drop', 'constant'
3. **Scaling options**: ScalerCrafter offers 'minmax', 'standard', 'robust' scaling
4. **Model choices**: ModelCrafter supports 'random_forest', 'logistic_regression', 'xgboost'
5. **Results dictionary**: All results are returned in a context dictionary
6. **Error handling**: Always wrap pipeline execution in try-catch blocks
7. **Logging**: Use setup_logger() to see detailed execution information

## Next Steps

- Learn about [User Guide](../user-guide/pipeline-basics.md) for more complex scenarios  
- Check out [API Reference](../api/mlfchain.md) to understand all options  
- Review [Getting Started Guide](../getting-started/quickstart.md) for additional examples 
