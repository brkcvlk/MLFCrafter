# Pipeline Basics

Learn the fundamental concepts of MLFCrafter pipelines and how to build effective ML workflows.

## What is a MLFChain?

A `MLFChain` is the core orchestrator in MLFCrafter that manages the execution of multiple processing steps (crafters) in sequence. It provides:

- **Sequential Processing**: Steps execute in order
- **Data Flow Management**: Automatic data passing between steps
- **Error Handling**: Graceful failure handling and rollback
- **State Management**: Track pipeline state and intermediate results
- **Logging**: Comprehensive logging of each step

## Basic Pipeline Structure

```python
from mlfcrafter import MLFChain

# Create a new pipeline
pipeline = MLFChain()

# Add processing steps
pipeline.add(CrafterA())
pipeline.add(CrafterB())
pipeline.add(CrafterC())

# Execute the pipeline
result = pipeline.execute()
```

## Crafter Lifecycle

Each crafter goes through several phases during execution:

1. **Initialization**: Crafter receives data from previous step
2. **Validation**: Input data is validated
3. **Processing**: Main logic execution
4. **Output**: Results are prepared for next step
5. **Cleanup**: Resources are released

## Data Flow Between Crafters

Data flows automatically between crafters:

```python
pipeline = MLFChain()
pipeline.add(DataIngestCrafter(data))      # Outputs: processed DataFrame
pipeline.add(CleanerCrafter())             # Inputs: DataFrame, Outputs: clean DataFrame  
pipeline.add(ScalerCrafter())              # Inputs: DataFrame, Outputs: scaled DataFrame
pipeline.add(ModelCrafter())               # Inputs: DataFrame, Outputs: trained model
```

## Pipeline Configuration

### Method Chaining

Use fluent interface for concise pipeline building:

```python
result = (MLFChain()
    .add(DataIngestCrafter(data))
    .add(CleanerCrafter(strategy='impute'))
    .add(ModelCrafter(algorithm='xgboost'))
    .execute())
```

### Pipeline Settings

Configure global pipeline behavior:

```python
pipeline = MLFChain(
    logging_level='INFO',
    fail_fast=True,
    parallel=False,
    cache_intermediate=True
)
```

## Error Handling

MLFCrafter provides robust error handling:

```python
try:
    result = pipeline.execute()
except MLFCrafterError as e:
    print(f"Pipeline failed at step: {e.step}")
    print(f"Error: {e.message}")
    # Access intermediate results
    partial_results = e.partial_results
```

## Pipeline States

Monitor pipeline execution state:

```python
pipeline = MLFChain()
pipeline.add(DataIngestCrafter(data))
pipeline.add(CleanerCrafter())

# Check state before execution
print(f"Pipeline state: {pipeline.state}")  # 'ready'

result = pipeline.execute()

# Check state after execution  
print(f"Pipeline state: {pipeline.state}")  # 'completed'
```

## Conditional Execution

Add conditional logic to pipelines:

```python
pipeline = MLFChain()
pipeline.add(DataIngestCrafter(data))

# Conditional cleaning based on data quality
if data.isnull().sum().sum() > 0:
    pipeline.add(CleanerCrafter(strategy='impute'))

pipeline.add(ModelCrafter())
```

## Pipeline Branching

Create complex workflows with branching:

```python
# Main pipeline
main_pipeline = MLFChain()
main_pipeline.add(DataIngestCrafter(data))
main_pipeline.add(CleanerCrafter())

# Branch for different models
rf_branch = MLFChain()
rf_branch.add(ModelCrafter(algorithm='random_forest'))

xgb_branch = MLFChain() 
xgb_branch.add(ModelCrafter(algorithm='xgboost'))

# Combine results
results = {
    'random_forest': main_pipeline.branch(rf_branch).execute(),
    'xgboost': main_pipeline.branch(xgb_branch).execute()
}
```

## Performance Optimization

### Caching

Enable caching for expensive operations:

```python
pipeline = MLFChain(cache_intermediate=True)
pipeline.add(DataIngestCrafter(data, cache_key='raw_data'))
pipeline.add(CleanerCrafter(cache_key='clean_data'))
```

### Parallel Processing

Enable parallel execution where possible:

```python
pipeline = MLFChain(parallel=True, n_jobs=4)
```

## Best Practices

### 1. Keep Crafters Focused

Each crafter should have a single responsibility:

```python
# Good: Focused crafters
pipeline.add(CleanerCrafter())         # Only cleaning
pipeline.add(ScalerCrafter())          # Only scaling  
pipeline.add(ModelCrafter())           # Only modeling

# Avoid: Multi-purpose crafters
pipeline.add(DataProcessorCrafter())   # Too many responsibilities
```

### 2. Use Meaningful Names

Name your pipeline steps clearly:

```python
pipeline.add(DataIngestCrafter(data), name='data_loading')
pipeline.add(CleanerCrafter(), name='missing_value_imputation')  
pipeline.add(ScalerCrafter(), name='feature_standardization')
```

### 3. Validate Inputs

Always validate data at key points:

```python
pipeline.add(DataIngestCrafter(data))
pipeline.add(ValidationCrafter(
    required_columns=['feature1', 'feature2', 'target'],
    data_types={'feature1': 'float64', 'target': 'int64'}
))
pipeline.add(CleanerCrafter())
```

### 4. Log Important Information

Use logging to track pipeline progress:

```python
import logging

pipeline = MLFChain(logging_level='INFO')
# MLFCrafter will automatically log step execution
```

## Common Patterns

### Data Preprocessing Pipeline

```python
preprocessing = (MLFChain()
    .add(DataIngestCrafter(data))
    .add(CleanerCrafter(strategy='impute'))
    .add(ScalerCrafter(method='standard'))
    .add(FeatureSelectorCrafter(method='variance')))
```

### Model Training Pipeline

```python
training = (MLFChain()
    .add(ModelCrafter(algorithm='random_forest'))
    .add(ScorerCrafter(metrics=['accuracy', 'f1']))
    .add(ValidationCrafter(method='cross_val')))
```

### End-to-End Pipeline

```python
complete_pipeline = (MLFChain()
    .add(DataIngestCrafter(data))
    .add(CleanerCrafter())
    .add(ScalerCrafter())
    .add(ModelCrafter())
    .add(ScorerCrafter())
    .add(DeployCrafter()))
```

## Next Steps

- Learn about specific [Data Processing](data-processing.md) techniques
- Explore [Model Training](model-training.md) options  
- Read about [Deployment](deployment.md) strategies
- Check [Logging & Debugging](logging.md) guide 
