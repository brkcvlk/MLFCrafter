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
pipeline.add_crafter(CrafterA())
pipeline.add_crafter(CrafterB())
pipeline.add_crafter(CrafterC())

# Run the pipeline
result = pipeline.run()
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
pipeline.add_crafter(DataIngestCrafter(data))      # Outputs: processed DataFrame
pipeline.add_crafter(CleanerCrafter())             # Inputs: DataFrame, Outputs: clean DataFrame  
pipeline.add_crafter(ScalerCrafter())              # Inputs: DataFrame, Outputs: scaled DataFrame
pipeline.add_crafter(ModelCrafter())               # Inputs: DataFrame, Outputs: trained model
```

## Conditional Execution

Add conditional logic to pipelines:

```python
pipeline = MLFChain()
pipeline.add_crafter(DataIngestCrafter(data))

# Conditional cleaning based on data quality
if data.isnull().sum().sum() > 0:
    pipeline.add_crafter(CleanerCrafter(strategy='impute'))

pipeline.add_crafter(ModelCrafter())
```

## Next Steps

- Explore [Model Training](model-training.md) options  
- Read about [Deployment](deployment.md) strategies
