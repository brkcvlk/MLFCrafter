# Your First Pipeline

In this tutorial, you'll build a complete end-to-end machine learning pipeline using MLFCrafter. We'll work with the Iris dataset and go through every step from data ingestion to model deployment.

## Dataset: Iris Classification

We'll use the classic Iris dataset to build a flower classification model.

```python
import pandas as pd
import tempfile
from sklearn.datasets import load_iris
from mlfcrafter import MLFChain, DataIngestCrafter, CleanerCrafter, ScalerCrafter, ModelCrafter, ScorerCrafter, DeployCrafter

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

print(f"Dataset shape: {data.shape}")
print(f"Features: {list(iris.feature_names)}")
print(f"Classes: {list(iris.target_names)}")
```

## Step 1: Save Data to File

Since DataIngestCrafter loads from files, we need to save our data first:

```python
# Save data to temporary CSV file
temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
data.to_csv(temp_file.name, index=False)
temp_file.close()

print(f"Data saved to: {temp_file.name}")
```

## Step 2: Build the Pipeline

Create a complete ML pipeline with all necessary steps:

```python
# Create the complete pipeline
pipeline = MLFChain(
    # Step 1: Load data
    DataIngestCrafter(data_path=temp_file.name, source_type="csv"),
    
    # Step 2: Handle missing values (though Iris has none)
    CleanerCrafter(strategy="auto"),

    # Step 3 : Encode categorica features for Scaler Crafter 
    CategoricalCrafter(encoder_type="onehot")
    
    # Step 4: Scale features for better model performance
    ScalerCrafter(scaler_type="standard"),
    
    # Step 5: Train the model
    ModelCrafter(
        model_name="random_forest",
        model_params={
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        },
        test_size=0.2,
        stratify=True
    ),
    
    # Step 6: Evaluate model performance
    ScorerCrafter(metrics=["accuracy", "precision", "recall", "f1"]),
    
    # Step 7: Save the trained model
    DeployCrafter(
        model_path="iris_classifier.joblib",
        include_scaler=True,
        include_metadata=True
    )
)
```

## Step 3: Run the Pipeline

Execute the complete pipeline:

```python
# Run the pipeline
results = pipeline.run(target_column="target")

print("Pipeline execution completed!")
print(f"Dataset shape: {results['original_shape']}")
print(f"Model: {results['model_name']}")
print(f"Features used: {results['features']}")
print(f"Training accuracy: {results['train_score']:.4f}")
print(f"Test accuracy: {results['test_score']:.4f}")
```

## Step 4: View Detailed Metrics

Check the detailed performance metrics:

```python
# Display detailed metrics
scores = results['scores']
print("\nDetailed Performance Metrics:")
print(f"Accuracy:  {scores['accuracy']:.4f}")
print(f"Precision: {scores['precision']:.4f}")
print(f"Recall:    {scores['recall']:.4f}")
print(f"F1 Score:  {scores['f1']:.4f}")

# Check if model was deployed successfully
if results['deployment_successful']:
    print(f"\nModel saved to: {results['deployment_path']}")
    print(f"Artifacts saved: {results['artifacts_saved']}")
else:
    print("Model deployment failed")
```

## Step 5: Make Predictions with Saved Model

Load the saved model and make predictions:

```python
# Load the saved model
artifacts = DeployCrafter.load_model("iris_classifier.joblib")
saved_model = artifacts["model"]
saved_scaler = artifacts["scaler"]
metadata = artifacts["metadata"]

print(f"Loaded model trained on: {metadata['timestamp']}")
print(f"Original features: {metadata['features']}")

# Prepare new data for prediction (using some test data)
new_data = pd.DataFrame({
    'sepal length (cm)': [5.1, 6.2, 4.9],
    'sepal width (cm)': [3.5, 3.4, 3.0],
    'petal length (cm)': [1.4, 5.4, 1.4],
    'petal width (cm)': [0.2, 2.3, 0.2]
})

# Scale the new data using the saved scaler
new_data_scaled = saved_scaler.transform(new_data)

# Make predictions
predictions = saved_model.predict(new_data_scaled)
prediction_probabilities = saved_model.predict_proba(new_data_scaled)

# Convert predictions to species names
species_names = iris.target_names
predicted_species = [species_names[pred] for pred in predictions]

print("\nPredictions:")
for i, (species, prob) in enumerate(zip(predicted_species, prediction_probabilities)):
    print(f"Sample {i+1}: {species} (confidence: {max(prob):.3f})")
```

## Alternative: Step-by-Step Pipeline

You can also build the pipeline step by step if needed:

```python
# Alternative approach: build pipeline incrementally
pipeline = MLFChain()
pipeline.add_crafter(DataIngestCrafter(data_path=temp_file.name))
pipeline.add_crafter(CleanerCrafter(strategy="auto"))
pipeline.add_crafter(ScalerCrafter(scaler_type="standard"))
pipeline.add_crafter(ModelCrafter(
    model_name="random_forest",
    model_params={"n_estimators": 100, "random_state": 42}
))
pipeline.add_crafter(ScorerCrafter())

# Run the step-by-step pipeline
results = pipeline.run(target_column="target")
```

## Complete Pipeline Code

Here's the complete pipeline in one block:

```python
import pandas as pd
import tempfile
from sklearn.datasets import load_iris
from mlfcrafter import MLFChain, DataIngestCrafter, CleanerCrafter, ScalerCrafter, ModelCrafter, ScorerCrafter, DeployCrafter, CategoricalCrafter

# Load and prepare data
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Save to temporary file
temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
data.to_csv(temp_file.name, index=False)
temp_file.close()

# Build and run pipeline
pipeline = MLFChain(
    DataIngestCrafter(data_path=temp_file.name),
    CleanerCrafter(strategy="auto"),
    CategoricalCrafter(encoder_type="onehot")
    ScalerCrafter(scaler_type="standard"),
    ModelCrafter(
        model_name="random_forest",
        model_params={"n_estimators": 100, "random_state": 42},
        test_size=0.2,
        stratify=True
    ),
    ScorerCrafter(metrics=["accuracy", "f1"]),
    DeployCrafter(model_path="iris_model.joblib")
)

results = pipeline.run(target_column="target")

# Display results
print(f"Test Accuracy: {results['test_score']:.4f}")
print(f"F1 Score: {results['scores']['f1']:.4f}")
print(f"Model saved: {results['deployment_successful']}")

# Clean up
import os
os.unlink(temp_file.name)
```

## Key Takeaways

1. **Simple API**: MLFCrafter uses constructor-based configuration
2. **File-Based Input**: DataIngestCrafter loads from CSV, Excel, or JSON files
3. **Context Flow**: Data flows automatically between crafters via context dictionary
4. **All-in-One**: Single `run()` call executes the entire pipeline
5. **Rich Results**: Results dictionary contains all model info, scores, and artifacts
6. **Model Persistence**: DeployCrafter saves models with metadata for later use

## Understanding the Results

The pipeline returns a dictionary with these key components:

- `original_shape`: Shape of loaded data
- `model_name`: Algorithm used ("random_forest", etc.)
- `features`: List of feature column names
- `train_score`, `test_score`: Training and test accuracy
- `scores`: Dictionary with detailed metrics (accuracy, precision, recall, f1)
- `deployment_path`: Where the model was saved
- `deployment_successful`: Whether model saving worked

## Next Steps

- Learn about [Data Processing](../user-guide/data-processing.md) techniques
- Explore [Model Training](../user-guide/model-training.md) options  
- Check out [Deployment](../user-guide/deployment.md) strategies
- Try [Basic Usage Examples](../examples/basic-usage.md) 