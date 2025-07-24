"""
MLFCrafter Usage Examples
=========================

This file demonstrates various ways to use MLFCrafter framework for ML pipeline automation.
Run different examples by uncommenting the desired section.
"""

import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from mlfcrafter import (
    CleanerCrafter,
    DataIngestCrafter,
    DeployCrafter,
    MLFChain,
    ModelCrafter,
    ScalerCrafter,
    ScorerCrafter,
    setup_logger,
)


def create_sample_data() -> str:
    """Create sample dataset and save to CSV file"""
    # Generate synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=3,
        n_informative=8,
        n_redundant=2,
        random_state=42,
    )

    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)  # type: ignore
    df["target"] = y

    # Add some missing values for cleaning demonstration
    np.random.seed(42)
    missing_indices = np.random.choice(1000, 100, replace=False)
    df.loc[missing_indices[:50], "feature_0"] = np.nan
    df.loc[missing_indices[50:], "feature_1"] = np.nan

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
    df.to_csv(temp_file.name, index=False)
    temp_file.close()

    return temp_file.name


def example_1_basic_pipeline() -> None:
    """Basic MLFCrafter pipeline with default settings"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Pipeline")
    print("=" * 60)

    # Create sample data
    data_path = create_sample_data()
    model_path = "basic_model.joblib"

    try:
        # Create pipeline
        chain = MLFChain(
            DataIngestCrafter(data_path=data_path),
            CleanerCrafter(strategy="auto"),
            ScalerCrafter(scaler_type="minmax"),
            ModelCrafter(model_name="random_forest"),
            ScorerCrafter(),
            DeployCrafter(model_path=model_path),
        )

        # Run pipeline
        results = chain.run(target_column="target")

        # Display results
        print(f"- Dataset shape: {results['original_shape']}")
        print(f"- Missing values handled: {results['missing_values_handled']}")
        print(f"- Model: {results['model_name']}")
        print(f"- Test accuracy: {results['test_score']:.4f}")
        print(f"- Model saved: {results.get('deployment_successful', False)}")

        print("Basic pipeline completed successfully!")

    finally:
        # Cleanup
        if os.path.exists(data_path):
            os.unlink(data_path)
        if os.path.exists(model_path):
            os.unlink(model_path)


def example_2_custom_parameters() -> None:
    """MLFCrafter pipeline with custom parameters"""
    print("=" * 60)
    print("EXAMPLE 2: Custom Parameters")
    print("=" * 60)

    # Create sample data
    data_path = create_sample_data()
    model_path = "custom_model.joblib"

    try:
        # Pipeline with custom parameters
        chain = MLFChain(
            DataIngestCrafter(data_path=data_path, source_type="auto"),
            CleanerCrafter(strategy="mean", str_fill="unknown", int_fill=-999),
            ScalerCrafter(scaler_type="standard"),
            ModelCrafter(
                model_name="xgboost",
                model_params={
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                },
                test_size=0.3,
                random_state=42,
            ),
            ScorerCrafter(metrics=["accuracy", "f1"]),
            DeployCrafter(
                model_path=model_path, save_format="joblib", include_metadata=True
            ),
        )

        results = chain.run(target_column="target")

        print(f"- Algorithm: {results['model_name']} with custom parameters")
        print(f"- Scaling: {results['scaler_type']}")
        print(f"- Test accuracy: {results['test_score']:.4f}")
        print(f"- F1-score: {results['scores'].get('f1', 'N/A'):.4f}")
        print("Custom pipeline completed successfully!")

    finally:
        # Cleanup
        if os.path.exists(data_path):
            os.unlink(data_path)
        if os.path.exists(model_path):
            os.unlink(model_path)


def example_3_with_logging() -> None:
    """MLFCrafter pipeline with detailed logging"""
    print("=" * 60)
    print("EXAMPLE 3: Pipeline with Logging")
    print("=" * 60)

    # Enable detailed logging
    setup_logger(level="INFO")

    # Create sample data
    data_path = create_sample_data()

    try:
        # Simple pipeline to demonstrate logging
        chain = MLFChain(
            DataIngestCrafter(data_path=data_path),
            CleanerCrafter(strategy="median"),
            ModelCrafter(
                model_name="logistic_regression", model_params={"max_iter": 1000}
            ),
        )

        results = chain.run(target_column="target")
        print(f"Final test score: {results['test_score']:.4f}")

    finally:
        # Cleanup
        if os.path.exists(data_path):
            os.unlink(data_path)


def example_4_model_loading() -> None:
    """Demonstrate saving and loading models"""
    print("=" * 60)
    print("EXAMPLE 4: Model Loading")
    print("=" * 60)

    # Create sample data
    data_path = create_sample_data()
    model_path = "loadable_model.joblib"

    try:
        # Train and save model
        chain = MLFChain(
            DataIngestCrafter(data_path=data_path),
            CleanerCrafter(strategy="auto"),
            ScalerCrafter(scaler_type="robust"),
            ModelCrafter(model_name="random_forest", model_params={"n_estimators": 50}),
            DeployCrafter(model_path=model_path, include_metadata=True),
        )

        results = chain.run(target_column="target")
        print(f"Model trained with accuracy: {results['test_score']:.4f}")

        if results.get("deployment_successful"):
            # Load the model back
            artifacts = DeployCrafter.load_model(model_path)

            print(f"Loaded model type: {type(artifacts['model']).__name__}")
            print(f"Saved artifacts: {list(artifacts.keys())}")

            if "metadata" in artifacts:
                metadata = artifacts["metadata"]
                print(
                    f"Model info: {metadata.get('model_name')} trained on {metadata.get('timestamp')}"
                )
                print(f"Features used: {len(metadata.get('features', []))}")

    finally:
        # Cleanup
        if os.path.exists(data_path):
            os.unlink(data_path)
        if os.path.exists(model_path):
            os.unlink(model_path)


if __name__ == "__main__":
    print("MLFCrafter Examples")
    print("Choose an example to run:")
    print("1. Basic pipeline (default settings)")
    print("2. Custom parameters")
    print("3. Pipeline with logging")
    print("4. Model saving and loading")

    # Uncomment the example you want to run:
    example_1_basic_pipeline()
    # example_2_custom_parameters()
    # example_3_with_logging()
    # example_4_model_loading()

    print("\nTo run tests: python -m pytest tests/test_mlfcrafter.py -v")
    print("All examples completed!")
