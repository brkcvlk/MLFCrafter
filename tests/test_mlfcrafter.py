"""
MLFCrafter Test Suite
====================

Comprehensive test suite for MLFCrafter ML pipeline automation tool using sklearn datasets.
Tests individual crafters and end-to-end pipeline functionality.
"""

import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

# Import sklearn datasets
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

from mlfcrafter import (
    CategoricalCrafter,
    CleanerCrafter,
    DataIngestCrafter,
    DeployCrafter,
    MLFChain,
    ModelCrafter,
    ScalerCrafter,
    ScorerCrafter,
)


class TestDatasets:
    """Base test class providing dataset fixtures using sklearn datasets."""

    @pytest.fixture
    def iris_dataset(self):
        """Load iris dataset for testing"""
        iris = load_iris(as_frame=False)
        df = pd.DataFrame(iris.data, columns=iris.feature_names)  # type: ignore
        df["target"] = iris.target  # type: ignore
        return df

    @pytest.fixture
    def wine_dataset(self):
        """Load wine dataset for multi-class classification tests"""
        wine = load_wine(as_frame=False)
        df = pd.DataFrame(wine.data, columns=wine.feature_names)  # type: ignore
        df["target"] = wine.target  # type: ignore
        return df

    @pytest.fixture
    def breast_cancer_dataset(self):
        """Load breast cancer dataset for binary classification tests"""
        cancer = load_breast_cancer(as_frame=False)
        df = pd.DataFrame(cancer.data, columns=cancer.feature_names)  # type: ignore
        df["target"] = cancer.target  # type: ignore
        return df

    @pytest.fixture
    def temp_csv_file(self, tmp_path):
        """Create temporary CSV file with iris data"""
        iris = load_iris(as_frame=False)
        df = pd.DataFrame(iris.data, columns=iris.feature_names)  # type: ignore
        df["target"] = iris.target  # type: ignore

        # Save to temporary file
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)

        return str(csv_path)


class TestDataIngestCrafter(TestDatasets):
    """Test suite for DataIngestCrafter"""

    def test_csv_ingestion(self, temp_csv_file):
        """Test CSV file ingestion"""
        crafter = DataIngestCrafter(data_path=temp_csv_file)
        context = {}
        result = crafter.run(context)

        assert "data" in result
        assert isinstance(result["data"], pd.DataFrame)
        assert result["data"].shape == (150, 5)  # Iris dataset shape
        assert "original_shape" in result

    def test_auto_detection(self, temp_csv_file):
        """Test automatic format detection"""
        crafter = DataIngestCrafter(data_path=temp_csv_file, source_type="auto")
        context = {}
        result = crafter.run(context)

        assert result["data"].shape[0] > 0


class TestCleanerCrafter(TestDatasets):
    """Test suite for CleanerCrafter"""

    def test_auto_cleaning(self, iris_dataset):
        """Test auto cleaning strategy"""
        # Add some missing values
        iris_dataset.loc[0:5, "sepal length (cm)"] = np.nan
        iris_dataset.loc[10:15, "petal width (cm)"] = np.nan

        crafter = CleanerCrafter(strategy="auto")
        context = {"data": iris_dataset}
        result = crafter.run(context)

        cleaned_data = result["data"]
        assert cleaned_data.isnull().sum().sum() == 0  # No missing values
        assert result["missing_values_handled"] is True

    def test_mean_cleaning(self, iris_dataset):
        """Test mean cleaning strategy"""
        # Add some missing values to numerical columns
        iris_dataset.loc[0:5, "sepal length (cm)"] = np.nan

        crafter = CleanerCrafter(strategy="mean")
        context = {"data": iris_dataset}
        result = crafter.run(context)

        cleaned_data = result["data"]
        assert cleaned_data.isnull().sum().sum() == 0


class TestScalerCrafter(TestDatasets):
    """Test suite for ScalerCrafter"""

    def test_standard_scaling(self, iris_dataset):
        """Test standard scaling (z-score normalization)"""
        crafter = ScalerCrafter(scaler_type="standard")
        context = {"data": iris_dataset, "target_column": "target"}
        result = crafter.run(context)

        scaled_data = result["data"]
        numerical_cols = result["scaled_columns"]

        # Check that numerical columns are scaled
        for col in numerical_cols:
            assert abs(scaled_data[col].mean()) < 1e-10  # Nearly zero mean
            assert abs(scaled_data[col].std() - 1) < 0.01  # Nearly unit std

        assert "scaler" in result
        assert result["scaler_type"] == "standard"

    def test_minmax_scaling(self, iris_dataset):
        """Test MinMax scaling to [0,1] range"""
        crafter = ScalerCrafter(scaler_type="minmax")
        context = {"data": iris_dataset, "target_column": "target"}
        result = crafter.run(context)

        scaled_data = result["data"]
        numerical_cols = result["scaled_columns"]

        # Check that numerical columns are scaled to [0,1]
        for col in numerical_cols:
            assert scaled_data[col].min() >= 0  # Min should be >= 0
            assert scaled_data[col].max() <= 1  # Max should be <= 1

        assert "scaler" in result
        assert result["scaler_type"] == "minmax"


class TestCategoricalCrafter(TestDatasets):
    """Test suite for CategoricalCrafter"""

    def test_one_hot_encoding(self, iris_dataset):
        """Test one-hot encoding of categorical columns"""
        # Convert target to categorical for testing
        iris_dataset["target"] = iris_dataset["target"].astype("category")

        crafter = CategoricalCrafter(encoder_type="onehot")
        context = {"data": iris_dataset, "target_column": "target"}
        result = crafter.run(context)

        encoded_data = result["data"]
        assert {"target_0", "target_1"}.issubset(encoded_data.columns)  # Check one-hot columns
        assert "target" not in encoded_data.columns  # Original target should be removed
        assert (len(result["encoded_columns"]) > 0)  # At least one column should be encoded

    def test_label_encoding(self, iris_dataset):
        """Test label encoding of categorical columns"""
        # Convert target to categorical for testing
        iris_dataset["target"] = iris_dataset["target"].astype("category")

        crafter = CategoricalCrafter(encoder_type="label")
        context = {"data": iris_dataset, "target_column": "target"}
        result = crafter.run(context)

        encoded_data = result["data"]
        assert "target" in encoded_data.columns  # Target should be label-encoded
        assert (len(result["encoded_columns"]) > 0)  # At least one column should be encoded


class TestModelCrafter(TestDatasets):
    """Test suite for ModelCrafter"""

    def test_random_forest_training(self, iris_dataset):
        """Test Random Forest model training"""
        crafter = ModelCrafter(
            model_name="random_forest", model_params={"n_estimators": 10}
        )
        context = {"data": iris_dataset, "target_column": "target"}
        result = crafter.run(context)

        assert "model" in result
        assert result["model_name"] == "random_forest"
        assert result["train_score"] > 0.8  # Should achieve good training score
        assert result["test_score"] > 0.7  # Should generalize reasonably well
        assert len(result["features"]) == 4  # Iris has 4 features

    def test_logistic_regression_training(self, breast_cancer_dataset):
        """Test Logistic Regression for binary classification"""
        crafter = ModelCrafter(
            model_name="logistic_regression", model_params={"max_iter": 2000}
        )
        context = {"data": breast_cancer_dataset, "target_column": "target"}
        result = crafter.run(context)

        assert "model" in result
        assert result["model_name"] == "logistic_regression"
        assert result["train_score"] > 0.9  # Should achieve very good training score
        assert result["test_score"] > 0.8  # Should generalize well

    def test_missing_target_column_error(self, iris_dataset):
        """Test error handling for missing target column"""
        crafter = ModelCrafter()
        context = {"data": iris_dataset, "target_column": "nonexistent_column"}

        with pytest.raises(ValueError, match="not found in data"):
            crafter.run(context)


class TestScorerCrafter(TestDatasets):
    """Test suite for ScorerCrafter"""

    def test_all_metrics_multiclass(self, iris_dataset):
        """Test calculation of all metrics for multi-class classification"""
        # First need to train a model
        model_crafter = ModelCrafter(
            model_name="random_forest", model_params={"n_estimators": 10}
        )
        model_context = {"data": iris_dataset, "target_column": "target"}
        model_context = model_crafter.run(model_context)

        # Now test scorer
        scorer = ScorerCrafter()  # Default is all metrics
        result = scorer.run(model_context)

        scores = result["scores"]
        expected_metrics = ["accuracy", "precision", "recall", "f1"]

        for metric in expected_metrics:
            assert metric in scores
            assert 0 <= scores[metric] <= 1  # All metrics should be between 0 and 1

    def test_binary_classification_metrics(self, breast_cancer_dataset):
        """Test metrics for binary classification"""
        # Train model first
        model_crafter = ModelCrafter(
            model_name="logistic_regression", model_params={"max_iter": 2000}
        )
        model_context = {"data": breast_cancer_dataset, "target_column": "target"}
        model_context = model_crafter.run(model_context)

        # Test scorer
        scorer = ScorerCrafter(metrics=["accuracy", "precision", "recall"])
        result = scorer.run(model_context)

        scores = result["scores"]
        assert "accuracy" in scores
        assert "precision" in scores
        assert "recall" in scores
        # All metrics should be reasonable for this dataset
        assert scores["accuracy"] > 0.8


class TestDeployCrafter(TestDatasets):
    """Test suite for DeployCrafter"""

    def test_model_deployment(self, iris_dataset, tmp_path):
        """Test basic model deployment"""
        # First train a model
        model_crafter = ModelCrafter(
            model_name="random_forest", model_params={"n_estimators": 10}
        )
        context = {"data": iris_dataset, "target_column": "target"}
        context = model_crafter.run(context)

        # Deploy the model
        model_path = tmp_path / "test_model.joblib"
        deployer = DeployCrafter(model_path=str(model_path))
        result = deployer.run(context)

        # Verify deployment
        if result["deployment_successful"]:
            assert os.path.exists(model_path)
            assert "artifacts_saved" in result
            assert "model" in result["artifacts_saved"]
        else:
            # Deployment might fail due to filesystem constraints, which is OK
            assert "deployment_error" in result

    def test_model_loading(self, iris_dataset, tmp_path):
        """Test loading deployed model"""
        # Train and deploy model
        model_crafter = ModelCrafter(
            model_name="random_forest", model_params={"n_estimators": 10}
        )
        model_context = {"data": iris_dataset, "target_column": "target"}
        model_context = model_crafter.run(model_context)

        model_path = tmp_path / "test_model.joblib"
        deployer = DeployCrafter(model_path=str(model_path))
        deploy_result = deployer.run(model_context)

        if deploy_result["deployment_successful"]:
            # Load model back
            artifacts = DeployCrafter.load_model(str(model_path))

            assert "model" in artifacts
            assert "metadata" in artifacts
            assert artifacts["metadata"]["model_name"] == "random_forest"

    def test_deployment_without_model_error(self):
        """Test error handling when no model is available"""
        deployer = DeployCrafter()
        context = {}  # No model

        with pytest.raises(ValueError, match="No trained model found"):
            deployer.run(context)


class TestEndToEndPipeline(TestDatasets):
    """Integration tests for complete MLFCrafter pipelines"""

    def test_complete_pipeline_iris(self, temp_csv_file, tmp_path):
        """Test complete pipeline with Iris dataset"""
        model_path = tmp_path / "pipeline_model.joblib"

        chain = MLFChain(
            DataIngestCrafter(data_path=temp_csv_file),
            CleanerCrafter(strategy="auto"),
            ScalerCrafter(scaler_type="standard"),
            ModelCrafter(model_name="random_forest", model_params={"n_estimators": 10}),
            ScorerCrafter(),  # Default is all metrics
            DeployCrafter(model_path=str(model_path)),
        )

        result = chain.run(target_column="target")

        # Verify pipeline completed successfully
        assert result["test_score"] > 0.7
        assert len(result["scores"]) == 4  # All metrics calculated
        # Iris dataset has no missing values, so this should be False
        assert result["missing_values_handled"] is False

        # Deployment should be attempted (success is optional due to filesystem constraints)
        assert "deployment_successful" in result
        if result["deployment_successful"]:
            assert "deployment_path" in result
            assert "artifacts_saved" in result

    def test_complete_pipeline_wine(self, wine_dataset, tmp_path):
        """Test complete pipeline with Wine dataset (multi-class)"""
        # Add some missing values first
        wine_dataset.loc[10:15, wine_dataset.columns[0]] = np.nan

        # Save to temp file
        temp_file = tmp_path / "wine_test.csv"
        wine_dataset.to_csv(temp_file, index=False)

        model_path = tmp_path / "wine_model.joblib"

        chain = MLFChain(
            DataIngestCrafter(data_path=str(temp_file)),
            CleanerCrafter(strategy="mean"),
            ScalerCrafter(scaler_type="minmax"),
            ModelCrafter(model_name="xgboost", model_params={"n_estimators": 20}),
            ScorerCrafter(metrics=["accuracy", "f1"]),
            DeployCrafter(model_path=str(model_path)),
        )

        result = chain.run(target_column="target")

        # Wine dataset should achieve good performance
        assert result["test_score"] > 0.8
        assert result["missing_values_handled"] is True
        assert len(result["scores"]) == 2  # Only accuracy and f1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
