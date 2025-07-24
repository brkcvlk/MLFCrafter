"""
MLFCrafter Logging System Tests
===============================

Tests for MLFCrafter's logging functionality using pytest.
"""

from io import StringIO
import logging
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from mlfcrafter import (
    CleanerCrafter,
    DataIngestCrafter,
    MLFChain,
    ModelCrafter,
    setup_logger,
)


class TestLoggingSystem:
    """Test suite for MLFCrafter logging functionality."""

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample test data for logging tests."""
        np.random.seed(42)
        data = {
            "feature1": np.random.normal(0, 1, 50),
            "feature2": np.random.normal(2, 1, 50),
            "target": np.random.choice([0, 1], 50),
        }
        df = pd.DataFrame(data)

        # Add some missing values for cleaner testing
        df.loc[5:8, "feature1"] = np.nan
        df.loc[10:12, "feature2"] = np.nan

        # Save to temporary file
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)

        return str(csv_path)

    @pytest.fixture
    def log_capture(self):
        """Capture log output for testing."""
        # Create string stream to capture logs
        log_stream = StringIO()

        # Setup logger to write to our stream
        logger = setup_logger(level="INFO")

        # Remove existing handlers and add our test handler
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        handler = logging.StreamHandler(log_stream)
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Also setup handlers for crafter loggers
        crafter_loggers = [
            "mlfcrafter.DataIngestCrafter",
            "mlfcrafter.CleanerCrafter",
            "mlfcrafter.ModelCrafter",
            "mlfcrafter.MLFChain",
        ]

        for crafter_name in crafter_loggers:
            crafter_logger = logging.getLogger(crafter_name)
            crafter_logger.setLevel(logging.INFO)
            for handler in crafter_logger.handlers[:]:
                crafter_logger.removeHandler(handler)
            crafter_logger.addHandler(handler)

        return log_stream

    def test_setup_logger(self):
        """Test logger setup functionality."""
        logger = setup_logger("test_logger", "DEBUG")

        assert logger.name == "test_logger"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) > 0

    def test_data_ingest_logging(self, sample_data, log_capture):
        """Test DataIngestCrafter logging output."""
        crafter = DataIngestCrafter(data_path=sample_data)
        context = {}

        result = crafter.run(context)

        log_output = log_capture.getvalue()

        # Check that key log messages are present
        assert "Starting data ingestion" in log_output
        assert "Loading data from:" in log_output
        assert "Data loaded successfully:" in log_output
        assert "Data ingestion completed" in log_output

        # Verify data was loaded
        assert "data" in result
        assert isinstance(result["data"], pd.DataFrame)

    def test_cleaner_logging(self, sample_data, log_capture):
        """Test CleanerCrafter logging output."""
        # First ingest data
        ingest_crafter = DataIngestCrafter(data_path=sample_data)
        context = ingest_crafter.run({})

        # Then clean it
        cleaner = CleanerCrafter(strategy="mean")
        _result = cleaner.run(context)

        log_output = log_capture.getvalue()

        # Check cleaning log messages
        assert "Starting data cleaning" in log_output
        assert "Total missing values found:" in log_output
        assert "Cleaning completed" in log_output
        assert "Final data shape:" in log_output

    def test_model_logging(self, sample_data, log_capture):
        """Test ModelCrafter logging output."""
        # Setup pipeline context
        ingest_crafter = DataIngestCrafter(data_path=sample_data)
        cleaner = CleanerCrafter(strategy="mean")

        context = ingest_crafter.run({})
        context = cleaner.run(context)
        context["target_column"] = "target"

        # Train model
        model_crafter = ModelCrafter(
            model_name="random_forest",
            model_params={"n_estimators": 5},  # Small for fast testing
        )
        model_crafter.run(context)

        log_output = log_capture.getvalue()

        # Check model training log messages
        assert "Starting model training" in log_output
        assert "Model: random_forest" in log_output
        assert "Dataset shape:" in log_output
        assert "Target column:" in log_output
        assert "Splitting data into train and test sets" in log_output
        assert "Training model" in log_output
        assert "Training accuracy:" in log_output
        assert "Test accuracy:" in log_output
        assert "Model training completed successfully" in log_output

    def test_mlfchain_logging(self, sample_data, log_capture):
        """Test MLFChain pipeline logging output."""
        chain = MLFChain(
            DataIngestCrafter(data_path=sample_data),
            CleanerCrafter(strategy="mean"),
            ModelCrafter(model_name="random_forest", model_params={"n_estimators": 5}),
        )

        chain.run(target_column="target")

        log_output = log_capture.getvalue()

        # Check MLFChain coordination messages
        assert "STARTING MLFCrafter PIPELINE" in log_output
        assert "MLFChain initialized with 3 crafters" in log_output
        assert "[1/3] Running DataIngestCrafter" in log_output
        assert "[2/3] Running CleanerCrafter" in log_output
        assert "[3/3] Running ModelCrafter" in log_output
        assert "MLFCrafter PIPELINE COMPLETED SUCCESSFULLY" in log_output

        # Verify we get individual crafter completions
        assert "DataIngestCrafter completed successfully" in log_output
        assert "CleanerCrafter completed successfully" in log_output
        assert "ModelCrafter completed successfully" in log_output

    def test_logging_with_errors(self, log_capture):
        """Test logging when errors occur."""
        # Try to run ModelCrafter without data (should fail)
        model_crafter = ModelCrafter()

        with pytest.raises(ValueError, match="No data found in context"):
            model_crafter.run({})

        log_output = log_capture.getvalue()
        assert "No data found in context" in log_output

    def test_different_log_levels(self, sample_data):
        """Test different logging levels (INFO vs DEBUG)."""
        # Test with DEBUG level
        debug_stream = StringIO()
        debug_logger = setup_logger("debug_test", "DEBUG")

        for handler in debug_logger.handlers[:]:
            debug_logger.removeHandler(handler)

        handler = logging.StreamHandler(debug_stream)
        handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        debug_logger.addHandler(handler)

        # Setup crafter logger to use our debug handler
        crafter_logger = logging.getLogger("mlfcrafter.DataIngestCrafter")
        crafter_logger.handlers = [handler]
        crafter_logger.setLevel(logging.DEBUG)

        # Run crafter
        crafter = DataIngestCrafter(data_path=sample_data)
        crafter.run({})

        debug_output = debug_stream.getvalue()

        # Should see DEBUG messages
        assert "DEBUG" in debug_output
        assert "Source type:" in debug_output or "Auto-detected" in debug_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
