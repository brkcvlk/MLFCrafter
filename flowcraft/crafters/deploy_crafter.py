from datetime import datetime
import logging
import os
from pathlib import Path
import pickle
from typing import Optional

import joblib

# Setup logger for this crafter
logger = logging.getLogger("flowcraft.DeployCrafter")


class DeployCrafter:
    """
    Model Deployment Crafter for saving trained models and artifacts.

    This crafter handles the final step of ML pipeline by saving the trained model
    along with associated artifacts (scaler, metadata) to disk for future use.
    It supports multiple serialization formats and provides utilities for loading
    saved models.

    Parameters:
        model_path (Optional[str]): Path where model should be saved
            - If None: auto-generates timestamp-based filename (default)
            - If provided: uses the specified path
            - Directory will be created automatically if it doesn't exist

        save_format (str): Serialization format for saving. Options:
            - "joblib": Use joblib (recommended for sklearn models, default)
            - "pickle": Use Python's pickle module

        include_scaler (bool): Whether to include fitted scaler in artifacts
            - True: Save scaler along with model (default)
            - False: Save only the model

        include_metadata (bool): Whether to include training metadata
            - True: Save training info, scores, features, etc. (default)
            - False: Save only model and scaler

    Context Input:
        - model (sklearn.base.BaseEstimator): Trained model (required)
        - scaler (sklearn transformer): Fitted scaler (optional)
        - model_name (str): Name of the model algorithm (optional)
        - features (list): Feature column names (optional)
        - train_score, test_score (float): Model performance scores (optional)
        - target_column (str): Target variable name (optional)

    Context Output:
        - deployment_path (str): Absolute path where model was saved
        - artifacts_saved (list): List of artifact keys that were saved
        - deployment_successful (bool): Whether deployment completed successfully

    Example Usage:
        # Auto-generate filename with joblib
        deployer = DeployCrafter()

        # Custom path with pickle
        deployer = DeployCrafter(
            model_path="models/my_model.pkl",
            save_format="pickle"
        )

        # Model only (no scaler or metadata)
        deployer = DeployCrafter(
            model_path="production_model.joblib",
            include_scaler=False,
            include_metadata=False
        )

    Workflow:
        1. Validate trained model exists in context
        2. Generate model path if not provided
        3. Create output directory if needed
        4. Collect artifacts (model + optional scaler + metadata)
        5. Save artifacts using specified format
        6. Update context with deployment results

    Static Methods:
        - load_model(): Load saved model and artifacts from file
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        save_format: str = "joblib",  # "joblib" or "pickle"
        include_scaler: bool = True,
        include_metadata: bool = True,
    ):
        self.model_path = model_path
        self.save_format = save_format.lower()
        self.include_scaler = include_scaler
        self.include_metadata = include_metadata

        if self.save_format not in ["joblib", "pickle"]:
            raise ValueError("save_format must be 'joblib' or 'pickle'")

    def run(self, context: dict) -> dict:
        """
        Save trained model and related artifacts
        Args:
            context: Pipeline context dict with 'model' key
        Returns:
            Updated context with deployment info
        """
        logger.info("Starting model deployment...")
        logger.debug(f"Save format: {self.save_format}")
        logger.debug(f"Include scaler: {self.include_scaler}")
        logger.debug(f"Include metadata: {self.include_metadata}")

        if "model" not in context:
            logger.error("No trained model found in context")
            raise ValueError(
                "No trained model found in context. Run ModelCrafter first."
            )

        # Generate default path if not provided
        if self.model_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = context.get("model_name", "model")
            extension = "joblib" if self.save_format == "joblib" else "pkl"
            self.model_path = f"{model_name}_{timestamp}.{extension}"
            logger.debug(f"Auto-generated model path: {self.model_path}")
        else:
            logger.info(f"Using provided model path: {self.model_path}")

        # Ensure directory exists
        model_path_obj = Path(self.model_path)
        model_dir = model_path_obj.parent
        if str(model_dir) != "." and str(model_dir) != "":
            logger.debug(f"Creating directory: {model_dir}")
            model_dir.mkdir(parents=True, exist_ok=True)

        # Prepare artifacts to save
        artifacts = {"model": context["model"]}
        logger.info("Added model to artifacts")

        # Include scaler if available and requested
        if (
            self.include_scaler
            and "scaler" in context
            and context["scaler"] is not None
        ):
            artifacts["scaler"] = context["scaler"]
            logger.info("Added scaler to artifacts")

        # Include metadata if requested
        if self.include_metadata:
            metadata = {
                "model_name": context.get("model_name"),
                "features": context.get("features", []),
                "target_column": context.get("target_column"),
                "train_score": context.get("train_score"),
                "test_score": context.get("test_score"),
                "original_shape": context.get("original_shape"),
                "scaler_type": context.get("scaler_type"),
                "timestamp": datetime.now().isoformat(),
            }
            artifacts["metadata"] = metadata
            logger.info("Added metadata to artifacts")
            logger.debug(f"Metadata keys: {list(metadata.keys())}")

        # Save artifacts
        try:
            logger.info(f"Saving artifacts using {self.save_format} format...")
            # Convert to string path for safety
            save_path = str(self.model_path)

            if self.save_format == "joblib":
                joblib.dump(artifacts, save_path)
            else:  # pickle
                with open(save_path, "wb") as f:
                    pickle.dump(artifacts, f)

            # Update context
            context["deployment_path"] = os.path.abspath(save_path)
            context["artifacts_saved"] = list(artifacts.keys())
            context["deployment_successful"] = True

            logger.info(f"Model successfully saved to: {os.path.abspath(save_path)}")
            logger.info(f"Artifacts saved: {list(artifacts.keys())}")
            logger.info("Model deployment completed successfully")

        except Exception as e:
            logger.error(f"Model deployment failed: {str(e)}")
            context["deployment_successful"] = False
            context["deployment_error"] = str(e)
            # Don't re-raise, just set the flag
            logger.warning("Continuing pipeline execution despite deployment failure")

        return context

    @staticmethod
    def load_model(model_path: str, load_format: str = "auto"):
        """
        Load saved model and artifacts
        Args:
            model_path: Path to saved model file
            load_format: Format to use ("joblib", "pickle", or "auto")
        Returns:
            Dictionary containing loaded artifacts
        """
        logger.info(f"Loading model from: {model_path}")

        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if load_format == "auto":
            # Auto-detect based on file extension
            extension = Path(model_path).suffix.lower()
            if extension == ".joblib":
                load_format = "joblib"
            elif extension in [".pkl", ".pickle"]:
                load_format = "pickle"
            else:
                load_format = "joblib"  # Default fallback
            logger.debug(f"Auto-detected format: {load_format}")

        try:
            if load_format == "joblib":
                artifacts = joblib.load(model_path)
            else:  # pickle
                with open(model_path, "rb") as f:
                    artifacts = pickle.load(f)

            logger.info(f"Successfully loaded artifacts: {list(artifacts.keys())}")
            return artifacts

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}") from e
