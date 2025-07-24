import logging
from typing import Dict, Optional

from sklearn.model_selection import train_test_split

# Setup logger for this crafter
logger = logging.getLogger("mlfcrafter.ModelCrafter")


class ModelCrafter:
    """
    Machine Learning Model Training Crafter.

    This crafter handles model selection, training, and evaluation. It supports
    multiple algorithms and provides comprehensive training results including
    train/test splits, predictions, and performance scores.

    Parameters:
        model_name (str): Machine learning algorithm to use. Options:
            - "random_forest": RandomForestClassifier - ensemble method, good baseline
            - "logistic_regression": LogisticRegression - linear classifier
            - "xgboost": XGBClassifier - gradient boosting, often high performance

        model_params (dict): Hyperparameters for the selected model
            - Default: {} (use sklearn defaults)
            - Examples:
              * RandomForest: {"n_estimators": 100, "max_depth": 10}
              * XGBoost: {"learning_rate": 0.1, "max_depth": 6}
              * LogisticRegression: {"C": 1.0, "max_iter": 1000}

        test_size (float): Proportion of data for testing (default: 0.2)

        random_state (int): Seed for reproducible results (default: 61)

        stratify (bool): Whether to maintain class proportions in train/test split
            - True: Recommended for imbalanced datasets (default)
            - False: Simple random split

    Context Input:
        - data (pd.DataFrame): Prepared dataset (required)
        - target_column (str): Name of target variable column (required)

    Context Output:
        - model (sklearn.base.BaseEstimator): Trained model object
        - X_train, X_test (pd.DataFrame): Feature splits
        - y_train, y_test (pd.Series): Target splits
        - y_pred (np.array): Predictions on test set
        - train_score (float): Training accuracy
        - test_score (float): Test accuracy
        - model_name (str): Name of algorithm used
        - features (list): List of feature column names

    Example Usage:
        # Basic Random Forest
        model = ModelCrafter(model_name="random_forest")

        # Tuned XGBoost
        model = ModelCrafter(
            model_name="xgboost",
            model_params={
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 6
            },
            test_size=0.25
        )

        # Logistic Regression with custom parameters
        model = ModelCrafter(
            model_name="logistic_regression",
            model_params={"C": 0.5, "max_iter": 2000},
            stratify=False
        )

    Workflow:
        1. Extract features (X) and target (y) from dataset
        2. Split data into train/test sets with optional stratification
        3. Initialize and train the specified model
        4. Generate predictions on test set
        5. Calculate training and testing accuracy scores
        6. Store all results in context for downstream use

    Supported Tasks:
        - Binary classification
        - Multi-class classification
        - Automatic handling of categorical targets
    """

    def __init__(
        self,
        model_name: str = "random_forest",
        model_params: Optional[Dict] = None,
        test_size: float = 0.2,
        random_state: int = 61,
        stratify: bool = True,
    ):
        self.model_name = model_name
        self.model_params = model_params if model_params is not None else {}
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    def run(self, context: dict) -> dict:
        """
        Train model using data and target from context
        Args:
            context: Pipeline context dict with 'data' and 'target_column' keys
        Returns:
            Updated context with model training results
        """
        logger.info("Starting model training...")
        logger.info(f"Model: {self.model_name}")
        logger.debug(f"Model parameters: {self.model_params}")
        logger.debug(f"Test size: {self.test_size}, Random state: {self.random_state}")

        if "data" not in context:
            logger.error("No data found in context")
            raise ValueError("No data found in context. Run DataIngestCrafter first.")
        if "target_column" not in context or context["target_column"] is None:
            logger.error("No target column specified")
            raise ValueError("target_column must be specified in context")

        df = context["data"]
        target_column = context["target_column"]

        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Target column: {target_column}")

        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in dataset")
            raise ValueError(f"Target column '{target_column}' not found in data")

        # Prepare features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        logger.info(f"Features: {X.shape[1]} columns")
        logger.info(f"Target distribution: {dict(y.value_counts())}")

        # Handle stratification
        stratify_param = y if self.stratify else None
        if self.stratify:
            logger.debug("Using stratified train-test split")
        else:
            logger.debug("Using simple random train-test split")

        # Split data
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param,
        )

        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        # Select and train model
        logger.info("Initializing model...")
        model = self._select_model()

        logger.info("Training model...")
        model.fit(X_train, y_train)

        # Calculate scores
        logger.info("Evaluating model performance...")
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        logger.info(f"Training accuracy: {train_score:.4f}")
        logger.info(f"Test accuracy: {test_score:.4f}")

        # Make predictions
        logger.debug("Generating predictions on test set...")
        y_pred = model.predict(X_test)

        # Update context with all training results
        context.update(
            {
                "model": model,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "y_pred": y_pred,
                "train_score": train_score,
                "test_score": test_score,
                "model_name": self.model_name,
                "features": list(X.columns),
            }
        )

        logger.info("Model training completed successfully")

        return context

    def _select_model(self):
        if self.model_name == "random_forest":
            logger.debug("Initializing RandomForestClassifier")
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(**self.model_params)

        elif self.model_name == "logistic_regression":
            logger.debug("Initializing LogisticRegression")
            from sklearn.linear_model import LogisticRegression

            # Default params to avoid convergence warnings
            default_params = {"max_iter": 5000, "solver": "liblinear"}
            default_params.update(self.model_params)
            return LogisticRegression(**default_params)

        elif self.model_name == "xgboost":
            logger.debug("Initializing XGBClassifier")
            from xgboost import XGBClassifier

            return XGBClassifier(**self.model_params)

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
