from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Optional
import numpy as np
import logging

# Setup logger for this crafter
logger = logging.getLogger("flowcraft.ScorerCrafter")

class ScorerCrafter:
    """
    Model Scoring and Evaluation Crafter.
    
    This crafter calculates various classification performance metrics to evaluate
    the trained model's performance. It automatically handles both binary and
    multi-class classification scenarios.
    
    Parameters:
        metrics (List[str]): List of metrics to calculate. Options:
            - "accuracy": Overall accuracy (correct predictions / total predictions)
            - "precision": Positive predictive value (TP / (TP + FP))
            - "recall": Sensitivity or true positive rate (TP / (TP + FN))
            - "f1": Harmonic mean of precision and recall
            - Default: ["accuracy", "precision", "recall", "f1"] (all metrics)
            
    Context Input:
        - y_test (array-like): True labels from test set (required)
        - y_pred (array-like): Predicted labels from model (required)
        
    Context Output:
        - scores (dict): Dictionary containing calculated metrics
            * Keys: metric names (e.g., "accuracy", "precision")
            * Values: calculated scores (float)
            
    Example Usage:
        # Calculate all metrics
        scorer = ScorerCrafter()
        
        # Calculate specific metrics only
        scorer = ScorerCrafter(metrics=["accuracy", "f1"])
        
        # Just accuracy
        scorer = ScorerCrafter(metrics=["accuracy"])
        
    Supported Classification Types:
        - Binary Classification: Uses "binary" averaging for precision/recall/f1
        - Multi-class Classification: Uses "macro" averaging for precision/recall/f1
        - Automatic detection based on number of unique classes in y_test
        
    Workflow:
        1. Extract y_test and y_pred from context (from ModelCrafter)
        2. Automatically detect binary vs multi-class classification
        3. Calculate requested metrics with appropriate averaging
        4. Handle edge cases (zero division warnings)
        5. Store results in context for further analysis
        
    Metrics Explanation:
        - Accuracy: Simple ratio of correct predictions
        - Precision: Of all positive predictions, how many were correct?
        - Recall: Of all actual positives, how many were found?
        - F1-Score: Balanced measure combining precision and recall
    """
    
    def __init__(self, metrics: Optional[List[str]] = None):
        if metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1"]
        else:
            self.metrics = metrics

    def run(self, context: dict) -> dict:
        """
        Calculate performance metrics using test predictions
        Args:
            context: Pipeline context with 'y_test' and 'y_pred' keys
        Returns:
            Updated context with 'scores' dict containing calculated metrics
        """
        logger.info("Starting model scoring...")
        logger.debug(f"Metrics to calculate: {self.metrics}")
        
        # Validate required data
        if "y_test" not in context:
            logger.error("No y_test found in context")
            raise ValueError("No y_test found in context. Run ModelCrafter first.")
        if "y_pred" not in context:
            logger.error("No y_pred found in context")
            raise ValueError("No y_pred found in context. Run ModelCrafter first.")
            
        y_test = context["y_test"]
        y_pred = context["y_pred"]
        
        logger.info(f"Test set size: {len(y_test)} samples")
        
        # Determine if problem is binary or multiclass
        unique_classes = np.unique(y_test)
        n_classes = len(unique_classes)
        average_type = "binary" if n_classes == 2 else "macro"
        
        logger.info(f"Classification type: {'Binary' if n_classes == 2 else 'Multi-class'} ({n_classes} classes)")
        logger.debug(f"Classes: {unique_classes}")
        logger.debug(f"Using average type: {average_type}")

        results = {}

        # Calculate each requested metric
        if "accuracy" in self.metrics:
            accuracy = accuracy_score(y_test, y_pred)
            results["accuracy"] = accuracy
            logger.info(f"Accuracy: {accuracy:.4f}")

        if "precision" in self.metrics:
            precision = precision_score(y_test, y_pred, average=average_type, zero_division="warn")
            results["precision"] = precision
            logger.info(f"Precision: {precision:.4f}")

        if "recall" in self.metrics:
            recall = recall_score(y_test, y_pred, average=average_type, zero_division="warn")
            results["recall"] = recall
            logger.info(f"Recall: {recall:.4f}")

        if "f1" in self.metrics:
            f1 = f1_score(y_test, y_pred, average=average_type, zero_division="warn")
            results["f1"] = f1
            logger.info(f"F1-Score: {f1:.4f}")

        # Store results in context
        context["scores"] = results
        
        logger.info("Model scoring completed successfully")
        logger.debug(f"Final scores: {results}")
        
        return context
