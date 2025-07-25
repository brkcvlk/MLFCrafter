from .crafters import (
    CategoricalCrafter,
    CleanerCrafter,
    DataIngestCrafter,
    DeployCrafter,
    ModelCrafter,
    ScalerCrafter,
    ScorerCrafter,
)
from .flow_chain import MLFChain
from .utils import setup_logger

__version__ = "0.1.1"

# Setup default logger when MLFCrafter is imported
_default_logger = setup_logger()

__all__ = [
    "MLFChain",
    "DataIngestCrafter",
    "CleanerCrafter",
    "ScalerCrafter",
    "ModelCrafter",
    "ScorerCrafter",
    "DeployCrafter",
    "setup_logger",
    "CategoricalCrafter",
]
