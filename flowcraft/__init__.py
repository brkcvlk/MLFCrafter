from .flow_chain import FlowChain
from .crafters import (
    DataIngestCrafter,
    CleanerCrafter,
    ScalerCrafter,
    ModelCrafter,
    ScorerCrafter,
    DeployCrafter
)
from .utils import setup_logger

__version__ = "0.1.0"

# Setup default logger when FlowCraft is imported
_default_logger = setup_logger()

__all__ = [
    'FlowChain',
    'DataIngestCrafter',
    'CleanerCrafter',
    'ScalerCrafter', 
    'ModelCrafter',
    'ScorerCrafter',
    'DeployCrafter',
    'setup_logger'
]
