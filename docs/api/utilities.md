# Utilities API Reference

FlowCraft provides logging utilities to support pipeline development and debugging.

## Logging Utilities

### `setup_logger(name="flowcraft", level="INFO")`

Setup FlowCraft logger with consistent formatting.

**Parameters:**

- `name` (str): Logger name. Default: "flowcraft"
- `level` (str): Log level ("DEBUG", "INFO", "WARNING", "ERROR"). Default: "INFO"

**Returns:** `logging.Logger` - Configured logger instance

**Example:**

```python
from flowcraft import setup_logger

# Setup default logger
logger = setup_logger()

# Setup with custom name and level
logger = setup_logger(name="my_pipeline", level="DEBUG")
```

### `get_crafter_logger(crafter_name)`

Get logger for specific crafter.

**Parameters:**

- `crafter_name` (str): Name of the crafter class

**Returns:** `logging.Logger` - Logger instance for the crafter

**Example:**

```python
from flowcraft.utils import get_crafter_logger

# Get logger for custom crafter
logger = get_crafter_logger("CustomCrafter")
logger.info("Custom crafter started")
```

## Logger Configuration

The FlowCraft logging system provides:

- **Consistent formatting**: All logs use the same format with timestamps
- **Hierarchical loggers**: Each crafter gets its own logger (e.g., "flowcraft.ModelCrafter")
- **Default level**: INFO level by default
- **Console output**: Logs are written to stdout by default

### Log Format

The default log format is:
```
HH:MM:SS - logger_name - LEVEL - message
```

**Example output:**
```
15:30:45 - flowcraft.DataIngestCrafter - INFO - Starting data ingestion...
15:30:45 - flowcraft.DataIngestCrafter - INFO - Loading data from: data.csv
15:30:46 - flowcraft.CleanerCrafter - INFO - Starting data cleaning...
```

## Usage Examples

### Basic Logging Setup

```python
from flowcraft import setup_logger

# Setup with default settings (INFO level)
logger = setup_logger()

# Setup with DEBUG level for detailed output
debug_logger = setup_logger(level="DEBUG")

# Create custom logger for your application
app_logger = setup_logger(name="my_app", level="WARNING")
```

### Using Logging in Custom Code

```python
from flowcraft.utils import get_crafter_logger

# In a custom crafter or script
logger = get_crafter_logger("CustomProcessor")

logger.info("Starting custom processing")
logger.debug("Processing 1000 records")
logger.warning("Some records had missing values")
logger.error("Failed to process batch")
```

### Pipeline Logging Example

```python
from flowcraft import FlowChain, setup_logger
from flowcraft.crafters import *

# Setup logging with DEBUG level to see detailed output
setup_logger(level="DEBUG")

# Create pipeline - all crafters will log their progress
pipeline = FlowChain(
    DataIngestCrafter(data_path="data.csv"),
    CleanerCrafter(strategy="auto"),
    ModelCrafter(model_name="random_forest")
)

# Run pipeline - logs will show progress of each step
results = pipeline.run(target_column="target")

# Expected log output:
# 15:30:45 - flowcraft.FlowChain - INFO - FlowChain initialized with 3 crafters
# 15:30:45 - flowcraft.FlowChain - INFO - STARTING FLOWCRAFT PIPELINE
# 15:30:45 - flowcraft.DataIngestCrafter - INFO - Starting data ingestion...
# 15:30:45 - flowcraft.DataIngestCrafter - INFO - Loading data from: data.csv
# 15:30:46 - flowcraft.CleanerCrafter - INFO - Starting data cleaning...
# ... and so on
```

## Available Loggers

FlowCraft automatically creates loggers for each crafter:

- `flowcraft.FlowChain`: Main pipeline logger
- `flowcraft.DataIngestCrafter`: Data loading logger
- `flowcraft.CleanerCrafter`: Data cleaning logger
- `flowcraft.ScalerCrafter`: Data scaling logger
- `flowcraft.ModelCrafter`: Model training logger
- `flowcraft.ScorerCrafter`: Model scoring logger
- `flowcraft.DeployCrafter`: Model deployment logger

Each logger provides detailed information about the operations performed by its corresponding crafter.

## Complete Example

```python
from flowcraft import FlowChain, setup_logger
from flowcraft.crafters import *

# Setup detailed logging
logger = setup_logger(level="DEBUG")

# Create and run pipeline
pipeline = FlowChain(
    DataIngestCrafter(data_path="data.csv"),
    CleanerCrafter(strategy="median"),
    ScalerCrafter(scaler_type="standard"),
    ModelCrafter(model_name="random_forest"),
    ScorerCrafter(metrics=["accuracy", "f1"]),
    DeployCrafter(model_path="model.joblib")
)

# Run with comprehensive logging
results = pipeline.run(target_column="target")

# Logger will output detailed information about each step
print(f"Final accuracy: {results['test_score']:.4f}")
``` 