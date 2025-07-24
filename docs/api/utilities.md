# Utilities API Reference

MLFCrafter provides logging utilities to support pipeline development and debugging.

## Logging Utilities

### `setup_logger(name="mlfcrafter", level="INFO")`

Setup MLFCrafter logger with consistent formatting.

**Parameters:**

- `name` (str): Logger name. Default: "mlfcrafter"
- `level` (str): Log level ("DEBUG", "INFO", "WARNING", "ERROR"). Default: "INFO"

**Returns:** `logging.Logger` - Configured logger instance

**Example:**

```python
from mlfcrafter import setup_logger

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
from mlfcrafter.utils import get_crafter_logger

# Get logger for custom crafter
logger = get_crafter_logger("CustomCrafter")
logger.info("Custom crafter started")
```

## Logger Configuration

The mlfcrafter logging system provides:

- **Consistent formatting**: All logs use the same format with timestamps
- **Hierarchical loggers**: Each crafter gets its own logger (e.g., "mlfcrafter.ModelCrafter")
- **Default level**: INFO level by default
- **Console output**: Logs are written to stdout by default

### Log Format

The default log format is:
```
HH:MM:SS - logger_name - LEVEL - message
```

**Example output:**
```
15:30:45 - mlfcrafter.DataIngestCrafter - INFO - Starting data ingestion...
15:30:45 - mlfcrafter.DataIngestCrafter - INFO - Loading data from: data.csv
15:30:46 - mlfcrafter.CleanerCrafter - INFO - Starting data cleaning...
```

## Usage Examples

### Basic Logging Setup

```python
from mlfcrafter import setup_logger

# Setup with default settings (INFO level)
logger = setup_logger()

# Setup with DEBUG level for detailed output
debug_logger = setup_logger(level="DEBUG")

# Create custom logger for your application
app_logger = setup_logger(name="my_app", level="WARNING")
```

### Using Logging in Custom Code

```python
from mlfcrafter.utils import get_crafter_logger

# In a custom crafter or script
logger = get_crafter_logger("CustomProcessor")

logger.info("Starting custom processing")
logger.debug("Processing 1000 records")
logger.warning("Some records had missing values")
logger.error("Failed to process batch")
```

### Pipeline Logging Example

```python
from mlfcrafter import MLFChain, setup_logger
from mlfcrafter.crafters import *

# Setup logging with DEBUG level to see detailed output
setup_logger(level="DEBUG")

# Create pipeline - all crafters will log their progress
pipeline = MLFChain(
    DataIngestCrafter(data_path="data.csv"),
    CleanerCrafter(strategy="auto"),
    ModelCrafter(model_name="random_forest")
)

# Run pipeline - logs will show progress of each step
results = pipeline.run(target_column="target")

# Expected log output:
# 15:30:45 - mlfcrafter.MLFChain - INFO - MLFChain initialized with 3 crafters
# 15:30:45 - mlfcrafter.MLFChain - INFO - STARTING mlfcrafter PIPELINE
# 15:30:45 - mlfcrafter.DataIngestCrafter - INFO - Starting data ingestion...
# 15:30:45 - mlfcrafter.DataIngestCrafter - INFO - Loading data from: data.csv
# 15:30:46 - mlfcrafter.CleanerCrafter - INFO - Starting data cleaning...
# ... and so on
```

## Available Loggers

mlfcrafter automatically creates loggers for each crafter:

- `mlfcrafter.MLFChain`: Main pipeline logger
- `mlfcrafter.DataIngestCrafter`: Data loading logger
- `mlfcrafter.CleanerCrafter`: Data cleaning logger
- `mlfcrafter.ScalerCrafter`: Data scaling logger
- `mlfcrafter.ModelCrafter`: Model training logger
- `mlfcrafter.ScorerCrafter`: Model scoring logger
- `mlfcrafter.DeployCrafter`: Model deployment logger

Each logger provides detailed information about the operations performed by its corresponding crafter.

## Complete Example

```python
from mlfcrafter import MLFChain, setup_logger
from mlfcrafter.crafters import *

# Setup detailed logging
logger = setup_logger(level="DEBUG")

# Create and run pipeline
pipeline = MLFChain(
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
