# FlowCraft - ML Pipeline Automation

**FlowCraft** is a Python library that simplifies machine learning pipeline creation through chainable "crafter" components. Build, train, and deploy ML models with minimal code.

## Features

- **Chainable Architecture**: Connect multiple processing steps seamlessly
- **Smart Data Handling**: Automatic data ingestion from CSV, Excel, JSON
- **Intelligent Cleaning**: Multiple strategies for missing value handling
- **Flexible Scaling**: MinMax, Standard, and Robust scaling options
- **Multiple Models**: Random Forest, XGBoost, Logistic Regression support
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- **Easy Deployment**: One-click model saving with metadata
- **Context-Based**: Seamless data flow between pipeline steps

## Installation

FlowCraft requires Python 3.8 or higher.

### From PyPI (Recommended)

```bash
pip install flowcraft
```

### From Source

```bash
git clone https://github.com/brkcvlk/flowcraft.git
cd flowcraft
pip install -r requirements.txt
pip install .
```

### Development Installation

For development work, you have two options:

#### Option 1: Using requirements-dev.txt (Recommended)
```bash
git clone https://github.com/brkcvlk/flowcraft.git
cd flowcraft
pip install -r requirements-dev.txt
```

#### Option 2: Using pyproject.toml optional dependencies
```bash
git clone https://github.com/brkcvlk/flowcraft.git
cd flowcraft
pip install -e ".[dev]"
```

### Dependency Groups

- **Production**: `pip install -r requirements.txt`
- **Development**: `pip install -r requirements-dev.txt` or `pip install -e ".[dev]"`  
- **Testing Only**: `pip install -e ".[test]"`
- **Documentation**: `pip install -e ".[docs]"`

### Development Commands

```bash
# Run tests
python -m pytest tests/ -v

# Run tests with coverage  
python -m pytest tests/ -v --cov=flowcraft --cov-report=html

# Check code quality
ruff check .

# Auto-fix code issues
ruff check --fix --unsafe-fixes .

# Format code
ruff format .

# Run example
python example.py
```

## Quick Start

### Basic Usage

```python
from flowcraft import FlowChain, DataIngestCrafter, CleanerCrafter, ScalerCrafter, ModelCrafter, ScorerCrafter, DeployCrafter

# Create ML pipeline in one line
chain = FlowChain(
    DataIngestCrafter(data_path="data/iris.csv"),
    CleanerCrafter(strategy="auto"),
    ScalerCrafter(scaler_type="standard"),
    ModelCrafter(model_name="random_forest"),
    ScorerCrafter(),
    DeployCrafter()
)

# Run entire pipeline
results = chain.run(target_column="species")
print(f"Test Score: {results['test_score']:.4f}")
```

### Advanced Configuration

```python
chain = FlowChain(
    DataIngestCrafter(data_path="data/titanic.csv", source_type="csv"),
    CleanerCrafter(strategy="mean", str_fill="Unknown"),
    ScalerCrafter(scaler_type="minmax", columns=["age", "fare"]),
    ModelCrafter(
        model_name="xgboost",
        model_params={"n_estimators": 200, "max_depth": 6},
        test_size=0.25
    ),
    ScorerCrafter(),
    DeployCrafter(model_path="models/titanic_model.joblib")
)

results = chain.run(target_column="survived")
```

## Components (Crafters)

### DataIngestCrafter
Loads data from various file formats:
```python
DataIngestCrafter(
    data_path="path/to/data.csv",
    source_type="auto"  # auto, csv, excel, json
)
```

### CleanerCrafter  
Handles missing values intelligently:
```python
CleanerCrafter(
    strategy="auto",    # auto, mean, median, mode, drop, constant
    str_fill="missing", # Fill value for strings
    int_fill=0.0       # Fill value for numbers
)
```

### ScalerCrafter
Scales numerical features:
```python
ScalerCrafter(
    scaler_type="standard",  # standard, minmax, robust
    columns=["age", "income"]  # Specific columns or None for all numeric
)
```

### ModelCrafter
Trains ML models:
```python
ModelCrafter(
    model_name="random_forest",  # random_forest, xgboost, logistic_regression
    model_params={"n_estimators": 100},
    test_size=0.2,
    stratify=True
)
```

### ScorerCrafter
Calculates performance metrics:
```python
ScorerCrafter(
    metrics=["accuracy", "precision", "recall", "f1"]  # Default: all metrics
)
```

### DeployCrafter
Saves trained models:
```python
DeployCrafter(
    model_path="model.joblib",
    save_format="joblib",  # joblib or pickle
    include_scaler=True,
    include_metadata=True
)
```

## Alternative Usage Patterns

### Step-by-Step Building
```python
chain = FlowChain()
chain.add_crafter(DataIngestCrafter(data_path="data.csv"))
chain.add_crafter(CleanerCrafter(strategy="median"))
chain.add_crafter(ModelCrafter(model_name="xgboost"))
results = chain.run(target_column="target")
```

### Loading Saved Models
```python
artifacts = DeployCrafter.load_model("model.joblib")
model = artifacts["model"]
metadata = artifacts["metadata"]
```

## Context Flow

FlowCraft uses a context dictionary that flows through each crafter:

```python
context = {
    "data": pandas.DataFrame,           # Current dataset
    "target_column": str,              # Target variable name
    "model": trained_model,            # Trained model object
    "scaler": fitted_scaler,          # Fitted scaler object
    "scores": dict,                   # Performance metrics
    "deployment_path": str,           # Saved model path
    # ... and more metadata
}
```

## Project Structure

```
flowcraft/
├── __init__.py
├── flow_chain.py              # Main FlowChain class
└── crafters/
    ├── __init__.py
    ├── data_ingest_crafter.py  # Data loading
    ├── cleaner_crafter.py      # Data cleaning
    ├── scaler_crafter.py       # Feature scaling
    ├── model_crafter.py        # Model training
    ├── scorer_crafter.py       # Performance metrics
    └── deploy_crafter.py       # Model deployment
```

## Use Cases

- **Rapid Prototyping**: Quick ML experiments with minimal boilerplate
- **Educational**: Learn ML pipelines step-by-step
- **Production**: Reproducible, configurable ML workflows
- **Experimentation**: Easy A/B testing of different configurations
- **Automation**: Scheduled model retraining and deployment

## Dependencies

- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- xgboost >= 2.0.0
- joblib >= 1.2.0

## Examples

Check out `example.py` for comprehensive usage examples.

## Testing

```bash
pytest tests/test_flowcraft.py -v
```

## Contributing

FlowCraft is designed to be extensible. Create custom crafters by implementing the `run(context: dict) -> dict` interface.

## License

MIT License - feel free to use in your projects.

---

*Built for the ML community* 