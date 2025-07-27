<h1 align="center" >
    MLFCrafter
</h1>

> **ML Pipeline Automation Tool - Chain together data processing, model training, and deployment with minimal code**

[![PyPI Version](https://img.shields.io/pypi/v/mlfcrafter?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/mlfcrafter/)
[![Python Support](https://img.shields.io/pypi/pyversions/mlfcrafter?logo=python&logoColor=white)](https://pypi.org/project/mlfcrafter/)
[![Tests](https://github.com/brkcvlk/mlfcrafter/workflows/🧪%20Tests%20&%20Code%20Quality/badge.svg)](https://github.com/brkcvlk/mlfcrafter/actions)
[![Documentation](https://github.com/brkcvlk/mlfcrafter/workflows/📚%20Deploy%20Documentation/badge.svg)](https://brkcvlk.github.io/mlfcrafter/)
[![License](https://img.shields.io/github/license/brkcvlk/mlfcrafter?color=green)](LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/mlfcrafter)](https://pepy.tech/projects/mlfcrafter)
---

## ⭐ **If you find MLFCrafter useful, please consider starring this repository!**

<a href="https://github.com/brkcvlk/mlfcrafter/stargazers">
  <img src="https://img.shields.io/github/stars/brkcvlk/mlfcrafter?style=social" alt="GitHub stars">
</a>

Your support helps us continue developing and improving MLFCrafter for the ML community.

---

## What is MLFCrafter?

MLFCrafter is a Python Tool that simplifies machine learning pipeline creation through chainable "crafter" components. Build, train, and deploy ML models with minimal code and maximum flexibility.

## Key Features

- **🔗 Chainable Architecture** - Connect multiple processing steps seamlessly
- **📊 Smart Data Handling** - Automatic data ingestion from CSV, Excel, JSON
- **🧹 Intelligent Cleaning** - Multiple strategies for missing value handling  
- **📏 Flexible Scaling** - MinMax, Standard, and Robust scaling options
- **🤖 Multiple Models** - Random Forest, XGBoost, Logistic Regression support
- **📈 Comprehensive Metrics** - Accuracy, Precision, Recall, F1-Score
- **💾 Easy Deployment** - One-click model saving with metadata
- **🔄 Context-Based** - Seamless data flow between pipeline steps

## Quick Start

### Installation

```bash
pip install mlfcrafter
```

### Basic Usage

```python
from mlfcrafter import MLFChain, DataIngestCrafter, CleanerCrafter, ScalerCrafter, ModelCrafter, ScorerCrafter, DeployCrafter

# Create ML pipeline in one line
chain = MLFChain(
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
chain = MLFChain(
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
chain = MLFChain()
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

## Requirements

- **Python**: 3.8 or higher
- **Core Dependencies**: pandas, scikit-learn, numpy, xgboost, joblib

## Development

### Setup Development Environment

```bash
git clone https://github.com/brkcvlk/mlfcrafter.git
cd mlfcrafter
pip install -r requirements-dev.txt
pip install -e .
```

### Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run tests with coverage  
python -m pytest tests/ -v --cov=mlfcrafter --cov-report=html

# Check code quality
ruff check .

# Auto-fix code issues
ruff check --fix .

# Format code
ruff format .
```

### Run Examples

```bash
python example.py
```

## Documentation

Complete documentation is available at [MLFCrafter Docs](https://brkcvlk.github.io/mlfcrafter/)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📖 **Documentation**: [MLFCrafter Docs](https://brkcvlk.github.io/mlfcrafter/)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/brkcvlk/mlfcrafter/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/brkcvlk/mlfcrafter/discussions)

---

**Made for the ML Community** 
