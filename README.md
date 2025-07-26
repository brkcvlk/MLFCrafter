<h1 align="center" >
    MLFCrafter
</h1>

> **ML Pipeline Automation Framework - Chain together data processing, model training, and deployment with minimal code**

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

MLFCrafter is a Python framework that simplifies machine learning pipeline creation through chainable "crafter" components. Build, train, and deploy ML models with minimal code and maximum flexibility.

### Key Features

- **🔗 Chainable Architecture** - Connect multiple processing steps seamlessly
- **📊 Smart Data Handling** - Automatic data ingestion from CSV, Excel, JSON
- **🧹 Intelligent Cleaning** - Multiple strategies for missing value handling  
- **📏 Flexible Scaling** - MinMax, Standard, and Robust scaling options
- **🤖 Multiple Models** - Random Forest, XGBoost, Logistic Regression support
- **📈 Comprehensive Metrics** - Accuracy, Precision, Recall, F1-Score
- **💾 Easy Deployment** - One-click model saving with metadata
- **🔄 Context-Based** - Seamless data flow between pipeline steps


## Why MLFCrafter?

Writing the same ML boilerplate again and again is exhausting — especially when juggling multiple datasets or experimenting with different models. MLFCrafter was created to solve exactly that.

Here’s why MLFCrafter might be the right tool for you:

✅ **Automation without Black Box**: You automate repetitive steps, but still keep visibility and control over each stage.

✅ **Modular by Design**: You can run only the steps you need. Don't want automatic data cleaning ? Just skip `CleanerCrafter` and plug in your own function.

✅ **Readable & Reusable**: The API is simple, clean, and built for easy experimentation and reproducibility.

✅ **Scikit-learn Compatible**: Use your favorite tools and estimators within the pipeline.

✅ **Open for Extension**: You can build your own custom crafters if needed.

✅ **Easy to Learn**: MLFCrafter’s intuitive API and clear component structure make it approachable even for users with basic machine learning knowledge. You don’t need to dive deep into complex frameworks to start building.

---

## Documentation

Complete documentation is available at [MLFCrafter Docs](https://brkcvlk.github.io/mlfcrafter/)

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
