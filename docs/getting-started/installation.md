# Installation

MLFCrafter can be installed using pip from PyPI or directly from the source.

## Requirements

- Python 3.8 or higher
- pip (Python package installer)

## Install from PyPI

```bash
pip install mlfcrafter
```

## Install from Source

If you want to install the latest development version or contribute to the project:

```bash
git clone https://github.com/brkcvlk/mlfcrafter.git
cd mlfcrafter
pip install -e .
```

## Development Installation

For development, install with development dependencies:

```bash
git clone https://github.com/brkcvlk/mlfcrafter.git
cd mlfcrafter
pip install -e ".[dev]"
```

## Verify Installation

To verify that MLFCrafter is installed correctly, run:

```python
import mlfcrafter
print(mlfcrafter.__version__)
```

## Dependencies

MLFCrafter automatically installs the following core dependencies:

- **pandas** (>=2.0.0) - Data manipulation and analysis
- **scikit-learn** (>=1.3.0) - Machine learning library
- **numpy** (>=1.24.0) - Numerical computing
- **xgboost** (>=2.0.0) - Gradient boosting framework
- **joblib** (>=1.2.0) - Lightweight pipelining

## Optional Dependencies

### Documentation

To build documentation locally:

```bash
pip install "mlfcrafter[docs]"
```

### Testing

For running tests:

```bash
pip install "mlfcrafter[test]"
```

## Troubleshooting

### Common Issues

1. **Permission errors**: Use `--user` flag or virtual environment
2. **Version conflicts**: Create a fresh virtual environment
3. **Missing dependencies**: Ensure pip is up to date

### Virtual Environment (Recommended)

We recommend using a virtual environment:

```bash
python -m venv mlfcrafter-env
source mlfcrafter-env/bin/activate  # On Windows: mlfcrafter-env\Scripts\activate
pip install mlfcrafter
```

## Next Steps

Once installed, check out the [Quick Start](quickstart.md) guide to begin building your first ML pipeline! 