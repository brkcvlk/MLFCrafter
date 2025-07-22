# Installation

FlowCraft can be installed using pip from PyPI or directly from the source.

## Requirements

- Python 3.8 or higher
- pip (Python package installer)

## Install from PyPI

```bash
pip install flowcraft
```

## Install from Source

If you want to install the latest development version or contribute to the project:

```bash
git clone https://github.com/brkcvlk/flowcraft.git
cd flowcraft
pip install -e .
```

## Development Installation

For development, install with development dependencies:

```bash
git clone https://github.com/brkcvlk/flowcraft.git
cd flowcraft
pip install -e ".[dev]"
```

## Verify Installation

To verify that FlowCraft is installed correctly, run:

```python
import flowcraft
print(flowcraft.__version__)
```

## Dependencies

FlowCraft automatically installs the following core dependencies:

- **pandas** (>=2.0.0) - Data manipulation and analysis
- **scikit-learn** (>=1.3.0) - Machine learning library
- **numpy** (>=1.24.0) - Numerical computing
- **xgboost** (>=2.0.0) - Gradient boosting framework
- **joblib** (>=1.2.0) - Lightweight pipelining

## Optional Dependencies

### Documentation

To build documentation locally:

```bash
pip install "flowcraft[docs]"
```

### Testing

For running tests:

```bash
pip install "flowcraft[test]"
```

## Troubleshooting

### Common Issues

1. **Permission errors**: Use `--user` flag or virtual environment
2. **Version conflicts**: Create a fresh virtual environment
3. **Missing dependencies**: Ensure pip is up to date

### Virtual Environment (Recommended)

We recommend using a virtual environment:

```bash
python -m venv flowcraft-env
source flowcraft-env/bin/activate  # On Windows: flowcraft-env\Scripts\activate
pip install flowcraft
```

## Next Steps

Once installed, check out the [Quick Start](quickstart.md) guide to begin building your first ML pipeline! 