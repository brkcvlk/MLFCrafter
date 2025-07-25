<div align="center">
  <h1 style="font-size: 3.5em; margin: 0.5em 0;">
    MLFCrafter
  </h1>
  <h2>🚀 ML Pipeline Automation Framework</h2>
  <p>Build, train, and deploy machine learning models with minimal code through chainable "crafter" components.</p>
</div>

---

## What is MLFCrafter?

**MLFCrafter** is a powerful Python framework that simplifies machine learning pipeline creation through a modular, chainable architecture. Instead of writing repetitive boilerplate code, you can compose sophisticated ML workflows by connecting specialized "crafters" together.

```python
from mlfcrafter import MLFChain, DataIngestCrafter, CleanerCrafter, ModelCrafter

# Create a complete ML pipeline in 4 lines
chain = MLFChain(
    DataIngestCrafter(data_path="data.csv"),
    CleanerCrafter(strategy="auto"),
    ModelCrafter(model_name="random_forest")
)

results = chain.run(target_column="target")
print(f"Model accuracy: {results['test_score']:.4f}")
```

## ✨ Key Features

### 🔗 **Chainable Architecture**
Connect multiple processing steps seamlessly with intelligent data flow between components.

### 🧠 **Smart Data Handling**
- **Auto-detection**: CSV, Excel, JSON formats automatically detected
- **Intelligent cleaning**: Multiple strategies for missing values
- **Flexible scaling**: MinMax, Standard, Robust scaling options

### 🤖 **Multiple ML Models**
- **Random Forest**: Ensemble method for robust predictions
- **XGBoost**: Gradient boosting for high performance
- **Logistic Regression**: Linear models for interpretability

### 📊 **Comprehensive Metrics**
Track model performance with accuracy, precision, recall, and F1-score automatically.

### 🚀 **Easy Deployment**
One-click model saving with metadata, scalers, and complete reproducibility.

### 🔍 **Professional Logging**
Detailed pipeline tracking with configurable log levels for debugging and monitoring.

---

## 🏗️ Architecture Overview

MLFCrafter uses a **context-based pipeline architecture** where each crafter:

1. **Receives** a context dictionary with data and metadata
2. **Processes** the data according to its specialization  
3. **Returns** an updated context for the next crafter

```
📊 DataIngestCrafter → 🧹 CleanerCrafter → CategoricalCrafter  → ⚖️ ScalerCrafter → 🤖 ModelCrafter → 📊 ScorerCrafter → 🚀 DeployCrafter
```

**Pipeline Flow:**
1. **DataIngestCrafter**: Load data from CSV/Excel/JSON files
2. **CleanerCrafter**: Handle missing values with various strategies  
3. **CategoricalCrafter**: Converts categorical variables into numerical data.
4. **ScalerCrafter**: Normalize numerical features (MinMax/Standard/Robust)
5. **ModelCrafter**: Train ML models (Random Forest, XGBoost, Logistic Regression)
6. **ScorerCrafter**: Calculate performance metrics (Accuracy, Precision, Recall, F1)
7. **DeployCrafter**: Save trained models with metadata for deployment

## 🛠️ Available Crafters

| Crafter | Purpose | Key Features |
|---------|---------|--------------|
| **[DataIngestCrafter](api/crafters/data-ingest-crafter.md)** | Data loading | CSV, Excel, JSON with auto-detection |
| **[CleanerCrafter](api/crafters/cleaner-crafter.md)** | Data cleaning | Multiple missing value strategies |
| **[CategoricalCrafter](api/crafters/categorical-crafter.md)** | Feature encoding | One-Hot Encoding, Label Encoding |
| **[ScalerCrafter](api/crafters/scaler-crafter.md)** | Feature scaling | MinMax, Standard, Robust scalers |
| **[ModelCrafter](api/crafters/model-crafter.md)** | Model training | RF, XGBoost, LogReg with hyperparameters |
| **[ScorerCrafter](api/crafters/scorer-crafter.md)** | Performance metrics | Accuracy, Precision, Recall, F1 |
| **[DeployCrafter](api/crafters/deploy-crafter.md)** | Model deployment | Joblib/Pickle with metadata |

---

## 🚀 Quick Start

### Installation

```bash
pip install mlfcrafter
```

### Your First Pipeline

```python
from mlfcrafter import *
import pandas as pd

# Load your data
chain = MLFChain(
    DataIngestCrafter(data_path="your_data.csv"),
    CleanerCrafter(strategy="mean"),           # Handle missing values
    CategoricalCrafter(encoder_type="onehot"), # Convert categorical values to numerical
    ScalerCrafter(scaler_type="standard"),     # Normalize features  
    ModelCrafter(model_name="random_forest"),  # Train model
    ScorerCrafter(),                          # Calculate metrics
    DeployCrafter(model_path="my_model.joblib") # Save model
)

# Run the entire pipeline
results = chain.run(target_column="your_target_column")

# Check results
print(f"🎯 Test Accuracy: {results['test_score']:.4f}")
print(f"📊 All Metrics: {results['scores']}")
print(f"💾 Model Saved: {results['deployment_successful']}")
```

---

## 📚 Learn More

<div class="grid cards" markdown>

-   **🚀 Getting Started**
    
    ---
    
    New to MLFCrafter? Start here with installation and your first pipeline.
    
    [→ Getting Started](getting-started/installation.md)

-   **📚 User Guide**
    
    ---
    
    Learn pipeline basics, data processing, and advanced features.
    
    [→ User Guide](user-guide/pipeline-basics.md)

-   **🔧 API Reference**
    
    ---
    
    Complete documentation of all crafters, parameters, and methods.
    
    [→ API Reference](api/mlfchain.md)

-   **💡 Examples**
    
    ---
    
    Real-world examples from basic usage to production deployments.
    
    [→ Examples](examples/basic-usage.md)

</div>

---

## 🎯 Why Choose MLFCrafter?

!!! success "Production Ready"
    MLFCrafter is built with production in mind - comprehensive logging, error handling, and testing ensure reliability.

!!! info "Developer Friendly" 
    Intuitive API design with clear documentation and helpful error messages make development fast and enjoyable.

!!! tip "Extensible"
    Create custom crafters easily by following the simple interface pattern.

!!! note "Well Tested"
    23 comprehensive tests ensure quality and reliability across all components.

---

## 🤝 Community & Support

- **GitHub**: [brkcvlk/mlfcrafter](https://github.com/brkcvlk/mlfcrafter)
- **Issues**: Report bugs or request features
- **Discussions**: Ask questions and share ideas  
- **PyPI**: [mlfcrafter package](https://pypi.org/project/mlfcrafter/)

---

*Documentation last updated: {{ git_revision_date_localized }}*

<div align="center">
  <p><strong>Ready to build your first ML pipeline?</strong></p>
  <a href="getting-started/installation/" class="md-button md-button--primary">Get Started</a>
  <a href="api/mlfchain/" class="md-button">API Reference</a>
</div>
