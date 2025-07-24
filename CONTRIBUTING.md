# 🤝 Contributing to MLFCrafter

Thank you for your interest in contributing to **MLFCrafter**! This document outlines how to contribute to our ML pipeline automation framework.

## 🚀 **Quick Start**

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/mlfcrafter.git`
3. **Set up** development environment
4. **Create** a feature branch
5. **Make** your changes
6. **Submit** a pull request

---

## 🛠️ **Development Setup**

### **Prerequisites**
- Python 3.9+ 
- Git
- Virtual environment (recommended)

### **Installation**
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/mlfcrafter.git
cd mlfcrafter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt
```

### **Verify Installation**
```bash
python -c "import mlfcrafter; print('✅ MLFCrafter imported successfully')"
python -m pytest tests/ -v
```

---

## 📝 **How to Contribute**

### **🐛 Bug Reports**
Use our **issue template** when reporting bugs:

- **Clear title** and description
- **Steps to reproduce** the issue
- **Expected vs actual** behavior
- **Environment info** (Python version, OS)
- **Code snippet** that reproduces the bug

### **✨ New Features**
Before working on new features:

1. **Check existing issues** - might already be planned
2. **Open a discussion** - describe your idea first
3. **Get feedback** from maintainers
4. **Follow architecture** - maintain chainable crafter pattern

### **📚 Documentation**
Documentation contributions are always welcome:

- **API docs** - docstring improvements
- **User guides** - tutorials and examples
- **README** updates
- **Code comments** for complex logic

---

## 🏗️ **Development Guidelines**

### **Code Style**
We use **Ruff** for linting and formatting:

```bash
# Format code
ruff format .

# Check linting
ruff check .

# Fix auto-fixable issues
ruff check . --fix
```

**Style Guidelines:**
- **PEP 8** compliance
- **Type hints** for public methods
- **Descriptive variable names**
- **Maximum line length**: 88 characters

### **Testing**
All contributions must include tests:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_mlfcrafter.py -v

# Run with coverage
python -m pytest tests/ --cov=mlfcrafter --cov-report=html
```

**Testing Requirements:**
- **Unit tests** for new functions/classes
- **Integration tests** for complete workflows
- **Edge case testing** for error handling
- **Coverage**: Aim for >80%

### **Commit Messages**
Follow **Conventional Commits** standard:

```bash
feat: add AutoMLCrafter for automated model selection
fix: resolve memory leak in DataIngestCrafter  
docs: update installation guide with conda instructions
test: add integration tests for scorer workflow
refactor: improve chain execution performance
ci: update GitHub Actions to latest versions
```

**Format:**
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

---

## 🔀 **Pull Request Process**

### **Before Submitting**
- [ ] **Tests pass**: `python -m pytest tests/ -v`
- [ ] **Linting clean**: `ruff check .`
- [ ] **Formatting applied**: `ruff format .`
- [ ] **Documentation updated** (if needed)
- [ ] **Changelog entry** added (if significant)

### **PR Template**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature  
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### **Review Process**
1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Feedback addressed** and discussion resolved
4. **Final approval** and merge

---

## 🏛️ **Architecture Guidelines**

### **Crafter Pattern**
All new components should follow the crafter pattern:

```python
from mlfcrafter.crafters.base_crafter import BaseCrafter

class NewCrafter(BaseCrafter):
    def __init__(self, param1: str, param2: int = 10):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def craft(self, context: dict) -> dict:
        """Main crafting logic here"""
        # Your implementation
        return context
```

**Design Principles:**
- **Single Responsibility** - Each crafter does one thing well
- **Chainability** - Works with MLFChain seamlessly  
- **Context Passing** - Maintains data flow between crafters
- **Error Handling** - Graceful failure with meaningful messages

### **Adding New Crafters**

1. **Create crafter file**: `mlfcrafter/crafters/your_crafter.py`
2. **Implement BaseCrafter**: Follow the pattern above
3. **Add to __init__.py**: Export in `mlfcrafter/crafters/__init__.py`
4. **Write tests**: Create `tests/test_your_crafter.py`
5. **Add documentation**: Update API docs and examples

---

## 🎯 **Areas for Contribution**

### **🔥 High Priority**
- **AutoML integration** (Auto-Sklearn, TPOT)
- **Deep learning crafters** (TensorFlow, PyTorch)
- **More deployment options** (Docker, Kubernetes)
- **Data validation** crafters
- **Feature engineering** automation

### **📊 Medium Priority**  
- **Visualization crafters** (plots, reports)
- **Model explainability** integration
- **Database connectors** (PostgreSQL, MongoDB)
- **Cloud storage** support (S3, GCP)
- **Performance optimizations**

### **📚 Documentation**
- **Tutorial videos** creation
- **Blog posts** and examples
- **Kaggle notebook** templates
- **API reference** improvements

---

## 💬 **Community**

### **Getting Help**
- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - General questions and ideas
- **Documentation** - Check our comprehensive guides

### **Code of Conduct**
Be respectful, inclusive, and constructive. We follow the **Contributor Covenant**.

---

## 🎉 **Recognition**

Contributors will be:
- **Listed** in our contributors section
- **Credited** in release notes  
- **Highlighted** in documentation (for significant contributions)
- **Invited** to maintainer discussions (for ongoing contributors)

---

## 📞 **Contact**

- **Maintainer**: [@brkcvlk](https://github.com/brkcvlk)
- **Issues**: [GitHub Issues](https://github.com/brkcvlk/mlfcrafter/issues)
- **Discussions**: [GitHub Discussions](https://github.com/brkcvlk/mlfcrafter/discussions)

---

**🚀 Ready to contribute? Fork the repo and let's build the future of ML automation together!** 