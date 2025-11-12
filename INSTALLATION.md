# Installation Guide for Q-Investment

This guide covers how to install and set up the Q-Investment package using modern Python tools.

## Prerequisites

- Python 3.8 or higher
- `uv` (recommended) or `pip`

## Installing uv (Recommended)

`uv` is a fast Python package installer and resolver. Install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or on macOS with Homebrew:

```bash
brew install uv
```

## Installation Options

### Option 1: Using uv (Recommended)

#### Basic Installation (Qmod only)

```bash
# Clone the repository
git clone https://github.com/llorracc/Q-Investment.git
cd Q-Investment

# Create a virtual environment
uv venv

# Install the package in editable mode
uv pip install -e .
```

#### With Development Tools

To include pytest, jupyter, and other development tools:

```bash
uv pip install -e ".[dev]"
```

#### With Dolo Support

To include Dolo support (required for 4 out of 5 example notebooks):

```bash
uv pip install -e ".[dolo]"
```

This installs the tested and working configuration for Dolo 0.4.9.12 with all compatible dependencies.

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/llorracc/Q-Investment.git
cd Q-Investment

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .

# Or with dev tools
pip install -e ".[dev]"
```

## Verifying Installation

### Test Qmod Installation

```python
from Qmod import Qmod

# Create a model instance
model = Qmod()
print(f"✅ Qmod imported successfully! Beta={model.beta}, Alpha={model.alpha}")

# Solve the model
model.solve()
print(f"✅ Model solved! Steady state capital: {model.kSS:.4f}")
```

### Test Dolo Installation (if installed)

```python
from dolo import yaml_import

# Load the Q-model
model = yaml_import("Dolo/Q-model.yaml")
print("✅ Dolo imported successfully!")
```

### Run Tests

```bash
# Run the test suite
pytest tests/

# Run with coverage
pytest tests/ --cov=Qmod --cov-report=html
```

### Run Notebooks

```bash
# Start Jupyter
jupyter notebook

# Navigate to Examples/ folder and open any notebook
# Start with: Qmod-basic-features.ipynb
```

## Package Structure

```
Q-Investment/
├── Qmod/                  # Main package
│   ├── __init__.py
│   └── q-investment.py    # Qmod class implementation
├── Dolo/                  # Dolo model definition
│   └── Q-model.yaml
├── Examples/              # Jupyter notebooks
├── tests/                 # Test suite
├── pyproject.toml         # Package configuration
└── README.md
```

## Common Issues

### Issue: `ModuleNotFoundError: No module named 'Qmod'`

**Solution**: Make sure you've installed the package with `-e` flag:
```bash
uv pip install -e .
```

### Issue: Dolo import fails

**Solution**: Install the Dolo dependencies:
```bash
uv pip install -e ".[dolo]"
```

The package includes tested versions that are known to work together.

### Issue: Tests fail with numerical warnings

**Solution**: Some warnings during model solving are expected and are handled gracefully by the code. As long as tests pass, the warnings can be ignored.

## Updating the Package

If you've made changes to the code:

```bash
# No need to reinstall with -e flag
# Just import the updated module

# To run tests after changes:
pytest tests/
```

## Uninstalling

```bash
uv pip uninstall q-investment
# or
pip uninstall q-investment
```

## Additional Resources

- [Project Repository](https://github.com/llorracc/Q-Investment)
- [Lecture Notes](http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/Investment/qModel/)
- [Binder (Interactive Notebooks)](https://mybinder.org/v2/gh/llorracc/Q-Investment/master)

## Development Workflow

For contributors:

```bash
# 1. Clone and install in dev mode
git clone https://github.com/llorracc/Q-Investment.git
cd Q-Investment
uv venv
uv pip install -e ".[dev]"

# 2. Make changes to the code

# 3. Run tests
pytest tests/

# 4. Run notebooks to verify
jupyter notebook Examples/

# 5. Commit and push
git add .
git commit -m "Your commit message"
git push
```

