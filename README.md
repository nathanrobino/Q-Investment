## Qmod: A Python class implementing the Abel-Hayashi "Marginal Q" model of investment.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/llorracc/Q-Investment/master)

This implementation follows Professor Christopher D. Carroll's [lecture notes](http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/Investment/qModel/).

## Installation

For installation instructions, see [INSTALLATION.md](INSTALLATION.md).

Quick start with `uv`:
```bash
git clone https://github.com/llorracc/Q-Investment.git
cd Q-Investment
uv venv
uv pip install -e ".[dev]"
```

### 1. Qmod

The Qmod folder includes a file defining the Qmod Python class, which represents a Q-model of capital investment. The Qmod class' current functions include:
- Solution of the model, obtaining its policy rule.
- Drawing of the model's phase diagram.
- Simulation of the model's dynamics starting from a given level of capital.

### 2. Dolo

The Dolo folder implements the model using [Dolo](https://dolo.readthedocs.io/en/latest/#).

### 3. Examples

Current examples include:
- [Qmod-basic-features](https://github.com/llorracc/Q-Investment/blob/master/Examples/Qmod-basic-features.ipynb): illustrates how to use Qmod and its main functions.
- [Dolo-simulations](https://github.com/llorracc/Q-Investment/blob/master/Examples/Dolo-simulations.ipynb): uses Dolo to conduct more complicated simulation exercises that could not be easily achieved using Qmod.
- [Structural-changes-Qmod-Dolo](https://github.com/llorracc/Q-Investment/blob/master/Examples/Structural-changes-Qmod-Dolo.ipynb): solves the dynamic exercises in Professor Christopher D. Carroll's [lecture notes](http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/Investment/qModel/) using both Qmod and Dolo.
