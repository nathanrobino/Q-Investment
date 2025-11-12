# Dolo Installation - Complete Working Configuration

This document describes the **tested and working** configuration for Dolo 0.4.9.12 with Python 3.10.

## Summary

After extensive debugging, **all 5 Jupyter notebooks now execute successfully** with the configuration documented here.

## Quick Install

```bash
cd Q-Investment
uv venv
uv pip install -e ".[dolo,dev]"
```

## Working Package Versions

The following versions are **tested and confirmed to work together**:

### Core Scientific Stack
- `numpy==1.23.5` (must be <1.25 for numba compatibility)
- `scipy>=1.7.0`
- `matplotlib>=3.3.0`

### Dolo and Direct Dependencies
- `dolo==0.4.9.12`
- `pandas==1.5.3` (<2.0 required)
- `numba==0.56.4` (critical: last version with `generated_jit`)
- `quantecon>=0.10.0` (>0.5 for Python 3.10 compatibility)
- `xarray==0.15.1`
- `ruamel.yaml>=0.16.10,<0.17` (older API required)
- `multipledispatch==0.6.0`
- `ipython>=7.34.0,<8.0`
- `dolang==0.0.10`
- `interpolation>=2.2.0`

## Key Compatibility Issues Solved

### 1. Numba Version
**Problem**: Dolo uses `generated_jit` which was removed in numba 0.57+  
**Solution**: Pin to `numba==0.56.4` (last version with this feature)

### 2. Numpy Version
**Problem**: numba 0.56.4 doesn't work with numpy 2.x or >=1.25  
**Solution**: Use `numpy>=1.23.5,<1.25`

### 3. Quantecon Compatibility
**Problem**: quantecon<0.5 uses `from fractions import gcd` (moved to `math` in Python 3.9+)  
**Solution**: Use `quantecon>=0.10.0` which fixes this

### 4. Ruamel.yaml API Changes
**Problem**: ruamel.yaml 0.17+ removed the old `load()` API that Dolo uses  
**Solution**: Use `ruamel.yaml>=0.16.10,<0.17`

### 5. PyYAML Build Issues
**Problem**: PyYAML 5.4.1 (Dolo's specified version) has build issues  
**Solution**: Let Dolo's dependencies be flexible; use newer PyYAML (6.x works)

## Verification

After installation, verify everything works:

```bash
# Test imports
python -c "from Qmod import Qmod; print('✅ Qmod works')"
python -c "from dolo import yaml_import; print('✅ Dolo works')"

# Test model solving
python -c "from Qmod import Qmod; m = Qmod(); m.solve(); print(f'✅ kss={m.kss:.4f}')"

# Test Dolo YAML import
python -c "from dolo import yaml_import; m = yaml_import('Dolo/Q-model.yaml'); print('✅ Dolo model loads')"

# Run all notebooks
cd Examples/
for nb in *.ipynb; do
  echo "Testing $nb..."
  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 "$nb" --output="/tmp/$nb"
done
```

## Notebook Test Results

All 5 notebooks execute successfully:

| Notebook | Status | Notes |
|----------|--------|-------|
| `Qmod-basic-features.ipynb` | ✅ Works | Qmod only, no Dolo needed |
| `Class-Figures.ipynb` | ✅ Works | Requires Dolo |
| `Problem-set-template.ipynb` | ✅ Works | Requires Dolo |
| `Dolo-simulations.ipynb` | ✅ Works | Requires Dolo |
| `Structural-changes-Qmod-Dolo.ipynb` | ✅ Works | Requires Dolo |

## Troubleshooting

### "cannot import name 'generated_jit' from 'numba'"
Your numba version is too new. Downgrade:
```bash
uv pip install "numba==0.56.4"
```

### "cannot import name 'gcd' from 'fractions'"
Your quantecon version is too old. Upgrade:
```bash
uv pip install "quantecon>=0.10.0"
```

### "load() has been removed" error from ruamel.yaml
Your ruamel.yaml version is too new. Downgrade:
```bash
uv pip install "ruamel.yaml>=0.16.10,<0.17"
```

### "Numba needs NumPy 1.26 or less"
But you have numpy 2.x installed. Downgrade:
```bash
uv pip install "numpy>=1.23.5,<1.25"
```

## Why These Specific Versions?

Dolo 0.4.9.12 was released in 2020 and has not been updated since. It was designed for:
- Python 3.7-3.8 era
- numpy 1.x
- numba 0.5x (when `generated_jit` existed)
- Old ruamel.yaml API

The configuration documented here represents the **newest possible versions** of dependencies that maintain compatibility with Dolo while working on Python 3.10.

## Alternative: Conda Environment

If you prefer using conda instead of uv, you can create an environment with:

```bash
conda create -n q-investment python=3.10
conda activate q-investment
conda install numpy=1.23.5 scipy matplotlib pandas=1.5.3 numba=0.56.4
pip install -e ".[dolo]"
```

## Future Considerations

Dolo 0.4.9.12 is unmaintained and increasingly difficult to use with modern Python. For long-term projects, consider:

1. **Focus on Qmod**: The pure Python implementation works perfectly
2. **Fork and Update Dolo**: Remove deprecated numba features
3. **Alternative Tools**: Use modern economic modeling frameworks

For this project, we've successfully made it work, but be aware these dependencies are frozen at 2020-era versions.

