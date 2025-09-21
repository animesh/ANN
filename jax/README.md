# JAX Neural Network Implementation

JAX implementations using automatic differentiation and functional programming.

## Files

### Core Implementations
- `ann_jax.py` - Main implementation with 3 methods:
  - Automatic differentiation (parallel updates)
  - Manual parallel updates
  - Manual sequential updates
- `compare_jax_implementations.py` - Compare all JAX methods
- `test_jax_installation.py` - Installation verification

## Network Specification

- **Input**: [0.05, 0.10]
- **Hidden**: 2 neurons, sigmoid activation
- **Output**: 2 neurons, sigmoid activation
- **Target**: [0.01, 0.99]
- **Learning Rate**: 0.5

## Usage

```bash
# Test installation
python test_jax_installation.py

# Run all 3 methods
python ann_jax.py

# Compare JAX methods
python compare_jax_implementations.py
```

## Methods

### 1. Automatic Differentiation
- Uses JAX's `grad()` function
- Parallel weight updates (natural autodiff behavior)
- Final loss (10 iter): 0.2288087457

### 2. Manual Parallel
- Manual gradient computation
- Parallel weight updates (matches automatic)
- Final loss (10 iter): 0.2288087457

### 3. Manual Sequential
- Manual gradient computation with gradient recalculation
- Sequential weight updates (matches NumPy sequential)
- Final loss (10 iter): 0.2288071513

## Key Results

- **JAX Auto = JAX Manual Parallel**: 0.0 difference (perfect match)
- **JAX = PyTorch**: 0.0 difference (perfect match for parallel methods)
- **JAX ≈ NumPy**: 5.2e-9 difference (parallel), 3.77e-8 difference (sequential)
- **Sequential vs Parallel**: 1.59e-6 difference (consistent with other frameworks)

## Installation

```bash
pip install jax jaxlib
# For GPU support (optional):
# pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## JAX Features

- **Functional Programming**: Pure functions, no side effects
- **Immutable Arrays**: Cannot modify arrays in-place
- **Automatic Differentiation**: `grad()` for gradient computation
- **JIT Compilation**: `@jax.jit` for performance optimization
- **Perfect Reproducibility**: Identical results to PyTorch when using same algorithms