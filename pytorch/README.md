# PyTorch Neural Network Implementation

PyTorch implementations with automatic differentiation and manual gradient computation.

## Files

### Core Implementations
- `ann_pytorch.py` - Main implementation with 3 methods:
  - Automatic differentiation (parallel updates)
  - Manual parallel updates
  - Manual sequential updates
- `compare_pytorch_implementations.py` - Compare all PyTorch methods
- `detailed_comparison.py` - Cross-framework comparison (all 6 frameworks)
- `test_pytorch_installation.py` - Installation verification

## Network Specification

- **Input**: [0.05, 0.10]
- **Hidden**: 2 neurons, sigmoid activation
- **Output**: 2 neurons, sigmoid activation
- **Target**: [0.01, 0.99]
- **Learning Rate**: 0.5

## Usage

```bash
# Test installation
python test_pytorch_installation.py

# Run all 3 methods
python ann_pytorch.py

# Run specific method
python ann_pytorch.py automatic
python ann_pytorch.py manual_parallel
python ann_pytorch.py manual_sequential

# Compare PyTorch methods
python compare_pytorch_implementations.py

# Compare all frameworks
python detailed_comparison.py
```

## Methods

### 1. Automatic Differentiation
- Uses PyTorch's `autograd` system
- Parallel weight updates (natural autodiff behavior)
- Final loss (10 iter): 0.2288087457

### 2. Manual Parallel
- Manual gradient computation
- Parallel weight updates (matches automatic)
- Final loss (10 iter): 0.2288087457

### 3. Manual Sequential
- Manual gradient computation
- Sequential weight updates (matches NumPy sequential)
- Final loss (10 iter): 0.2288071513

## Key Results

- **Automatic = Manual Parallel**: 0.0 difference (perfect match)
- **Sequential vs Parallel**: 1.59e-6 difference
- **PyTorch vs NumPy**: Within floating-point precision (1e-8 to 1e-6)

## Installation

```bash
pip install torch torchvision numpy matplotlib
# or
pip install -r requirements.txt
```

## Cross-Framework Comparison

The `detailed_comparison.py` script compares all 12 implementations across 6 frameworks:
- NumPy (3 implementations)
- PyTorch (3 implementations)
- TensorFlow (1 implementation)
- Keras (1 implementation)
- JAX (3 implementations)
- C# (1 implementation)

Results show PyTorch matches NumPy behavior exactly when using the same algorithms.