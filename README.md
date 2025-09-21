# Cross-Framework Neural Network Comparison

Implementation of a simple 2-layer neural network across 6 machine learning frameworks to analyze numerical differences in identical algorithms.

## Network Specification

- **Architecture**: 2 inputs → 2 hidden → 2 outputs
- **Input**: [0.05, 0.10]
- **Target**: [0.01, 0.99]
- **Activation**: Sigmoid
- **Learning Rate**: 0.5
- **Iterations**: 10 (default)

## Implementations

1. **NumPy** - 3 implementations (Sequential, Parallel, No-Loop)
2. **PyTorch** - 3 implementations (Auto, Manual Parallel, Manual Sequential)
3. **TensorFlow** - 1 implementation (MSE loss)
4. **Keras** - 1 implementation (Custom loss)
5. **JAX** - 3 implementations (Auto, Manual Parallel, Manual Sequential)
6. **C#** - 1 implementation (.NET)

**Total**: 12 implementations across 6 frameworks

## Installation

```bash
# Core dependencies
pip install numpy matplotlib

# Framework-specific
pip install torch torchvision
pip install tensorflow keras
pip install jax jaxlib

# For C#: Install .NET 6.0+ SDK
```

## Running Programs

### NumPy
```bash
cd numpy
python ann_numpy.py 10                    # Sequential updates
python ann_numpy_original.py 10           # Parallel updates
python ann_numpy_no_loop.py               # Step-by-step (2 iterations)
```

### PyTorch
```bash
cd pytorch
python ann_pytorch.py                     # All 3 methods
python compare_pytorch_implementations.py # Comparison
```

### TensorFlow/Keras
```bash
cd tensorflow
python ann_tensorflow.py                  # TensorFlow implementation
python ann_keras.py                       # Keras implementation
python compare_implementations.py         # Comparison
```

### JAX
```bash
cd jax
python ann_jax.py                         # All 3 methods
python compare_jax_implementations.py     # Comparison
```

### C#
```bash
cd c#/ann
dotnet run                                # .NET CLI
# or: csc Program.cs && Program.exe
```

### Complete Comparison
```bash
cd pytorch
python detailed_comparison.py             # All frameworks comparison
```

## Key Results

- **Perfect Match**: JAX Auto/Manual Parallel = PyTorch Auto/Manual Parallel (0.0 difference)
- **Closest to NumPy**: JAX/PyTorch Manual Parallel ≈ NumPy Parallel (5.2e-9 difference)
- **Sequential vs Parallel**: Consistent 1.59e-6 difference across all frameworks
- **TensorFlow Outlier**: 5.42e-3 difference due to MSE loss scaling

## Repository Structure

```
├── numpy/          # NumPy implementations and analysis tools
├── pytorch/        # PyTorch implementations and detailed comparison
├── tensorflow/     # TensorFlow and Keras implementations
├── jax/           # JAX implementations with automatic differentiation
├── c#/            # C# .NET implementation
├── EXACT_COMPARISON_RESULTS.md  # Complete numerical analysis
└── README.md      # This file
```

## Analysis Tools

Each directory contains analysis and visualization scripts:

- **NumPy**: `convergence_test.py`, `analyze_convergence.py`, `plot_*.py`
- **TensorFlow**: `compare_implementations.py`, `plot_tensorflow_results.py`
- **PyTorch**: `compare_pytorch_implementations.py`, `detailed_comparison.py`
- **JAX**: `compare_jax_implementations.py`

## Documentation

- `EXACT_COMPARISON_RESULTS.md` - Complete numerical analysis across all frameworks
- Individual README files in each directory with specific instructions
- Based on [Matt Mazur's backpropagation tutorial](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)