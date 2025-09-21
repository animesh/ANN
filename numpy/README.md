# NumPy Neural Network Implementation

NumPy-based implementations of a 2-layer neural network with manual backpropagation.

## Files

### Core Implementations
- `ann_numpy.py` - Sequential weight updates (default NumPy approach)
- `ann_numpy_original.py` - Parallel weight updates (matches autodiff behavior)
- `ann_numpy_no_loop.py` - Step-by-step implementation (2 iterations only)

### Analysis Tools
- `convergence_test.py` - Generate comparison tables
- `analyze_convergence.py` - Comprehensive numerical analysis
- `plot_convergence.py` - Error convergence plots
- `plot_all_variables.py` - Variable evolution tracking
- `plot_key_differences.py` - Sequential vs parallel comparison

## Network Specification

- **Input**: [0.05, 0.10]
- **Hidden**: 2 neurons, sigmoid activation
- **Output**: 2 neurons, sigmoid activation
- **Target**: [0.01, 0.99]
- **Learning Rate**: 0.5

## Usage

```bash
# Sequential updates (default)
python ann_numpy.py 10

# Parallel updates
python ann_numpy_original.py 10

# Step-by-step (2 iterations)
python ann_numpy_no_loop.py

# Generate analysis
python convergence_test.py
python analyze_convergence.py

# Create visualizations
python plot_convergence.py
python plot_all_variables.py
python plot_key_differences.py
```

## Key Differences

### Sequential Updates (`ann_numpy.py`)
- Updates w2 first, then w1 using updated w2
- Slightly faster convergence
- Final loss (10 iter): 0.2288071136

### Parallel Updates (`ann_numpy_original.py`)
- Updates all weights simultaneously using original values
- Matches automatic differentiation behavior
- Final loss (10 iter): 0.2288087405

### Difference: 1.63e-6 (sequential vs parallel)

## Installation

```bash
pip install numpy matplotlib
```

## Generated Files

Analysis scripts create visualization files:
- `simple_convergence.png`
- `convergence_comparison.png`
- `key_differences.png`
- `all_variables_evolution.png`
- Various other analysis plots

Based on [Matt Mazur's backpropagation tutorial](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)