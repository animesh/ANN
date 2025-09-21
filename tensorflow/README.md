# TensorFlow/Keras Neural Network Implementation

TensorFlow and Keras implementations using automatic differentiation and custom loss functions.

## Files

### Core Implementations
- `ann_tensorflow.py` - TensorFlow with MSE loss (built-in scaling)
- `ann_keras.py` - Keras with custom loss function (matches NumPy)
- `compare_implementations.py` - TensorFlow vs Keras comparison
- `plot_tensorflow_results.py` - Visualization tools
- `test_installation.py` - Installation verification

## Network Specification

- **Input**: [0.05, 0.10]
- **Hidden**: 2 neurons, sigmoid activation
- **Output**: 2 neurons, sigmoid activation
- **Target**: [0.01, 0.99]
- **Learning Rate**: 0.5

## Usage

```bash
# Test installation
python test_installation.py

# TensorFlow implementation
python ann_tensorflow.py

# Keras implementation
python ann_keras.py

# Compare both
python compare_implementations.py

# Generate plots
python plot_tensorflow_results.py
```

## Key Differences

### TensorFlow (`ann_tensorflow.py`)
- Uses built-in MSE loss: `tf.reduce_mean(tf.square(y_true - y_pred))`
- Automatic differentiation with parallel updates
- Final loss (10 iter): 0.2342312336
- **Significantly different** due to MSE loss scaling (2x gradient effect)

### Keras (`ann_keras.py`)
- Uses custom loss: `0.5 * tf.reduce_sum(tf.square(y_true - y_pred))`
- Matches NumPy loss function exactly
- Final loss (10 iter): 0.2288087308
- **Very close to NumPy/PyTorch** (9.7e-9 difference from NumPy parallel)

## Results Comparison

| Implementation | Final Loss (10 iter) | Difference from NumPy Sequential |
|---------------|---------------------|----------------------------------|
| **TensorFlow** | 0.2342312336 | 5.42e-3 (MSE scaling) |
| **Keras** | 0.2288087308 | 1.62e-6 (precision) |
| **NumPy Sequential** | 0.2288071136 | Reference |
| **NumPy Parallel** | 0.2288087405 | 1.63e-6 |

## Installation

```bash
pip install tensorflow keras numpy matplotlib
# or
pip install -r requirements.txt
```

## Key Insights

1. **TensorFlow MSE loss** creates 2x gradient scaling effect
2. **Keras custom loss** can match NumPy behavior exactly
3. **Automatic differentiation** naturally implements parallel updates
4. **Framework choice** affects numerical results even with same algorithms
5. **Loss function implementation** is critical for cross-framework consistency