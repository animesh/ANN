import numpy as np
import sys

# Based on ann_numpy_no_loop.py but with loop structure
x = np.array([0.05, 0.10])
w1 = np.array([[0.15, 0.20], [0.25, 0.3]])
w2 = np.array([[0.4, 0.45], [0.5, 0.55]])
y = np.array([0.01, 0.99])
bias = np.array([0.35, 0.6])
lr = 0.5

iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 1

for iteration in range(iterations):
    # Forward pass
    h = 1/(1+np.exp(-(x.dot(w1.T)+bias[0])))
    y_pred = 1/(1+np.exp(-(h.dot(w2.T)+bias[1])))
    error = 0.5*np.square(y_pred - y).sum()
    
    # Original implementation: parallel weight updates (same as no-loop version)
    # Calculate both updates using original weights
    w2_update = lr*np.outer((y_pred - y)*(1-y_pred)*y_pred, h)
    w1_update = lr*np.outer(w2.T.dot((y_pred - y)*(1-y_pred)*y_pred)*h*(1-h), x)
    # Apply updates in parallel
    w2 = w2 - w2_update
    w1 = w1 - w1_update
    
    # Print ALL iterations
    print(f"Iteration {iteration+1}: Error = {error:.10f}")

# Calculate final prediction with updated weights (same as no-loop version)
h_final = 1/(1+np.exp(-(x.dot(w1.T)+bias[0])))
y_pred_final = 1/(1+np.exp(-(h_final.dot(w2.T)+bias[1])))
final_error = 0.5*np.square(y_pred_final - y).sum()

print(f"Expected Output:\n{y}")
print(f"Predicted Output:\n{y_pred_final}")
print(f"Final Error: {final_error}")