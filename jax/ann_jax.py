#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from jax import grad, jit
import numpy as np

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class SimpleNeuralNetwork:
    
    def __init__(self, seed=42):
        self.key = jax.random.PRNGKey(seed)
        
        self.w1 = jnp.array([[0.15, 0.20], [0.25, 0.30]])
        self.w2 = jnp.array([[0.40, 0.45], [0.50, 0.55]])
        
        self.bias = jnp.array([0.35, 0.60])
        
        self.x = jnp.array([0.05, 0.10])
        self.y_target = jnp.array([0.01, 0.99])
        
        self.learning_rate = 0.5
        
        self.loss_history = []
        self.weight_history = []
    
    def forward(self):
        self.h = sigmoid(jnp.dot(self.x, self.w1.T) + self.bias[0])
        self.y_pred = sigmoid(jnp.dot(self.h, self.w2.T) + self.bias[1])
        return self.h, self.y_pred
    
    def compute_loss(self, y_pred):
        return 0.5 * jnp.sum((y_pred - self.y_target) ** 2)
    
    def compute_gradients_manual(self, h, y_pred):
        output_error = (y_pred - self.y_target) * (1 - y_pred) * y_pred
        w2_gradient = jnp.outer(output_error, h)
        hidden_error = jnp.dot(self.w2.T, output_error) * h * (1 - h)
        w1_gradient = jnp.outer(hidden_error, self.x)
        return w1_gradient, w2_gradient
    
    def update_weights_parallel(self, w1_gradient, w2_gradient):
        w1_old = self.w1
        w2_old = self.w2
        self.w1 = w1_old - self.learning_rate * w1_gradient
        self.w2 = w2_old - self.learning_rate * w2_gradient
    
    def update_weights_sequential(self, h, y_pred):
        output_error = (y_pred - self.y_target) * (1 - y_pred) * y_pred
        w2_gradient = jnp.outer(output_error, h)
        self.w2 = self.w2 - self.learning_rate * w2_gradient
        
        hidden_error_updated = jnp.dot(self.w2.T, output_error) * h * (1 - h)
        w1_gradient_updated = jnp.outer(hidden_error_updated, self.x)
        self.w1 = self.w1 - self.learning_rate * w1_gradient_updated
    
    def train_manual_parallel(self, epochs=10, verbose=True):
        if verbose:
            print("Training with JAX Manual Parallel Updates:")
            print("=" * 50)
        
        for epoch in range(epochs):
            h, y_pred = self.forward()
            loss = self.compute_loss(y_pred)
            self.loss_history.append(float(loss))
            
            self.weight_history.append({
                'w1': np.array(self.w1),
                'w2': np.array(self.w2)
            })
            
            if verbose:
                print(f"Iteration {epoch + 1}: Error = {loss:.10f}")
            
            w1_gradient, w2_gradient = self.compute_gradients_manual(h, y_pred)
            self.update_weights_parallel(w1_gradient, w2_gradient)
        
        return self.loss_history
    
    def train_manual_sequential(self, epochs=10, verbose=True):
        if verbose:
            print("Training with JAX Manual Sequential Updates:")
            print("=" * 50)
        
        for epoch in range(epochs):
            h, y_pred = self.forward()
            loss = self.compute_loss(y_pred)
            self.loss_history.append(float(loss))
            
            self.weight_history.append({
                'w1': np.array(self.w1),
                'w2': np.array(self.w2)
            })
            
            if verbose:
                print(f"Iteration {epoch + 1}: Error = {loss:.10f}")
            
            self.update_weights_sequential(h, y_pred)
        
        return self.loss_history

def create_jax_auto_network():
    
    def init_params():
        return {
            'w1': jnp.array([[0.15, 0.20], [0.25, 0.30]]),
            'w2': jnp.array([[0.40, 0.45], [0.50, 0.55]])
        }
    
    def forward_fn(params, x, bias):
        h = sigmoid(jnp.dot(x, params['w1'].T) + bias[0])
        y_pred = sigmoid(jnp.dot(h, params['w2'].T) + bias[1])
        return y_pred
    
    def loss_fn(params, x, y_target, bias):
        y_pred = forward_fn(params, x, bias)
        return 0.5 * jnp.sum((y_pred - y_target) ** 2)
    
    grad_fn = grad(loss_fn)
    
    return init_params, forward_fn, loss_fn, grad_fn

def train_jax_auto(epochs=10, learning_rate=0.5, verbose=True):
    if verbose:
        print("Training with JAX Automatic Differentiation:")
        print("=" * 50)
    
    init_params, forward_fn, loss_fn, grad_fn = create_jax_auto_network()
    params = init_params()
    
    x = jnp.array([0.05, 0.10])
    y_target = jnp.array([0.01, 0.99])
    bias = jnp.array([0.35, 0.60])
    
    loss_history = []
    weight_history = []
    
    for epoch in range(epochs):
        y_pred = forward_fn(params, x, bias)
        loss = loss_fn(params, x, y_target, bias)
        loss_history.append(float(loss))
        
        weight_history.append({
            'w1': np.array(params['w1']),
            'w2': np.array(params['w2'])
        })
        
        if verbose:
            print(f"Iteration {epoch + 1}: Error = {loss:.10f}")
        
        grads = grad_fn(params, x, y_target, bias)
        
        params = {
            'w1': params['w1'] - learning_rate * grads['w1'],
            'w2': params['w2'] - learning_rate * grads['w2']
        }
    
    return loss_history, weight_history, params

def main():
    print("JAX Neural Network Implementation")
    print("=" * 50)
    
    print("Input: [0.05, 0.10]")
    print("Target: [0.01, 0.99]")
    print()
    
    print("1. Manual Parallel Updates")
    print("-" * 30)
    nn_parallel = SimpleNeuralNetwork()
    loss_parallel = nn_parallel.train_manual_parallel(epochs=10)
    print()
    
    print("2. Manual Sequential Updates")
    print("-" * 30)
    nn_sequential = SimpleNeuralNetwork()
    loss_sequential = nn_sequential.train_manual_sequential(epochs=10)
    print()
    
    print("3. Automatic Differentiation")
    print("-" * 30)
    loss_auto, weights_auto, final_params = train_jax_auto(epochs=10)
    print()
    
    print("Final Comparison:")
    print("=" * 50)
    
    h_parallel, pred_parallel = nn_parallel.forward()
    h_sequential, pred_sequential = nn_sequential.forward()
    
    init_params, forward_fn, _, _ = create_jax_auto_network()
    x = jnp.array([0.05, 0.10])
    bias = jnp.array([0.35, 0.60])
    pred_auto = forward_fn(final_params, x, bias)
    
    print(f"Manual Parallel Final:   Loss = {loss_parallel[-1]:.10f}")
    print(f"Manual Sequential Final: Loss = {loss_sequential[-1]:.10f}")
    print(f"Automatic Diff Final:    Loss = {loss_auto[-1]:.10f}")
    
    print(f"\nExpected Output: [0.01, 0.99]")
    print(f"Manual Parallel Pred:   {pred_parallel}")
    print(f"Manual Sequential Pred: {pred_sequential}")
    print(f"Automatic Diff Pred:    {pred_auto}")
    
    print(f"\nDifferences:")
    print(f"Auto vs Manual Parallel: {abs(loss_auto[-1] - loss_parallel[-1]):.2e}")
    print(f"Auto vs Manual Sequential: {abs(loss_auto[-1] - loss_sequential[-1]):.2e}")
    print(f"Manual Parallel vs Sequential: {abs(loss_parallel[-1] - loss_sequential[-1]):.2e}")
    
    return {
        'manual_parallel': loss_parallel,
        'manual_sequential': loss_sequential,
        'automatic': loss_auto
    }

if __name__ == "__main__":
    main()