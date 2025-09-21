#!/usr/bin/env python3
"""
Compare JAX implementations with other frameworks
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ann_jax import SimpleNeuralNetwork, train_jax_auto
import numpy as np

def run_jax_comparison():
    """Run JAX implementations and return results for comparison"""
    
    print("JAX Cross-Framework Comparison")
    print("=" * 50)
    
    # Manual Parallel Updates
    print("1. JAX Manual Parallel Updates")
    print("-" * 40)
    nn_parallel = SimpleNeuralNetwork()
    loss_parallel = nn_parallel.train_manual_parallel(epochs=10, verbose=True)
    print()
    
    # Manual Sequential Updates  
    print("2. JAX Manual Sequential Updates")
    print("-" * 40)
    nn_sequential = SimpleNeuralNetwork()
    loss_sequential = nn_sequential.train_manual_sequential(epochs=10, verbose=True)
    print()
    
    # Automatic Differentiation
    print("3. JAX Automatic Differentiation")
    print("-" * 40)
    loss_auto, weights_auto, final_params = train_jax_auto(epochs=10, verbose=True)
    print()
    
    # Return results in format compatible with detailed_comparison.py
    return {
        'manual_parallel': loss_parallel,
        'manual_sequential': loss_sequential, 
        'automatic': loss_auto
    }

def get_jax_iteration_results():
    """Get JAX results for iteration-by-iteration comparison"""
    
    # Manual Parallel
    nn_parallel = SimpleNeuralNetwork()
    loss_parallel = nn_parallel.train_manual_parallel(epochs=10, verbose=False)
    
    # Manual Sequential
    nn_sequential = SimpleNeuralNetwork()
    loss_sequential = nn_sequential.train_manual_sequential(epochs=10, verbose=False)
    
    # Automatic Differentiation
    loss_auto, _, _ = train_jax_auto(epochs=10, verbose=False)
    
    return {
        'manual_parallel': loss_parallel,
        'manual_sequential': loss_sequential,
        'automatic': loss_auto
    }

if __name__ == "__main__":
    results = run_jax_comparison()
    
    print("Final JAX Results Summary:")
    print("=" * 50)
    print(f"Manual Parallel Final Loss:   {results['manual_parallel'][-1]:.10f}")
    print(f"Manual Sequential Final Loss: {results['manual_sequential'][-1]:.10f}")
    print(f"Automatic Diff Final Loss:    {results['automatic'][-1]:.10f}")
    
    print(f"\nJAX Internal Differences:")
    print(f"Auto vs Manual Parallel:      {abs(results['automatic'][-1] - results['manual_parallel'][-1]):.2e}")
    print(f"Auto vs Manual Sequential:    {abs(results['automatic'][-1] - results['manual_sequential'][-1]):.2e}")
    print(f"Manual Parallel vs Sequential: {abs(results['manual_parallel'][-1] - results['manual_sequential'][-1]):.2e}")