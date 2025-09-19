import numpy as np

def run_neural_network(iterations, use_sequential=True):
    """
    Run neural network for specified iterations
    
    Args:
        iterations: Number of training iterations
        use_sequential: If True, use sequential updates; if False, use parallel updates
    
    Returns:
        tuple: (final_prediction, final_error)
    """
    # Initialize network parameters
    x = np.array([0.05, 0.10])
    w1 = np.array([[0.15, 0.20], [0.25, 0.3]])
    w2 = np.array([[0.4, 0.45], [0.5, 0.55]])
    y = np.array([0.01, 0.99])
    bias = np.array([0.35, 0.6])
    lr = 0.5
    
    for iteration in range(iterations):
        # Forward pass
        h = 1/(1+np.exp(-(x.dot(w1.T)+bias[0])))
        y_pred = 1/(1+np.exp(-(h.dot(w2.T)+bias[1])))
        error = 0.5*np.square(y_pred - y).sum()
        
        if use_sequential:
            # Sequential updates: update w2 first, then w1 using updated w2
            w2 = w2 - lr*np.outer((y_pred - y)*(1-y_pred)*y_pred, h)
            w1 = w1 - lr*np.outer(w2.T.dot((y_pred - y)*(1-y_pred)*y_pred)*h*(1-h), x)
        else:
            # Parallel updates: calculate both updates using original weights
            w2_update = lr*np.outer((y_pred - y)*(1-y_pred)*y_pred, h)
            w1_update = lr*np.outer(w2.T.dot((y_pred - y)*(1-y_pred)*y_pred)*h*(1-h), x)
            w2 = w2 - w2_update
            w1 = w1 - w1_update
    
    # Final forward pass to get prediction
    h_final = 1/(1+np.exp(-(x.dot(w1.T)+bias[0])))
    y_pred_final = 1/(1+np.exp(-(h_final.dot(w2.T)+bias[1])))
    final_error = 0.5*np.square(y_pred_final - y).sum()
    
    return y_pred_final, final_error

def generate_convergence_table():
    """Generate the convergence comparison table as shown in README"""
    
    # Test iterations from the README table
    test_iterations = [1, 2, 5, 10, 100, 1000, 10000]
    target = np.array([0.01, 0.99])
    
    print("# Convergence Results Table")
    print("\n## Detailed Comparison Table")
    print("\n| Iterations | Sequential Updates | Parallel Updates | Target |")
    print("|---|---|---|---|")
    
    sequential_results = []
    parallel_results = []
    
    for iterations in test_iterations:
        # Run sequential approach
        seq_pred, seq_error = run_neural_network(iterations, use_sequential=True)
        sequential_results.append((seq_pred, seq_error))
        
        # Run parallel approach  
        par_pred, par_error = run_neural_network(iterations, use_sequential=False)
        parallel_results.append((par_pred, par_error))
        
        # Format for table
        if iterations >= 1000:
            iter_str = f"**{iterations:,}**"
        else:
            iter_str = f"**{iterations}**"
            
        seq_str = f"`[{seq_pred[0]:.8f}, {seq_pred[1]:.8f}]`"
        par_str = f"`[{par_pred[0]:.8f}, {par_pred[1]:.8f}]`"
        target_str = f"`[{target[0]:.2f}, {target[1]:.2f}]`"
        
        print(f"| {iter_str} | {seq_str} | {par_str} | {target_str} |")
    
    print("\n## Error Progression")
    print("\n| Iterations | Sequential Error | Parallel Error |")
    print("|---|---|---|")
    
    for i, iterations in enumerate(test_iterations):
        seq_error = sequential_results[i][1]
        par_error = parallel_results[i][1]
        
        if iterations >= 1000:
            iter_str = f"**{iterations:,}**"
        else:
            iter_str = f"**{iterations}**"
            
        print(f"| {iter_str} | `{seq_error:.10f}` | `{par_error:.10f}` |")
    
    print("\n## Summary Statistics")
    print(f"\nTarget: {target}")
    print(f"\nAfter 10,000 iterations:")
    final_seq = sequential_results[-1][0]
    final_par = parallel_results[-1][0]
    print(f"Sequential: [{final_seq[0]:.8f}, {final_seq[1]:.8f}]")
    print(f"Parallel:   [{final_par[0]:.8f}, {final_par[1]:.8f}]")
    print(f"Sequential errors: [{abs(final_seq[0] - target[0]):.8f}, {abs(final_seq[1] - target[1]):.8f}]")
    print(f"Parallel errors:   [{abs(final_par[0] - target[0]):.8f}, {abs(final_par[1] - target[1]):.8f}]")

def compare_single_iteration():
    """Compare single iteration results with no-loop version"""
    print("\n# Single Iteration Verification")
    
    # Run single iteration with both approaches
    seq_pred, seq_error = run_neural_network(1, use_sequential=True)
    par_pred, par_error = run_neural_network(1, use_sequential=False)
    
    print(f"\nAfter 1 iteration:")
    print(f"Sequential: [{seq_pred[0]:.8f}, {seq_pred[1]:.8f}] (Error: {seq_error:.10f})")
    print(f"Parallel:   [{par_pred[0]:.8f}, {par_pred[1]:.8f}] (Error: {par_error:.10f})")
    print(f"Difference: [{abs(seq_pred[0] - par_pred[0]):.8f}, {abs(seq_pred[1] - par_pred[1]):.8f}]")

if __name__ == "__main__":
    print("Neural Network Convergence Analysis")
    print("=" * 50)
    
    generate_convergence_table()
    compare_single_iteration()
    
    print(f"\nNote: This script generates the comparison tables shown in the README.md file.")
    print(f"Both approaches converge to nearly identical results with excellent accuracy.")