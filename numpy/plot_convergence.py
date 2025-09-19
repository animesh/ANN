import numpy as np
import matplotlib.pyplot as plt

def run_neural_network_with_history(iterations, use_sequential=True):
    """
    Run neural network and return error history
    
    Args:
        iterations: Number of training iterations
        use_sequential: If True, use sequential updates; if False, use parallel updates
    
    Returns:
        list: Error values for each iteration
    """
    # Initialize network parameters
    x = np.array([0.05, 0.10])
    w1 = np.array([[0.15, 0.20], [0.25, 0.3]])
    w2 = np.array([[0.4, 0.45], [0.5, 0.55]])
    y = np.array([0.01, 0.99])
    bias = np.array([0.35, 0.6])
    lr = 0.5
    
    errors = []
    
    for iteration in range(iterations):
        # Forward pass
        h = 1/(1+np.exp(-(x.dot(w1.T)+bias[0])))
        y_pred = 1/(1+np.exp(-(h.dot(w2.T)+bias[1])))
        error = 0.5*np.square(y_pred - y).sum()
        errors.append(error)
        
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
    
    return errors

def get_no_loop_error():
    """Get the single iteration error from no-loop version"""
    # This matches the no-loop implementation exactly
    x = np.array([0.05, 0.10])
    w1 = np.array([[0.15, 0.20], [0.25, 0.3]])
    w2 = np.array([[0.4, 0.45], [0.5, 0.55]])
    y = np.array([0.01, 0.99])
    bias = np.array([0.35, 0.6])
    lr = 0.5
    
    # Initial error
    h = 1/(1+np.exp(-(x.dot(w1.T)+bias[0])))
    y_pred = 1/(1+np.exp(-(h.dot(w2.T)+bias[1])))
    initial_error = 0.5*np.square(y_pred - y).sum()
    
    # After weight updates (parallel approach)
    w2_new = w2 - lr*np.outer((y_pred - y)*(1-y_pred)*y_pred, h)
    w1_new = w1 - lr*np.outer(w2.T.dot((y_pred - y)*(1-y_pred)*y_pred)*h*(1-h), x)
    
    # Final error
    h_final = 1/(1+np.exp(-(x.dot(w1_new.T)+bias[0])))
    y_pred_final = 1/(1+np.exp(-(h_final.dot(w2_new.T)+bias[1])))
    final_error = 0.5*np.square(y_pred_final - y).sum()
    
    return initial_error, final_error

def create_convergence_plot():
    """Create convergence plot comparing all three approaches"""
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Test different iteration ranges for better visualization
    iterations_short = 50  # For detailed view of early convergence
    iterations_long = 1000  # For full convergence view
    
    # Get error histories
    print("Generating error histories...")
    sequential_errors_short = run_neural_network_with_history(iterations_short, use_sequential=True)
    parallel_errors_short = run_neural_network_with_history(iterations_short, use_sequential=False)
    
    sequential_errors_long = run_neural_network_with_history(iterations_long, use_sequential=True)
    parallel_errors_long = run_neural_network_with_history(iterations_long, use_sequential=False)
    
    # Get no-loop single iteration result
    no_loop_initial, no_loop_final = get_no_loop_error()
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Neural Network Error Convergence Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: First 50 iterations (linear scale)
    ax1.plot(range(1, iterations_short + 1), sequential_errors_short, 'b-', linewidth=2, label='Sequential Updates')
    ax1.plot(range(1, iterations_short + 1), parallel_errors_short, 'r--', linewidth=2, label='Parallel Updates')
    ax1.scatter([1], [no_loop_initial], color='green', s=100, marker='o', label='No-Loop Initial', zorder=5)
    ax1.scatter([1], [no_loop_final], color='green', s=100, marker='s', label='No-Loop Final', zorder=5)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Error')
    ax1.set_title('Error Convergence - First 50 Iterations (Linear Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: First 50 iterations (log scale)
    ax2.semilogy(range(1, iterations_short + 1), sequential_errors_short, 'b-', linewidth=2, label='Sequential Updates')
    ax2.semilogy(range(1, iterations_short + 1), parallel_errors_short, 'r--', linewidth=2, label='Parallel Updates')
    ax2.scatter([1], [no_loop_initial], color='green', s=100, marker='o', label='No-Loop Initial', zorder=5)
    ax2.scatter([1], [no_loop_final], color='green', s=100, marker='s', label='No-Loop Final', zorder=5)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Error (log scale)')
    ax2.set_title('Error Convergence - First 50 Iterations (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Full 1000 iterations (linear scale)
    ax3.plot(range(1, iterations_long + 1), sequential_errors_long, 'b-', linewidth=2, label='Sequential Updates')
    ax3.plot(range(1, iterations_long + 1), parallel_errors_long, 'r--', linewidth=2, label='Parallel Updates')
    ax3.scatter([1], [no_loop_initial], color='green', s=100, marker='o', label='No-Loop Initial', zorder=5)
    ax3.scatter([1], [no_loop_final], color='green', s=100, marker='s', label='No-Loop Final', zorder=5)
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Error')
    ax3.set_title('Error Convergence - 1000 Iterations (Linear Scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Full 1000 iterations (log scale)
    ax4.semilogy(range(1, iterations_long + 1), sequential_errors_long, 'b-', linewidth=2, label='Sequential Updates')
    ax4.semilogy(range(1, iterations_long + 1), parallel_errors_long, 'r--', linewidth=2, label='Parallel Updates')
    ax4.scatter([1], [no_loop_initial], color='green', s=100, marker='o', label='No-Loop Initial', zorder=5)
    ax4.scatter([1], [no_loop_final], color='green', s=100, marker='s', label='No-Loop Final', zorder=5)
    ax4.set_xlabel('Iterations')
    ax4.set_ylabel('Error (log scale)')
    ax4.set_title('Error Convergence - 1000 Iterations (Log Scale)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('numpy/convergence_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'numpy/convergence_comparison.png'")
    
    # Show final statistics
    print(f"\nFinal Results after 1000 iterations:")
    print(f"Sequential Updates: {sequential_errors_long[-1]:.10f}")
    print(f"Parallel Updates:   {parallel_errors_long[-1]:.10f}")
    print(f"No-Loop (1 iter):   {no_loop_final:.10f}")
    print(f"Difference (Seq-Par): {sequential_errors_long[-1] - parallel_errors_long[-1]:.2e}")
    
    # Show the plot
    plt.show()

def create_simple_comparison_plot():
    """Create a simpler comparison plot focusing on key differences"""
    
    plt.figure(figsize=(10, 6))
    
    # Get error histories for 100 iterations
    iterations = 100
    sequential_errors = run_neural_network_with_history(iterations, use_sequential=True)
    parallel_errors = run_neural_network_with_history(iterations, use_sequential=False)
    
    # Get no-loop result
    no_loop_initial, no_loop_final = get_no_loop_error()
    
    # Plot the convergence
    plt.semilogy(range(1, iterations + 1), sequential_errors, 'b-', linewidth=2.5, label='Sequential Updates (Loop)')
    plt.semilogy(range(1, iterations + 1), parallel_errors, 'r--', linewidth=2.5, label='Parallel Updates (Original)')
    
    # Add no-loop points
    plt.scatter([0.8], [no_loop_initial], color='green', s=120, marker='o', label='No-Loop Initial', zorder=5)
    plt.scatter([1.2], [no_loop_final], color='green', s=120, marker='s', label='No-Loop After 1 Iter', zorder=5)
    
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Error (log scale)', fontsize=12)
    plt.title('Neural Network Error Convergence Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add text annotations
    plt.text(50, 0.1, f'Final Sequential: {sequential_errors[-1]:.6f}', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    plt.text(50, 0.05, f'Final Parallel: {parallel_errors[-1]:.6f}', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('numpy/simple_convergence.png', dpi=300, bbox_inches='tight')
    print("Simple plot saved as 'numpy/simple_convergence.png'")
    plt.show()

if __name__ == "__main__":
    print("Creating convergence comparison plots...")
    
    try:
        # Create comprehensive plot
        create_convergence_plot()
        
        # Create simple plot
        create_simple_comparison_plot()
        
    except ImportError:
        print("Error: matplotlib is required for plotting.")
        print("Install it with: pip install matplotlib")
        print("\nAlternatively, here are the numerical results:")
        
        # Show numerical comparison instead
        iterations = 100
        sequential_errors = run_neural_network_with_history(iterations, use_sequential=True)
        parallel_errors = run_neural_network_with_history(iterations, use_sequential=False)
        no_loop_initial, no_loop_final = get_no_loop_error()
        
        print(f"\nError progression (first 10 iterations):")
        print(f"{'Iter':<5} {'Sequential':<12} {'Parallel':<12} {'Difference':<12}")
        print("-" * 45)
        for i in range(min(10, len(sequential_errors))):
            diff = sequential_errors[i] - parallel_errors[i]
            print(f"{i+1:<5} {sequential_errors[i]:<12.8f} {parallel_errors[i]:<12.8f} {diff:<12.2e}")
        
        print(f"\nNo-Loop version:")
        print(f"Initial error: {no_loop_initial:.8f}")
        print(f"After 1 iter:  {no_loop_final:.8f}")
        
        print(f"\nFinal errors after {iterations} iterations:")
        print(f"Sequential: {sequential_errors[-1]:.8f}")
        print(f"Parallel:   {parallel_errors[-1]:.8f}")