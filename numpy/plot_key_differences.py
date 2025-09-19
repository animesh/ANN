import numpy as np
import matplotlib.pyplot as plt

def run_neural_network_detailed(iterations, use_sequential=True):
    """Run neural network with detailed tracking of key variables"""
    x = np.array([0.05, 0.10])
    w1 = np.array([[0.15, 0.20], [0.25, 0.3]])
    w2 = np.array([[0.4, 0.45], [0.5, 0.55]])
    y = np.array([0.01, 0.99])
    bias = np.array([0.35, 0.6])
    lr = 0.5
    
    history = {
        'errors': [],
        'predictions': [],
        'weight_norms': {'w1': [], 'w2': []},
        'weight_changes': {'w1': [], 'w2': []},
        'gradients': {'w1': [], 'w2': []}
    }
    
    w1_initial = w1.copy()
    w2_initial = w2.copy()
    
    for iteration in range(iterations):
        # Forward pass
        h = 1/(1+np.exp(-(x.dot(w1.T)+bias[0])))
        y_pred = 1/(1+np.exp(-(h.dot(w2.T)+bias[1])))
        error = 0.5*np.square(y_pred - y).sum()
        
        # Calculate gradients (before weight updates)
        grad_w2 = lr*np.outer((y_pred - y)*(1-y_pred)*y_pred, h)
        grad_w1 = lr*np.outer(w2.T.dot((y_pred - y)*(1-y_pred)*y_pred)*h*(1-h), x)
        
        # Store values
        history['errors'].append(error)
        history['predictions'].append(y_pred.copy())
        history['weight_norms']['w1'].append(np.linalg.norm(w1))
        history['weight_norms']['w2'].append(np.linalg.norm(w2))
        history['weight_changes']['w1'].append(np.linalg.norm(w1 - w1_initial))
        history['weight_changes']['w2'].append(np.linalg.norm(w2 - w2_initial))
        history['gradients']['w1'].append(np.linalg.norm(grad_w1))
        history['gradients']['w2'].append(np.linalg.norm(grad_w2))
        
        # Weight updates
        if use_sequential:
            w2 = w2 - grad_w2
            # Recalculate w1 gradient with updated w2
            grad_w1_updated = lr*np.outer(w2.T.dot((y_pred - y)*(1-y_pred)*y_pred)*h*(1-h), x)
            w1 = w1 - grad_w1_updated
        else:
            w2 = w2 - grad_w2
            w1 = w1 - grad_w1
    
    return history

def plot_key_differences():
    """Create focused plots showing the most important differences"""
    
    iterations = 100
    print("Analyzing key differences between approaches...")
    
    seq_history = run_neural_network_detailed(iterations, use_sequential=True)
    par_history = run_neural_network_detailed(iterations, use_sequential=False)
    
    # Create the plot
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Key Differences: Sequential vs Parallel Weight Updates', 
                 fontsize=16, fontweight='bold')
    
    iterations_range = range(1, iterations + 1)
    target = np.array([0.01, 0.99])
    
    # Plot 1: Error comparison
    ax = axes[0, 0]
    ax.semilogy(iterations_range, seq_history['errors'], 'b-', linewidth=2, label='Sequential')
    ax.semilogy(iterations_range, par_history['errors'], 'r--', linewidth=2, label='Parallel')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Error (log scale)')
    ax.set_title('Error Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Prediction accuracy for output 1
    ax = axes[0, 1]
    seq_pred_0 = [pred[0] for pred in seq_history['predictions']]
    par_pred_0 = [pred[0] for pred in par_history['predictions']]
    ax.plot(iterations_range, seq_pred_0, 'b-', linewidth=2, label='Sequential')
    ax.plot(iterations_range, par_pred_0, 'r--', linewidth=2, label='Parallel')
    ax.axhline(y=target[0], color='green', linestyle=':', linewidth=2, label='Target (0.01)')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Prediction y[0]')
    ax.set_title('Output 1 Convergence to Target')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Prediction accuracy for output 2
    ax = axes[0, 2]
    seq_pred_1 = [pred[1] for pred in seq_history['predictions']]
    par_pred_1 = [pred[1] for pred in par_history['predictions']]
    ax.plot(iterations_range, seq_pred_1, 'b-', linewidth=2, label='Sequential')
    ax.plot(iterations_range, par_pred_1, 'r--', linewidth=2, label='Parallel')
    ax.axhline(y=target[1], color='green', linestyle=':', linewidth=2, label='Target (0.99)')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Prediction y[1]')
    ax.set_title('Output 2 Convergence to Target')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Weight norm evolution w1
    ax = axes[1, 0]
    ax.plot(iterations_range, seq_history['weight_norms']['w1'], 'b-', linewidth=2, label='Sequential')
    ax.plot(iterations_range, par_history['weight_norms']['w1'], 'r--', linewidth=2, label='Parallel')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('||w1|| (Frobenius norm)')
    ax.set_title('w1 Weight Matrix Norm Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Weight norm evolution w2
    ax = axes[1, 1]
    ax.plot(iterations_range, seq_history['weight_norms']['w2'], 'b-', linewidth=2, label='Sequential')
    ax.plot(iterations_range, par_history['weight_norms']['w2'], 'r--', linewidth=2, label='Parallel')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('||w2|| (Frobenius norm)')
    ax.set_title('w2 Weight Matrix Norm Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Cumulative weight changes
    ax = axes[1, 2]
    ax.plot(iterations_range, seq_history['weight_changes']['w1'], 'b-', linewidth=2, label='Sequential w1')
    ax.plot(iterations_range, par_history['weight_changes']['w1'], 'r--', linewidth=2, label='Parallel w1')
    ax.plot(iterations_range, seq_history['weight_changes']['w2'], 'b:', linewidth=2, label='Sequential w2')
    ax.plot(iterations_range, par_history['weight_changes']['w2'], 'r:', linewidth=2, label='Parallel w2')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('||w - w_initial||')
    ax.set_title('Cumulative Weight Changes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 7: Gradient magnitudes w1
    ax = axes[2, 0]
    ax.semilogy(iterations_range, seq_history['gradients']['w1'], 'b-', linewidth=2, label='Sequential')
    ax.semilogy(iterations_range, par_history['gradients']['w1'], 'r--', linewidth=2, label='Parallel')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('||∇w1|| (log scale)')
    ax.set_title('w1 Gradient Magnitude Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Gradient magnitudes w2
    ax = axes[2, 1]
    ax.semilogy(iterations_range, seq_history['gradients']['w2'], 'b-', linewidth=2, label='Sequential')
    ax.semilogy(iterations_range, par_history['gradients']['w2'], 'r--', linewidth=2, label='Parallel')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('||∇w2|| (log scale)')
    ax.set_title('w2 Gradient Magnitude Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 9: Prediction error from target
    ax = axes[2, 2]
    seq_errors_from_target = [np.linalg.norm(pred - target) for pred in seq_history['predictions']]
    par_errors_from_target = [np.linalg.norm(pred - target) for pred in par_history['predictions']]
    ax.semilogy(iterations_range, seq_errors_from_target, 'b-', linewidth=2, label='Sequential')
    ax.semilogy(iterations_range, par_errors_from_target, 'r--', linewidth=2, label='Parallel')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('||y_pred - y_target|| (log scale)')
    ax.set_title('Distance from Target')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('numpy/key_differences.png', dpi=300, bbox_inches='tight')
    print("Key differences plot saved as 'numpy/key_differences.png'")
    
    # Print numerical summary
    print(f"\n" + "="*60)
    print("KEY DIFFERENCES SUMMARY")
    print("="*60)
    
    print(f"\nFinal Error (after {iterations} iterations):")
    print(f"  Sequential: {seq_history['errors'][-1]:.8f}")
    print(f"  Parallel:   {par_history['errors'][-1]:.8f}")
    print(f"  Improvement: {(par_history['errors'][-1] - seq_history['errors'][-1])/par_history['errors'][-1]*100:.4f}% better (Sequential)")
    
    print(f"\nFinal Predictions:")
    seq_final = seq_history['predictions'][-1]
    par_final = par_history['predictions'][-1]
    print(f"  Target:     [{target[0]:.6f}, {target[1]:.6f}]")
    print(f"  Sequential: [{seq_final[0]:.6f}, {seq_final[1]:.6f}]")
    print(f"  Parallel:   [{par_final[0]:.6f}, {par_final[1]:.6f}]")
    
    seq_target_error = np.linalg.norm(seq_final - target)
    par_target_error = np.linalg.norm(par_final - target)
    print(f"\nDistance from Target:")
    print(f"  Sequential: {seq_target_error:.6f}")
    print(f"  Parallel:   {par_target_error:.6f}")
    print(f"  Improvement: {(par_target_error - seq_target_error)/par_target_error*100:.4f}% better (Sequential)")
    
    print(f"\nFinal Weight Norms:")
    print(f"  w1 Sequential: {seq_history['weight_norms']['w1'][-1]:.6f}")
    print(f"  w1 Parallel:   {par_history['weight_norms']['w1'][-1]:.6f}")
    print(f"  w2 Sequential: {seq_history['weight_norms']['w2'][-1]:.6f}")
    print(f"  w2 Parallel:   {par_history['weight_norms']['w2'][-1]:.6f}")
    
    print(f"\nTotal Weight Changes:")
    print(f"  w1 Sequential: {seq_history['weight_changes']['w1'][-1]:.6f}")
    print(f"  w1 Parallel:   {par_history['weight_changes']['w1'][-1]:.6f}")
    print(f"  w2 Sequential: {seq_history['weight_changes']['w2'][-1]:.6f}")
    print(f"  w2 Parallel:   {par_history['weight_changes']['w2'][-1]:.6f}")

if __name__ == "__main__":
    print("Creating key differences analysis...")
    
    try:
        plot_key_differences()
        print(f"\nAnalysis complete! Check 'numpy/key_differences.png' for detailed comparison.")
        
    except ImportError:
        print("Error: matplotlib is required for plotting.")
        print("Install it with: pip install matplotlib")
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        
        # Fallback numerical analysis
        print("Generating numerical analysis instead...")
        seq_history = run_neural_network_detailed(100, use_sequential=True)
        par_history = run_neural_network_detailed(100, use_sequential=False)
        
        target = np.array([0.01, 0.99])
        seq_final = seq_history['predictions'][-1]
        par_final = par_history['predictions'][-1]
        
        print(f"\nFinal Results (100 iterations):")
        print(f"Sequential Error: {seq_history['errors'][-1]:.8f}")
        print(f"Parallel Error:   {par_history['errors'][-1]:.8f}")
        print(f"Sequential Pred:  [{seq_final[0]:.6f}, {seq_final[1]:.6f}]")
        print(f"Parallel Pred:    [{par_final[0]:.6f}, {par_final[1]:.6f}]")
        print(f"Target:           [{target[0]:.6f}, {target[1]:.6f}]")