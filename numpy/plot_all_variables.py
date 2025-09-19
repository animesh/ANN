import numpy as np
import matplotlib.pyplot as plt

def run_neural_network_full_history(iterations, use_sequential=True):
    """
    Run neural network and return complete history of all variables
    
    Returns:
        dict: Contains histories of all variables (errors, h, y_pred, w1, w2)
    """
    # Initialize network parameters
    x = np.array([0.05, 0.10])
    w1 = np.array([[0.15, 0.20], [0.25, 0.3]])
    w2 = np.array([[0.4, 0.45], [0.5, 0.55]])
    y = np.array([0.01, 0.99])
    bias = np.array([0.35, 0.6])
    lr = 0.5
    
    # History storage
    history = {
        'errors': [],
        'h': [],  # Hidden layer outputs
        'y_pred': [],  # Network predictions
        'w1': [],  # Input-to-hidden weights
        'w2': []   # Hidden-to-output weights
    }
    
    for iteration in range(iterations):
        # Forward pass
        h = 1/(1+np.exp(-(x.dot(w1.T)+bias[0])))
        y_pred = 1/(1+np.exp(-(h.dot(w2.T)+bias[1])))
        error = 0.5*np.square(y_pred - y).sum()
        
        # Store current values
        history['errors'].append(error)
        history['h'].append(h.copy())
        history['y_pred'].append(y_pred.copy())
        history['w1'].append(w1.copy())
        history['w2'].append(w2.copy())
        
        # Weight updates
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
    
    return history

def plot_variable_evolution():
    """Create comprehensive plots showing evolution of all variables"""
    
    # Get histories for both approaches
    iterations = 50  # Use 50 iterations for clear visualization
    print("Generating variable histories...")
    
    seq_history = run_neural_network_full_history(iterations, use_sequential=True)
    par_history = run_neural_network_full_history(iterations, use_sequential=False)
    
    # Create the comprehensive plot
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Neural Network Variable Evolution Comparison\n(Sequential vs Parallel Updates)', 
                 fontsize=16, fontweight='bold')
    
    # Define colors
    seq_color = 'blue'
    par_color = 'red'
    target_color = 'green'
    
    # Plot 1: Error evolution
    ax1 = plt.subplot(4, 4, 1)
    plt.semilogy(range(1, iterations + 1), seq_history['errors'], 
                color=seq_color, linewidth=2, label='Sequential')
    plt.semilogy(range(1, iterations + 1), par_history['errors'], 
                color=par_color, linewidth=2, linestyle='--', label='Parallel')
    plt.xlabel('Iterations')
    plt.ylabel('Error (log scale)')
    plt.title('Error Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Hidden layer output h[0]
    ax2 = plt.subplot(4, 4, 2)
    seq_h0 = [h[0] for h in seq_history['h']]
    par_h0 = [h[0] for h in par_history['h']]
    plt.plot(range(1, iterations + 1), seq_h0, color=seq_color, linewidth=2, label='Sequential')
    plt.plot(range(1, iterations + 1), par_h0, color=par_color, linewidth=2, linestyle='--', label='Parallel')
    plt.xlabel('Iterations')
    plt.ylabel('Hidden Output h[0]')
    plt.title('Hidden Layer Output h[0]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Hidden layer output h[1]
    ax3 = plt.subplot(4, 4, 3)
    seq_h1 = [h[1] for h in seq_history['h']]
    par_h1 = [h[1] for h in par_history['h']]
    plt.plot(range(1, iterations + 1), seq_h1, color=seq_color, linewidth=2, label='Sequential')
    plt.plot(range(1, iterations + 1), par_h1, color=par_color, linewidth=2, linestyle='--', label='Parallel')
    plt.xlabel('Iterations')
    plt.ylabel('Hidden Output h[1]')
    plt.title('Hidden Layer Output h[1]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Network prediction y_pred[0]
    ax4 = plt.subplot(4, 4, 4)
    seq_y0 = [y[0] for y in seq_history['y_pred']]
    par_y0 = [y[0] for y in par_history['y_pred']]
    plt.plot(range(1, iterations + 1), seq_y0, color=seq_color, linewidth=2, label='Sequential')
    plt.plot(range(1, iterations + 1), par_y0, color=par_color, linewidth=2, linestyle='--', label='Parallel')
    plt.axhline(y=0.01, color=target_color, linestyle=':', linewidth=2, label='Target')
    plt.xlabel('Iterations')
    plt.ylabel('Prediction y_pred[0]')
    plt.title('Network Prediction y_pred[0]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Network prediction y_pred[1]
    ax5 = plt.subplot(4, 4, 5)
    seq_y1 = [y[1] for y in seq_history['y_pred']]
    par_y1 = [y[1] for y in par_history['y_pred']]
    plt.plot(range(1, iterations + 1), seq_y1, color=seq_color, linewidth=2, label='Sequential')
    plt.plot(range(1, iterations + 1), par_y1, color=par_color, linewidth=2, linestyle='--', label='Parallel')
    plt.axhline(y=0.99, color=target_color, linestyle=':', linewidth=2, label='Target')
    plt.xlabel('Iterations')
    plt.ylabel('Prediction y_pred[1]')
    plt.title('Network Prediction y_pred[1]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plots 6-9: w1 weights (2x2 matrix)
    w1_titles = ['w1[0,0] (Input 0→Hidden 0)', 'w1[0,1] (Input 1→Hidden 0)', 
                 'w1[1,0] (Input 0→Hidden 1)', 'w1[1,1] (Input 1→Hidden 1)']
    
    for i in range(2):
        for j in range(2):
            ax = plt.subplot(4, 4, 6 + i*2 + j)
            seq_w1_ij = [w[i,j] for w in seq_history['w1']]
            par_w1_ij = [w[i,j] for w in par_history['w1']]
            plt.plot(range(1, iterations + 1), seq_w1_ij, color=seq_color, linewidth=2, label='Sequential')
            plt.plot(range(1, iterations + 1), par_w1_ij, color=par_color, linewidth=2, linestyle='--', label='Parallel')
            plt.xlabel('Iterations')
            plt.ylabel(f'Weight w1[{i},{j}]')
            plt.title(w1_titles[i*2 + j])
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # Plots 10-13: w2 weights (2x2 matrix)
    w2_titles = ['w2[0,0] (Hidden 0→Output 0)', 'w2[0,1] (Hidden 1→Output 0)', 
                 'w2[1,0] (Hidden 0→Output 1)', 'w2[1,1] (Hidden 1→Output 1)']
    
    for i in range(2):
        for j in range(2):
            ax = plt.subplot(4, 4, 10 + i*2 + j)
            seq_w2_ij = [w[i,j] for w in seq_history['w2']]
            par_w2_ij = [w[i,j] for w in par_history['w2']]
            plt.plot(range(1, iterations + 1), seq_w2_ij, color=seq_color, linewidth=2, label='Sequential')
            plt.plot(range(1, iterations + 1), par_w2_ij, color=par_color, linewidth=2, linestyle='--', label='Parallel')
            plt.xlabel('Iterations')
            plt.ylabel(f'Weight w2[{i},{j}]')
            plt.title(w2_titles[i*2 + j])
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # Plots 14-16: Weight differences
    ax14 = plt.subplot(4, 4, 14)
    w1_diffs = []
    for i in range(iterations):
        diff = np.linalg.norm(seq_history['w1'][i] - par_history['w1'][i])
        w1_diffs.append(diff)
    plt.semilogy(range(1, iterations + 1), w1_diffs, color='purple', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('||w1_seq - w1_par|| (log)')
    plt.title('w1 Weight Difference (Frobenius Norm)')
    plt.grid(True, alpha=0.3)
    
    ax15 = plt.subplot(4, 4, 15)
    w2_diffs = []
    for i in range(iterations):
        diff = np.linalg.norm(seq_history['w2'][i] - par_history['w2'][i])
        w2_diffs.append(diff)
    plt.semilogy(range(1, iterations + 1), w2_diffs, color='orange', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('||w2_seq - w2_par|| (log)')
    plt.title('w2 Weight Difference (Frobenius Norm)')
    plt.grid(True, alpha=0.3)
    
    ax16 = plt.subplot(4, 4, 16)
    pred_diffs = []
    for i in range(iterations):
        diff = np.linalg.norm(seq_history['y_pred'][i] - par_history['y_pred'][i])
        pred_diffs.append(diff)
    plt.semilogy(range(1, iterations + 1), pred_diffs, color='brown', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('||y_pred_seq - y_pred_par|| (log)')
    plt.title('Prediction Difference (Euclidean Norm)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('numpy/all_variables_evolution.png', dpi=300, bbox_inches='tight')
    print("Comprehensive plot saved as 'numpy/all_variables_evolution.png'")
    
    return seq_history, par_history

def plot_weight_trajectories():
    """Create 3D trajectory plots for weight evolution"""
    
    iterations = 100
    seq_history = run_neural_network_full_history(iterations, use_sequential=True)
    par_history = run_neural_network_full_history(iterations, use_sequential=False)
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Weight Evolution Trajectories (3D)', fontsize=16, fontweight='bold')
    
    # Plot w1 weight trajectories
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    seq_w1_00 = [w[0,0] for w in seq_history['w1']]
    seq_w1_01 = [w[0,1] for w in seq_history['w1']]
    seq_w1_10 = [w[1,0] for w in seq_history['w1']]
    
    par_w1_00 = [w[0,0] for w in par_history['w1']]
    par_w1_01 = [w[0,1] for w in par_history['w1']]
    par_w1_10 = [w[1,0] for w in par_history['w1']]
    
    ax1.plot(seq_w1_00, seq_w1_01, seq_w1_10, 'b-', linewidth=2, label='Sequential')
    ax1.plot(par_w1_00, par_w1_01, par_w1_10, 'r--', linewidth=2, label='Parallel')
    ax1.set_xlabel('w1[0,0]')
    ax1.set_ylabel('w1[0,1]')
    ax1.set_zlabel('w1[1,0]')
    ax1.set_title('w1 Weight Trajectory')
    ax1.legend()
    
    # Plot w2 weight trajectories
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    seq_w2_00 = [w[0,0] for w in seq_history['w2']]
    seq_w2_01 = [w[0,1] for w in seq_history['w2']]
    seq_w2_10 = [w[1,0] for w in seq_history['w2']]
    
    par_w2_00 = [w[0,0] for w in par_history['w2']]
    par_w2_01 = [w[0,1] for w in par_history['w2']]
    par_w2_10 = [w[1,0] for w in par_history['w2']]
    
    ax2.plot(seq_w2_00, seq_w2_01, seq_w2_10, 'b-', linewidth=2, label='Sequential')
    ax2.plot(par_w2_00, par_w2_01, par_w2_10, 'r--', linewidth=2, label='Parallel')
    ax2.set_xlabel('w2[0,0]')
    ax2.set_ylabel('w2[0,1]')
    ax2.set_zlabel('w2[1,0]')
    ax2.set_title('w2 Weight Trajectory')
    ax2.legend()
    
    # Plot prediction trajectory
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    seq_y0 = [y[0] for y in seq_history['y_pred']]
    seq_y1 = [y[1] for y in seq_history['y_pred']]
    iterations_range = list(range(iterations))
    
    par_y0 = [y[0] for y in par_history['y_pred']]
    par_y1 = [y[1] for y in par_history['y_pred']]
    
    ax3.plot(seq_y0, seq_y1, iterations_range, 'b-', linewidth=2, label='Sequential')
    ax3.plot(par_y0, par_y1, iterations_range, 'r--', linewidth=2, label='Parallel')
    ax3.scatter([0.01], [0.99], [0], color='green', s=100, label='Target')
    ax3.set_xlabel('y_pred[0]')
    ax3.set_ylabel('y_pred[1]')
    ax3.set_zlabel('Iteration')
    ax3.set_title('Prediction Trajectory')
    ax3.legend()
    
    # Plot error surface
    ax4 = fig.add_subplot(2, 2, 4)
    plt.plot(seq_y0, seq_y1, 'b-', linewidth=2, label='Sequential Path')
    plt.plot(par_y0, par_y1, 'r--', linewidth=2, label='Parallel Path')
    plt.scatter([0.01], [0.99], color='green', s=100, marker='*', label='Target', zorder=5)
    plt.scatter([seq_y0[0]], [seq_y1[0]], color='blue', s=50, marker='o', label='Start', zorder=5)
    plt.scatter([seq_y0[-1]], [seq_y1[-1]], color='blue', s=50, marker='s', label='Seq End', zorder=5)
    plt.scatter([par_y0[-1]], [par_y1[-1]], color='red', s=50, marker='s', label='Par End', zorder=5)
    plt.xlabel('y_pred[0]')
    plt.ylabel('y_pred[1]')
    plt.title('Prediction Path (2D View)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('numpy/weight_trajectories_3d.png', dpi=300, bbox_inches='tight')
    print("3D trajectory plot saved as 'numpy/weight_trajectories_3d.png'")

def create_summary_statistics(seq_history, par_history):
    """Create summary statistics for all variables"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE VARIABLE ANALYSIS SUMMARY")
    print("="*80)
    
    iterations = len(seq_history['errors'])
    
    # Final values comparison
    print(f"\nFINAL VALUES AFTER {iterations} ITERATIONS:")
    print("-" * 50)
    
    print(f"Error:")
    print(f"  Sequential: {seq_history['errors'][-1]:.8f}")
    print(f"  Parallel:   {par_history['errors'][-1]:.8f}")
    print(f"  Difference: {seq_history['errors'][-1] - par_history['errors'][-1]:.2e}")
    
    print(f"\nHidden Layer Outputs (h):")
    seq_h_final = seq_history['h'][-1]
    par_h_final = par_history['h'][-1]
    print(f"  Sequential: [{seq_h_final[0]:.6f}, {seq_h_final[1]:.6f}]")
    print(f"  Parallel:   [{par_h_final[0]:.6f}, {par_h_final[1]:.6f}]")
    print(f"  Difference: [{seq_h_final[0] - par_h_final[0]:.2e}, {seq_h_final[1] - par_h_final[1]:.2e}]")
    
    print(f"\nNetwork Predictions (y_pred):")
    seq_y_final = seq_history['y_pred'][-1]
    par_y_final = par_history['y_pred'][-1]
    target = np.array([0.01, 0.99])
    print(f"  Target:     [{target[0]:.6f}, {target[1]:.6f}]")
    print(f"  Sequential: [{seq_y_final[0]:.6f}, {seq_y_final[1]:.6f}]")
    print(f"  Parallel:   [{par_y_final[0]:.6f}, {par_y_final[1]:.6f}]")
    print(f"  Seq Error:  [{abs(seq_y_final[0] - target[0]):.6f}, {abs(seq_y_final[1] - target[1]):.6f}]")
    print(f"  Par Error:  [{abs(par_y_final[0] - target[0]):.6f}, {abs(par_y_final[1] - target[1]):.6f}]")
    
    print(f"\nWeight Matrix w1 (Input → Hidden):")
    seq_w1_final = seq_history['w1'][-1]
    par_w1_final = par_history['w1'][-1]
    print(f"  Sequential: [[{seq_w1_final[0,0]:.6f}, {seq_w1_final[0,1]:.6f}],")
    print(f"               [{seq_w1_final[1,0]:.6f}, {seq_w1_final[1,1]:.6f}]]")
    print(f"  Parallel:   [[{par_w1_final[0,0]:.6f}, {par_w1_final[0,1]:.6f}],")
    print(f"               [{par_w1_final[1,0]:.6f}, {par_w1_final[1,1]:.6f}]]")
    
    print(f"\nWeight Matrix w2 (Hidden → Output):")
    seq_w2_final = seq_history['w2'][-1]
    par_w2_final = par_history['w2'][-1]
    print(f"  Sequential: [[{seq_w2_final[0,0]:.6f}, {seq_w2_final[0,1]:.6f}],")
    print(f"               [{seq_w2_final[1,0]:.6f}, {seq_w2_final[1,1]:.6f}]]")
    print(f"  Parallel:   [[{par_w2_final[0,0]:.6f}, {par_w2_final[0,1]:.6f}],")
    print(f"               [{par_w2_final[1,0]:.6f}, {par_w2_final[1,1]:.6f}]]")
    
    # Calculate weight change magnitudes
    seq_w1_change = np.linalg.norm(seq_history['w1'][-1] - seq_history['w1'][0])
    par_w1_change = np.linalg.norm(par_history['w1'][-1] - par_history['w1'][0])
    seq_w2_change = np.linalg.norm(seq_history['w2'][-1] - seq_history['w2'][0])
    par_w2_change = np.linalg.norm(par_history['w2'][-1] - par_history['w2'][0])
    
    print(f"\nWeight Change Magnitudes (Frobenius Norm):")
    print(f"  w1 Sequential: {seq_w1_change:.6f}")
    print(f"  w1 Parallel:   {par_w1_change:.6f}")
    print(f"  w2 Sequential: {seq_w2_change:.6f}")
    print(f"  w2 Parallel:   {par_w2_change:.6f}")

if __name__ == "__main__":
    print("Creating comprehensive variable evolution plots...")
    
    try:
        # Create main variable evolution plot
        seq_history, par_history = plot_variable_evolution()
        
        # Create 3D trajectory plots
        plot_weight_trajectories()
        
        # Print summary statistics
        create_summary_statistics(seq_history, par_history)
        
        print(f"\nPlots created successfully:")
        print(f"- all_variables_evolution.png: Comprehensive 16-panel comparison")
        print(f"- weight_trajectories_3d.png: 3D trajectory visualization")
        
    except ImportError:
        print("Error: matplotlib is required for plotting.")
        print("Install it with: pip install matplotlib")
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        print("Generating numerical summary instead...")
        
        # Fallback to numerical analysis
        seq_history = run_neural_network_full_history(50, use_sequential=True)
        par_history = run_neural_network_full_history(50, use_sequential=False)
        create_summary_statistics(seq_history, par_history)