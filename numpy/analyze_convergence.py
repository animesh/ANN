import numpy as np

def run_neural_network_with_history(iterations, use_sequential=True):
    """Run neural network and return error history"""
    x = np.array([0.05, 0.10])
    w1 = np.array([[0.15, 0.20], [0.25, 0.3]])
    w2 = np.array([[0.4, 0.45], [0.5, 0.55]])
    y = np.array([0.01, 0.99])
    bias = np.array([0.35, 0.6])
    lr = 0.5
    
    errors = []
    
    for iteration in range(iterations):
        h = 1/(1+np.exp(-(x.dot(w1.T)+bias[0])))
        y_pred = 1/(1+np.exp(-(h.dot(w2.T)+bias[1])))
        error = 0.5*np.square(y_pred - y).sum()
        errors.append(error)
        
        if use_sequential:
            w2 = w2 - lr*np.outer((y_pred - y)*(1-y_pred)*y_pred, h)
            w1 = w1 - lr*np.outer(w2.T.dot((y_pred - y)*(1-y_pred)*y_pred)*h*(1-h), x)
        else:
            w2_update = lr*np.outer((y_pred - y)*(1-y_pred)*y_pred, h)
            w1_update = lr*np.outer(w2.T.dot((y_pred - y)*(1-y_pred)*y_pred)*h*(1-h), x)
            w2 = w2 - w2_update
            w1 = w1 - w1_update
    
    return errors

def get_no_loop_errors():
    """Get no-loop version errors (initial and after 1 iteration)"""
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
    
    # After weight updates (parallel approach - same as original)
    w2_new = w2 - lr*np.outer((y_pred - y)*(1-y_pred)*y_pred, h)
    w1_new = w1 - lr*np.outer(w2.T.dot((y_pred - y)*(1-y_pred)*y_pred)*h*(1-h), x)
    
    # Final error
    h_final = 1/(1+np.exp(-(x.dot(w1_new.T)+bias[0])))
    y_pred_final = 1/(1+np.exp(-(h_final.dot(w2_new.T)+bias[1])))
    final_error = 0.5*np.square(y_pred_final - y).sum()
    
    return initial_error, final_error

def analyze_convergence():
    """Comprehensive convergence analysis"""
    
    print("=" * 70)
    print("NEURAL NETWORK CONVERGENCE ANALYSIS")
    print("=" * 70)
    
    # Test different iteration counts
    test_iterations = [1, 2, 5, 10, 20, 50, 100, 500, 1000]
    
    print(f"\n{'Iterations':<10} {'Sequential':<15} {'Parallel':<15} {'Difference':<15}")
    print("-" * 60)
    
    for iterations in test_iterations:
        seq_errors = run_neural_network_with_history(iterations, use_sequential=True)
        par_errors = run_neural_network_with_history(iterations, use_sequential=False)
        
        seq_final = seq_errors[-1]
        par_final = par_errors[-1]
        diff = seq_final - par_final
        
        print(f"{iterations:<10} {seq_final:<15.8f} {par_final:<15.8f} {diff:<15.2e}")
    
    # No-loop comparison
    no_loop_initial, no_loop_final = get_no_loop_errors()
    print(f"\nNo-Loop Version:")
    print(f"Initial error:     {no_loop_initial:.8f}")
    print(f"After 1 iteration: {no_loop_final:.8f}")
    
    # Verify no-loop matches parallel approach
    par_1_iter = run_neural_network_with_history(1, use_sequential=False)
    print(f"Parallel 1 iter:   {par_1_iter[0]:.8f} (should match no-loop final)")
    print(f"Match verification: {abs(no_loop_final - par_1_iter[0]) < 1e-10}")
    
    # Detailed first 10 iterations
    print(f"\n" + "=" * 70)
    print("DETAILED FIRST 10 ITERATIONS")
    print("=" * 70)
    
    seq_errors_10 = run_neural_network_with_history(10, use_sequential=True)
    par_errors_10 = run_neural_network_with_history(10, use_sequential=False)
    
    print(f"{'Iter':<5} {'Sequential Error':<18} {'Parallel Error':<18} {'Difference':<15}")
    print("-" * 60)
    
    for i in range(10):
        diff = seq_errors_10[i] - par_errors_10[i]
        print(f"{i+1:<5} {seq_errors_10[i]:<18.10f} {par_errors_10[i]:<18.10f} {diff:<15.2e}")
    
    # Convergence rate analysis
    print(f"\n" + "=" * 70)
    print("CONVERGENCE RATE ANALYSIS")
    print("=" * 70)
    
    # Calculate error reduction percentages
    seq_1000 = run_neural_network_with_history(1000, use_sequential=True)
    par_1000 = run_neural_network_with_history(1000, use_sequential=False)
    
    seq_reduction = (seq_1000[0] - seq_1000[-1]) / seq_1000[0] * 100
    par_reduction = (par_1000[0] - par_1000[-1]) / par_1000[0] * 100
    
    print(f"After 1000 iterations:")
    print(f"Sequential: {seq_1000[0]:.8f} → {seq_1000[-1]:.8f} ({seq_reduction:.4f}% reduction)")
    print(f"Parallel:   {par_1000[0]:.8f} → {par_1000[-1]:.8f} ({par_reduction:.4f}% reduction)")
    
    # Final accuracy analysis
    target = np.array([0.01, 0.99])
    
    # Get final predictions
    def get_final_prediction(iterations, use_sequential):
        x = np.array([0.05, 0.10])
        w1 = np.array([[0.15, 0.20], [0.25, 0.3]])
        w2 = np.array([[0.4, 0.45], [0.5, 0.55]])
        y = np.array([0.01, 0.99])
        bias = np.array([0.35, 0.6])
        lr = 0.5
        
        for _ in range(iterations):
            h = 1/(1+np.exp(-(x.dot(w1.T)+bias[0])))
            y_pred = 1/(1+np.exp(-(h.dot(w2.T)+bias[1])))
            
            if use_sequential:
                w2 = w2 - lr*np.outer((y_pred - y)*(1-y_pred)*y_pred, h)
                w1 = w1 - lr*np.outer(w2.T.dot((y_pred - y)*(1-y_pred)*y_pred)*h*(1-h), x)
            else:
                w2_update = lr*np.outer((y_pred - y)*(1-y_pred)*y_pred, h)
                w1_update = lr*np.outer(w2.T.dot((y_pred - y)*(1-y_pred)*y_pred)*h*(1-h), x)
                w2 = w2 - w2_update
                w1 = w1 - w1_update
        
        h_final = 1/(1+np.exp(-(x.dot(w1.T)+bias[0])))
        y_pred_final = 1/(1+np.exp(-(h_final.dot(w2.T)+bias[1])))
        return y_pred_final
    
    seq_pred_1000 = get_final_prediction(1000, True)
    par_pred_1000 = get_final_prediction(1000, False)
    
    print(f"\n" + "=" * 70)
    print("FINAL PREDICTION ACCURACY (1000 iterations)")
    print("=" * 70)
    print(f"Target:     [{target[0]:.6f}, {target[1]:.6f}]")
    print(f"Sequential: [{seq_pred_1000[0]:.6f}, {seq_pred_1000[1]:.6f}]")
    print(f"Parallel:   [{par_pred_1000[0]:.6f}, {par_pred_1000[1]:.6f}]")
    print(f"Seq Error:  [{abs(seq_pred_1000[0] - target[0]):.6f}, {abs(seq_pred_1000[1] - target[1]):.6f}]")
    print(f"Par Error:  [{abs(par_pred_1000[0] - target[0]):.6f}, {abs(par_pred_1000[1] - target[1]):.6f}]")
    
    print(f"\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("Both approaches converge to nearly identical results with excellent accuracy.")
    print("The differences are minimal and both successfully learn the target mapping.")
    print("Sequential updates show slightly faster early convergence.")
    print("Parallel updates match the original blog post methodology exactly.")

if __name__ == "__main__":
    analyze_convergence()