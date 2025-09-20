import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ann_tensorflow import SimpleNeuralNetwork as TFNetwork
from ann_keras import KerasNeuralNetwork

def plot_convergence_comparison():
    """Create convergence plots comparing TensorFlow and Keras implementations"""
    
    # Test different iteration counts
    iteration_counts = [10, 50, 100, 500, 1000]
    
    tf_results = []
    keras_results = []
    
    print("Generating convergence data...")
    
    for iterations in iteration_counts:
        print(f"Testing {iterations} iterations...")
        
        # TensorFlow implementation
        tf_network = TFNetwork(learning_rate=0.5)
        tf_history = tf_network.train(iterations=iterations, verbose=False)
        tf_final_pred = tf_network.predict()
        tf_results.append({
            'iterations': iterations,
            'final_loss': tf_history['losses'][-1],
            'final_prediction': tf_final_pred[0],
            'history': tf_history
        })
        
        # Keras implementation
        keras_network = KerasNeuralNetwork(learning_rate=0.5)
        keras_history = keras_network.train(iterations=iterations, verbose=False)
        keras_final_pred = keras_network.predict()
        keras_results.append({
            'iterations': iterations,
            'final_loss': keras_history['losses'][-1],
            'final_prediction': keras_final_pred[0],
            'history': keras_history
        })
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('TensorFlow vs Keras Neural Network Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Final loss comparison
    ax1 = axes[0, 0]
    iterations_list = [r['iterations'] for r in tf_results]
    tf_losses = [r['final_loss'] for r in tf_results]
    keras_losses = [r['final_loss'] for r in keras_results]
    
    ax1.semilogy(iterations_list, tf_losses, 'b-o', linewidth=2, markersize=6, label='TensorFlow')
    ax1.semilogy(iterations_list, keras_losses, 'r--s', linewidth=2, markersize=6, label='Keras')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Final Loss (log scale)')
    ax1.set_title('Final Loss vs Iterations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction accuracy
    ax2 = axes[0, 1]
    target = np.array([0.01, 0.99])
    
    tf_errors = [np.linalg.norm(r['final_prediction'] - target) for r in tf_results]
    keras_errors = [np.linalg.norm(r['final_prediction'] - target) for r in keras_results]
    
    ax2.semilogy(iterations_list, tf_errors, 'b-o', linewidth=2, markersize=6, label='TensorFlow')
    ax2.semilogy(iterations_list, keras_errors, 'r--s', linewidth=2, markersize=6, label='Keras')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Distance from Target (log scale)')
    ax2.set_title('Prediction Accuracy vs Iterations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Detailed convergence for 1000 iterations
    ax3 = axes[1, 0]
    tf_1000 = tf_results[-1]['history']['losses']
    keras_1000 = keras_results[-1]['history']['losses']
    
    ax3.semilogy(range(1, len(tf_1000) + 1), tf_1000, 'b-', linewidth=1.5, label='TensorFlow', alpha=0.8)
    ax3.semilogy(range(1, len(keras_1000) + 1), keras_1000, 'r--', linewidth=1.5, label='Keras', alpha=0.8)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss (log scale)')
    ax3.set_title('Detailed Convergence (1000 iterations)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final predictions comparison
    ax4 = axes[1, 1]
    tf_pred_1 = [r['final_prediction'][0] for r in tf_results]
    tf_pred_2 = [r['final_prediction'][1] for r in tf_results]
    keras_pred_1 = [r['final_prediction'][0] for r in keras_results]
    keras_pred_2 = [r['final_prediction'][1] for r in keras_results]
    
    ax4.plot(iterations_list, tf_pred_1, 'b-o', linewidth=2, markersize=6, label='TensorFlow Output 1')
    ax4.plot(iterations_list, tf_pred_2, 'b-s', linewidth=2, markersize=6, label='TensorFlow Output 2')
    ax4.plot(iterations_list, keras_pred_1, 'r--o', linewidth=2, markersize=6, label='Keras Output 1')
    ax4.plot(iterations_list, keras_pred_2, 'r--s', linewidth=2, markersize=6, label='Keras Output 2')
    ax4.axhline(y=0.01, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Target 1 (0.01)')
    ax4.axhline(y=0.99, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Target 2 (0.99)')
    ax4.set_xlabel('Iterations')
    ax4.set_ylabel('Prediction Value')
    ax4.set_title('Final Predictions vs Iterations')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tensorflow/tensorflow_keras_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'tensorflow/tensorflow_keras_comparison.png'")
    
    return tf_results, keras_results

def plot_training_dynamics():
    """Plot detailed training dynamics for both implementations"""
    
    iterations = 500
    
    # Run both implementations
    tf_network = TFNetwork(learning_rate=0.5)
    tf_history = tf_network.train(iterations=iterations, verbose=False)
    
    keras_network = KerasNeuralNetwork(learning_rate=0.5)
    keras_history = keras_network.train(iterations=iterations, verbose=False)
    
    # Create detailed plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('TensorFlow Neural Network Training Dynamics', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss evolution
    ax1 = axes[0, 0]
    ax1.semilogy(range(1, iterations + 1), tf_history['losses'], 'b-', linewidth=2, label='TensorFlow')
    ax1.semilogy(range(1, iterations + 1), keras_history['losses'], 'r--', linewidth=2, label='Keras')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Loss Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction evolution - Output 1
    ax2 = axes[0, 1]
    tf_pred_1 = [pred[0] for pred in tf_history['predictions']]
    keras_pred_1 = [pred[0] for pred in keras_history['predictions']]
    ax2.plot(range(1, iterations + 1), tf_pred_1, 'b-', linewidth=2, label='TensorFlow')
    ax2.plot(range(1, iterations + 1), keras_pred_1, 'r--', linewidth=2, label='Keras')
    ax2.axhline(y=0.01, color='green', linestyle=':', linewidth=2, label='Target')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Prediction Value')
    ax2.set_title('Output 1 Evolution (Target: 0.01)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prediction evolution - Output 2
    ax3 = axes[0, 2]
    tf_pred_2 = [pred[1] for pred in tf_history['predictions']]
    keras_pred_2 = [pred[1] for pred in keras_history['predictions']]
    ax3.plot(range(1, iterations + 1), tf_pred_2, 'b-', linewidth=2, label='TensorFlow')
    ax3.plot(range(1, iterations + 1), keras_pred_2, 'r--', linewidth=2, label='Keras')
    ax3.axhline(y=0.99, color='green', linestyle=':', linewidth=2, label='Target')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Prediction Value')
    ax3.set_title('Output 2 Evolution (Target: 0.99)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Error from target
    ax4 = axes[1, 0]
    target = np.array([0.01, 0.99])
    tf_errors = [np.linalg.norm(np.array(pred) - target) for pred in tf_history['predictions']]
    keras_errors = [np.linalg.norm(np.array(pred) - target) for pred in keras_history['predictions']]
    ax4.semilogy(range(1, iterations + 1), tf_errors, 'b-', linewidth=2, label='TensorFlow')
    ax4.semilogy(range(1, iterations + 1), keras_errors, 'r--', linewidth=2, label='Keras')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Distance from Target (log scale)')
    ax4.set_title('Error from Target')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Hidden layer evolution (TensorFlow only, as it tracks hidden outputs)
    ax5 = axes[1, 1]
    if 'hidden_outputs' in tf_history:
        tf_hidden_1 = [hidden[0] for hidden in tf_history['hidden_outputs']]
        tf_hidden_2 = [hidden[1] for hidden in tf_history['hidden_outputs']]
        ax5.plot(range(1, iterations + 1), tf_hidden_1, 'b-', linewidth=2, label='Hidden Neuron 1')
        ax5.plot(range(1, iterations + 1), tf_hidden_2, 'g-', linewidth=2, label='Hidden Neuron 2')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Hidden Activation')
        ax5.set_title('Hidden Layer Evolution (TensorFlow)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Hidden layer data\nnot available', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Hidden Layer Evolution')
    
    # Plot 6: Difference between implementations
    ax6 = axes[1, 2]
    loss_diff = [abs(tf_history['losses'][i] - keras_history['losses'][i]) for i in range(iterations)]
    pred_diff = [np.linalg.norm(np.array(tf_history['predictions'][i]) - np.array(keras_history['predictions'][i])) 
                 for i in range(iterations)]
    ax6.semilogy(range(1, iterations + 1), loss_diff, 'purple', linewidth=2, label='Loss Difference')
    ax6.semilogy(range(1, iterations + 1), pred_diff, 'orange', linewidth=2, label='Prediction Difference')
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Absolute Difference (log scale)')
    ax6.set_title('TensorFlow vs Keras Differences')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tensorflow/training_dynamics.png', dpi=300, bbox_inches='tight')
    print("Training dynamics plot saved as 'tensorflow/training_dynamics.png'")
    
    return tf_history, keras_history

def main():
    """Main function to generate all plots"""
    print("Generating TensorFlow visualization plots...")
    
    try:
        # Generate comparison plots
        tf_results, keras_results = plot_convergence_comparison()
        
        # Generate training dynamics plots
        tf_history, keras_history = plot_training_dynamics()
        
        # Print summary
        print(f"\n📊 Visualization Summary:")
        print(f"✅ Convergence comparison plot generated")
        print(f"✅ Training dynamics plot generated")
        print(f"📁 Check tensorflow/ directory for PNG files")
        
        # Final comparison
        final_tf_loss = tf_results[-1]['final_loss']
        final_keras_loss = keras_results[-1]['final_loss']
        
        # Extract scalar values if they're lists/arrays
        final_tf_loss = final_tf_loss[0] if isinstance(final_tf_loss, (list, np.ndarray)) and len(final_tf_loss) > 0 else final_tf_loss
        final_keras_loss = final_keras_loss[0] if isinstance(final_keras_loss, (list, np.ndarray)) and len(final_keras_loss) > 0 else final_keras_loss
        
        print(f"\n🎯 Final Results (1000 iterations):")
        print(f"TensorFlow Final Loss: {float(final_tf_loss):.10f}")
        print(f"Keras Final Loss: {float(final_keras_loss):.10f}")
        print(f"Difference: {abs(float(final_tf_loss) - float(final_keras_loss)):.2e}")
        
    except ImportError:
        print("Error: matplotlib is required for plotting.")
        print("Install it with: pip install matplotlib")
        
    except Exception as e:
        print(f"Error generating plots: {e}")

if __name__ == "__main__":
    main()