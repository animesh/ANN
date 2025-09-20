import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys

class KerasNeuralNetwork:
    """
    Simple 2-2-2 Neural Network that mirrors the NumPy implementation exactly
    Uses manual gradient computation to match NumPy behavior
    """
    
    def __init__(self, learning_rate=0.5):
        """Initialize the neural network to match NumPy implementation"""
        self.learning_rate = learning_rate
        
        # Initialize weights exactly like NumPy (as Variables for gradient computation)
        self.w1 = tf.Variable([[0.15, 0.20], [0.25, 0.30]], dtype=tf.float32, name='w1')
        self.w2 = tf.Variable([[0.40, 0.45], [0.50, 0.55]], dtype=tf.float32, name='w2')
        
        # Constant biases (never updated)
        self.bias = tf.constant([0.35, 0.60], dtype=tf.float32, name='bias')
        
        # Input and target data
        self.x = tf.constant([0.05, 0.10], dtype=tf.float32)
        self.y_target = tf.constant([0.01, 0.99], dtype=tf.float32)
    
    def forward_pass(self):
        """Forward pass exactly matching NumPy implementation"""
        # Hidden layer: h = sigmoid(x.dot(w1.T) + bias[0])
        h = tf.sigmoid(tf.tensordot(self.x, tf.transpose(self.w1), axes=1) + self.bias[0])
        
        # Output layer: y_pred = sigmoid(h.dot(w2.T) + bias[1])
        y_pred = tf.sigmoid(tf.tensordot(h, tf.transpose(self.w2), axes=1) + self.bias[1])
        
        return h, y_pred
    
    def compute_loss(self, y_pred):
        """Compute loss exactly like NumPy: 0.5 * sum((y_pred - y)^2)"""
        return 0.5 * tf.reduce_sum(tf.square(y_pred - self.y_target))
    
    def manual_gradient_update(self):
        """Manual gradient computation and weight update to match NumPy exactly"""
        h, y_pred = self.forward_pass()
        
        # Compute gradients exactly like NumPy
        # Output layer gradient: (y_pred - y) * (1 - y_pred) * y_pred
        output_error = (y_pred - self.y_target) * (1 - y_pred) * y_pred
        
        # Weight update for w2: w2 = w2 - lr * outer(output_error, h)
        w2_gradient = tf.tensordot(output_error, h, axes=0)  # outer product
        
        # Hidden layer gradient: w2.T.dot(output_error) * h * (1 - h)
        hidden_error = tf.tensordot(tf.transpose(self.w2), output_error, axes=1) * h * (1 - h)
        
        # Weight update for w1: w1 = w1 - lr * outer(hidden_error, x)
        w1_gradient = tf.tensordot(hidden_error, self.x, axes=0)  # outer product
        
        # Apply updates
        self.w2.assign_sub(self.learning_rate * w2_gradient)
        self.w1.assign_sub(self.learning_rate * w1_gradient)
        
        return self.compute_loss(y_pred), y_pred, h
    
    def train(self, iterations=1000, verbose=True):
        """Train the network using manual gradients to match NumPy exactly"""
        history = {
            'losses': [],
            'predictions': []
        }
        
        for i in range(iterations):
            # Manual gradient step (matches NumPy exactly)
            loss, prediction, hidden = self.manual_gradient_update()
            
            history['losses'].append(loss.numpy())
            history['predictions'].append(prediction.numpy())
            
            if verbose and (i == 0 or i == iterations-1 or (i+1) % max(1, iterations//10) == 0):
                print(f"Iteration {i+1:4d}: Loss = {float(loss):.10f}, Prediction = [{float(prediction[0]):.8f}, {float(prediction[1]):.8f}]")
        
        return history
    
    def get_weights(self):
        """Get current weights and biases"""
        return {
            'hidden_weights': self.w1.numpy(),
            'hidden_bias': np.array([self.bias[0].numpy(), self.bias[0].numpy()]),  # Same bias for both neurons
            'output_weights': self.w2.numpy(),
            'output_bias': np.array([self.bias[1].numpy(), self.bias[1].numpy()])   # Same bias for both neurons
        }
    
    def predict(self, x_input=None):
        """Make prediction with current weights"""
        if x_input is not None:
            # Update input temporarily
            old_x = self.x
            self.x = tf.constant(x_input.flatten(), dtype=tf.float32)
            
        _, y_pred = self.forward_pass()
        result = y_pred.numpy().reshape(1, -1)
        
        if x_input is not None:
            # Restore original input
            self.x = old_x
            
        return result
    
    def summary(self):
        """Print model summary"""
        print("Manual Keras Implementation (mirrors NumPy exactly)")
        print(f"w1 shape: {self.w1.shape}")
        print(f"w2 shape: {self.w2.shape}")
        print(f"bias shape: {self.bias.shape}")
        print(f"Total parameters: {self.w1.shape[0] * self.w1.shape[1] + self.w2.shape[0] * self.w2.shape[1] + 2}")

def main():
    """Main function to run the Keras neural network that mirrors NumPy exactly"""
    # Get number of iterations from command line or use default
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    
    print("Keras Neural Network Implementation (NumPy Mirror)")
    print("=" * 50)
    print(f"Network Architecture: 2-2-2 (Input-Hidden-Output)")
    print(f"Input: [0.05, 0.10]")
    print(f"Target: [0.01, 0.99]")
    print(f"Learning Rate: 0.5")
    print(f"Iterations: {iterations}")
    print("=" * 50)
    
    # Create and train the network
    nn = KerasNeuralNetwork(learning_rate=0.5)
    
    # Show model architecture
    print(f"\nModel Architecture:")
    nn.summary()
    
    # Show initial state
    initial_weights = nn.get_weights()
    print(f"\nInitial Weights:")
    print(f"w1 (Input→Hidden):\n{initial_weights['hidden_weights']}")
    print(f"w2 (Hidden→Output):\n{initial_weights['output_weights']}")
    print(f"bias: [{nn.bias[0].numpy():.2f}, {nn.bias[1].numpy():.2f}]")
    
    # Train the network
    print(f"\nTraining Progress:")
    history = nn.train(iterations=iterations, verbose=True)
    
    # Show final results
    final_weights = nn.get_weights()
    final_prediction = nn.predict()
    final_loss = history['losses'][-1]
    
    print(f"\nFinal Results:")
    print(f"Final Loss: {float(final_loss):.10f}")
    print(f"Final Prediction: [{float(final_prediction[0][0]):.8f}, {float(final_prediction[0][1]):.8f}]")
    print(f"Target:           [0.01000000, 0.99000000]")
    print(f"Error:            [{abs(float(final_prediction[0][0]) - 0.01):.8f}, {abs(float(final_prediction[0][1]) - 0.99):.8f}]")
    
    print(f"\nFinal Weights:")
    print(f"w1 (Input→Hidden):\n{final_weights['hidden_weights']}")
    print(f"w2 (Hidden→Output):\n{final_weights['output_weights']}")
    print(f"bias: [{nn.bias[0].numpy():.2f}, {nn.bias[1].numpy():.2f}] (constant)")
    
    return history, nn

if __name__ == "__main__":
    history, network = main()