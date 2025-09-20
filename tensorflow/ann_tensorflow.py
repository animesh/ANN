import tensorflow as tf
import numpy as np
import sys

class SimpleNeuralNetwork:
    """
    Simple 2-2-2 Neural Network using TensorFlow
    Based on Matt Mazur's backpropagation example
    """
    
    def __init__(self, learning_rate=0.5):
        """Initialize the neural network with specified learning rate"""
        self.learning_rate = learning_rate
        
        # Initialize weights and biases to match the original example
        # Input to hidden weights (2x2)
        self.w1 = tf.Variable([[0.15, 0.20], [0.25, 0.30]], dtype=tf.float32, name='w1')
        
        # Hidden to output weights (2x2)  
        self.w2 = tf.Variable([[0.40, 0.45], [0.50, 0.55]], dtype=tf.float32, name='w2')
        
        # Biases - constant values that don't get updated (to match NumPy implementation)
        self.b1 = tf.constant(0.35, dtype=tf.float32, name='b1')  # Hidden layer bias
        self.b2 = tf.constant(0.60, dtype=tf.float32, name='b2')  # Output layer bias
        
        # Input and target data
        self.x = tf.constant([[0.05, 0.10]], dtype=tf.float32)
        self.y_target = tf.constant([[0.01, 0.99]], dtype=tf.float32)
        
        # Optimizer
        self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
        
    def forward_pass(self, x):
        """Forward pass through the network"""
        # Hidden layer: sigmoid(x * w1 + b1)
        hidden_input = tf.matmul(x, self.w1) + self.b1
        hidden_output = tf.sigmoid(hidden_input)
        
        # Output layer: sigmoid(hidden * w2 + b2)
        output_input = tf.matmul(hidden_output, self.w2) + self.b2
        output = tf.sigmoid(output_input)
        
        return hidden_output, output
    
    def compute_loss(self, y_pred, y_target):
        """Compute mean squared error loss"""
        return tf.reduce_mean(tf.square(y_target - y_pred))
    
    def train_step(self):
        """Single training step using automatic differentiation"""
        with tf.GradientTape() as tape:
            hidden_output, y_pred = self.forward_pass(self.x)
            loss = self.compute_loss(y_pred, self.y_target)
        
        # Compute gradients only for weights (biases are constant)
        gradients = tape.gradient(loss, [self.w1, self.w2])
        
        # Apply gradients only to weights
        self.optimizer.apply_gradients(zip(gradients, [self.w1, self.w2]))
        
        return loss.numpy(), y_pred.numpy(), hidden_output.numpy()
    
    def train(self, iterations=1000, verbose=True):
        """Train the network for specified iterations"""
        history = {
            'losses': [],
            'predictions': [],
            'hidden_outputs': []
        }
        
        for i in range(iterations):
            loss, prediction, hidden = self.train_step()
            
            history['losses'].append(loss)
            history['predictions'].append(prediction[0])  # Remove batch dimension
            history['hidden_outputs'].append(hidden[0])   # Remove batch dimension
            
            if verbose and (i == 0 or i == iterations-1 or (i+1) % max(1, iterations//10) == 0):
                print(f"Iteration {i+1:4d}: Loss = {loss:.10f}, Prediction = [{prediction[0][0]:.8f}, {prediction[0][1]:.8f}]")
        
        return history
    
    def get_weights(self):
        """Get current weights and biases"""
        return {
            'w1': self.w1.numpy(),
            'w2': self.w2.numpy(),
            'b1': self.b1.numpy(),
            'b2': self.b2.numpy()
        }
    
    def predict(self, x_input=None):
        """Make prediction with current weights"""
        if x_input is None:
            x_input = self.x
        else:
            x_input = tf.constant(x_input, dtype=tf.float32)
            
        _, output = self.forward_pass(x_input)
        return output.numpy()

def main():
    """Main function to run the neural network"""
    # Get number of iterations from command line or use default
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    
    print("TensorFlow Neural Network Implementation")
    print("=" * 50)
    print(f"Network Architecture: 2-2-2 (Input-Hidden-Output)")
    print(f"Input: [0.05, 0.10]")
    print(f"Target: [0.01, 0.99]")
    print(f"Learning Rate: 0.5")
    print(f"Iterations: {iterations}")
    print("=" * 50)
    
    # Create and train the network
    nn = SimpleNeuralNetwork(learning_rate=0.5)
    
    # Show initial state
    initial_weights = nn.get_weights()
    print(f"\nInitial Weights:")
    print(f"w1 (Input→Hidden):\n{initial_weights['w1']}")
    print(f"w2 (Hidden→Output):\n{initial_weights['w2']}")
    print(f"b1 (Hidden bias): {initial_weights['b1']}")
    print(f"b2 (Output bias): {initial_weights['b2']}")
    
    # Train the network
    print(f"\nTraining Progress:")
    history = nn.train(iterations=iterations, verbose=True)
    
    # Show final results
    final_weights = nn.get_weights()
    final_prediction = nn.predict()
    final_loss = history['losses'][-1]
    
    print(f"\nFinal Results:")
    print(f"Final Loss: {final_loss:.10f}")
    print(f"Final Prediction: [{final_prediction[0][0]:.8f}, {final_prediction[0][1]:.8f}]")
    print(f"Target:           [0.01000000, 0.99000000]")
    print(f"Error:            [{abs(final_prediction[0][0] - 0.01):.8f}, {abs(final_prediction[0][1] - 0.99):.8f}]")
    
    print(f"\nFinal Weights:")
    print(f"w1 (Input→Hidden):\n{final_weights['w1']}")
    print(f"w2 (Hidden→Output):\n{final_weights['w2']}")
    print(f"b1 (Hidden bias): {final_weights['b1']}")
    print(f"b2 (Output bias): {final_weights['b2']}")
    
    return history, nn

if __name__ == "__main__":
    history, network = main()