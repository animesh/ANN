import torch
import numpy as np
import sys

class PyTorchNeuralNetwork:
    
    def __init__(self, learning_rate=0.5, update_method='parallel'):
        self.learning_rate = learning_rate
        self.update_method = update_method
        
        self.w1 = torch.tensor([[0.15, 0.20], [0.25, 0.30]], dtype=torch.float32, requires_grad=True)
        self.w2 = torch.tensor([[0.40, 0.45], [0.50, 0.55]], dtype=torch.float32, requires_grad=True)
        
        self.bias = torch.tensor([0.35, 0.60], dtype=torch.float32, requires_grad=False)
        
        self.x = torch.tensor([0.05, 0.10], dtype=torch.float32)
        self.y_target = torch.tensor([0.01, 0.99], dtype=torch.float32)
        
    def forward_pass(self):
        h = torch.sigmoid(torch.matmul(self.x, self.w1.t()) + self.bias[0])
        y_pred = torch.sigmoid(torch.matmul(h, self.w2.t()) + self.bias[1])
        return h, y_pred
    
    def compute_loss(self, y_pred):
        return 0.5 * torch.sum((y_pred - self.y_target) ** 2)
    
    def train_step_automatic(self):
        if self.w1.grad is not None:
            self.w1.grad.zero_()
        if self.w2.grad is not None:
            self.w2.grad.zero_()
        
        h, y_pred = self.forward_pass()
        loss = self.compute_loss(y_pred)
        
        loss.backward()
        
        with torch.no_grad():
            self.w1 -= self.learning_rate * self.w1.grad
            self.w2 -= self.learning_rate * self.w2.grad
        
        return loss.item(), y_pred.detach().numpy(), h.detach().numpy()
    
    def train_step_manual_parallel(self):
        with torch.no_grad():
            h, y_pred = self.forward_pass()
            loss = self.compute_loss(y_pred)
            
            output_error = (y_pred - self.y_target) * (1 - y_pred) * y_pred
            w2_gradient = torch.outer(output_error, h)
            hidden_error = torch.matmul(self.w2.t(), output_error) * h * (1 - h)
            w1_gradient = torch.outer(hidden_error, self.x)
            
            self.w2 -= self.learning_rate * w2_gradient
            self.w1 -= self.learning_rate * w1_gradient
            
            return loss.item(), y_pred.numpy(), h.numpy()
    
    def train_step_manual_sequential(self):
        with torch.no_grad():
            h, y_pred = self.forward_pass()
            loss = self.compute_loss(y_pred)
            
            output_error = (y_pred - self.y_target) * (1 - y_pred) * y_pred
            
            w2_gradient = torch.outer(output_error, h)
            self.w2 -= self.learning_rate * w2_gradient
            
            hidden_error = torch.matmul(self.w2.t(), output_error) * h * (1 - h)
            w1_gradient = torch.outer(hidden_error, self.x)
            self.w1 -= self.learning_rate * w1_gradient
            
            return loss.item(), y_pred.numpy(), h.numpy()
    
    def train(self, iterations=1000, verbose=True, method='automatic'):
        """Train the network for specified iterations"""
        history = {
            'losses': [],
            'predictions': [],
            'hidden_outputs': []
        }
        
        if method == 'automatic':
            train_step = self.train_step_automatic
        elif method == 'manual_parallel':
            train_step = self.train_step_manual_parallel
        elif method == 'manual_sequential':
            train_step = self.train_step_manual_sequential
        else:
            raise ValueError("Method must be 'automatic', 'manual_parallel', or 'manual_sequential'")
        
        for i in range(iterations):
            loss, prediction, hidden = train_step()
            
            history['losses'].append(loss)
            history['predictions'].append(prediction)
            history['hidden_outputs'].append(hidden)
            
            if verbose and (i == 0 or i == iterations-1 or (i+1) % max(1, iterations//10) == 0):
                print(f"Iteration {i+1:4d}: Loss = {loss:.10f}, Prediction = [{prediction[0]:.8f}, {prediction[1]:.8f}]")
        
        return history
    
    def get_weights(self):
        return {
            'w1': self.w1.detach().numpy(),
            'w2': self.w2.detach().numpy(),
            'bias': self.bias.numpy()
        }
    
    def predict(self):
        with torch.no_grad():
            _, y_pred = self.forward_pass()
            return y_pred.numpy()

def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    method = sys.argv[2] if len(sys.argv) > 2 else 'automatic'
    
    method_names = {
        'automatic': 'Automatic Differentiation (Parallel Updates)',
        'manual_parallel': 'Manual Gradients (Parallel Updates - matches NumPy parallel)',
        'manual_sequential': 'Manual Gradients (Sequential Updates - matches NumPy sequential)'
    }
    
    print("PyTorch Neural Network Implementation")
    print("=" * 60)
    print(f"Network Architecture: 2-2-2 (Input-Hidden-Output)")
    print(f"Input: [0.05, 0.10]")
    print(f"Target: [0.01, 0.99]")
    print(f"Learning Rate: 0.5")
    print(f"Iterations: {iterations}")
    print(f"Method: {method_names.get(method, method)}")
    print("=" * 60)
    
    nn = PyTorchNeuralNetwork(learning_rate=0.5)
    
    initial_weights = nn.get_weights()
    print(f"\nInitial Weights:")
    print(f"w1 (Input→Hidden):\n{initial_weights['w1']}")
    print(f"w2 (Hidden→Output):\n{initial_weights['w2']}")
    print(f"bias: [{initial_weights['bias'][0]:.2f}, {initial_weights['bias'][1]:.2f}] (constant)")
    
    print(f"\nTraining Progress:")
    history = nn.train(iterations=iterations, verbose=True, method=method)
    
    final_weights = nn.get_weights()
    final_prediction = nn.predict()
    final_loss = history['losses'][-1]
    
    print(f"\nFinal Results:")
    print(f"Final Loss: {final_loss:.10f}")
    print(f"Final Prediction: [{final_prediction[0]:.8f}, {final_prediction[1]:.8f}]")
    print(f"Target:           [0.01000000, 0.99000000]")
    print(f"Error:            [{abs(final_prediction[0] - 0.01):.8f}, {abs(final_prediction[1] - 0.99):.8f}]")
    
    print(f"\nFinal Weights:")
    print(f"w1 (Input→Hidden):\n{final_weights['w1']}")
    print(f"w2 (Hidden→Output):\n{final_weights['w2']}")
    print(f"bias: [{final_weights['bias'][0]:.2f}, {final_weights['bias'][1]:.2f}] (constant)")
    
    return history, nn

if __name__ == "__main__":
    history, network = main()