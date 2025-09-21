#!/usr/bin/env python3
"""
PyTorch Installation Test Script
Tests PyTorch installation and basic functionality for neural network implementation
"""

import sys

def test_pytorch_installation():
    """Test PyTorch installation and basic operations"""
    
    print("PyTorch Installation Test")
    print("=" * 40)
    
    # Test PyTorch import
    try:
        import torch
        print("✅ PyTorch imported successfully")
        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError as e:
        print("❌ PyTorch import failed!")
        print(f"Error: {e}")
        print("\nTo install PyTorch:")
        print("pip install torch")
        print("or visit: https://pytorch.org/get-started/locally/")
        return False
    
    # Test basic tensor operations
    try:
        # Create tensors
        x = torch.tensor([0.05, 0.10], dtype=torch.float32)
        w = torch.tensor([[0.15, 0.20], [0.25, 0.30]], dtype=torch.float32)
        
        # Test matrix multiplication
        result = torch.matmul(x, w.t())
        expected = torch.tensor([0.0275, 0.0425], dtype=torch.float32)
        
        if torch.allclose(result, expected, atol=1e-6):
            print("✅ Basic tensor operations work:", result.tolist())
        else:
            print("❌ Basic tensor operations failed")
            print(f"Expected: {expected.tolist()}, Got: {result.tolist()}")
            return False
            
    except Exception as e:
        print("❌ Basic tensor operations failed!")
        print(f"Error: {e}")
        return False
    
    # Test automatic differentiation
    try:
        # Create tensor with gradient tracking
        x = torch.tensor([2.0], requires_grad=True)
        y = x ** 2
        y.backward()
        
        if torch.allclose(x.grad, torch.tensor([4.0]), atol=1e-6):
            print("✅ Automatic differentiation works:", x.grad.item())
        else:
            print("❌ Automatic differentiation failed")
            print(f"Expected gradient: 4.0, Got: {x.grad.item()}")
            return False
            
    except Exception as e:
        print("❌ Automatic differentiation failed!")
        print(f"Error: {e}")
        return False
    
    # Test GPU availability (optional)
    try:
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"ℹ️  CUDA version: {torch.version.cuda}")
        else:
            print("ℹ️  CUDA not available (CPU only)")
    except Exception as e:
        print(f"ℹ️  Could not check CUDA availability: {e}")
    
    # Test MPS availability (Apple Silicon)
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon) available")
        else:
            print("ℹ️  MPS not available")
    except Exception as e:
        print(f"ℹ️  Could not check MPS availability: {e}")
    
    return True

def test_dependencies():
    """Test other required dependencies"""
    
    print("\n" + "=" * 40)
    print("Other Dependencies Test")
    print("=" * 40)
    
    # Test NumPy
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
        
        # Test basic NumPy operation
        arr = np.array([1, 2, 3])
        result = np.sum(arr)
        if result == 6:
            print("✅ NumPy operations work")
        else:
            print("❌ NumPy operations failed")
            
    except ImportError:
        print("❌ NumPy not available")
        print("Install with: pip install numpy")
    
    # Test Matplotlib (optional for visualization)
    try:
        import matplotlib
        print(f"✅ Matplotlib version: {matplotlib.__version__}")
    except ImportError:
        print("ℹ️  Matplotlib not available (optional for visualization)")
        print("Install with: pip install matplotlib")

def test_neural_network_components():
    """Test specific components needed for neural network implementation"""
    
    print("\n" + "=" * 40)
    print("Neural Network Components Test")
    print("=" * 40)
    
    try:
        import torch
        
        # Test sigmoid activation
        x = torch.tensor([0.0, 1.0, -1.0])
        sigmoid_result = torch.sigmoid(x)
        expected = torch.tensor([0.5, 0.7311, 0.2689], dtype=torch.float32)
        
        if torch.allclose(sigmoid_result, expected, atol=1e-4):
            print("✅ Sigmoid activation works")
        else:
            print("❌ Sigmoid activation failed")
        
        # Test gradient computation for neural network
        w1 = torch.tensor([[0.15, 0.20], [0.25, 0.30]], requires_grad=True)
        w2 = torch.tensor([[0.40, 0.45], [0.50, 0.55]], requires_grad=True)
        x = torch.tensor([0.05, 0.10])
        y_target = torch.tensor([0.01, 0.99])
        
        # Forward pass
        h = torch.sigmoid(torch.matmul(x, w1.t()) + 0.35)
        y_pred = torch.sigmoid(torch.matmul(h, w2.t()) + 0.60)
        loss = 0.5 * torch.sum((y_pred - y_target) ** 2)
        
        # Backward pass
        loss.backward()
        
        if w1.grad is not None and w2.grad is not None:
            print("✅ Neural network gradient computation works")
            print(f"✅ Initial loss: {loss.item():.6f}")
        else:
            print("❌ Neural network gradient computation failed")
            
    except Exception as e:
        print("❌ Neural network components test failed!")
        print(f"Error: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    
    print("🧠 PyTorch Neural Network Implementation Test")
    print("Testing PyTorch installation and neural network components...")
    print()
    
    # Run all tests
    pytorch_ok = test_pytorch_installation()
    
    if pytorch_ok:
        test_dependencies()
        components_ok = test_neural_network_components()
        
        if components_ok:
            print("\n" + "🎉 All tests passed! PyTorch is ready for neural network implementation.")
            print("\n" + "=" * 60)
            print("Usage Examples")
            print("=" * 60)
            print("📝 Automatic differentiation (parallel updates):")
            print("   python ann_pytorch.py 100 automatic")
            print("📝 Manual gradients (parallel updates - matches NumPy parallel):")
            print("   python ann_pytorch.py 100 manual_parallel")
            print("📝 Manual gradients (sequential updates - matches NumPy sequential):")
            print("   python ann_pytorch.py 100 manual_sequential")
            print("📝 Compare all PyTorch methods:")
            print("   python compare_pytorch_implementations.py 1000")
            return True
        else:
            print("\n❌ Some neural network components failed. Check PyTorch installation.")
            return False
    else:
        print("\n❌ PyTorch installation test failed. Please install PyTorch first.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)