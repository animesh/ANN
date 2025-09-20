#!/usr/bin/env python3
"""
Test script to check TensorFlow installation and provide setup instructions
"""

def test_tensorflow_installation():
    """Test if TensorFlow is properly installed"""
    print("TensorFlow Installation Test")
    print("=" * 40)
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Test basic operations
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        print(f"✅ Basic operations work: {c.numpy()}")
        
        # Test GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU support available: {len(gpus)} GPU(s) found")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("ℹ️  GPU support not available (CPU only)")
        
        print("\n🎉 TensorFlow is properly installed and working!")
        return True
        
    except ImportError:
        print("❌ TensorFlow is not installed")
        print("\n📦 Installation Instructions:")
        print("   pip install tensorflow")
        print("   # or for GPU support:")
        print("   pip install tensorflow-gpu")
        print("\n📋 Or install all requirements:")
        print("   pip install -r requirements.txt")
        return False
    
    except Exception as e:
        print(f"❌ TensorFlow installation error: {e}")
        return False

def test_other_dependencies():
    """Test other required dependencies"""
    print("\n" + "=" * 40)
    print("Other Dependencies Test")
    print("=" * 40)
    
    dependencies = [
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib')
    ]
    
    all_good = True
    
    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name} version: {version}")
        except ImportError:
            print(f"❌ {display_name} is not installed")
            print(f"   Install with: pip install {module_name}")
            all_good = False
    
    return all_good

def show_usage_examples():
    """Show usage examples for the TensorFlow implementations"""
    print("\n" + "=" * 40)
    print("Usage Examples")
    print("=" * 40)
    
    examples = [
        ("Basic TensorFlow implementation", "python ann_tensorflow.py 100"),
        ("Keras implementation", "python ann_keras.py 100"),
        ("Compare implementations", "python compare_implementations.py 500"),
        ("Generate visualizations", "python plot_tensorflow_results.py"),
    ]
    
    for description, command in examples:
        print(f"📝 {description}:")
        print(f"   {command}")
        print()

def main():
    """Main test function"""
    tf_ok = test_tensorflow_installation()
    deps_ok = test_other_dependencies()
    
    if tf_ok and deps_ok:
        print("\n🚀 All dependencies are installed! You can run the TensorFlow implementations.")
        show_usage_examples()
    else:
        print("\n⚠️  Please install missing dependencies before running the implementations.")
        print("\n📦 Quick install command:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()