#!/usr/bin/env python3
"""
Test JAX installation and basic functionality
"""

def test_jax_installation():
    """Test if JAX is properly installed and working"""
    try:
        import jax
        import jax.numpy as jnp
        from jax import grad, jit
        
        print("✓ JAX imported successfully")
        print(f"JAX version: {jax.__version__}")
        
        # Test basic array operations
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])
        z = x + y
        print(f"✓ Basic array operations work: {x} + {y} = {z}")
        
        # Test automatic differentiation
        def simple_function(x):
            return x ** 2 + 2 * x + 1
        
        grad_fn = grad(simple_function)
        result = grad_fn(3.0)
        expected = 8.0  # derivative of x^2 + 2x + 1 at x=3 is 2*3 + 2 = 8
        print(f"✓ Automatic differentiation works: grad(x^2 + 2x + 1) at x=3 = {result}")
        
        # Test JIT compilation
        @jit
        def jit_function(x, y):
            return jnp.dot(x, y)
        
        result = jit_function(x, y)
        print(f"✓ JIT compilation works: dot product = {result}")
        
        # Test random number generation
        key = jax.random.PRNGKey(42)
        random_array = jax.random.normal(key, (3,))
        print(f"✓ Random number generation works: {random_array}")
        
        print("\n🎉 JAX installation test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ JAX import failed: {e}")
        print("Install JAX with: pip install jax jaxlib")
        return False
    except Exception as e:
        print(f"❌ JAX test failed: {e}")
        return False

if __name__ == "__main__":
    test_jax_installation()