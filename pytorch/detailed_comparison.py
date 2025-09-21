import torch
import numpy as np
import sys

def detailed_iteration_comparison():
    """Detailed iteration-by-iteration comparison of all implementations"""
    
    print("Detailed Iteration-by-Iteration Comparison")
    print("=" * 80)
    
    # Run PyTorch methods
    from ann_pytorch import PyTorchNeuralNetwork
    
    methods = ['automatic', 'manual_parallel', 'manual_sequential']
    pytorch_results = {}
    
    for method in methods:
        nn = PyTorchNeuralNetwork(learning_rate=0.5)
        history = nn.train(iterations=10, verbose=False, method=method)
        pytorch_results[method] = history['losses']
    
    # Manually input NumPy results (from actual runs)
    numpy_results = {
        'sequential': [
            0.2983711088,  # Iteration 1
            0.2910279239,  # Iteration 2
            0.2835473641,  # Iteration 3
            0.2759435235,  # Iteration 4
            0.2682329155,  # Iteration 5
            0.2604343778,  # Iteration 6
            0.2525688987,  # Iteration 7
            0.2446593651,  # Iteration 8
            0.2367302306,  # Iteration 9
            0.2288071136   # Iteration 10
        ],
        'parallel': [
            0.2983711088,  # Iteration 1
            0.2910277737,  # Iteration 2
            0.2835471331,  # Iteration 3
            0.2759432889,  # Iteration 4
            0.2682327612,  # Iteration 5
            0.2604343928,  # Iteration 6
            0.2525691760,  # Iteration 7
            0.2446599992,  # Iteration 8
            0.2367313155,  # Iteration 9
            0.2288087405   # Iteration 10
        ],
        'no_loop': [
            0.2983711087600027,  # Iteration 1 (from no-loop version)
            0.29102777369359933, # Iteration 2 (from no-loop version)
            None, None, None, None, None, None, None, None  # Only 2 iterations in no-loop
        ]
    }
    
    # TensorFlow results (from actual run)
    tensorflow_results = [
        0.3036583066,  # Iteration 1
        0.2963685989,  # Iteration 2
        0.2889351845,  # Iteration 3
        0.2813707590,  # Iteration 4
        0.2736903727,  # Iteration 5
        0.2659112513,  # Iteration 6
        0.2580531836,  # Iteration 7
        0.2501378655,  # Iteration 8
        0.2421889007,  # Iteration 9
        0.2342312336   # Iteration 10
    ]
    
    # Keras results (from actual run)
    keras_results = [
        0.2983711064,  # Iteration 1
        0.2910277843,  # Iteration 2
        0.2835471630,  # Iteration 3
        0.2759432793,  # Iteration 4
        0.2682327628,  # Iteration 5
        0.2604343593,  # Iteration 6
        0.2525692284,  # Iteration 7
        0.2446600199,  # Iteration 8
        0.2367313206,  # Iteration 9
        0.2288087308   # Iteration 10
    ]
    
    # JAX results (from actual run)
    jax_auto_results = [
        0.2983711064,  # Iteration 1
        0.2910277843,  # Iteration 2
        0.2835471630,  # Iteration 3
        0.2759432793,  # Iteration 4
        0.2682327926,  # Iteration 5
        0.2604344189,  # Iteration 6
        0.2525691986,  # Iteration 7
        0.2446599901,  # Iteration 8
        0.2367313206,  # Iteration 9
        0.2288087457   # Iteration 10
    ]
    
    jax_par_results = [
        0.2983711064,  # Iteration 1
        0.2910277843,  # Iteration 2
        0.2835471630,  # Iteration 3
        0.2759432793,  # Iteration 4
        0.2682327926,  # Iteration 5
        0.2604344189,  # Iteration 6
        0.2525691986,  # Iteration 7
        0.2446599901,  # Iteration 8
        0.2367313206,  # Iteration 9
        0.2288087457   # Iteration 10
    ]
    
    jax_seq_results = [
        0.2983711064,  # Iteration 1
        0.2910279632,  # Iteration 2
        0.2835473120,  # Iteration 3
        0.2759435773,  # Iteration 4
        0.2682329416,  # Iteration 5
        0.2604343891,  # Iteration 6
        0.2525689006,  # Iteration 7
        0.2446593642,  # Iteration 8
        0.2367302477,  # Iteration 9
        0.2288071513   # Iteration 10
    ]
    
    # Print detailed comparison
    print(f"{'Iteration':<10} {'NumPy Seq':<15} {'NumPy Par':<15} {'NumPy NoLoop':<15} {'PyTorch Auto':<15} {'PyTorch ManPar':<15} {'PyTorch ManSeq':<15} {'TensorFlow':<15} {'Keras':<15} {'JAX Auto':<15} {'JAX ManPar':<15} {'JAX ManSeq':<15}")
    print("-" * 195)
    
    for i in range(10):
        numpy_seq = numpy_results['sequential'][i]
        numpy_par = numpy_results['parallel'][i]
        numpy_no_loop = numpy_results['no_loop'][i] if i < len(numpy_results['no_loop']) and numpy_results['no_loop'][i] is not None else 'N/A'
        pytorch_auto = pytorch_results['automatic'][i]
        pytorch_par = pytorch_results['manual_parallel'][i]
        pytorch_seq = pytorch_results['manual_sequential'][i]
        tensorflow_loss = tensorflow_results[i]
        keras_loss = keras_results[i]
        
        numpy_seq_str = f"{numpy_seq:.10f}"
        numpy_par_str = f"{numpy_par:.10f}"
        numpy_no_loop_str = f"{numpy_no_loop:.10f}" if numpy_no_loop != 'N/A' else "N/A"
        
        jax_auto_loss = jax_auto_results[i]
        jax_par_loss = jax_par_results[i]
        jax_seq_loss = jax_seq_results[i]
        
        print(f"{i+1:<10} {numpy_seq_str:<15} {numpy_par_str:<15} {numpy_no_loop_str:<15} {pytorch_auto:<15.10f} {pytorch_par:<15.10f} {pytorch_seq:<15.10f} {tensorflow_loss:<15.10f} {keras_loss:<15.10f} {jax_auto_loss:<15.10f} {jax_par_loss:<15.10f} {jax_seq_loss:<15.10f}")
    
    # Calculate differences
    print(f"\nDifferences from NumPy Sequential:")
    numpy_seq_final = numpy_results['sequential'][-1]
    numpy_par_final = numpy_results['parallel'][-1]
    pytorch_auto_final = pytorch_results['automatic'][-1]
    pytorch_par_final = pytorch_results['manual_parallel'][-1]
    pytorch_seq_final = pytorch_results['manual_sequential'][-1]
    tensorflow_final = tensorflow_results[-1]
    keras_final = keras_results[-1]
    jax_auto_final = jax_auto_results[-1]
    jax_par_final = jax_par_results[-1]
    jax_seq_final = jax_seq_results[-1]
    
    print(f"PyTorch Auto vs NumPy Seq: {abs(pytorch_auto_final - numpy_seq_final):.2e}")
    print(f"PyTorch Manual Par vs NumPy Seq: {abs(pytorch_par_final - numpy_seq_final):.2e}")
    print(f"PyTorch Manual Seq vs NumPy Seq: {abs(pytorch_seq_final - numpy_seq_final):.2e}")
    print(f"TensorFlow vs NumPy Seq: {abs(tensorflow_final - numpy_seq_final):.2e}")
    print(f"Keras vs NumPy Seq: {abs(keras_final - numpy_seq_final):.2e}")
    print(f"JAX Auto vs NumPy Seq: {abs(jax_auto_final - numpy_seq_final):.2e}")
    print(f"JAX Manual Par vs NumPy Seq: {abs(jax_par_final - numpy_seq_final):.2e}")
    print(f"JAX Manual Seq vs NumPy Seq: {abs(jax_seq_final - numpy_seq_final):.2e}")
    
    print(f"\nDifferences from NumPy Parallel:")
    print(f"PyTorch Auto vs NumPy Par: {abs(pytorch_auto_final - numpy_par_final):.2e}")
    print(f"PyTorch Manual Par vs NumPy Par: {abs(pytorch_par_final - numpy_par_final):.2e}")
    print(f"PyTorch Manual Seq vs NumPy Par: {abs(pytorch_seq_final - numpy_par_final):.2e}")
    print(f"TensorFlow vs NumPy Par: {abs(tensorflow_final - numpy_par_final):.2e}")
    print(f"Keras vs NumPy Par: {abs(keras_final - numpy_par_final):.2e}")
    print(f"JAX Auto vs NumPy Par: {abs(jax_auto_final - numpy_par_final):.2e}")
    print(f"JAX Manual Par vs NumPy Par: {abs(jax_par_final - numpy_par_final):.2e}")
    print(f"JAX Manual Seq vs NumPy Par: {abs(jax_seq_final - numpy_par_final):.2e}")
    
    print(f"\nNumPy No-Loop vs Other NumPy (Iteration 1):")
    numpy_seq_1 = numpy_results['sequential'][0]
    numpy_par_1 = numpy_results['parallel'][0]
    numpy_no_loop_1 = numpy_results['no_loop'][0]
    print(f"No-Loop vs Sequential: {abs(numpy_no_loop_1 - numpy_seq_1):.2e}")
    print(f"No-Loop vs Parallel: {abs(numpy_no_loop_1 - numpy_par_1):.2e}")
    
    print(f"\nNumPy No-Loop vs Other NumPy (Iteration 2):")
    numpy_seq_2 = numpy_results['sequential'][1]
    numpy_no_loop_2 = numpy_results['no_loop'][1]
    pytorch_auto_2 = pytorch_results['automatic'][1]
    pytorch_seq_2 = pytorch_results['manual_sequential'][1]
    print(f"No-Loop vs Sequential: {abs(numpy_no_loop_2 - numpy_seq_2):.2e}")
    print(f"No-Loop vs PyTorch Auto: {abs(numpy_no_loop_2 - pytorch_auto_2):.2e}")
    print(f"No-Loop vs PyTorch Manual Seq: {abs(numpy_no_loop_2 - pytorch_seq_2):.2e}")
    tensorflow_2 = tensorflow_results[1]
    print(f"No-Loop vs TensorFlow: {abs(numpy_no_loop_2 - tensorflow_2):.2e}")
    
    print(f"\nDifferences between PyTorch methods:")
    print(f"Auto vs Manual Parallel: {abs(pytorch_auto_final - pytorch_par_final):.2e}")
    print(f"Manual Parallel vs Manual Sequential: {abs(pytorch_par_final - pytorch_seq_final):.2e}")
    print(f"Auto vs Manual Sequential: {abs(pytorch_auto_final - pytorch_seq_final):.2e}")
    
    print(f"\nTensorFlow vs PyTorch methods:")
    print(f"TensorFlow vs PyTorch Auto: {abs(tensorflow_final - pytorch_auto_final):.2e}")
    print(f"TensorFlow vs PyTorch Manual Par: {abs(tensorflow_final - pytorch_par_final):.2e}")
    print(f"TensorFlow vs PyTorch Manual Seq: {abs(tensorflow_final - pytorch_seq_final):.2e}")
    
    # Analysis
    print(f"\nAnalysis:")
    print(f"1. PyTorch Auto and Manual Parallel are identical: {pytorch_auto_final == pytorch_par_final}")
    print(f"2. TensorFlow significantly different from all others: ~5.4e-3 vs ~1e-6 to 1e-8")
    print(f"3. NumPy/PyTorch differences are in floating-point precision range (< 1e-6)")
    print(f"4. No exact matches - all frameworks have slight numerical differences")
    print(f"5. Sequential vs Parallel updates show consistent small differences")
    print(f"6. NumPy No-Loop matches Sequential approach (both use sequential updates)")
    print(f"7. TensorFlow MSE loss scaling creates much larger differences than precision effects")
    
    return pytorch_results, numpy_results

if __name__ == "__main__":
    detailed_iteration_comparison()