import torch
import time
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('CUDA is available.')
else:
    device = torch.device("cpu")
    print('CUDA is not available.')




matrix_size = 10000
A = torch.randn(matrix_size, matrix_size)
B = torch.randn(matrix_size, matrix_size)





def cpu_matrix_multiplication(A, B):
    start_time = time.time()
    result = torch.matmul(A, B)  # Matrix multiplication
    end_time = time.time()
    return result, end_time - start_time

# Function to measure time on GPU (CUDA)
def cuda_matrix_multiplication(A, B):
    # Move tensors to GPU
    A_cuda = A.to('cuda')
    B_cuda = B.to('cuda')
    
    # Synchronize before starting time
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Matrix multiplication on GPU
    result = torch.matmul(A_cuda, B_cuda)
    
    # Synchronize before ending time
    torch.cuda.synchronize()
    end_time = time.time()
    
    return result, end_time - start_time


# Measure CPU time
cpu_result, cpu_time = cpu_matrix_multiplication(A, B)

# Measure CUDA time
if torch.cuda.is_available():
    cuda_result, cuda_time = cuda_matrix_multiplication(A, B)
else:
    cuda_time = None
    print("CUDA is not available")

# Print results
print(f"CPU Time: {cpu_time:.6f} seconds")
if cuda_time is not None:
    print(f"CUDA Time: {cuda_time:.6f} seconds")

# Check if results are similar (they should be)
assert torch.allclose(cpu_result, cuda_result.cpu(), atol=1e-6), "Results differ between CPU and GPU!"
