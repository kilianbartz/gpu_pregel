#include <hip/hip_runtime.h>
#include <iostream>
#include "PregelClasses.h"

// Kernel to demonstrate the use of ThreadSafeVector
__global__ void kernel(ThreadSafeVector<int> *vector, int num_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements)
    {
        vector->append(tid); // Append thread ID to the vector
    }
}

int main()
{
    const int num_elements = 1024;
    const int capacity = 2048;

    // Create a ThreadSafeVector on the host
    ThreadSafeVector<int> *d_vector;
    hipMalloc(&d_vector, sizeof(ThreadSafeVector<int>));
    ThreadSafeVector<int> h_vector(capacity);

    // Copy the vector to the device
    hipMemcpy(d_vector, &h_vector, sizeof(ThreadSafeVector<int>), hipMemcpyHostToDevice);

    // Launch the kernel
    hipLaunchKernelGGL(kernel, dim3(32), dim3(32), 0, 0, d_vector, num_elements);

    // Wait for GPU to finish before accessing on host
    hipDeviceSynchronize();

    // Copy the size and data from device to host
    int vector_size = h_vector.getSize(); // Retrieve the size from the device
    int *host_data = new int[capacity];
    hipMemcpy(host_data, h_vector.data(), vector_size * sizeof(int), hipMemcpyDeviceToHost);

    // Print the results
    std::cout << "Vector size: " << vector_size << std::endl;
    for (int i = 0; i < vector_size; ++i)
    {
        std::cout << host_data[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    hipFree(d_vector);
    delete[] host_data;

    return 0;
}
