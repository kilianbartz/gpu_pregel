#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <iostream>
#include <vector>

#define DELTA 0.85

__global__ void compute_pagerank_superstep(double *vertex_values,
                                           const int *neighbors_offsets,
                                           const int *neighbors,
                                           const double *old_messages,
                                           double *new_messages,
                                           const int *message_offsets,
                                           int num_vertices)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_vertices)
        return;

    // Compute the new PageRank value for the vertex
    double sum = 0;
    int start = message_offsets[tid];
    int end = (tid == num_vertices - 1) ? message_offsets[num_vertices] : message_offsets[tid + 1];

    for (int i = start; i < end; i++)
    {
        sum += old_messages[i];
    }
    vertex_values[tid] = DELTA * sum + (1 - DELTA) / num_vertices;

    // Send the new PageRank value to all neighbors
    int neighbor_start = neighbors_offsets[tid];
    int neighbor_end = (tid == num_vertices - 1) ? neighbors_offsets[0] : neighbors_offsets[tid + 1];
    int num_neighbors = neighbor_end - neighbor_start;

    for (int i = neighbor_start; i < neighbor_end; i++)
    {
        int neighbor_id = neighbors[i];
        atomicAdd(&new_messages[neighbor_id], vertex_values[tid] / num_neighbors);
    }
}

int main()
{
    const int num_iterations = 1;
    const int num_vertices = 2;

    // Host vectors
    thrust::host_vector<double> h_vertex_values(num_vertices, 1.0);
    thrust::host_vector<int> h_neighbors = {1, 0};
    thrust::host_vector<int> h_neighbors_offsets(3);
    h_neighbors_offsets[0] = 0;
    h_neighbors_offsets[1] = 1;
    h_neighbors_offsets[2] = 2;
    thrust::host_vector<double> h_old_messages(num_vertices, 0.0);
    thrust::host_vector<double> h_new_messages(num_vertices, 0.0);
    thrust::host_vector<int> h_message_offsets(3);
    h_message_offsets[0] = 0;
    h_message_offsets[1] = 1;
    h_message_offsets[2] = 2; // end of messages

    // Device vectors
    thrust::device_vector<double> d_vertex_values = h_vertex_values;
    thrust::device_vector<int> d_neighbors = h_neighbors;
    thrust::device_vector<int> d_neighbors_offsets = h_neighbors_offsets;
    thrust::device_vector<double> d_old_messages = h_old_messages;
    thrust::device_vector<double> d_new_messages = h_new_messages;
    thrust::device_vector<int> d_message_offsets = h_message_offsets;

    // Pointers for kernel launch
    double *raw_d_vertex_values = thrust::raw_pointer_cast(d_vertex_values.data());
    const int *raw_d_neighbors_offsets = thrust::raw_pointer_cast(d_neighbors_offsets.data());
    const int *raw_d_neighbors = thrust::raw_pointer_cast(d_neighbors.data());
    const double *raw_d_old_messages = thrust::raw_pointer_cast(d_old_messages.data());
    double *raw_d_new_messages = thrust::raw_pointer_cast(d_new_messages.data());
    const int *raw_d_message_offsets = thrust::raw_pointer_cast(d_message_offsets.data());

    // Launch kernel
    for (int i = 0; i < num_iterations; i++)
    {
        hipLaunchKernelGGL(compute_pagerank_superstep, dim3(1), dim3(num_vertices), 0, 0,
                           raw_d_vertex_values, raw_d_neighbors_offsets, raw_d_neighbors,
                           raw_d_old_messages, raw_d_new_messages, raw_d_message_offsets, num_vertices);

        hipDeviceSynchronize();

        // Swap old and new messages

        // thrust::fill(d_new_messages.begin(), d_new_messages.end(), 0.0);
        d_old_messages.swap(d_new_messages);
    }

    // Copy results back to host
    h_vertex_values = d_vertex_values;
    h_old_messages = d_old_messages;
    h_new_messages = d_new_messages;

    // Print results
    for (int i = 0; i < num_vertices; i++)
    {
        std::cout << "Vertex " << i << " has PageRank value " << h_vertex_values[i] << std::endl;
    }

    // Print old messages
    std::cout << "Old messages: ";
    for (const auto &msg : h_old_messages)
    {
        std::cout << msg << " ";
    }
    std::cout << std::endl;

    // Print new messages
    std::cout << "New messages: ";
    for (const auto &msg : h_new_messages)
    {
        std::cout << msg << " ";
    }
    std::cout << std::endl;

    return 0;
}