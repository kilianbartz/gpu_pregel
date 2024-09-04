#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include "ThreadSafeVector.h"
#include <array>

#define DELTA 0.85

__global__ void compute_pagerank_superstep(double *vertex_values, ThreadSafeVector<int> *neighbors, ThreadSafeVector<double> *old_messages, ThreadSafeVector<double> *new_messages, int num_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_elements)
        return;

    // compute the new pagerank value for the vertex
    double sum = 0;
    for (int i = 0; i < *old_messages[tid].size(); i++)
    {
        sum += old_messages[tid][i];
    }
    vertex_values[tid] = DELTA * sum + (1 - DELTA) / num_elements;

    // send the new pagerank value to all neighbors
    int num_neighbors = *neighbors[tid].size();
    for (int i = 0; i < *neighbors[tid].size(); i++)
    {
        int neighbor_id = neighbors[tid][i];
        new_messages[neighbor_id].append(vertex_values[tid] / num_neighbors);
    }
}

int main()
{

    const int num_iterations = 1;

    // host copies: arrays needed for pagerank algorithm
    const int num_vertices = 2;
    double vertex_values[num_vertices] = {1.0, 1.0};
    std::array<ThreadSafeVector<int>, num_vertices> neighbors = {ThreadSafeVector<int>({1}), ThreadSafeVector<int>({0})};
    std::array<ThreadSafeVector<double>, num_vertices> old_messages;
    std::array<ThreadSafeVector<double>, num_vertices> new_messages;

    // create device copies
    double *d_vertex_values;
    hipMalloc(&d_vertex_values, num_vertices * sizeof(double));
    ThreadSafeVector<int> *d_neighbors;
    hipMalloc(&d_neighbors, num_vertices * sizeof(ThreadSafeVector<int>));
    ThreadSafeVector<double> *d_oldMessages;
    hipMalloc(&d_oldMessages, num_vertices * sizeof(ThreadSafeVector<double>));
    ThreadSafeVector<double> *d_newMessages;
    hipMalloc(&d_newMessages, num_vertices * sizeof(ThreadSafeVector<double>));

    // copy data to device
    hipMemcpy(d_vertex_values, &vertex_values, num_vertices * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_neighbors, &neighbors, num_vertices * sizeof(ThreadSafeVector<int>), hipMemcpyHostToDevice);
    hipMemcpy(d_oldMessages, &old_messages, num_vertices * sizeof(ThreadSafeVector<double>), hipMemcpyHostToDevice);
    hipMemcpy(d_newMessages, &new_messages, num_vertices * sizeof(ThreadSafeVector<double>), hipMemcpyHostToDevice);

    // launch kernel

    for (int i = 0; i < num_iterations; i++)
    {
        compute_pagerank_superstep<<<1, num_vertices>>>(d_vertex_values, d_neighbors, d_oldMessages, d_newMessages, num_vertices);
        hipDeviceSynchronize();
        // swap old and new messages
        std::swap(d_oldMessages, d_newMessages);
    }

    // copy data back to host
    hipMemcpy(&vertex_values, d_vertex_values, num_vertices * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(&old_messages, d_oldMessages, num_vertices * sizeof(ThreadSafeVector<double>), hipMemcpyDeviceToHost);
    hipMemcpy(&new_messages, d_newMessages, num_vertices * sizeof(ThreadSafeVector<double>), hipMemcpyDeviceToHost);

    // print results
    for (int i = 0; i < num_vertices; i++)
    {
        std::cout << "Vertex " << i << " has pagerank value " << vertex_values[i] << std::endl;
    }
    // print oldMessages
    for (int i = 0; i < num_vertices; i++)
    {
        std::cout << "Vertex " << i << " has old messages: ";
        for (int j = 0; j < *old_messages[i].size(); j++)
        {
            std::cout << old_messages[i][j] << " ";
        }
        std::cout << std::endl;
    }
    // print newMessages
    for (int i = 0; i < num_vertices; i++)
    {
        std::cout << "Vertex " << i << " has new messages: ";
        for (int j = 0; j < *new_messages[i].size(); j++)
        {
            std::cout << new_messages[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // clean up
    hipFree(d_vertex_values);
    hipFree(d_neighbors);
    hipFree(d_oldMessages);
    hipFree(d_newMessages);

    return 0;
}