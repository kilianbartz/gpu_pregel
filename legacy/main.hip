#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <cstring>
#include "../ImportGraph.h"
#include <chrono>

#define DELTA 0.85
#define EPSILON 1e-6

__global__ void pagerank_compute_new_rank(double *current_values, double *sums, bool *active_old, bool *active_new, double *total_change, int num_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_elements || !active_old[tid])
        return;

    // Compute the new PageRank value for the vertex
    double new_value = DELTA * sums[tid] + (1 - DELTA) / num_elements;
    double change = abs(new_value - current_values[tid]);
    current_values[tid] = new_value;
    atomicAdd(total_change, change);
    sums[tid] = 0;
    active_new[tid] = false;
}

__global__ void pagerank_distribute_new_rank(double *current_values, int *neighbors, int *neighbor_offsets, double *sums, bool *active_old, bool *active_new, int num_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_elements || !active_old[tid])
        return;
    int num_neighbors = neighbor_offsets[tid + 1] - neighbor_offsets[tid];
    for (int i = neighbor_offsets[tid]; i < neighbor_offsets[tid + 1]; i++)
    {
        int neighbor_id = neighbors[i];
        atomicAdd(&sums[neighbor_id], current_values[tid] / num_neighbors);
        active_new[neighbor_id] = true;
    }
}

int main(int argc, char **argv)
{
    // read cli args
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    const char *filename = argv[1];

    // init host data
    const int num_iterations = 2;

    // default example
    // const int num_vertices = 2;
    // double current_values[num_vertices] = {0, 0};
    // double sums[num_vertices] = {};
    // bool active_old[num_vertices] = {true, true};
    // bool active_new[num_vertices] = {};
    // int neighbors[num_vertices] = {1, 0};
    // int neighbors_offsets[num_vertices + 1] = {0, 1, 2};

    Graph graph(filename);
    int num_vertices = graph.num_vertices;
    std::vector<double> current_values(num_vertices, 1);
    std::vector<double> sums(num_vertices);
    std::vector<char> active_old(num_vertices, true);
    std::vector<int> neighbors = graph.neighbors;
    std::vector<int> neighbors_offsets = graph.offsets;

    double total_change = 0;

    // init device data
    double *d_current_values;
    hipMalloc(&d_current_values, num_vertices * sizeof(double));
    hipMemcpy(d_current_values, &current_values[0], num_vertices * sizeof(double), hipMemcpyHostToDevice);

    double *d_sums;
    hipMalloc(&d_sums, num_vertices * sizeof(double));
    hipMemcpy(d_sums, &sums[0], num_vertices * sizeof(double), hipMemcpyHostToDevice);

    bool *d_active_old;
    hipMalloc(&d_active_old, num_vertices * sizeof(bool));
    hipMemcpy(d_active_old, active_old.data(), num_vertices * sizeof(bool), hipMemcpyHostToDevice);

    bool *d_active_new;
    hipMalloc(&d_active_new, num_vertices * sizeof(bool));
    hipMemcpy(d_active_new, active_old.data(), num_vertices * sizeof(bool), hipMemcpyHostToDevice);

    int *d_neighbors;
    hipMalloc(&d_neighbors, num_vertices * sizeof(int));
    hipMemcpy(d_neighbors, neighbors.data(), num_vertices * sizeof(int), hipMemcpyHostToDevice);

    int *d_neighbors_offsets;
    hipMalloc(&d_neighbors_offsets, (num_vertices + 1) * sizeof(int));
    hipMemcpy(d_neighbors_offsets, neighbors_offsets.data(), (num_vertices + 1) * sizeof(int), hipMemcpyHostToDevice);

    double *d_total_change;
    hipMalloc((void **)&d_total_change, sizeof(double));
    hipMemcpy(d_total_change, &total_change, sizeof(double), hipMemcpyHostToDevice);

    // run kernel
    int iterations_needed = 0;
    for (; iterations_needed < num_iterations; iterations_needed++)
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        hipLaunchKernelGGL(pagerank_compute_new_rank, dim3((num_vertices + 255) / 256), dim3(256), 0, 0, d_current_values, d_sums, d_active_old, d_active_new, d_total_change, num_vertices);
        hipDeviceSynchronize();
        // could this be optimized to prevent copying back and forth? probably only check every 10 iterations or so
        hipMemcpy(&total_change, d_total_change, sizeof(double), hipMemcpyDeviceToHost);
        if (total_change < EPSILON)
            break;
        total_change = 0;
        hipMemcpy(d_total_change, &total_change, sizeof(double), hipMemcpyHostToDevice);
        hipLaunchKernelGGL(pagerank_distribute_new_rank, dim3((num_vertices + 255) / 256), dim3(256), 0, 0, d_current_values, d_neighbors, d_neighbors_offsets, d_sums, d_active_old, d_active_new, num_vertices);
        hipDeviceSynchronize();
        std::swap(d_active_old, d_active_new);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Iteration " << iterations_needed << " took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    }
    std::cout << "Converged after " << iterations_needed << " iterations" << std::endl;

    // copy back
    hipMemcpy(&current_values[0], d_current_values, num_vertices * sizeof(double), hipMemcpyDeviceToHost);

    // print result
    // sort by highest pagerank
    std::sort(current_values.begin(), current_values.end(), std::greater<double>());
    for (double value : current_values)
    {
        if (value < .001)
            break;
        std::cout << value << std::endl;
    }

    // clean up
    hipFree(d_current_values);
    hipFree(d_sums);
    hipFree(d_active_old);
    hipFree(d_active_new);
    hipFree(d_neighbors);
    hipFree(d_neighbors_offsets);

    return 0;
}