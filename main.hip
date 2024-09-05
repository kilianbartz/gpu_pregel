#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>

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

int main()
{
    // init host data
    const int num_vertices = 2;
    const int num_iterations = 100;

    double current_values[num_vertices] = {0, 0};
    double sums[num_vertices] = {};
    bool active_old[num_vertices] = {true, true};
    bool active_new[num_vertices] = {};
    int neighbors[num_vertices] = {1, 0};
    int neighbors_offsets[num_vertices + 1] = {0, 1, 2};

    double total_change = 0;

    // init device data
    double *d_current_values;
    hipMalloc(&d_current_values, num_vertices * sizeof(double));
    hipMemcpy(d_current_values, current_values, num_vertices * sizeof(double), hipMemcpyHostToDevice);

    double *d_sums;
    hipMalloc(&d_sums, num_vertices * sizeof(double));
    hipMemcpy(d_sums, sums, num_vertices * sizeof(double), hipMemcpyHostToDevice);

    bool *d_active_old;
    hipMalloc(&d_active_old, num_vertices * sizeof(bool));
    hipMemcpy(d_active_old, active_old, num_vertices * sizeof(bool), hipMemcpyHostToDevice);

    bool *d_active_new;
    hipMalloc(&d_active_new, num_vertices * sizeof(bool));
    hipMemcpy(d_active_new, active_old, num_vertices * sizeof(bool), hipMemcpyHostToDevice);

    int *d_neighbors;
    hipMalloc(&d_neighbors, num_vertices * sizeof(int));
    hipMemcpy(d_neighbors, neighbors, num_vertices * sizeof(int), hipMemcpyHostToDevice);

    int *d_neighbors_offsets;
    hipMalloc(&d_neighbors_offsets, (num_vertices + 1) * sizeof(int));
    hipMemcpy(d_neighbors_offsets, neighbors_offsets, (num_vertices + 1) * sizeof(int), hipMemcpyHostToDevice);

    double *d_total_change;
    hipMalloc((void **)&d_total_change, sizeof(double));
    hipMemcpy(d_total_change, &total_change, sizeof(double), hipMemcpyHostToDevice);

    // run kernel
    int iterations_needed = 0;
    for (; iterations_needed < num_iterations; iterations_needed++)
    {
        hipLaunchKernelGGL(pagerank_compute_new_rank, dim3((num_vertices + 255) / 256), dim3(256), 0, 0, d_current_values, d_sums, d_active_old, d_active_new, d_total_change, num_vertices);
        hipDeviceSynchronize();
        // could this be optimized to prevent copying back and forth?
        hipMemcpy(&total_change, d_total_change, sizeof(double), hipMemcpyDeviceToHost);
        if (total_change < EPSILON)
            break;
        total_change = 0;
        hipMemcpy(d_total_change, &total_change, sizeof(double), hipMemcpyHostToDevice);
        hipLaunchKernelGGL(pagerank_distribute_new_rank, dim3((num_vertices + 255) / 256), dim3(256), 0, 0, d_current_values, d_neighbors, d_neighbors_offsets, d_sums, d_active_old, d_active_new, num_vertices);
        hipDeviceSynchronize();
        std::swap(d_active_old, d_active_new);
    }
    std::cout << "Converged after " << iterations_needed << " iterations" << std::endl;

    // copy back
    hipMemcpy(current_values, d_current_values, num_vertices * sizeof(double), hipMemcpyDeviceToHost);

    // print result
    for (int i = 0; i < num_vertices; i++)
    {
        std::cout << "Vertex " << i << " has PageRank " << current_values[i] << std::endl;
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