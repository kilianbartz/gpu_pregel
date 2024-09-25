#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <utility>
#include <cmath>

// Include the Graph class header here
#include "ImportGraph.h"

const double DAMPING_FACTOR = 0.85;
const double EPSILON = 1e-6;
const int MAX_ITERATIONS = 1000;

// CUDA kernel for PageRank computation
__global__ void pagerank_kernel(const int *src_indices, const int *dest_indices, const int *out_degrees,
                                double *pagerank, double *new_pagerank, int num_edges, int num_vertices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_edges)
    {
        int src = src_indices[idx];
        int dest = dest_indices[idx];
        atomicAdd(&new_pagerank[dest], pagerank[src] / out_degrees[src]);
    }
}

// CUDA kernel for applying damping factor
__global__ void apply_damping_kernel(const double *new_pagerank, int num_vertices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices)
    {
        new_pagerank[idx] = (1.0f - DAMPING_FACTOR) / num_vertices + DAMPING_FACTOR * new_pagerank[idx];
    }
}

class PageRank
{
private:
    int num_vertices;
    int num_edges;

    // Device vectors
    thrust::device_vector<int> d_src_indices;
    thrust::device_vector<int> d_dest_indices;
    thrust::device_vector<int> d_out_degrees;
    thrust::device_vector<double> d_pagerank;
    thrust::device_vector<double> d_new_pagerank;

public:
    PageRank(const Graph &graph)
    {
        num_vertices = graph.num_vertices;
        num_edges = graph.neighbors.size();

        std::vector<int> src_indices, dest_indices;
        std::vector<int> out_degrees(num_vertices, 0);

        // Construct the src and dest indices, and calculate out-degrees
        for (int src = 0; src < num_vertices; ++src)
        {
            for (int edge_idx = graph.offsets[src]; edge_idx < graph.offsets[src + 1]; ++edge_idx)
            {
                int dest = graph.neighbors[edge_idx];
                src_indices.push_back(src);
                dest_indices.push_back(dest);
                out_degrees[src]++;
            }
        }

        // Copy data to device vectors
        d_src_indices = src_indices;
        d_dest_indices = dest_indices;
        d_out_degrees = out_degrees;

        // Initialize pagerank vectors
        d_pagerank.resize(num_vertices);
        d_new_pagerank.resize(num_vertices);
        thrust::fill(d_pagerank.begin(), d_pagerank.end(), 1.0f / num_vertices);
    }
    std::vector<double> compute()
    {
        int block_size = 256;
        int num_blocks = (num_edges + block_size - 1) / block_size;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration)
        {
            cudaEventRecord(start);

            thrust::fill(d_new_pagerank.begin(), d_new_pagerank.end(), 0.0f);

            // Launch PageRank kernel
            pagerank_kernel<<<num_blocks, block_size>>>(
                thrust::raw_pointer_cast(d_src_indices.data()),
                thrust::raw_pointer_cast(d_dest_indices.data()),
                thrust::raw_pointer_cast(d_out_degrees.data()),
                thrust::raw_pointer_cast(d_pagerank.data()),
                thrust::raw_pointer_cast(d_new_pagerank.data()),
                num_edges,
                num_vertices);

            // Launch damping factor kernel
            apply_damping_kernel<<<(num_vertices + block_size - 1) / block_size, block_size>>>(
                thrust::raw_pointer_cast(d_new_pagerank.data()),
                num_vertices);

            // Check for convergence
            double sum_diff = thrust::transform_reduce(
                thrust::make_zip_iterator(thrust::make_tuple(d_pagerank.begin(), d_new_pagerank.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(d_pagerank.end(), d_new_pagerank.end())),
                [] __host__ __device__(const thrust::tuple<double, double> &t) -> double
                {
                    return fabs(thrust::get<0>(t) - thrust::get<1>(t));
                },
                0.0,
                thrust::plus<double>());

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float elapsed_time;
            cudaEventElapsedTime(&elapsed_time, start, stop);
            std::cout << "Iteration " << iteration + 1 << " time: " << elapsed_time << " ms, sum_diff: " << sum_diff << std::endl;

            if (sum_diff < EPSILON)
            {
                std::cout << "Converged after " << iteration + 1 << " iterations" << std::endl;
                break;
            }

            // Swap pagerank vectors for the next iteration
            d_pagerank.swap(d_new_pagerank);
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Copy result back to host
        std::vector<double> pagerank(num_vertices);
        thrust::copy(d_pagerank.begin(), d_pagerank.end(), pagerank.begin());
        return pagerank;
    }
};

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <graph_file>" << std::endl;
        return 1;
    }

    // Load graph from file
    std::string filename = argv[1];
    Graph graph(filename);

    std::cout << "Creating PageRank instance..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    PageRank pr(graph);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "PageRank instance created in " << duration.count() << " ms" << std::endl;

    std::cout << "\nComputing PageRank..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();

    std::vector<double> pagerank = pr.compute();

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "PageRank computed in " << duration.count() << " ms" << std::endl;

    std::cout << "\nTop 10 vertices by PageRank:" << std::endl;
    std::vector<std::pair<int, double>> ranked_vertices;
    for (int i = 0; i < graph.num_vertices; ++i)
    {
        ranked_vertices.push_back({i, pagerank[i]});
    }
    std::partial_sort(ranked_vertices.begin(), ranked_vertices.begin() + 10, ranked_vertices.end(),
                      [](const auto &a, const auto &b)
                      { return a.second > b.second; });

    for (int i = 0; i < 10; ++i)
    {
        std::cout << "Vertex " << ranked_vertices[i].first << ": " << ranked_vertices[i].second << std::endl;
    }

    return 0;
}
