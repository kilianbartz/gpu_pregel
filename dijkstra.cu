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
#include <limits>
#include <iomanip>

// Include the updated Graph class header here
#include "ImportGraph_weighted.h"

const int INF = std::numeric_limits<int>::max();
const int BLOCK_SIZE = 256;

// CUDA kernel for SSSP relaxation
__global__ void sssp_kernel(const int *src_indices, const int *dest_indices, const int *weights,
                            int *distances, bool *changed, int num_edges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_edges)
    {
        int src = src_indices[idx];
        int dest = dest_indices[idx];
        int weight = weights[idx];

        if (distances[src] != INF)
        {
            int new_dist = distances[src] + weight;
            if (new_dist < distances[dest])
            {
                atomicMin(&distances[dest], new_dist);
                *changed = true;
            }
        }
    }
}

class SSSP
{
private:
    int num_vertices;
    int num_edges;

    // Device vectors
    thrust::device_vector<int> d_src_indices;
    thrust::device_vector<int> d_dest_indices;
    thrust::device_vector<int> d_weights;
    thrust::device_vector<int> d_distances;
    thrust::device_vector<bool> d_changed;

public:
    SSSP(const Graph &graph)
    {
        num_vertices = graph.num_vertices;
        num_edges = graph.neighbors.size();

        std::vector<int> src_indices, dest_indices;

        // Construct the src and dest indices
        for (int src = 0; src < num_vertices; ++src)
        {
            for (int edge_idx = graph.offsets[src]; edge_idx < graph.offsets[src + 1]; ++edge_idx)
            {
                int dest = graph.neighbors[edge_idx];
                src_indices.push_back(src);
                dest_indices.push_back(dest);
            }
        }

        // Copy data to device vectors
        d_src_indices = src_indices;
        d_dest_indices = dest_indices;
        d_weights = graph.weights;

        // Initialize distances vector
        d_distances.resize(num_vertices);
        d_changed.resize(1);
    }

    std::vector<int> compute(int source)
    {
        thrust::fill(d_distances.begin(), d_distances.end(), INF);
        thrust::fill(d_changed.begin(), d_changed.end(), false);

        // Set source distance to 0
        d_distances[source] = 0;

        int num_blocks = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;

        bool changed = true;
        while (changed)
        {
            changed = false;
            thrust::fill(d_changed.begin(), d_changed.end(), false);

            // Launch SSSP kernel
            sssp_kernel<<<num_blocks, BLOCK_SIZE>>>(
                thrust::raw_pointer_cast(d_src_indices.data()),
                thrust::raw_pointer_cast(d_dest_indices.data()),
                thrust::raw_pointer_cast(d_weights.data()),
                thrust::raw_pointer_cast(d_distances.data()),
                thrust::raw_pointer_cast(d_changed.data()),
                num_edges);

            // Check if any distances were updated
            thrust::copy(d_changed.begin(), d_changed.end(), &changed);
        }

        // Copy result back to host
        std::vector<int> distances(num_vertices);
        thrust::copy(d_distances.begin(), d_distances.end(), distances.begin());
        return distances;
    }
};

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <graph_file> <source_vertex>" << std::endl;
        return 1;
    }

    // Load graph from file
    std::string filename = argv[1];
    Graph graph(filename);

    int source = std::stoi(argv[2]);

    std::cout << "Creating SSSP instance..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    SSSP sssp(graph);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "SSSP instance created in " << duration.count() << " ms" << std::endl;

    std::cout << "\nComputing Single-Source Shortest Paths..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();

    std::vector<int> distances = sssp.compute(source);

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "SSSP computed in " << duration.count() << " ms" << std::endl;

    std::cout << "\nTop 10 closest vertices to source " << source << ":" << std::endl;
    std::vector<std::pair<int, int>> ranked_vertices;
    for (int i = 0; i < graph.num_vertices; ++i)
    {
        if (i != source && distances[i] != INF)
        {
            ranked_vertices.push_back({i, distances[i]});
        }
    }
    std::partial_sort(ranked_vertices.begin(), ranked_vertices.begin() + std::min(10, static_cast<int>(ranked_vertices.size())), ranked_vertices.end(),
                      [](const auto &a, const auto &b)
                      { return a.second < b.second; });

    std::cout << std::setw(10) << "Vertex" << std::setw(15) << "Distance" << std::endl;
    std::cout << std::string(25, '-') << std::endl;
    for (int i = 0; i < std::min(10, static_cast<int>(ranked_vertices.size())); ++i)
    {
        std::cout << std::setw(10) << ranked_vertices[i].first
                  << std::setw(15) << ranked_vertices[i].second << std::endl;
    }

    return 0;
}