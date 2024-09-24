#include <omp.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

// Include the Graph class header here
#include "ImportGraph.h"

const float DAMPING_FACTOR = 0.85;
const float EPSILON = 1e-6;
const int MAX_ITERATIONS = 1000;

class PageRank
{
private:
    int num_vertices;
    int num_edges;
    std::vector<int> src_indices;
    std::vector<int> dest_indices;
    std::vector<int> out_degrees;
    std::vector<float> pagerank;
    std::vector<float> new_pagerank;

public:
    PageRank(const Graph &graph) : num_vertices(graph.num_vertices), num_edges(0)
    {
        out_degrees.resize(num_vertices, 0);

        // Construct the src and dest indices, and calculate out-degrees
        for (int src = 0; src < num_vertices; ++src)
        {
            for (int edge_idx = graph.offsets[src]; edge_idx < graph.offsets[src + 1]; ++edge_idx)
            {
                int dest = graph.neighbors[edge_idx];
                src_indices.push_back(src);
                dest_indices.push_back(dest);
                out_degrees[src]++;
                num_edges++;
            }
        }

        // Initialize pagerank vectors
        pagerank.resize(num_vertices, 1.0f / num_vertices);
        new_pagerank.resize(num_vertices, 0.0f);
    }

    std::vector<float> compute()
    {

        for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration)
        {
            auto start_time = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_vertices; ++i)
            {
                new_pagerank[i] = 0.0f;
            }

// Compute PageRank in parallel
#pragma omp parallel for
            for (int i = 0; i < num_edges; ++i)
            {
                int src = src_indices[i];
                int dest = dest_indices[i];
#pragma omp atomic
                new_pagerank[dest] += pagerank[src] / out_degrees[src];
            }

// Apply damping factor in parallel
#pragma omp parallel for
            for (int i = 0; i < num_vertices; ++i)
            {
                new_pagerank[i] = (1.0f - DAMPING_FACTOR) / num_vertices + DAMPING_FACTOR * new_pagerank[i];
            }

            // Check for convergence in parallel with reduction
            float sum_diff = 0.0f;
#pragma omp parallel for reduction(+ : sum_diff)
            for (int i = 0; i < num_vertices; ++i)
            {
                sum_diff += std::abs(pagerank[i] - new_pagerank[i]);
            }
            std::cout << "Iteration " << iteration + 1 << " sum_diff: " << sum_diff << std::endl;

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "Iteration " << iteration + 1 << " time: " << duration.count() << " ms" << std::endl;

            if (sum_diff < EPSILON)
            {
                std::cout << "Converged after " << iteration + 1 << " iterations" << std::endl;
                break;
            }

            // Swap pagerank vectors for the next iteration
            std::swap(pagerank, new_pagerank);
        }

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

    std::vector<float> pagerank = pr.compute();

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "PageRank computed in " << duration.count() << " ms" << std::endl;

    std::cout << "\nTop 10 vertices by PageRank:" << std::endl;
    std::vector<std::pair<int, float>> ranked_vertices;
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
