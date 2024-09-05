#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <cmath>
#include "ImportGraph.h"

#define DELTA 0.85
#define EPSILON 1e-8

void pagerank_compute_new_rank(std::vector<double> &current_values, const std::vector<double> &sums,
                               std::vector<bool> &active_old, std::vector<bool> &active_new,
                               double &total_change, int num_elements)
{
    for (int tid = 0; tid < num_elements; ++tid)
    {
        if (!active_old[tid])
            continue;

        double new_value = DELTA * sums[tid] + (1 - DELTA) / num_elements;
        double change = std::abs(new_value - current_values[tid]);
        current_values[tid] = new_value;
        total_change += change;
        active_new[tid] = false;
    }
}

void pagerank_distribute_new_rank(const std::vector<double> &current_values,
                                  const std::vector<int> &neighbors,
                                  const std::vector<int> &neighbor_offsets,
                                  std::vector<double> &sums,
                                  const std::vector<bool> &active_old,
                                  std::vector<bool> &active_new,
                                  int num_elements)
{
    for (int tid = 0; tid < num_elements; ++tid)
    {
        if (!active_old[tid])
            continue;

        int num_neighbors = neighbor_offsets[tid + 1] - neighbor_offsets[tid];
        for (int i = neighbor_offsets[tid]; i < neighbor_offsets[tid + 1]; ++i)
        {
            int neighbor_id = neighbors[i];
            sums[neighbor_id] += current_values[tid] / num_neighbors;
            active_new[neighbor_id] = true;
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    const char *filename = argv[1];
    const int num_iterations = 100;

    Graph graph(filename);
    int num_vertices = graph.num_vertices;
    std::vector<double> current_values(num_vertices, 1.0);
    std::vector<double> sums(num_vertices, 0.0);
    std::vector<bool> active_old(num_vertices, true);
    std::vector<bool> active_new(num_vertices, false);
    std::vector<int> neighbors = graph.neighbors;
    std::vector<int> neighbors_offsets = graph.offsets;

    double total_change = 0.0;

    int iterations_needed = 0;
    auto start = std::chrono::steady_clock::now();
    for (; iterations_needed < num_iterations; ++iterations_needed)
    {
        auto begin = std::chrono::steady_clock::now();

        pagerank_compute_new_rank(current_values, sums, active_old, active_new, total_change, num_vertices);

        if (total_change < EPSILON)
            break;

        total_change = 0.0;
        std::fill(sums.begin(), sums.end(), 0.0);

        pagerank_distribute_new_rank(current_values, neighbors, neighbors_offsets, sums, active_old, active_new, num_vertices);

        active_old.swap(active_new);
        std::fill(active_new.begin(), active_new.end(), false);

        auto end = std::chrono::steady_clock::now();
        std::cout << "Iteration " << iterations_needed << " took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                  << "ms" << std::endl;
    }

    std::cout << "Converged after " << iterations_needed << " iterations" << std::endl;
    auto end = std::chrono::steady_clock::now();
    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // Sort by highest PageRank
    std::sort(current_values.begin(), current_values.end(), std::greater<double>());
    for (double value : current_values)
    {
        if (value < .001)
            break;
        std::cout << value << std::endl;
    }

    return 0;
}