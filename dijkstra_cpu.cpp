#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include "ImportGraph_weighted.h"

const int INF = std::numeric_limits<int>::max();

class SSSP
{
private:
    int num_vertices;
    std::vector<std::vector<std::pair<int, int>>> adj_list;

public:
    SSSP(const Graph &graph) : num_vertices(graph.num_vertices)
    {
        adj_list.resize(num_vertices);
        for (int i = 0; i < num_vertices; ++i)
        {
            for (int j = graph.offsets[i]; j < graph.offsets[i + 1]; ++j)
            {
                int dest = graph.neighbors[j];
                int weight = graph.weights[j];
                adj_list[i].emplace_back(dest, weight);
            }
        }
    }

    std::vector<int> compute(int source)
    {
        std::vector<int> distances(num_vertices, INF);
        std::vector<bool> visited(num_vertices, false);
        std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> pq;

        distances[source] = 0;
        pq.emplace(0, source);

        while (!pq.empty())
        {
            int u = pq.top().second;
            pq.pop();

            if (visited[u])
                continue;
            visited[u] = true;

            for (const auto &[v, weight] : adj_list[u])
            {
                if (!visited[v] && distances[u] != INF && distances[u] + weight < distances[v])
                {
                    distances[v] = distances[u] + weight;
                    pq.emplace(distances[v], v);
                }
            }
        }

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

    std::string filename = argv[1];
    int source = std::stoi(argv[2]);

    std::cout << "Loading graph..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    Graph graph(filename);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Graph loaded in " << duration.count() << " ms" << std::endl;

    std::cout << "Creating SSSP instance..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();

    SSSP sssp(graph);

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
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
            ranked_vertices.emplace_back(i, distances[i]);
        }
    }
    std::partial_sort(ranked_vertices.begin(),
                      ranked_vertices.begin() + std::min(10, static_cast<int>(ranked_vertices.size())),
                      ranked_vertices.end(),
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