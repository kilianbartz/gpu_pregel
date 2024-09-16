#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <algorithm>
#include <mutex>

class Graph
{
public:
    int num_vertices = 0;

    Graph(std::string filename)
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        std::string line;
        std::ifstream file(filename);
        std::vector<std::pair<int, int>> edges; // Store all edges

        // Step 1: Read the file sequentially and store edges
        while (std::getline(file, line))
        {
            if (line[0] == '#')
                continue;
            std::stringstream iss(line);
            int a, b;
            iss >> a >> b;
            edges.push_back({a, b});
        }
        file.close();
        std::cout << "Read the file. Number of edges: " << edges.size() << std::endl;

        // Step 2: Determine the size of the adjacency list
        int max_vertex = 0;
        for (const auto &edge : edges)
        {
            max_vertex = std::max(max_vertex, std::max(edge.first, edge.second));
        }

        adj_list.resize(max_vertex + 1); // Resize adj_list to fit the largest vertex

// Step 3: Parallelize the insertion of edges into the adjacency list
#pragma omp parallel for
        for (size_t i = 0; i < edges.size(); ++i)
        {
            int a = edges[i].first;
            int b = edges[i].second;

#pragma omp critical // Ensure only one thread modifies the list at a time
            {
                adj_list[a].push_back(b);
            }
        }

        num_vertices = adj_list.size();
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Time to read and build adjacency list: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

        // Step 4: Process adjacency lists to generate offsets and neighbors
        std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
        int counter = 0;
        offsets.push_back(0);

// This part can also be parallelized
#pragma omp parallel for reduction(+ : counter)
        for (int i = 0; i < num_vertices; i++)
        {
            std::lock_guard<std::mutex> guard(mutex); // Synchronize access to shared resources
            if (adj_list[i].size() == 0)
            {
                offsets.push_back(counter);
            }
            else
            {
                for (auto j : adj_list[i])
                {
                    neighbors.push_back(j);
                }
                counter += adj_list[i].size();
                offsets.push_back(counter);
            }
        }

        std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
        std::cout << "Time to process adjacency lists (offsets): " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - begin2).count() << "[ms]" << std::endl;
    }

    std::vector<int> neighbors;
    std::vector<int> offsets;

private:
    std::vector<std::vector<int>> adj_list;
    std::mutex mutex; // To guard against race conditions during parallel execution
};
