#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>
#include <limits>

class Graph
{
public:
    int num_vertices = 0;
    std::vector<int> neighbors;
    std::vector<int> offsets;
    std::vector<int> weights; // New vector to store edge weights

    Graph(std::string filename)
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        std::string line;
        std::ifstream file(filename);
        int a, b, w;
        while (std::getline(file, line))
        {
            if (line[0] == '#')
                continue;
            std::stringstream iss(line);
            if (!(iss >> a >> b >> w)) // Try to read weight
            {
                iss.clear();
                iss.str(line);
                iss >> a >> b;
                w = 1; // Default weight if not provided
            }
            int m = std::max(a, b);
            if (m >= adj_list.size())
            {
                adj_list.resize(m + 1);
                adj_weights.resize(m + 1); // Resize weights list as well
            }
            adj_list[a].push_back(b);
            adj_weights[a].push_back(w);
        }
        file.close();
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Time to read the file: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

        num_vertices = adj_list.size();
        int counter = 0;
        offsets.push_back(0);
        for (int i = 0; i < num_vertices; i++)
        {
            if (adj_list[i].empty())
            {
                offsets.push_back(counter);
            }
            else
            {
                for (size_t j = 0; j < adj_list[i].size(); j++)
                {
                    neighbors.push_back(adj_list[i][j]);
                    weights.push_back(adj_weights[i][j]);
                }
                counter += adj_list[i].size();
                offsets.push_back(counter);
            }
        }
        std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
        std::cout << "Time to process adjacency lists: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end).count() << "[ms]" << std::endl;
    }

private:
    std::vector<std::vector<int>> adj_list;
    std::vector<std::vector<int>> adj_weights; // New vector to store weights in adjacency list
};