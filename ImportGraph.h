#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
class Graph
{
public:
    int num_vertices = 0;

    Graph(std::string filename)
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        std::string line;
        std::ifstream file(filename);
        int a, b;
        while (std::getline(file, line))
        {
            if (line[0] == '#')
                continue;
            std::stringstream iss(line);
            iss >> a >> b;
            // check if a is in the adj_list
            int m = std::max(a, b);
            if (m >= adj_list.size())
            {
                adj_list.resize(m + 1);
            }
            adj_list[a].push_back(b);
        }
        file.close();
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Time to read the file: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
        num_vertices = adj_list.size();
        int counter = 0;
        offsets.push_back(0);
        for (int i = 0; i < num_vertices; i++)
        {
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
        std::cout << "Time to process adjacency lists: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end).count() << "[ms]" << std::endl;
    }

    std::vector<int> neighbors;
    std::vector<int> offsets;

private:
    std::vector<std::vector<int>> adj_list;
};