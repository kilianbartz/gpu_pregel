#include <string>
#include <vector>
#include <fstream>
#include <sstream>
class Graph
{
public:
    int num_vertices;

    Graph(std::string filename)
    {
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
    }

    int *get_neighbors()
    {
        int *n = &neighbors[0];
        return n;
    }

    int *get_offsets()
    {
        int *o = &offsets[0];
        return o;
    }

private:
    std::vector<std::vector<int>> adj_list;
    std::vector<int> neighbors;
    std::vector<int> offsets;
};