#include "ImportGraph.h"
#include <iostream>

int main()
{
    Graph g("test.txt");

    std::vector<int> neighbors = g.neighbors;
    std::vector<int> offsets = g.offsets;

    for (int i = 0; i < neighbors.size(); i++)
    {
        std::cout << neighbors[i] << " ";
    }

    std::cout << std::endl;

    for (int i = 0; i < offsets.size(); i++)
    {
        std::cout << offsets[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}