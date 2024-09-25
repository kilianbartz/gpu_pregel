#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include "PregelClasses.h"
#include "ThreadSafeVector.h"

#define DELTA 0.85

__global__ void
pagerank_superstep(PregelVertex<double, double, double> *vertices, ThreadSafeVector<double> *messageBoardOld, ThreadSafeVector<double> *messageBoardNew, int num_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_elements)
        return;

    PregelVertex<double, double, double> *vertex = &vertices[tid];

    if (vertex->IsActive() || *messageBoardOld[tid].size() > 0)
    {
        vertex->SetActive(true);
        double sum = 0;
        for (int i = 0; i < messageBoardOld[tid].getSize(); i++)
        {
            sum += messageBoardOld[tid][i];
        }
        vertex->SetValue((1 - DELTA) / num_elements + DELTA * sum);
        vertex->SetActive(false);
    }

    int num_neighbors = vertex->GetEdges().size();
    for (auto &edge : vertex->GetEdges())
    {
        auto neighbor_id = edge.first;
        messageBoardNew[neighbor_id].append(vertex->GetValue() / num_neighbors);
    }
}

int main()
{
    const int num_elements = 2;
    const int num_iterations = 10;

    PregelVertex<double, double, double> vertex1(0, 1.0);
    PregelVertex<double, double, double> vertex2(1, 1.0);
    vertex1.AddNeighbor(1, 1.0);
    vertex2.AddNeighbor(0, 1.0);
    PregelVertex<double, double, double> vertices[num_elements] = {vertex1, vertex2};

    ThreadSafeVector<double> messageBoardOld[num_elements] = {ThreadSafeVector<double>(10), ThreadSafeVector<double>(10)};
    ThreadSafeVector<double> messageBoardNew[num_elements] = {ThreadSafeVector<double>(10), ThreadSafeVector<double>(10)};

    PregelVertex<double, double, double> *d_vertices;
    hipMalloc(&d_vertices, num_elements * sizeof(PregelVertex<double, double, double>));
    hipMemcpy(d_vertices, vertices, num_elements * sizeof(PregelVertex<double, double, double>), hipMemcpyHostToDevice);

    ThreadSafeVector<double> *d_messageBoardOld;
    hipMalloc(&d_messageBoardOld, num_elements * sizeof(ThreadSafeVector<double>));
    hipMemcpy(d_messageBoardOld, messageBoardOld, num_elements * sizeof(ThreadSafeVector<double>), hipMemcpyHostToDevice);

    ThreadSafeVector<double> *d_messageBoardNew;
    hipMalloc(&d_messageBoardNew, num_elements * sizeof(ThreadSafeVector<double>));
    hipMemcpy(d_messageBoardNew, messageBoardNew, num_elements * sizeof(ThreadSafeVector<double>), hipMemcpyHostToDevice);

    for (int i = 0; i < num_iterations; i++)
    {
        hipLaunchKernelGGL(pagerank_superstep, dim3(32), dim3(32), 0, 0, d_vertices, d_messageBoardOld, d_messageBoardNew, num_elements);
        hipDeviceSynchronize();
        hipMemcpy(d_messageBoardOld, d_messageBoardNew, num_elements * sizeof(ThreadSafeVector<double>), hipMemcpyDeviceToDevice);
    }

    hipMemcpy(vertices, d_vertices, num_elements * sizeof(PregelVertex<double, double, double>), hipMemcpyDeviceToHost);

    for (int i = 0; i < num_elements; i++)
    {
        std::cout << "Vertex " << i << " has value " << vertices[i].GetValue() << std::endl;
    }

    return 0;
}
