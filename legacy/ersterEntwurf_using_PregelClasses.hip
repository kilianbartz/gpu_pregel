#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>

#ifndef PREGEL_VERTEX_H
#define PREGEL_VERTEX_H

// Templated thread-safe vector class
template <typename T>
class ThreadSafeVector
{
public:
    __host__ ThreadSafeVector(size_t capacity) : size_(0), capacity_(capacity)
    {
        hipMalloc(&data_, capacity * sizeof(T));
        hipMalloc(&d_size_, sizeof(int));
        hipMemcpy(d_size_, &size_, sizeof(int), hipMemcpyHostToDevice);
    }

    __host__ __device__ ~ThreadSafeVector()
    {
        if (!hipGetLastError())
        {
            hipFree(data_);
            hipFree(d_size_);
        }
    }

    __device__ void append(const T &value)
    {
        int index = atomicAdd(d_size_, 1);
        if (index < capacity_)
        {
            data_[index] = value;
        }
        else
        {
            atomicSub(d_size_, 1);
        }
    }

    __host__ int getSize() const
    {
        int host_size;
        hipMemcpy(&host_size, d_size_, sizeof(int), hipMemcpyDeviceToHost);
        return host_size;
    }

    __host__ __device__ T *data() const { return data_; }
    __device__ int *size() const { return d_size_; }
    __device__ T &operator[](int index) { return data_[index]; }

private:
    T *data_;
    int *d_size_;
    int size_;
    size_t capacity_;
};

template <typename VertexValue, typename EdgeValue, typename MessageValue>
class PregelVertex
{
public:
    using VertexId = int;

    PregelVertex(VertexId id, VertexValue value)
        : id_(id), value_(value), active_(true) {}

    virtual ~PregelVertex() = default;

    __host__ __device__ VertexId GetId() const { return id_; }
    __host__ __device__ VertexValue &GetValue() { return value_; }
    __host__ __device__ const VertexValue &GetValue() const { return value_; }
    __host__ __device__ bool IsActive() const { return active_; }

    __host__ __device__ void SetActive(bool active) { active_ = active; }

    __host__ __device__ void AddNeighbor(VertexId neighborId, EdgeValue edgeValue)
    {
        edges_.emplace_back(neighborId, edgeValue);
    }

    __host__ __device__ std::vector<std::pair<VertexId, EdgeValue>> &GetEdges() { return edges_; }

protected:
    VertexId id_;
    VertexValue value_;
    bool active_;
    std::vector<std::pair<VertexId, EdgeValue>> edges_;
    std::queue<std::pair<VertexId, MessageValue>> outgoing_messages_;
};

#endif // PREGEL_VERTEX_H

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
