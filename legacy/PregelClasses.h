
#ifndef PREGEL_VERTEX_H
#define PREGEL_VERTEX_H
#include <hip/hip_runtime.h>
#include <vector>
#include <queue>
#include <algorithm>
#include "ThreadSafeVector.h"
template <typename VertexValue, typename EdgeValue, typename MessageValue>
class PregelVertex
{
public:
    using VertexId = int;

    PregelVertex(VertexId id, VertexValue value)
        : id_(id), value_(value), active_(true) {}

    virtual ~PregelVertex() = default;

    // Accessor methods
    VertexId GetId() const { return id_; }
    __host__ __device__ VertexValue &GetValue() { return value_; }
    const VertexValue &GetValue() const { return value_; }
    bool IsActive() const { return active_; }

    // Methods for the Pregel system to use
    void SetActive(bool active) { active_ = active; }
    void SetValue(const VertexValue &value) { value_ = value; }
    std::queue<std::pair<VertexId, MessageValue>> &GetOutgoingMessages() { return outgoing_messages_; }
    void ClearOutgoingMessages()
    {
        while (!outgoing_messages_.empty())
            outgoing_messages_.pop();
    }

    // Method to add a neighbor to the vertex
    void AddNeighbor(VertexId neighborId, EdgeValue edgeValue)
    {
        edges_.emplace_back(neighborId, edgeValue);
    }
    using NeighborIterator = typename std::vector<std::pair<VertexId, EdgeValue>>::iterator;

    __host__ __device__ std::vector<std::pair<VertexId, EdgeValue>> &GetEdges() { return edges_; }

protected:
    VertexId id_;
    VertexValue value_;
    bool active_;
    std::vector<std::pair<VertexId, EdgeValue>> edges_;
    std::queue<std::pair<VertexId, MessageValue>> outgoing_messages_;
};
#endif // PREGEL_VERTEX_H