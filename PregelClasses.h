#ifndef PREGEL_VERTEX_H
#define PREGEL_VERTEX_H

#include <vector>
#include <queue>
#include <algorithm>

template <typename VertexValue, typename EdgeValue, typename MessageValue>
class PregelVertex
{
public:
    using VertexId = int;

    PregelVertex(VertexId id, VertexValue value)
        : id_(id), value_(value), active_(true) {}

    virtual ~PregelVertex() = default;

    // The main compute function to be implemented by the user
    virtual void Compute(const std::vector<MessageValue> &messages) = 0;

    // Send a message to another vertex
    void SendMessage(VertexId target, const MessageValue &message)
    {
        outgoing_messages_.push({target, message});
    }

    // Vote to halt the computation for this vertex
    void VoteToHalt()
    {
        active_ = false;
    }

    // Accessor methods
    VertexId GetId() const { return id_; }
    VertexValue &GetValue() { return value_; }
    const VertexValue &GetValue() const { return value_; }
    bool IsActive() const { return active_; }

    // Methods for the Pregel system to use
    void SetActive(bool active) { active_ = active; }
    std::queue<std::pair<VertexId, MessageValue>> &GetOutgoingMessages() { return outgoing_messages_; }
    void ClearOutgoingMessages()
    {
        while (!outgoing_messages_.empty())
            outgoing_messages_.pop();
    }

protected:
    VertexId id_;
    VertexValue value_;
    bool active_;
    std::vector<std::pair<VertexId, EdgeValue>> edges_;
    std::queue<std::pair<VertexId, MessageValue>> outgoing_messages_;
};

#endif // PREGEL_VERTEX_H