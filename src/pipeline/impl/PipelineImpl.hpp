#ifndef NEXUSFLOW_PIPELINE_PIPELINEIMPL_HPP
#define NEXUSFLOW_PIPELINE_PIPELINEIMPL_HPP

#include "base/Define.hpp"
#include "base/Graph.hpp"
#include "core/Worker.hpp"
#include "dispatcher/Dispatcher.hpp"
#include "module/ModuleActor.hpp"
#include <nexusflow/Pipeline.hpp>
#include <set>
#include <string>

namespace nexusflow {

// Forward declarations
class Pipeline;

// An internal struct to group all runtime components related to a single module.
using ActorName = std::string;
using ActorNode = ModuleActor;

// --- Pipeline's Private Implementation (m_pImpl) ---
class Pipeline::Impl {
public:
    std::unique_ptr<Graph> graph;
    std::vector<MessageQueueUPtr> queues;

    std::set<std::shared_ptr<ActorNode>> actorOrderedNodes;

    ErrorCode Init();

private:
    std::shared_ptr<ActorNode> GetOrCreateActorNode(const std::shared_ptr</*Graph::*/ Node>& node);

    std::unordered_map<ActorName, std::shared_ptr<ActorNode>> actorModuleMap;
};

} // namespace nexusflow

#endif // NEXUSFLOW_PIPELINE_PIPELINEIMPL_HPP
