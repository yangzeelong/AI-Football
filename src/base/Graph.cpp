#include "Graph.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

void Graph::addEdge(const std::shared_ptr<Node>& srcNodePtr, const std::shared_ptr<Node>& dstNodePtr) {
    if (srcNodePtr == nullptr || dstNodePtr == nullptr) return;

    m_nodeMap[srcNodePtr->name] = srcNodePtr;
    m_nodeMap[dstNodePtr->name] = dstNodePtr;

    m_adjList[srcNodePtr].emplace_back(dstNodePtr);
}

bool Graph::hasCycle() const { return checkCycleAndConvertToEdgeList(nullptr).first; }

std::vector<Edge> Graph::toEdgeListBFS(const std::shared_ptr<Node>& inputNodePtr) const {
    return checkCycleAndConvertToEdgeList(inputNodePtr).second;
}

std::pair<bool, std::vector<Edge>> Graph::checkCycleAndConvertToEdgeList(const std::shared_ptr<Node>& inputNodePtr) const {
    std::vector<Edge> edgeList;
    std::unordered_map<std::shared_ptr<Node>, int> inDegree;
    std::unordered_set<std::shared_ptr<Node>> allNodes;

    for (const auto& entry : m_adjList) {
        allNodes.insert(entry.first);
        if (inDegree.find(entry.first) == inDegree.end()) {
            inDegree[entry.first] = 0;
        }
        for (const auto& neighbor : entry.second) {
            allNodes.insert(neighbor);
            inDegree[neighbor]++;
        }
    }

    // 如果指定inputNodePtr, 则根从这个节点为根节点开始搜索.
    std::queue<std::shared_ptr<Node>> nodeQueue;
    if (inputNodePtr != nullptr) {
        nodeQueue.push(inputNodePtr);
        inDegree[inputNodePtr] = 0;
    } else {
        for (const auto& node : allNodes) {
            if (inDegree[node] == 0) {
                nodeQueue.push(node);
            }
        }
    }

    int visitedCount = 0;
    while (!nodeQueue.empty()) {
        auto node = nodeQueue.front();
        nodeQueue.pop();
        visitedCount++;

        auto iter = m_adjList.find(node);
        if (iter != m_adjList.end()) {
            // 为当前节点的邻居创建一个临时的集合，用于去重
            std::unordered_set<std::shared_ptr<Node>> processedNeighbors;

            for (const auto& neighbor : iter->second) {
                // 1. 入度计算：这部分逻辑必须对每一条边都执行，即使是重复的。
                if (--inDegree[neighbor] == 0) {
                    nodeQueue.push(neighbor);
                }

                // 2. 边列表构建：这部分逻辑只对唯一的邻居执行一次。
                //    我们检查这个邻居是否在本次循环中被处理过。
                if (processedNeighbors.find(neighbor) == processedNeighbors.end()) {
                    edgeList.push_back({node, neighbor});
                    processedNeighbors.insert(neighbor); // 标记为已处理
                }
            }
        }
    }

    // 如果访问的节点数量小于图中的节点数量，则存在环
    return {visitedCount != allNodes.size(), std::move(edgeList)};
}

std::string Graph::toString() const {
    auto edgeList = toEdgeListBFS();

    std::ostringstream oss;
    oss << "[" + m_name + "]: name=" + m_name + ", graph: \n";

    for (auto&& edge : edgeList) {
        auto srcNodePtr = edge.srcNodePtr.lock();
        auto dstNodePtr = edge.dstNodePtr.lock();
        oss << "  " << srcNodePtr->name + " -> " << dstNodePtr->name << "\n";
    }
    return oss.str();
}
