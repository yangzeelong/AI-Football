#include "GraphUtils.hpp"
#include "base/Define.hpp"
#include "base/Graph.hpp"
#include "nexusflow/Any.hpp"
#include "nexusflow/Config.hpp"
#include "utils/logging.hpp"
#include "yaml-cpp/node/node.h"
#include "yaml-cpp/yaml.h"

#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

namespace graphutils {

namespace detail {

// 判断字符串是否是整数
inline bool isInteger(const std::string& str) {
    if (str.empty()) return false;
    size_t i = (str[0] == '-' || str[0] == '+') ? 1 : 0;
    for (; i < str.size(); ++i) {
        if (!std::isdigit(str[i])) return false;
    }
    return true;
}

// 判断字符串是否是浮点数
inline bool isFloat(const std::string& str) {
    std::istringstream iss(str);
    float f;
    char c;
    return (iss >> f) && !(iss >> c);
}

// 判断是否是 bool
inline bool isBool(const std::string& str, bool& out) {
    if (str == "true" || str == "True" || str == "yes") {
        out = true;
        return true;
    }
    if (str == "false" || str == "False" || str == "no") {
        out = false;
        return true;
    }
    return false;
}

} // namespace detail

// TODO: 待优化
nexusflow::Any convertYamlNodeToAny(const YAML::Node& node) {
    using nexusflow::Any;
    switch (node.Type()) {
        case YAML::NodeType::Null: return Any();

        case YAML::NodeType::Scalar: {
            std::string val = node.Scalar();
            bool b;
            if (detail::isBool(val, b)) return Any(b);
            if (detail::isInteger(val)) {
                try {
                    return Any(std::stoi(val));
                } catch (...) {
                }
            }
            if (detail::isFloat(val)) {
                try {
                    return Any(std::stod(val));
                } catch (...) {
                }
            }
            return Any(val); // fallback: string
        }

        case YAML::NodeType::Sequence: {
            std::vector<Any> vec;
            for (const auto& item : node) {
                vec.push_back(convertYamlNodeToAny(item));
            }
            return Any(std::move(vec));
        }

        case YAML::NodeType::Map: {
            std::map<std::string, Any> map;
            for (const auto& kv : node) {
                std::string key = kv.first.as<std::string>();
                map[key] = convertYamlNodeToAny(kv.second);
            }
            return Any(std::move(map));
        }

        case YAML::NodeType::Undefined:
        default: throw std::runtime_error("Unsupported or undefined YAML node type.");
    }
}

std::unique_ptr<Graph> CreateGraphFromYaml(const std::string& configPath) {
    try {
        YAML::Node root = YAML::LoadFile(configPath);

        const YAML::Node& graph_yaml = root["graph"];
        if (!graph_yaml) {
            LOG_ERROR("YAML configuration must contain a 'graph' root node in '{}'", configPath);
            return nullptr;
        }

        auto graph = std::make_unique<Graph>();

        // 1. 设置图的名称
        if (!graph_yaml["name"]) {
            LOG_ERROR("Graph configuration must have a 'name' under 'graph' section in '{}'", configPath);
            return nullptr;
        }
        graph->setName(graph_yaml["name"].as<std::string>());
        LOG_INFO("Start creating graph '{}' from config: {}", graph->getName(), configPath);

        // 2. 创建所有节点
        std::unordered_map<std::string, std::shared_ptr<Node>> tempNodeMap;
        const YAML::Node& modules_yaml = graph_yaml["modules"];
        if (!modules_yaml || !modules_yaml.IsSequence()) {
            LOG_ERROR("'modules' section is missing or not a sequence in '{}'", configPath);
            return nullptr;
        }

        for (const auto& module_item : modules_yaml) {
            std::string nodeName = module_item["name"].as<std::string>();
            std::string moduleClassName = module_item["class"].as<std::string>();

            // parse custom config.
            nexusflow::Config config;
            const YAML::Node& configsNode = module_item["config"];
            if (configsNode && configsNode.IsMap()) {
                for (const auto& kv : configsNode) {
                    std::string key = kv.first.as<std::string>();
                    config.Add(key, convertYamlNodeToAny(kv.second));
                }
            }

            auto node = std::make_shared<NodeWithModuleClassName>(nodeName, moduleClassName, std::move(config));

            auto item = tempNodeMap.emplace(nodeName, node);
            if (!item.second) {
                LOG_ERROR("Duplicate module name found: {} in '{}'", nodeName, configPath);
                return nullptr;
            }
        }
        LOG_INFO("Created {} nodes from 'modules' section.", tempNodeMap.size());

        // 3. 创建边，并计算度
        std::unordered_map<std::string, int> inDegree;
        std::unordered_map<std::string, int> outDegree;
        for (const auto& pair : tempNodeMap) {
            inDegree[pair.first] = 0;
            outDegree[pair.first] = 0;
        }

        const YAML::Node& connections_yaml = graph_yaml["connections"];
        if (connections_yaml && connections_yaml.IsSequence()) {
            for (const auto& connection_item : connections_yaml) {
                std::string fromName = connection_item["from"].as<std::string>();
                std::string toName = connection_item["to"].as<std::string>();

                auto srcIt = tempNodeMap.find(fromName);
                auto dstIt = tempNodeMap.find(toName);
                if (srcIt == tempNodeMap.end() || dstIt == tempNodeMap.end()) {
                    LOG_ERROR("Connection '{} -> {}' refers to a non-existent module.", fromName, toName);
                    return nullptr;
                }

                // 假设 addEdge 会将节点添加到 Graph 的内部 m_nodeMap 中
                graph->addEdge(srcIt->second, dstIt->second);
                outDegree[fromName]++;
                inDegree[toName]++;
            }
            LOG_INFO("Created {} connections.", connections_yaml.size());
        }

        // TODO:
        // // 4. 验证图的拓扑结构 (这是关键步骤！)
        // // 尽管 Graph 不再存储输入/输出节点，但我们必须验证它们是否存在且唯一
        // std::vector<std::shared_ptr<Node>> sourceNodes;
        // for (const auto& pair : tempNodeMap) {
        //     if (inDegree.at(pair.first) == 0) {
        //         sourceNodes.push_back(pair.second);
        //     }
        // }

        // if (sourceNodes.size() != 1) {
        //     LOG_ERROR("Graph must have exactly one source node (in-degree 0). Found {}.", sourceNodes.size());
        //     return nullptr;
        // }
        // LOG_INFO("Graph validation passed: Found a single source node '{}'.", sourceNodes[0]->name);

        // // 可选：同样可以验证汇节点（输出）
        // // std::vector<std::shared_ptr<Node>> sinkNodes;
        // // for (const auto& pair : tempNodeMap) {
        // //     if (outDegree.at(pair.first) == 0) {
        // //         sinkNodes.push_back(pair.second);
        // //     }
        // // }
        // // if (sinkNodes.size() != 1) {
        // //     LOG_ERROR("Graph must have exactly one sink node (out-degree 0). Found {}.", sinkNodes.size());
        // //     return nullptr;
        // // }
        // // LOG_INFO("Graph validation passed: Found a single sink node '{}'.", sinkNodes[0]->name);

        // 5. 最终校验
        if (graph->hasCycle()) {
            LOG_ERROR("The constructed graph '{}' has a cycle.", graph->getName());
            return nullptr;
        }
        if (graph->isEmpty()) {
            LOG_ERROR("The constructed graph '{}' is empty or incomplete.", graph->getName());
            return nullptr;
        }

        LOG_INFO("Successfully created and validated graph '{}'.", graph->getName());
        return graph;

    } catch (const YAML::Exception& e) {
        LOG_ERROR("Failed to process YAML file '{}' due to a parsing error: {}", configPath, e.what());
        return nullptr;
    }
}

} // namespace graphutils