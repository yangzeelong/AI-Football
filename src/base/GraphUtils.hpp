
#pragma once

#include "base/Graph.hpp"
#include "nexusflow/Any.hpp"
#include <yaml-cpp/yaml.h>

namespace graphutils {

nexusflow::Any convertYamlNodeToAny(const YAML::Node& node);
std::unique_ptr<Graph> CreateGraphFromYaml(const std::string& configPath);

}
