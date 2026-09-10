
#pragma once

#include "base/Graph.hpp"

namespace graphutils {

std::unique_ptr<Graph> CreateGraphFromYaml(const std::string& configPath);

}
