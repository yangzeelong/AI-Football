#ifndef NEXUSFLOW_CONFIG_HPP
#define NEXUSFLOW_CONFIG_HPP

#include "nexusflow/Any.hpp"
#include <string>
#include <unordered_map>
namespace nexusflow {

class Config {
public:
    void Reset(std::unordered_map<std::string, Any> cfgMap) { m_cfgMap = std::move(cfgMap); }

    template <typename T>
    void Add(const std::string& key, const T& value) {
        m_cfgMap[key] = value;
    }

    template <typename T>
    T GetValueOrDefault(const std::string& key, const T& defaultValue) const {
        auto it = m_cfgMap.find(key);
        if (it == m_cfgMap.end() || !it->second.hasValue<T>()) {
            return defaultValue;
        }

        auto* res = it->second.get<T>();
        if (res != nullptr) {
            return *res;
        } else {
            return defaultValue;
        }
    }

    const std::unordered_map<std::string, Any>& GetConfigMap() const { return m_cfgMap; }

private:
    std::unordered_map<std::string, Any> m_cfgMap;
};

} // namespace nexusflow

#endif