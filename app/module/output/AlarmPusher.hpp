#pragma once

#include <nexusflow/Nexusflow.hpp>

namespace ns = nexusflow;

class AlarmPusher : public ns::Module {
public:
    AlarmPusher(const std::string& name);

    ~AlarmPusher() override;

    // --- Lifecycle ---
    ns::ErrorCode Configure(const ns::Config& config) override;

protected:
    void Process(ns::Message& inputMessage) override;
};

NEXUSFLOW_REGISTER_MODULE(AlarmPusher);
