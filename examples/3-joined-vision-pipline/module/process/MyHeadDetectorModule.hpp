#pragma once

#include "nexusflow/ErrorCode.hpp"
#include <nexusflow/Message.hpp>
#include <nexusflow/Module.hpp>

class MyHeadDetectorModule : public nexusflow::Module {
public:
    MyHeadDetectorModule(const std::string& name);
    ~MyHeadDetectorModule() override;

    nexusflow::ErrorCode Configure(const nexusflow::Config& config) override;

    nexusflow::ErrorCode Init() override;

protected:
    void Process(nexusflow::Message& inputMessage) override;

private:
    std::string m_modelPath;
};
