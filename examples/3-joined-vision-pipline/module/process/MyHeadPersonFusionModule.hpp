#pragma once

#include "../MyMessage.hpp"
#include "nexusflow/ErrorCode.hpp"

#include <nexusflow/Message.hpp>
#include <nexusflow/Module.hpp>

class MyHeadPersonFusionModule : public nexusflow::Module {
    

public:
    MyHeadPersonFusionModule(const std::string& name);
    ~MyHeadPersonFusionModule() override;

    nexusflow::ErrorCode Configure(const nexusflow::Config& config) override;

    nexusflow::ErrorCode Init() override;

protected:
    void Process(nexusflow::Message& inputMessage) override;

private:
    InferenceMessage DoFusion(const InferenceMessage& headMessage, const InferenceMessage& personMessage) const;

private:
    std::string m_modelPath;
};
