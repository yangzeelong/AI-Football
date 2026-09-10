
#pragma once
#include "nexusflow/Message.hpp"
#include <nexusflow/Module.hpp>

class MockOutputModule : public nexusflow::Module {
public:
    MockOutputModule(const std::string& name);
    ~MockOutputModule() override;

protected:
    void Process(nexusflow::Message& inputMessage) override;
};
