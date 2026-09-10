#pragma once


#include <nexusflow/Message.hpp>
#include <nexusflow/Module.hpp>

class MyDecoderModule : public nexusflow::Module {
    

public:
    MyDecoderModule(const std::string& name);
    ~MyDecoderModule() override;

    nexusflow::ErrorCode Configure(const nexusflow::Config& config) override;

protected:
    void Process(nexusflow::Message& inputMessage) override;

private:
    uint32_t m_skipInterval = 1; // skip every n-th message
    uint32_t m_frameIdx = 0; // frame index
};
