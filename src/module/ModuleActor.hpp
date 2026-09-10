#ifndef NEXUSFLOW_MODULE_ACTOR_HPP
#define NEXUSFLOW_MODULE_ACTOR_HPP

#include "core/Worker.hpp"
#include "dispatcher/Dispatcher.hpp"
#include "nexusflow/ErrorCode.hpp"
#include <nexusflow/Module.hpp>
#include <thread>

namespace nexusflow {

class ModuleActor {
public:
    ModuleActor(const std::shared_ptr<Module>& module, const Config& config);

    ~ModuleActor();

    void AddInputQueue(const std::string& name, ViewPtr<MessageQueue> queue) { m_worker->AddQueue(name, queue); }

    void AddOutputQueue(const std::string& name, ViewPtr<MessageQueue> queue) { m_dispatcher->AddSubscriber(name, queue); }

    std::shared_ptr<Module>& GetModule() { return m_module; };

    std::string GetModuleName() const { return m_module->GetModuleName(); }

    ErrorCode Init();

    ErrorCode DeInit();

    ErrorCode Start();

    ErrorCode Stop();

private:
    std::shared_ptr<core::Worker>& GetWorker() { return m_worker; }
    std::shared_ptr<dispatcher::Dispatcher>& GetDispatcher() { return m_dispatcher; }

    std::shared_ptr<Module> m_module;
    std::shared_ptr<core::Worker> m_worker;
    std::shared_ptr<dispatcher::Dispatcher> m_dispatcher;
    std::unique_ptr<Config> m_config;

    std::thread m_workThread;
};
} // namespace nexusflow

#endif