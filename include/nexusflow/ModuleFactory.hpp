#ifndef NEXUSFLOW_MODULE_FACTORY_HPP
#define NEXUSFLOW_MODULE_FACTORY_HPP

#include <nexusflow/Module.hpp>

#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace nexusflow {

/**
 * @class ModuleFactory
 * @brief A singleton factory for creating Module instances from class names.
 *
 * This factory allows users to register their custom Module-derived classes
 * with a string identifier. The framework can then dynamically create instances
 * of these modules at runtime based on their class name.
 *
 * This version of the factory assumes all registerable modules have a public
 * constructor with the signature: `MyModule(std::string name)`.
 */
class ModuleFactory {
public:
    /**
     * @brief Gets the singleton instance of the factory.
     * @return A reference to the single ModuleFactory instance.
     */
    static ModuleFactory& GetInstance();

    // Delete copy/move constructors and assignment operators for the singleton pattern.
    ModuleFactory(const ModuleFactory&) = delete;
    void operator=(const ModuleFactory&) = delete;

    /**
     * @brief Registers a Module-derived class with the factory.
     *
     * This method is a template and its definition must reside in the header file.
     * It registers a creator function for any class `T` that has a public
     * constructor `T(std::string)`.
     *
     * @tparam T The Module-derived class type to register.
     * @param className The string name to associate with the class.
     *
     * @example
     *   // In your application's setup code:
     *   nexusflow::ModuleFactory::getInstance().Register<MySimpleModule>("MySimpleModule");
     */
    template <typename T>
    void Register(const std::string& className) {
        // Compile-time check to ensure T is derived from Module.
        static_assert(std::is_base_of<Module, T>::value, "Registered type must be a descendant of nexusflow::Module.");

        // Compile-time check to ensure T has the required constructor T(std::string).
        static_assert(std::is_constructible<T, std::string>::value,
                      "Registered type must have a public constructor like MyModule(std::string name)");

        m_creators[className] = [](const std::string& instanceName) { return std::make_shared<T>(instanceName); };
    }

    /**
     * @brief Creates an instance of a registered module using its class name.
     *
     * @param className The string name of the class to instantiate.
     * @param moduleName The unique instance name to pass to the module's constructoro.
     * @param config The modules's configuration.
     * @return A std::shared_ptr to the newly created Module instance, or nullptr if the
     *         class name is not registered.
     */
    std::shared_ptr<Module> CreateModule(const std::string& className, const std::string& moduleName, const Config& config);

private:
    // A type alias for the creator function. It takes an instance name.
    using CreatorFunc = std::function<std::shared_ptr<Module>(const std::string&)>;

    // A map from class name to its corresponding creator function.
    std::unordered_map<std::string, CreatorFunc> m_creators;

    // Private constructor to enforce the singleton pattern.
    ModuleFactory() = default;
};

} // namespace nexusflow

#define NEXUSFLOW_REGISTER_MODULE(moduleClassName)                                           \
    static bool registrar_##moduleClassName = []() {                                         \
        nexusflow::ModuleFactory::GetInstance().Register<moduleClassName>(#moduleClassName); \
        return true;                                                                         \
    }();

#endif // NEXUSFLOW_MODULEFACTORY_HPP