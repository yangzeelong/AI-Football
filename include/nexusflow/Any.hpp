#ifndef NEXUSFLOW_ANY_HPP
#define NEXUSFLOW_ANY_HPP

#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>

namespace nexusflow {

/**
 * @class Any
 * @brief A type-erased container for holding a single value of any type.
 *
 * A simplified, C++14-compatible implementation inspired by std::any. It allows
 * storing different types in a type-safe manner. It is used to pass
 * module-specific parameters from configuration to module instances.
 */
class Any {
public:
    // --- Constructors ---
    Any() noexcept = default;

    template <typename T, typename = typename std::enable_if<!std::is_same<typename std::decay<T>::type, Any>::value>::type>
    Any(T&& value) : m_content(std::make_unique<Model<typename std::decay<T>::type>>(std::forward<T>(value))) {}

    // --- Copy and Move Semantics ---
    Any(const Any& other) : m_content(other.m_content ? other.m_content->clone() : nullptr) {}

    Any& operator=(const Any& other) {
        if (this != &other) {
            m_content = other.m_content ? other.m_content->clone() : nullptr;
        }
        return *this;
    }

    Any(Any&& other) noexcept = default;
    Any& operator=(Any&& other) noexcept = default;

    // --- Public Interface ---
    bool hasValue() const noexcept { return m_content != nullptr; }

    template <typename T>
    bool hasValue() const noexcept {
        return m_content && typeid(T) == m_content->getTypeInfo();
    }

    const std::type_info& getType() const noexcept { return m_content ? m_content->getTypeInfo() : typeid(void); }

    /**
     * @brief Type-safe cast to the contained value.
     * @tparam T The type to cast to.
     * @return A pointer to the value if the cast is successful, otherwise nullptr.
     */
    template <typename T>
    T* get() noexcept {
        if (!m_content || typeid(T) != m_content->getTypeInfo()) {
            return nullptr;
        }
        return &static_cast<Model<T>*>(m_content.get())->m_data;
    }

    template <typename T>
    const T* get() const noexcept {
        // Implementation for const version
        if (!m_content || typeid(T) != m_content->getTypeInfo()) {
            return nullptr;
        }
        return &static_cast<const Model<T>*>(m_content.get())->m_data;
    }

private:
    // --- Internal Type-Erasure Implementation ---
    struct Concept {
        virtual ~Concept() = default;
        virtual const std::type_info& getTypeInfo() const noexcept = 0;
        virtual std::unique_ptr<Concept> clone() const = 0;
    };

    template <typename T>
    struct Model final : Concept {
        explicit Model(T data) : m_data(std::move(data)) {}
        const std::type_info& getTypeInfo() const noexcept override { return typeid(T); }
        std::unique_ptr<Concept> clone() const override { return std::make_unique<Model<T>>(m_data); }
        T m_data;
    };

    std::unique_ptr<Concept> m_content;
};

} // namespace nexusflow

#endif
