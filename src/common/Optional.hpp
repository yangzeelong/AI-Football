#ifndef OPTIONAL_HPP_
#define OPTIONAL_HPP_

#include <memory>
#include <type_traits>
#include <stdexcept>

template <class T>
struct is_always_false : std::false_type {};

namespace {
struct OptionalNullStruct {};
} // namespace

static constexpr OptionalNullStruct nullOpt{};

// Optional is a class template that provides a way to represent an optional value.
// It can be used to avoid null pointer dereferencing and to handle cases where a value may or may not be present.
// It is similar to the Optional class in the Rust programming language.
template <class T>
class Optional {
public:
    // Constructors and destructors
    Optional() noexcept : m_hasValue(false) {}

    Optional(const OptionalNullStruct&) noexcept : m_hasValue(false) {}

    Optional(T value) : m_hasValue(true) { new (&m_value) T(std::move(value)); }

    Optional(const Optional& other) : m_hasValue(other.m_hasValue) {
        if (m_hasValue) {
            new (&m_value) T(other.m_value);
        }
    }

    template <typename... Args, typename = std::enable_if_t<std::is_constructible<T, Args...>::value>>
    Optional(Args&&... args) : m_hasValue(true) {
        new (&m_value) T(std::forward<Args>(args)...);
    }

    Optional(Optional&& other) noexcept(std::is_nothrow_move_constructible<T>::value) : m_hasValue(other.m_hasValue) {
        if (m_hasValue) {
            new (&m_value) T(std::move(other.m_value));
            other.m_hasValue = false;
        }
    }

    ~Optional() {
        if (m_hasValue) {
            m_value.~T();
            m_hasValue = false;
        }
    }

    // Operators overloading
    Optional& operator=(const Optional& other) {
        if (this != &other) {
            if (m_hasValue) {
                m_value.~T();
            }

            m_hasValue = other.m_hasValue;
            if (m_hasValue) {
                new (&m_value) T(other.m_value);
            }
        }
        return *this;
    }

    Optional& operator=(Optional&& other) noexcept(std::is_nothrow_move_constructible<T>::value) {
        if (this != &other) {
            if (m_hasValue) {
                m_value.~T();
            }

            m_hasValue = other.m_hasValue;
            if (m_hasValue) {
                new (&m_value) T(std::move(other.m_value));
                other.m_hasValue = false;
            }
        }
        return *this;
    }

    explicit operator bool() const noexcept { return m_hasValue; }

public:
    // Member functions
    inline bool hasValue() const noexcept { return m_hasValue; }

    void reset() noexcept {
        if (m_hasValue) {
            m_value.~T();
            m_hasValue = false;
        }
    }

    template <class U, typename = std::enable_if_t<std::is_constructible<T, U>::value || std::is_assignable<T, U>::value ||
                                                   std::is_enum<T>::value>>
    void reset(U&& u) {
        if (m_hasValue) {
            m_value.~T();
        }
        new (&m_value) T(std::forward<U>(u));
        m_hasValue = true;
    }

    template <typename... Args>
    T& emplace(Args&&... args) {
        reset();
        new (&m_value) T(std::forward<Args>(args)...);
        m_hasValue = true;
        return m_value;
    }

    T const& value() const {
        if (!m_hasValue) throw std::runtime_error("Option does not contain a value");
        return m_value;
    }

    T& value() {
        if (!m_hasValue) throw std::runtime_error("Option does not contain a value");
        return m_value;
    }

    T orElse(T val) const {
        if (hasValue()) {
            return value();
        }
        return val;
    }

    T& operator*() { return value(); }

    T const& operator*() const { return value(); }

    T* operator->() { return &value(); }

    T const* operator->() const { return &value(); }

private:
    union {
        T m_value;
    };
    bool m_hasValue;
};

template <class T>
class Optional<T*> {
    static_assert(is_always_false<T>::value, "Not Support Pointer Type Now.");
};

template <class T>
class Optional<std::reference_wrapper<T>> {
    static_assert(is_always_false<T>::value, "Not Support Refernce Type Now.");
};

#endif // OPTIONAL_HPP_