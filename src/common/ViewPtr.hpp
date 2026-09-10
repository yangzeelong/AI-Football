#ifndef VIEW_PTR_HPP
#define VIEW_PTR_HPP

#include <cassert>
#include <functional>
#include <type_traits>
#include <utility>

/**
 * @class ViewPtr
 * @brief A lightweight, non‑owning pointer wrapper for C++14.
 *
 * This class explicitly models "just observing a pointer" semantics.
 * It is a zero‑cost abstraction (same size & performance as a raw pointer)
 * and can be used wherever you would otherwise keep a naked T* that you
 * do **not** own.
 *
 * API intentionally mirrors a subset of std::observer_ptr from C++23 while
 * staying strictly within the C++14 feature set.
 *
 * @tparam T Pointed‑to type.
 */

template <typename T>
class ViewPtr {
public:
    // — STL‑style aliases —
    using element_type = T;
    using pointer = T*;
    using reference = T&;

    // — Constructors —
    constexpr ViewPtr() noexcept : m_ptr(nullptr) {}
    // constexpr ViewPtr(std::nullptr_t) noexcept : m_ptr(nullptr) {}
    explicit constexpr ViewPtr(pointer ptr) noexcept : m_ptr(ptr) {}

    // Allow implicit conversion from ViewPtr<Derived> → ViewPtr<Base>
    template <typename U, typename = typename std::enable_if<std::is_convertible<U*, T*>::value>::type>
    constexpr ViewPtr(const ViewPtr<U>& other) noexcept : m_ptr(other.get()) {}

    // — Observers —
    reference operator*() const {
        assert(m_ptr != nullptr && "Cannot dereference a null ViewPtr");
        return *m_ptr;
    }

    pointer operator->() const noexcept { return m_ptr; }

    constexpr pointer get() const noexcept { return m_ptr; }

    explicit constexpr operator bool() const noexcept { return m_ptr != nullptr; }

    // — Modifiers —
    void reset(pointer ptr = nullptr) noexcept { m_ptr = ptr; }

    void swap(ViewPtr& other) noexcept { std::swap(m_ptr, other.m_ptr); }

private:
    pointer m_ptr;
};

// — Free comparison operators —

template <typename T, typename U>
inline bool operator==(const ViewPtr<T>& lhs, const ViewPtr<U>& rhs) {
    return lhs.get() == rhs.get();
}

template <typename T, typename U>
inline bool operator!=(const ViewPtr<T>& lhs, const ViewPtr<U>& rhs) {
    return !(lhs == rhs);
}

template <typename T>
inline bool operator==(const ViewPtr<T>& p, std::nullptr_t) noexcept {
    return !p;
}

template <typename T>
inline bool operator==(std::nullptr_t, const ViewPtr<T>& p) noexcept {
    return !p;
}

template <typename T>
inline bool operator!=(const ViewPtr<T>& p, std::nullptr_t) noexcept {
    return static_cast<bool>(p);
}

template <typename T>
inline bool operator!=(std::nullptr_t, const ViewPtr<T>& p) noexcept {
    return static_cast<bool>(p);
}

// Optional but often handy: strict total order on address value

template <typename T, typename U>
inline bool operator<(const ViewPtr<T>& lhs, const ViewPtr<U>& rhs) {
    return lhs.get() < rhs.get();
}

// — Hash support so ViewPtr can be a key in unordered containers —

namespace std {
template <typename T>
struct hash<ViewPtr<T>> {
    std::size_t operator()(const ViewPtr<T>& p) const noexcept { return std::hash<typename ViewPtr<T>::pointer>()(p.get()); }
};
} // namespace std

// — Convenience helper —
template <typename T>
inline ViewPtr<T> makeViewPtr(T* ptr) noexcept {
    return ViewPtr<T>(ptr);
}

#endif // VIEW_PTR_HPP