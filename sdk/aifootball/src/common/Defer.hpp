#pragma once

/**
 * @file Defer.hpp
 * @brief Go-style defer for C++ — executes a lambda at scope exit.
 *
 * Usage:
 *   void foo() {
 *       FILE* f = fopen("x.txt", "r");
 *       defer { fclose(f); };
 *       // ... f is automatically closed when foo() returns
 *   }
 *
 *   void bar() {
 *       cudaMalloc(&ptr, size);
 *       defer { cudaFree(ptr); };
 *       // ... ptr is freed on any exit path (return, exception, etc.)
 *   }
 *
 * Multiple defers in the same scope execute in reverse order (LIFO),
 * matching Go semantics.
 *
 * Implementation: a temporary deferrer object whose destructor invokes
 * the captured lambda. The `defer` macro expands to a uniquely-named
 * variable so multiple defers can coexist in one scope.
 */

#ifndef NEXUSFLOW_COMMON_DEFER_HPP
#define NEXUSFLOW_COMMON_DEFER_HPP

namespace nexusflow {
namespace detail {

struct DeferDummy {};

template <class F>
struct Deferrer {
    F f;
    ~Deferrer() { f(); }
};

template <class F>
Deferrer<F> operator*(DeferDummy, F f) {
    return {f};
}

} // namespace detail
} // namespace nexusflow

#define NEXUSFLOW_DEFER_(LINE) nexusflow_defer_##LINE
#define NEXUSFLOW_DEFER(LINE)  NEXUSFLOW_DEFER_(LINE)

/**
 * @brief Go-style defer. Captures by reference [&] by default.
 *
 * Example:
 *   defer { cleanup(); };
 *   defer { LOG_INFO("leaving scope"); };
 */
#define defer auto NEXUSFLOW_DEFER(__LINE__) = ::nexusflow::detail::DeferDummy{} *[&]()

#endif // NEXUSFLOW_COMMON_DEFER_HPP
