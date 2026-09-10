#ifndef NEXUS_FLOW_TYPE_TRAITS_HPP
#define NEXUS_FLOW_TYPE_TRAITS_HPP

#include <type_traits>

namespace nexusflow {

template <typename T, typename... Ts>
struct is_any_of : std::false_type {};

template <typename T, typename First, typename... Rest>
struct is_any_of<T, First, Rest...> : std::conditional<std::is_same<T, First>::value, std::true_type, is_any_of<T, Rest...>>::type {};

} // namespace nexusflow

#endif
