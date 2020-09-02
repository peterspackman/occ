#pragma once
#include <cmath>

namespace craso::util {

template<typename T>
constexpr bool isclose(T a, T b, T rtol=1e-5, T atol=1e-8)
{
    return abs(a - b) <= (atol + rtol * abs(b));
}

template<typename T>
inline auto deg2rad(T x) {
    return static_cast<T>(x * M_PI / 180.0);
}

template<typename T>
inline auto rad2deg(T x) {
    return static_cast<T>(x * 180.0 / M_PI);
}
}

