#pragma once
// #include <cmath>

namespace trajan::units {
constexpr double PI = 3.14159265358979323846;

template <typename T> constexpr auto radians(T x) { return x * PI / 180; }

template <typename T> constexpr auto degrees(T x) { return x * 180 / PI; }
} // namespace trajan::units
