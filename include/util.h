#pragma once
#include <cmath>
#include <vector>

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

static inline std::vector<std::string> tokenize(
    const std::string& str, const std::string& delimiters) 
{
    std::vector<std::string> tokens;
    auto last_position = str.find_first_not_of(delimiters, 0);
    auto position = str.find_first_of(delimiters, last_position);
    while(std::string::npos != position || std::string::npos != last_position)
    {
        tokens.push_back(str.substr(last_position, position - last_position));
        last_position = str.find_first_not_of(delimiters, position);
        position = str.find_first_of(delimiters, last_position);
    }
    return tokens;
}

}

