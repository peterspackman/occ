#pragma once
#include <cmath>
#include <vector>
#include <algorithm>
#include <cctype>
#include <locale>
#include <string>

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

std::string join(const std::vector<std::string>& seq, const std::string& sep)
{
    std::string res;
    for(size_t i = 0; i < seq.size(); ++i)
        res += seq[i] + ((i != seq.size() - 1) ? sep : "");
    return res;
}

// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

// trim from start (copying)
static inline std::string ltrim_copy(std::string s) {
    ltrim(s);
    return s;
}

// trim from end (copying)
static inline std::string rtrim_copy(std::string s) {
    rtrim(s);
    return s;
}

// trim from both ends (copying)
static inline std::string trim_copy(std::string s) {
    trim(s);
    return s;
}

static inline void capitalize(std::string &s) {
    s[0] = std::toupper(s[0]);
    std::transform(s.begin() + 1, s.end() , s.begin() + 1,
                   [](unsigned char c) { return std::tolower(c); });
}

static inline std::string capitalize_copy(std::string s)
{
    capitalize(s);
    return s;
}

static inline void to_lower(std::string &s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c){ return std::tolower(c); });
}

static inline void to_upper(std::string &s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c){ return std::toupper(c); });
}

static inline std::string to_lower_copy(std::string s) {
    to_lower(s);
    return s;
}
static inline std::string to_upper_copy(std::string s) {
    to_upper(s);
    return s;
}
}

