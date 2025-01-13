#pragma once
#include <algorithm>
#include <cctype>
#include <chrono>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <locale>
#include <numeric>
#include <occ/core/linear_algebra.h>
#include <string>
#include <vector>

namespace occ::util {

template <typename TA, typename TB>
bool all_close(const Eigen::DenseBase<TA> &a, const Eigen::DenseBase<TB> &b,
               const typename TA::RealScalar &rtol =
                   Eigen::NumTraits<typename TA::RealScalar>::dummy_precision(),
               const typename TA::RealScalar &atol =
                   Eigen::NumTraits<typename TA::RealScalar>::epsilon()) {
  return ((a.derived() - b.derived()).array().abs() <=
          (atol + rtol * b.derived().array().abs()))
      .all();
}

template <typename T>
constexpr bool is_close(T a, T b,
                        const T rtol = Eigen::NumTraits<T>::dummy_precision(),
                        const T atol = Eigen::NumTraits<T>::epsilon()) {
  return abs(a - b) <= (atol + rtol * abs(b));
}

template <typename T> constexpr bool is_even(T a) { return a % 2 == 0; }

template <typename T> constexpr bool is_odd(T a) { return !is_even(a); }

static inline std::vector<std::string> tokenize(const std::string &str,
                                                const std::string &delimiters) {
  std::vector<std::string> tokens;
  auto last_position = str.find_first_not_of(delimiters, 0);
  auto position = str.find_first_of(delimiters, last_position);
  while (std::string::npos != position || std::string::npos != last_position) {
    tokens.push_back(str.substr(last_position, position - last_position));
    last_position = str.find_first_not_of(delimiters, position);
    position = str.find_first_of(delimiters, last_position);
  }
  return tokens;
}

static inline std::string join(const std::vector<std::string> &seq,
                               const std::string &sep) {
  std::string res;
  for (size_t i = 0; i < seq.size(); ++i)
    res += seq[i] + ((i != seq.size() - 1) ? sep : "");
  return res;
}

// trim from start (in place)
static inline void ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                  [](int ch) { return !std::isspace(ch); }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](int ch) { return !std::isspace(ch); })
              .base(),
          s.end());
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
  std::transform(s.begin() + 1, s.end(), s.begin() + 1,
                 [](unsigned char c) { return std::tolower(c); });
}

static inline std::string capitalize_copy(std::string s) {
  capitalize(s);
  return s;
}

static inline void to_lower(std::string &s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
}

static inline void to_upper(std::string &s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::toupper(c); });
}

static inline void remove_character_occurences(std::string &s, char c) {
  s.erase(std::remove(s.begin(), s.end(), c), s.end());
}

static inline std::string to_lower_copy(std::string s) {
  to_lower(s);
  return s;
}
static inline std::string to_upper_copy(std::string s) {
  to_upper(s);
  return s;
}

inline bool startswith(const std::string &h, const std::string &prefix,
                       bool trimmed = true) {
  if (trimmed) {
    auto trimmed_str = trim_copy(h);
    return trimmed_str.rfind(prefix, 0) == 0;
  }
  return h.rfind(prefix, 0) == 0;
}

template <typename T>
std::string human_readable_size(T number, const std::string &suffix) {
  double num = static_cast<double>(number);
  const auto units = {"", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"};
  for (const auto &unit : units) {
    if (abs(num) < 1024.0) {
      return fmt::format("{:3.2f}{}{}", num, unit, suffix);
    }
    num /= 1024.0;
  }
  return fmt::format("{:.1f}{}{}", num, "Yi", suffix);
}

inline std::string human_readable_time(std::chrono::milliseconds duration) {
  typedef std::chrono::duration<int, std::ratio<86400>> days;
  auto d = std::chrono::duration_cast<days>(duration);
  duration -= d;
  auto h = std::chrono::duration_cast<std::chrono::hours>(duration);
  duration -= h;
  auto m = std::chrono::duration_cast<std::chrono::minutes>(duration);
  duration -= m;
  auto s = std::chrono::duration_cast<std::chrono::seconds>(duration);
  duration -= s;

  auto dc = d.count();
  auto hc = h.count();
  auto mc = m.count();
  auto sc = s.count();
  auto msc = duration.count();

  std::string result;
  if (dc) {
    result = fmt::format("{}d", dc);
  }
  if (hc) {
    result = fmt::format("{}{}{}h", result, result.empty() ? "" : " ", hc);
  }
  if (mc) {
    result = fmt::format("{}{}{}m", result, result.empty() ? "" : " ", mc);
  }
  if (sc) {
    result = fmt::format("{}{}{}s", result, result.empty() ? "" : " ", sc);
  }
  if (msc) {
    result = fmt::format("{}{}{}ms", result, result.empty() ? "" : " ", msc);
  }
  return result;
}

template <class M, class N>
constexpr std::common_type_t<M, N> smallest_common_factor(M m, N n) {
  if (n == 0)
    return std::abs(m);
  return smallest_common_factor(n, m % n);
}

static inline double double_factorial(int l) {
  switch (l) {
  case 0:
    return 1.0;
  case 1:
    return 1.0;
  case 2:
    return 3.0;
  case 3:
    return 15.0;
  case 4:
    return 105.0;
  case 5:
    return 945.0;
  default:
    double result = 10395.0;
    for (int i = 7; i <= l; i++) {
      result *= (2 * i - 1);
    }
    return result;
  }
}

static inline double factorial(int l) {
  switch (l) {
  case 0:
    return 1.0;
  case 1:
    return 1.0;
  case 2:
    return 2.0;
  case 3:
    return 6.0;
  case 4:
    return 24.0;
  case 5:
    return 120.0;
  default:
    double result = 720.0;
    for (int i = 7; i <= l; i++) {
      result *= i;
    }
    return result;
  }
}

static inline int multinomial_coefficient(std::initializer_list<int> args) {
  int result = 1;
  int numerator = std::accumulate(args.begin(), args.end(), 0);

  for (const auto &k : args) {
    for (int i = 0; i < k; i++) {
      result *= numerator;
      result /= 1 + i;
      numerator--;
    }
  }
  return result;
}

template <typename T> size_t index_of(T x, const std::vector<T> &vec) {
  return std::distance(vec.begin(), std::find(vec.begin(), vec.end(), x));
}

} // namespace occ::util
