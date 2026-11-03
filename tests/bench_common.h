#pragma once
// Shared by the timing harnesses (*_bench.cpp): wall-clock timing and a
// minimal command line of positional arguments, boolean flags and
// "--name value" options. Anything unrecognised prints the usage and fails.
#include <algorithm>
#include <chrono>
#include <fmt/core.h>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace occ::bench {

template <typename F> double seconds(F &&f) {
  const auto t0 = std::chrono::steady_clock::now();
  f();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

inline double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

inline double minimum(const std::vector<double> &v) {
  return *std::min_element(v.begin(), v.end());
}

class Args {
public:
  // `flags` take no value, `options` take one; every harness also accepts
  // --threads N.
  Args(int argc, char **argv, std::string usage,
       std::set<std::string> flags = {}, std::set<std::string> options = {})
      : m_usage(std::move(usage)), m_flags(std::move(flags)),
        m_options(std::move(options)) {
    m_options.insert("--threads");
    m_program = argv[0];
    for (int i = 1; i < argc; i++) {
      const std::string a = argv[i];
      if (a.rfind("--", 0) != 0) {
        m_positional.push_back(a);
      } else if (m_flags.count(a)) {
        m_set.insert(a);
      } else if (m_options.count(a) && i + 1 < argc) {
        m_values[a] = argv[++i];
      } else {
        m_error = "unrecognised or incomplete option " + a;
        return;
      }
    }
  }

  // False (after printing the usage) unless the command line parsed and has
  // between min and max positional arguments.
  bool valid(size_t min_positional, size_t max_positional) const {
    if (m_error.empty() && m_positional.size() >= min_positional &&
        m_positional.size() <= max_positional)
      return true;
    if (!m_error.empty())
      fmt::print(stderr, "{}\n", m_error);
    fmt::print(stderr, "usage: {} {}\n", m_program, m_usage);
    return false;
  }

  const std::string &positional(size_t i) const { return m_positional[i]; }
  size_t num_positional() const { return m_positional.size(); }
  bool has(const std::string &flag) const { return m_set.count(flag) > 0; }
  std::string get(const std::string &option, const std::string &def) const {
    auto it = m_values.find(option);
    return it == m_values.end() ? def : it->second;
  }
  int get_int(const std::string &option, int def) const {
    auto it = m_values.find(option);
    return it == m_values.end() ? def : std::stoi(it->second);
  }
  double get_double(const std::string &option, double def) const {
    auto it = m_values.find(option);
    return it == m_values.end() ? def : std::stod(it->second);
  }
  int threads() const { return get_int("--threads", 1); }
  int reps() const { return get_int("--reps", 3); }

private:
  std::string m_program, m_usage, m_error;
  std::set<std::string> m_flags, m_options, m_set;
  std::map<std::string, std::string> m_values;
  std::vector<std::string> m_positional;
};

} // namespace occ::bench
