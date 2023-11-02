#pragma once
#include <Eigen/Core>
#include <fmt/ostream.h>

template <typename T>
requires std::is_base_of_v<Eigen::DenseBase<T>, T>
struct fmt::formatter<T> : ostream_formatter {
};
