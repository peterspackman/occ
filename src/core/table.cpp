#include <occ/core/table.h>

namespace occ::io {

Table::Table() {}

Table::Table(const std::vector<std::string> &column_names) {
  for (const auto &name : column_names) {
    m_columns[name] = std::vector<std::string>{};
    m_column_config[name] = {};
  }
}

size_t Table::num_cols() const { return m_columns.size(); }

size_t Table::num_rows() const {
  size_t n = 0;
  for (const auto &kv : m_columns) {
    n = std::max(n, kv.second.size());
  }
  return n;
}

void Table::print() const {
  for (const auto &key : m_column_order) {
    std::string fs = m_column_config.at(key).format_string();
    fmt::print(fmt::runtime(fs), key);
  }
  fmt::print("\n");
  for (size_t row = 0; row < num_rows(); row++) {
    for (const std::string &key : m_column_order) {
      const auto &c = m_columns.at(key);
      const auto &config = m_column_config.at(key);
      fmt::print(fmt::runtime(config.format_string()),
                 (row < c.size()) ? c[row] : config.fill_value);
    }
    fmt::print("\n");
  }
}

} // namespace occ::io
