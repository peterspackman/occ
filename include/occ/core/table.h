#pragma once
#include <ankerl/unordered_dense.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <occ/core/linear_algebra.h>
#include <optional>
#include <string>
#include <vector>

namespace occ::io {

struct ColumnConfiguration {
    struct Border {
        std::string left{""};
        std::string right{" "};
    };
    enum class Alignment : char {
        left = '<',
        right = '>',
        center = '^',
    };

    uint_fast8_t width{12};
    Alignment alignment{Alignment::left};
    Border border;
    std::string pad{""};
    std::string fill_value{""};

    std::string format_string() const {
        std::string result = fmt::format("{0}{{:{1}{2}{3}s}}{4}", border.left,
                                         pad, static_cast<char>(alignment),
                                         static_cast<int>(width), border.right);
        return result;
    }
    uint_fast8_t column_width() const {
        return width + border.left.size() + border.right.size();
    }
};

struct RowConfiguration {
    struct Border {
        std::optional<std::string> left;
        std::optional<std::string> right;
        std::optional<std::string> top;
        std::optional<std::string> bottom;
    };
    Border border;
};

class Table {
  public:
    Table();
    Table(const std::vector<std::string> &column_names);

    template <typename T>
    int set_column(const std::string &name, const std::vector<T> &column,
                   const std::string &fmt_string = "{}") {
        size_t num_values = column.size();
        m_column_order.push_back(name);
        m_columns[name] = std::vector<std::string>{};
        ColumnConfiguration config;
        if constexpr (std::is_arithmetic<T>::value)
            config.alignment = ColumnConfiguration::Alignment::right;
        auto &col = m_columns[name];
        col.reserve(num_values);
        size_t num_added{0};
        size_t cell_width = name.size();
        for (const auto &val : column) {
            std::string cell = fmt::format(fmt::runtime(fmt_string), val);
            cell_width = std::max(cell_width, cell.size());
            col.push_back(cell);
            num_added++;
        }
        config.width = static_cast<uint_fast8_t>(cell_width);
        m_column_config[name] = config;
        return num_added;
    }

    template <typename TA>
    int set_column(const std::string &name, const Eigen::DenseBase<TA> &a,
                   std::string fmt_string = "{}") {
        if constexpr (std::is_floating_point<typename TA::Scalar>::value) {
            fmt_string = "{: 12.6f}";
        }
        size_t num_values = a.rows();
        for (Eigen::Index c = 0; c < a.cols(); c++) {
            std::string colname = fmt::format("{}{}", name, c);
            m_column_order.push_back(colname);
            m_columns[colname] = std::vector<std::string>{};
            ColumnConfiguration config;
            config.alignment = ColumnConfiguration::Alignment::right;
            auto &col = m_columns[colname];
            col.reserve(num_values);
            size_t cell_width = colname.size();
            for (Eigen::Index r = 0; r < num_values; r++) {
                std::string cell =
                    fmt::format(fmt::runtime(fmt_string), a(r, c));
                cell_width = std::max(cell_width, cell.size());
                col.push_back(cell);
            }
            config.width = static_cast<uint_fast8_t>(cell_width);
            m_column_config[colname] = config;
        }
        return num_values;
    }

    size_t num_rows() const;
    size_t num_cols() const;

    size_t width() const;
    size_t height() const;

    void print() const;

  private:
    std::vector<std::string> m_column_order;
    RowConfiguration m_row_config;
    ankerl::unordered_dense::map<std::string, std::vector<std::string>>
        m_columns;
    ankerl::unordered_dense::map<std::string, ColumnConfiguration>
        m_column_config;
};
} // namespace occ::io
