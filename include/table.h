#pragma once
#include <string>
#include <vector>
#include "robin_hood.h"
#include <fmt/core.h>
#include <fmt/ostream.h>

namespace tonto::io {

struct ColumnConfiguration
{
    struct Border {
        std::optional<std::string> left;
        std::optional<std::string> right;
    };
    enum class Alignment: char {
        left = '<',
        right = '>',
        center = '^',
    };

    uint_fast8_t width{12};
    Alignment alignment{Alignment::center};
    Border border;
    std::optional<std::string> pad;
    std::string fill_value{""};

    std::string format_string() const {
        std::string result = fmt::format(
            "{0}{{:{1}{2}{3}s}}{4}",
            border.left.value_or(""),
            pad.value_or(""),
            alignment,
            static_cast<int>(width),
            border.right.value_or("")
        );
        return result;
    }
    uint_fast8_t column_width() const {
        uint_fast8_t w = 0;
        if(border.left) w++;
        if(border.right) w++;
        return width + w;
    }
};

struct RowConfiguration
{
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
    Table(const std::vector<std::string>& column_names);

    template<typename T>
    int set_column(const std::string& name, const std::vector<T>& column, const std::string& fmt_string = "{}")
    {
        size_t num_values = column.size();
        m_columns[name] = std::vector<std::string>{};
        m_column_config[name] = {};
        auto& col = m_columns[name];
        col.reserve(num_values);
        size_t num_added = 0;
        for(const auto& val: column)
        {
            col.push_back(fmt::format(fmt_string, val));
            num_added++;
        }
        return num_added;
    }

    size_t num_rows() const;
    size_t num_cols() const;

    size_t width() const;
    size_t height() const;

    void print() const;

private:
    RowConfiguration m_row_config;
    robin_hood::unordered_map<std::string, std::vector<std::string>> m_columns;
    robin_hood::unordered_map<std::string, ColumnConfiguration> m_column_config;

};
}
