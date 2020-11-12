#pragma once
#include <string>
#include <vector>
#include "robin_hood.h"
#include <fmt/core.h>
#include <fmt/ostream.h>

namespace tonto::io {

struct ColumnConfiguration
{
    std::string fill_value{""};
    std::string format_string{"{:<20s}"};
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

    void print() const;

private:
    robin_hood::unordered_map<std::string, std::vector<std::string>> m_columns;
    robin_hood::unordered_map<std::string, ColumnConfiguration> m_column_config;

};
}
