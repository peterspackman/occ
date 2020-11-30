#pragma once
#include <string>
#include <array>
#include <vector>
#include <istream>

namespace tonto::io {

struct GaussianInputFile
{
    using position = std::array<double, 3>;
    GaussianInputFile(const std::string&);
    GaussianInputFile(std::istream&);

    std::vector<std::pair<std::string, std::string>> link0_commands;
    std::string method, basis_name;
    std::vector<std::string> keywords;
    std::vector<uint_fast8_t> atomic_numbers();
    std::vector<position> atomic_positions;
    std::string route_tag{"#"};
    std::string comment;
    uint_fast8_t charge{0}, multiplicity{1};

private:
    void parse(std::istream&);
    void parse_link0(const std::string&);
    void parse_route_line(const std::string&);
};

}
