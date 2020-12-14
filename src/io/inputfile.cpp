#include <tonto/io/inputfile.h>
#include <tonto/core/logger.h>
#include <scn/scn.h>
#include <fstream>
#include <regex>
#include <tonto/core/util.h>
#include <tonto/core/element.h>

namespace tonto::io {

using tonto::qm::SpinorbitalKind;

GaussianInputFile::GaussianInputFile(const std::string &filename)
{
    std::ifstream file(filename);
    parse(file);
}

GaussianInputFile::GaussianInputFile(std::istream &stream)
{
    parse(stream);
}

void GaussianInputFile::parse(std::istream &stream)
{
    using tonto::util::trim;
    std::string line;
    while(std::getline(stream, line)) {
        trim(line);
        if(line[0] == '%') parse_link0(line);
        else if(line[0] == '#') {
            parse_route_line(line);
            tonto::log::debug("Found route line, breaking");
            break;
        }
    }
    std::getline(stream, line);
    std::getline(stream, comment);
    std::getline(stream, line);
    std::getline(stream, line);
    parse_charge_multiplicity_line(line);
    while(std::getline(stream, line)) {
        trim(line);
        if(line.empty()) break;
        else {
            parse_atom_line(line);
        }
    }
}

void GaussianInputFile::parse_link0(const std::string &line)
{
    using tonto::util::trim_copy;
    auto eq = line.find('=');
    std::string cmd = line.substr(1, eq - 1);
    std::string arg = line.substr(eq + 1);
    tonto::log::debug("Found link0 command {} = {}", cmd, arg);
    link0_commands.push_back({std::move(cmd), std::move(arg)});
}

void GaussianInputFile::parse_route_line(const std::string &line)
{
    using tonto::util::to_lower_copy;
    using tonto::util::trim;
    auto ltrim = to_lower_copy(line);
    trim(ltrim);
    if(ltrim.size() == 0) return;

    const std::regex method_regex(R"###(([\w-\+\*\(\)]+)\s*\/\s*([\w-\+\*\(\)]+))###", std::regex_constants::ECMAScript);
    std::smatch sm;
    if(std::regex_search(ltrim, sm, method_regex))
    {
        method = sm[1].str();
        basis_name = sm[2].str();
    }
    else
    {
        tonto::log::error("Did not find method/basis in route line in gaussian input!");
    }
    tonto::log::debug("Found route command: method = {} basis = {}", method, basis_name);
}

void GaussianInputFile::parse_charge_multiplicity_line(const std::string &line)
{
    scn::scan(line, "{} {}", charge, multiplicity);
}

void GaussianInputFile::parse_atom_line(const std::string &line)
{
    double x, y, z;
    std::string symbol;
    scn::scan(line, "{} {} {} {}", symbol, x, y, z);
    atomic_positions.push_back({x, y, z});
    tonto::chem::Element elem(symbol);
    atomic_numbers.push_back(elem.atomic_number());
}


SpinorbitalKind GaussianInputFile::spinorbital_kind() const
{
    if(multiplicity != 1) return SpinorbitalKind::Unrestricted;
    if(method[0] == 'u') return SpinorbitalKind::Unrestricted;
    return SpinorbitalKind::Restricted;
}

}
