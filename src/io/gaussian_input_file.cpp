#include <fstream>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/util.h>
#include <occ/io/gaussian_input_file.h>
#include <occ/io/occ_input.h>
#include <regex>
#include <scn/scan.h>

namespace occ::io {

using occ::qm::SpinorbitalKind;

GaussianInputFile::GaussianInputFile(const std::string &filename) {
  std::ifstream file(filename);
  parse(file);
}

GaussianInputFile::GaussianInputFile(std::istream &stream) { parse(stream); }

void GaussianInputFile::parse(std::istream &stream) {
  using occ::util::trim;
  std::string line;
  while (std::getline(stream, line)) {
    trim(line);
    if (line[0] == '%')
      parse_link0(line);
    else if (line[0] == '#') {
      parse_route_line(line);
      occ::log::debug("Found route line, breaking");
      break;
    }
  }
  std::getline(stream, line);
  std::getline(stream, comment);
  std::getline(stream, line);
  std::getline(stream, line);
  parse_charge_multiplicity_line(line);
  while (std::getline(stream, line)) {
    trim(line);
    if (line.empty())
      break;
    else {
      parse_atom_line(line);
    }
  }
}

void GaussianInputFile::parse_link0(const std::string &line) {
  using occ::util::trim_copy;
  auto eq = line.find('=');
  std::string cmd = line.substr(1, eq - 1);
  std::string arg = line.substr(eq + 1);
  occ::log::debug("Found link0 command {} = {}", cmd, arg);
  link0_commands.push_back({std::move(cmd), std::move(arg)});
}

void GaussianInputFile::parse_route_line(const std::string &line) {
  using occ::util::to_lower_copy;
  using occ::util::trim;
  auto ltrim = to_lower_copy(line);
  trim(ltrim);
  if (ltrim.size() == 0)
    return;

  const std::regex method_regex(R"(([\w\-\+\*\(\)]+)\s*\/\s*([\w\-\+\*\(\)]+))",
                                std::regex_constants::ECMAScript);
  std::smatch sm;
  if (std::regex_search(ltrim, sm, method_regex)) {
    method = sm[1].str();
    basis_name = sm[2].str();
  } else {
    occ::log::error(
        "Did not find method/basis in route line in gaussian input!");
  }
  occ::log::debug("Found route command: method = {} basis = {}", method,
                  basis_name);
  if (method == "hf" || method == "uhf" || method == "ghf")
    method_type = MethodType::HF;
  else
    method_type = MethodType::DFT;
}

void GaussianInputFile::parse_charge_multiplicity_line(
    const std::string &line) {
  auto result = scn::scan<int, int>(line, "{} {}");
  std::tie(charge, multiplicity) = result->values();
}

void GaussianInputFile::parse_atom_line(const std::string &line) {
  auto result =
      scn::scan<std::string, double, double, double>(line, "{} {} {} {}");
  auto &[symbol, x, y, z] = result->values();
  atomic_positions.push_back({x, y, z});
  elements.emplace_back(occ::core::Element(symbol));
}

SpinorbitalKind GaussianInputFile::spinorbital_kind() const {
  if (method == "ghf")
    return SpinorbitalKind::General;
  if (multiplicity != 1)
    return SpinorbitalKind::Unrestricted;
  if (method[0] == 'u')
    return SpinorbitalKind::Unrestricted;
  return SpinorbitalKind::Restricted;
}

void GaussianInputFile::update_occ_input(OccInput &result) const {
  result.geometry.positions = atomic_positions;
  result.geometry.elements = elements;
  result.method.name = method;
  result.electronic.charge = charge;
  result.electronic.multiplicity = multiplicity;
  result.basis.name = basis_name;
}

OccInput GaussianInputFile::as_occ_input() const {
  OccInput result;
  update_occ_input(result);
  return result;
}

} // namespace occ::io
