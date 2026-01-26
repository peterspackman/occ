#pragma once
#include <array>
#include <istream>
#include <occ/core/element.h>
#include <occ/qm/spinorbital.h>
#include <string>
#include <vector>

namespace occ::io {
using occ::core::Element;
using Position = std::array<double, 3>;

struct OccInput;

struct GaussianInputFile {
  enum MethodType { HF, DFT, Other };
  GaussianInputFile(const std::string &);
  GaussianInputFile(std::istream &);

  std::vector<std::pair<std::string, std::string>> link0_commands;
  std::string method, basis_name;
  std::vector<std::string> keywords;
  std::vector<Element> elements;
  std::vector<Position> atomic_positions;
  std::string route_tag{"#"};
  std::string comment;
  MethodType method_type{Other};
  int charge{0}, multiplicity{1};
  occ::qm::SpinorbitalKind spinorbital_kind() const;

  OccInput as_occ_input() const;
  void update_occ_input(OccInput &) const;

private:
  void parse(std::istream &);
  void parse_link0(const std::string &);
  void parse_route_line(const std::string &);
  void parse_charge_multiplicity_line(const std::string &);
  void parse_atom_line(const std::string &);
};

} // namespace occ::io
