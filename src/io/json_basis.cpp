#include <cctype>
#include <fmt/core.h>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/io/json_basis.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace occ::io {

void from_json(const nlohmann::json &J, ElectronShell &shell) {
  if (J.contains("function_type"))
    J.at("function_type").get_to(shell.function_type);
  if (J.contains("region"))
    J.at("region").get_to(shell.region);
  if (J.contains("angular_momentum")) {
    for (const auto &x : J.at("angular_momentum")) {
      shell.angular_momentum.push_back(x);
    }
  }
  if (J.contains("exponents")) {
    for (const auto &x : J.at("exponents")) {
      if (x.is_string()) {
        std::string s = x;
        shell.exponents.push_back(std::stod(s));
      } else {
        shell.exponents.push_back(x);
      }
    }
  }
  if (J.contains("coefficients")) {
    for (const auto &x : J.at("coefficients")) {
      shell.coefficients.push_back({});
      for (const auto &c : x) {
        if (c.is_string()) {
          std::string s = c;
          shell.coefficients.back().push_back(std::stod(s));
        } else {
          shell.coefficients.back().push_back(c);
        }
      }
    }
  }
}

void from_json(const nlohmann::json &J, ECPShell &shell) {
  if (J.contains("ecp_type"))
    J.at("ecp_type").get_to(shell.ecp_type);
  if (J.contains("angular_momentum")) {
    for (const auto &x : J.at("angular_momentum")) {
      shell.angular_momentum.push_back(x);
    }
  }
  if (J.contains("gaussian_exponents")) {
    for (const auto &x : J.at("gaussian_exponents")) {
      if (x.is_string()) {
        std::string s = x;
        shell.exponents.push_back(std::stod(s));
      } else {
        shell.exponents.push_back(x);
      }
    }
  }
  if (J.contains("coefficients")) {
    for (const auto &x : J.at("coefficients")) {
      shell.coefficients.push_back({});
      for (const auto &c : x) {
        if (c.is_string()) {
          std::string s = c;
          shell.coefficients.back().push_back(std::stod(s));
        } else {
          shell.coefficients.back().push_back(c);
        }
      }
    }
  }
  if (J.contains("r_exponents")) {
    for (const auto &x : J.at("r_exponents")) {
      shell.r_exponents.push_back(x);
    }
  }
}

void from_json(const nlohmann::json &J, ReferenceData &ref) {
  if (J.contains("reference_description"))
    J.at("reference_description").get_to(ref.description);
  if (J.contains("reference_keys")) {
    for (const auto &x : J.at("reference_keys")) {
      ref.keys.push_back(x);
    }
  }
}

void from_json(const nlohmann::json &J, ElementBasis &basis) {
  if (J.contains("references")) {
    for (const auto &x : J.at("references")) {
      basis.references.push_back(x);
    }
  }
  for (const auto &x : J.at("electron_shells")) {
    basis.electron_shells.push_back(x);
  }
  if (J.contains("ecp_potentials")) {
    occ::log::trace("Reading ECP potentials");
    for (const auto &x : J.at("ecp_potentials")) {
      basis.ecp_shells.push_back(x);
    }
  }
  if (J.contains("ecp_electrons")) {
    basis.ecp_electrons = J.at("ecp_electrons");
    occ::log::trace("ECP contains {} electrons", basis.ecp_electrons);
  }
}

JsonBasisReader::JsonBasisReader(const std::string &filename)
    : m_filename(filename) {
  occ::timing::start(occ::timing::category::io);
  std::ifstream file(filename);
  if (!file.good())
    throw std::runtime_error("JsonBasisReader file stream: bad");
  occ::log::trace("Loading JSON basis from file {}", filename);
  parse(file);
  occ::timing::stop(occ::timing::category::io);
}

JsonBasisReader::JsonBasisReader(std::istream &file) : m_filename("_istream_") {
  occ::timing::start(occ::timing::category::io);
  if (!file.good())
    throw std::runtime_error("JsonBasisReader file stream: bad");
  parse(file);
  occ::timing::stop(occ::timing::category::io);
}

void JsonBasisReader::parse(std::istream &is) {
  nlohmann::json j;
  is >> j;
  if (!j.contains("elements"))
    throw std::runtime_error("JSON basis has no key 'elements'");
  occ::log::trace("JSON basis has {} elements", j["elements"].size());
  for (auto it = j["elements"].begin(); it != j["elements"].end(); ++it) {
    int atomic_number = 1;
    if (std::isdigit(it.key()[0])) {
      atomic_number = std::stoi(it.key());
      occ::log::trace("Reading JSON basis Z = {}", atomic_number);
    } else {
      Element el(it.key());
      occ::log::trace("Reading JSON basis el = {}", it.key());
      atomic_number = el.atomic_number();
    }
    occ::log::trace("inserting: basis for Z = {}", atomic_number);
    json_basis.elements.insert({atomic_number, it.value().get<ElementBasis>()});
  }
}

const ElementMap &JsonBasisReader::element_map() const {
  return json_basis.elements;
}

const ElementBasis &JsonBasisReader::element_basis(int number) {
  Element el(number);
  return json_basis.elements.at(el.atomic_number());
}
const ElementBasis &JsonBasisReader::element_basis(const Element &element) {
  return json_basis.elements.at(element.atomic_number());
}

} // namespace occ::io
