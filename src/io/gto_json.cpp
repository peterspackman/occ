#include <nlohmann/json.hpp>
#include <occ/io/gto_json.h>
#include <occ/io/eigen_json.h>

namespace occ::gto {

void to_json(nlohmann::json &J, const Shell &shell) {
  J["spherical"] = shell.kind == Shell::Kind::Spherical;
  J["l"] = shell.l;
  J["origin"] = shell.origin;
  J["exponents"] = shell.exponents;
  J["contraction coefficients"] = shell.contraction_coefficients;
  J["unnormalized contraction coefficients"] = shell.u_coefficients;
  if (shell.max_ln_coefficient.size() > 0) {
    J["max ln coefficient"] = shell.max_ln_coefficient;
  }
  if (shell.ecp_r_exponents.size() > 0) {
    J["ecp r exponents"] = shell.ecp_r_exponents;
  }
  J["extent"] = shell.extent;
}

void from_json(const nlohmann::json &J, Shell &shell) {
  if (J.at("spherical")) {
    shell.kind = Shell::Kind::Spherical;
  } else {
    shell.kind = Shell::Kind::Cartesian;
  }
  J.at("l").get_to(shell.l);
  J.at("origin").get_to(shell.origin);
  J.at("exponents").get_to(shell.exponents);
  J.at("contraction coefficients").get_to(shell.contraction_coefficients);
  J.at("unnormalized contraction coefficients").get_to(shell.u_coefficients);
  if (J.contains("max ln coefficient")) {
    J.at("max ln coefficient").get_to(shell.max_ln_coefficient);
  }
  if (J.contains("ecp r exponents")) {
    J.at("ecp r exponents").get_to(shell.ecp_r_exponents);
  }
  J.at("extent").get_to(shell.extent);
}

} // namespace occ::gto
