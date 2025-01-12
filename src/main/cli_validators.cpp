#include <occ/core/util.h>
#include <occ/io/pc.h>
#include <occ/main/cli_validators.h>
#include <occ/qm/spinorbital.h>

namespace occ::main {
MultiplicityValidator::MultiplicityValidator() {
  name_ = "MULTIPLICITY";
  func_ = [](const std::string &s) {
    int mult = std::stoi(s);
    if (mult < 1)
      return std::string{"Spin multiplicity must be >= 1"};
    else
      return std::string{};
  };
}

SpinorbitalKindValidator::SpinorbitalKindValidator() {
  name_ = "SK";
  func_ = [](const std::string &s) {
    occ::qm::SpinorbitalKind sk;
    if (occ::qm::get_spinorbital_kind_from_string(s, sk)) {
      return std::string{};
    } else {
      return std::string{"Spinorbital kind must be one of restricted, "
                         "unrestricted or general."};
    }
  };
}

} // namespace occ::main
