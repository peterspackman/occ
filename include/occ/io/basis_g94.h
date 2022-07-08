#pragma once
#include <occ/qm/occshell.h>
#include <string>
#include <vector>

namespace occ::io::basis::g94 {

std::vector<std::vector<occ::qm::OccShell>>
read_occshell(const std::string &, bool force_cartesian_d = false,
              std::string locale_name = "POSIX");

} // namespace occ::io::basis::g94
