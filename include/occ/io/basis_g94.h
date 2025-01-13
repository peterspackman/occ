#pragma once
#include <occ/qm/shell.h>
#include <string>
#include <vector>

namespace occ::io::basis::g94 {

std::vector<std::vector<occ::qm::Shell>>
read_shell(const std::string &, bool force_cartesian_d = false,
           std::string locale_name = "POSIX");

} // namespace occ::io::basis::g94
