#pragma once
#include <libint2/shell.h>
#include <string>
#include <vector>

namespace occ::io::basis::g94 {

using libint2::Shell;

std::vector<std::vector<Shell>> read(const std::string &,
                                     bool force_cartesian_d = false,
                                     std::string locale_name = "POSIX");

} // namespace occ::io::basis::g94
