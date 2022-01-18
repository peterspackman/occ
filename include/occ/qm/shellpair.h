#pragma once
#include <occ/3rdparty/robin_hood.h>
#include <libint2/shell.h>

namespace occ::qm {

using ShellPairList = robin_hood::unordered_map<size_t, std::vector<size_t>>;
using ShellPairData = std::vector<std::vector<std::shared_ptr<libint2::ShellPair>>>; // in same order as shellpair_list_t
}
