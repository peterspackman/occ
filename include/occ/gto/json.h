#pragma once
#include <nlohmann/json_fwd.hpp>
#include <occ/gto/shell.h>

namespace occ::gto {

void to_json(nlohmann::json &J, const Shell &shell);
void from_json(const nlohmann::json &J, Shell &shell);

} // namespace occ::gto
