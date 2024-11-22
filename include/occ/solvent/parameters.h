#pragma once
#include <ankerl/unordered_dense.h>
#include <occ/io/solvent_json.h>
#include <occ/solvent/smd_parameters.h>
#include <string>

namespace occ::solvent {

double get_dielectric(const std::string &name);
SMDSolventParameters get_smd_parameters(const std::string &name);

void list_available_solvents();

nlohmann::json load_draco_parameters();

} // namespace occ::solvent
