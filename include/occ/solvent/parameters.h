#pragma once
#include <ankerl/unordered_dense.h>
#include <occ/solvent/json.h>
#include <occ/solvent/smd_parameters.h>
#include <string>

namespace occ::solvent {

/// Directory holding the shipped solvent data, `$OCC_DATA_PATH/solvent`,
/// falling back to the working directory.
std::string solvent_data_path();
void override_data_path_directory(const std::string &);

double get_dielectric(const std::string &name);
SMDSolventParameters get_smd_parameters(const std::string &name);

void list_available_solvents();

nlohmann::json load_draco_parameters();

} // namespace occ::solvent
