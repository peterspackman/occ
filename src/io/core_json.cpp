#include <occ/core/dimer.h>
#include <occ/io/core_json.h>
#include <occ/io/eigen_json.h>

namespace occ::core {

void to_json(nlohmann::json &j, const Dimer &dimer) {
    j["mol_a"] = dimer.a();
    j["mol_b"] = dimer.b();
    j["interaction_energy"] = dimer.interaction_energy();
    j["interaction_id"] = dimer.interaction_id();
}

void to_json(nlohmann::json &j, const Molecule &mol) {

    nlohmann::json elements;
    for (const auto &el : mol.elements()) {
        elements.push_back(el.symbol());
    }
    j["name"] = mol.name();
    j["elements"] = elements;
    Mat pos = mol.positions().transpose();
    j["positions"] = pos;
}

} // namespace occ::core
