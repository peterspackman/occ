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

    if(mol.asymmetric_molecule_idx() > -1) {
	j["asym mol"] = mol.asymmetric_molecule_idx();
    }

    if(mol.unit_cell_molecule_idx() > -1) {
	j["uc mol"] = mol.unit_cell_molecule_idx();
    }

    const auto &asym_idx = mol.asymmetric_unit_idx();
    if(asym_idx.size() > 0) {
	j["asym atom"] = asym_idx.transpose();
    }

    const auto &uc_idx = mol.unit_cell_idx();
    if(asym_idx.size() > 0) {
	j["uc atom"] = uc_idx.transpose();
    }

    const auto &cell_shift = mol.cell_shift();
    j["cell shift"] = cell_shift;

}

} // namespace occ::core
