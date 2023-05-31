#include <nlohmann/json.hpp>
#include <occ/core/dimer.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/qm/wavefunction.h>
#include <string>
#include <vector>

namespace occ::main {

using occ::crystal::Crystal;
using occ::crystal::CrystalDimers;

enum class SolvationPartitionScheme {
    NearestAtom,
    NearestAtomDnorm,
    ElectronDensity,
};

struct SolvatedSurfaceProperties {
    double esolv{0.0};
    double dg_ele{0.0};
    double dg_gas{0.0};
    double dg_correction{0.0};
    occ::Mat3N coulomb_pos;
    occ::Mat3N cds_pos;
    occ::Vec e_coulomb;
    occ::Vec e_cds;
    occ::Vec e_ele;
    occ::Vec a_coulomb;
    occ::Vec a_cds;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SolvatedSurfaceProperties, esolv, dg_ele,
                                   dg_gas, dg_correction, coulomb_pos, cds_pos,
                                   e_coulomb, e_cds, e_ele, a_coulomb, a_cds)

std::pair<std::vector<SolvatedSurfaceProperties>,
          std::vector<occ::qm::Wavefunction>>
calculate_solvated_surfaces(const std::string &basename,
                            const std::vector<occ::core::Molecule> &mols,
                            const std::vector<occ::qm::Wavefunction> &wfns,
                            const std::string &solvent_name,
                            const std::string &method = "b3lyp",
                            const std::string &basis_name = "6-31g**");

struct SolventNeighborContribution {
    struct AsymPair {
        double ab{0.0}, ba{0.0};
        inline double total() const { return ab + ba; }
        void assign(AsymPair &other) {
            other.ba = ab;
            ba = other.ab;
        }
    };

    inline double total() const {
        double t = coulomb.total() + cds.total();
        double difference = coulomb.ab + cds.ab - coulomb.ba - cds.ba;
        return t + 0.5 * difference;
    }

    inline double total_kjmol() const {
        return occ::units::AU_TO_KJ_PER_MOL * total();
    }

    AsymPair coulomb;
    AsymPair cds;
    AsymPair cds_area;
    AsymPair coulomb_area;
    bool neighbor_set{false};

    void assign(SolventNeighborContribution &other) {
        coulomb.assign(other.coulomb);
        cds.assign(other.cds);
        coulomb_area.assign(other.coulomb_area);
        cds_area.assign(other.cds_area);
    }
};

std::vector<SolventNeighborContribution> partition_solvent_surface(
    SolvationPartitionScheme scheme, const Crystal &crystal,
    const std::string &mol_name, const std::vector<occ::qm::Wavefunction> &wfns,
    const SolvatedSurfaceProperties &surface,
    const CrystalDimers::MoleculeNeighbors &neighbors,
    const CrystalDimers::MoleculeNeighbors &nearest_neighbors,
    const std::string &solvent);

} // namespace occ::main
