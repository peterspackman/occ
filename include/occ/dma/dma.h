#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
#include <occ/core/timings.h>
#include <occ/dma/mult.h>
#include <occ/qm/wavefunction.h>
#include <occ/qm/integral_engine.h>
#include <vector>

namespace occ::dma {


struct DMASettings {
    int max_rank{4};
    double big_exponent{4.0};
    bool include_nuclei{true};
};

struct DMAResult {
    int max_rank{4};
    std::vector<Mult> multipoles;
};

class DMACalculator {
public:
    DMACalculator(const qm::Wavefunction &wfn);
    void update_settings(const DMASettings &settings);
    inline const auto &settings() const { return m_settings; }

    DMAResult compute_multipoles();

private:
    std::vector<occ::core::Atom> m_atoms;
    Mat3N m_site_positions;
    IVec m_atom_indices;
    qm::AOBasis m_basis;
    qm::MolecularOrbitals m_mo;
    DMASettings m_settings;
    IVec m_site_limits;

};

} // namespace occ::dma
