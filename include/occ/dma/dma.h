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

struct DMASites {
    inline auto size() const { return positions.cols(); }
    inline auto num_atoms() const { return atoms.size(); }

    std::vector<occ::core::Atom> atoms;
    Mat3N positions;
    IVec atom_indices;
    Vec radii;
    IVec limits;
};

class DMACalculator {
public:
    DMACalculator(const qm::Wavefunction &wfn);
    void update_settings(const DMASettings &settings);
    inline const auto &settings() const { return m_settings; }
    void set_radius_for_element(int atomic_number, double radius_angs);
    void set_limit_for_element(int atomic_number, int limit);

    DMAResult compute_multipoles();

private:
    DMASites m_sites;
    qm::AOBasis m_basis;
    qm::MolecularOrbitals m_mo;
    DMASettings m_settings;

};

} // namespace occ::dma
