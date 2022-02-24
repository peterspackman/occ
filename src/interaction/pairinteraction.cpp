#include <occ/core/timings.h>
#include <occ/interaction/disp.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/qm/hf.h>
#include <occ/qm/merge.h>

namespace occ::interaction {
using qm::SpinorbitalKind;

CEModelInteraction::CEModelInteraction(const CEParameterizedModel &facs)
    : scale_factors(facs) {}

template <SpinorbitalKind kind>
void compute_ce_model_energies(Wavefunction &wfn, occ::hf::HartreeFock &hf, double precision, const Mat &Schwarz) {
    if (wfn.have_energies)
        return;
    using occ::qm::expectation;
    using occ::qm::matrix_dimensions;
    if constexpr (kind == SpinorbitalKind::Restricted) {
        wfn.V = hf.compute_nuclear_attraction_matrix();
        wfn.energy.nuclear_attraction = 2 * expectation<kind>(wfn.mo.D, wfn.V);
        wfn.T = hf.compute_kinetic_matrix();
        wfn.energy.kinetic = 2 * expectation<kind>(wfn.mo.D, wfn.T);
        wfn.H = wfn.V + wfn.T;
        wfn.energy.core = 2 * expectation<kind>(wfn.mo.D, wfn.H);
        std::tie(wfn.J, wfn.K) = hf.compute_JK(kind, wfn.mo, precision, Schwarz);
        wfn.energy.coulomb = expectation<kind>(wfn.mo.D, wfn.J);
        wfn.energy.exchange = -expectation<kind>(wfn.mo.D, wfn.K);
        wfn.energy.nuclear_repulsion = hf.nuclear_repulsion_energy();
    } else {
	namespace block = occ::qm::block;
        size_t rows, cols;
        std::tie(rows, cols) =
            matrix_dimensions<SpinorbitalKind::Unrestricted>(wfn.nbf);
        wfn.T = Mat(rows, cols);
        wfn.V = Mat(rows, cols);
        block::a(wfn.T) = hf.compute_kinetic_matrix();
        block::b(wfn.T) = block::a(wfn.T);
        block::a(wfn.V) = hf.compute_nuclear_attraction_matrix();
        block::b(wfn.V) = block::a(wfn.V);
        wfn.H = wfn.V + wfn.T;
        wfn.energy.nuclear_attraction = 2 * expectation<kind>(wfn.mo.D, wfn.V);
        wfn.energy.kinetic = 2 * expectation<kind>(wfn.mo.D, wfn.T);
        wfn.energy.core = 2 * expectation<kind>(wfn.mo.D, wfn.H);
        std::tie(wfn.J, wfn.K) = hf.compute_JK(kind, wfn.mo, precision, Schwarz);
        wfn.energy.coulomb = expectation<kind>(wfn.mo.D, wfn.J);
        wfn.energy.exchange = -expectation<kind>(wfn.mo.D, wfn.K);
        wfn.energy.nuclear_repulsion = hf.nuclear_repulsion_energy();
    }
    wfn.have_energies = true;
}

void compute_ce_model_energies(Wavefunction &wfn, occ::hf::HartreeFock &hf, double precision, const Mat &Schwarz) {
    if (wfn.is_restricted())
        return compute_ce_model_energies<SpinorbitalKind::Restricted>(wfn, hf, precision, Schwarz);
    else
        return compute_ce_model_energies<SpinorbitalKind::Unrestricted>(wfn,
                                                                        hf, precision, Schwarz);
}

void CEModelInteraction::use_density_fitting() {
    m_use_density_fitting = true;
}

void dump_matrix(const Mat& matrix) {
    size_t maxdim = std::max(matrix.rows(), matrix.cols());
    Eigen::Index fields = 5;

    if (fields==0) fields = matrix.rows();
    size_t n_block = static_cast<size_t>((matrix.cols() - 0.1)/fields) + 1;
    for(size_t block = 0; block < n_block; block++) {
	Eigen::Index f = fields * block;
	Eigen::Index l = std::min(f + fields, matrix.cols());
	fmt::print("{:8s}", " ");
	for(size_t j = f; j < l; j++) {
	    fmt::print(" {:8d}", j);
	}
	fmt::print("\n");
	for(size_t i = 0; i < matrix.rows(); i++) {
	    fmt::print("{:8d}", i);
	    for(size_t j = f; j < l; j++) {
		fmt::print(" {:8.5f}", matrix(i, j));
	    }
	    fmt::print("\n");
	}
    }
}

CEEnergyComponents
CEModelInteraction::operator()(Wavefunction &A, Wavefunction &B) const {
    using occ::disp::ce_model_dispersion_energy;
    using occ::hf::HartreeFock;
    using occ::qm::Energy;
    constexpr double precision = std::numeric_limits<double>::epsilon();

    HartreeFock hf_a(A.atoms, A.basis);
    HartreeFock hf_b(B.atoms, B.basis);
    if(m_use_density_fitting) {
	hf_a.set_density_fitting_basis("cc-pvdz-jkfit");
	hf_b.set_density_fitting_basis("cc-pvdz-jkfit");
    }

    Mat schwarz_a = hf_a.compute_schwarz_ints();
    Mat schwarz_b = hf_b.compute_schwarz_ints();

    compute_ce_model_energies(A, hf_a, precision, schwarz_a);
    compute_ce_model_energies(B, hf_b, precision, schwarz_b);

    Wavefunction ABn(A, B);

    // Can reuse the same HartreeFock object for both merged wfns: same basis
    // and atoms
    auto hf_AB = HartreeFock(ABn.atoms, ABn.basis);
    Mat schwarz_ab = hf_AB.compute_schwarz_ints();
    if(m_use_density_fitting) {
	hf_AB.set_density_fitting_basis("cc-pvdz-jkfit");
    }

    Wavefunction ABo = ABn;
    Mat S_AB = hf_AB.compute_overlap_matrix();
    ABo.symmetric_orthonormalize_molecular_orbitals(S_AB);

    ABn.compute_density_matrix();
    ABo.compute_density_matrix();

    compute_ce_model_energies(ABn, hf_AB, precision, schwarz_ab);
    compute_ce_model_energies(ABo, hf_AB, precision, schwarz_ab);

    Energy E_ABn = ABn.energy - (A.energy + B.energy);

    CEEnergyComponents energy;
    energy.coulomb =
        E_ABn.coulomb + E_ABn.nuclear_attraction + E_ABn.nuclear_repulsion;
    double eABn = ABn.energy.core + ABn.energy.exchange + ABn.energy.coulomb;
    double eABo = ABo.energy.core + ABo.energy.exchange + ABo.energy.coulomb;
    double E_rep = eABo - eABn;
    energy.exchange_repulsion = E_ABn.exchange + E_rep;
    energy.dispersion = ce_model_dispersion_energy(A.atoms, B.atoms);
    energy.polarization = compute_polarization_energy(A, hf_a, B, hf_b);

    energy.total =
        scale_factors.scaled_total(energy.coulomb, energy.exchange_repulsion,
                                   energy.polarization, energy.dispersion);
    return energy;
}

} // namespace occ::interaction
