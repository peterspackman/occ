#include <fmt/core.h>
#include <fmt/ostream.h>
#include <occ/core/eigenp.h>
#include <occ/core/element.h>
#include <occ/core/logger.h>
#include <occ/core/timings.h>
#include <occ/io/conversion.h>
#include <occ/io/fchkreader.h>
#include <occ/io/fchkwriter.h>
#include <occ/io/moldenreader.h>
#include <occ/qm/merge.h>
#include <occ/qm/orb.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/wavefunction.h>
#include <occ/qm/ints.h>

namespace occ::qm {

using occ::io::FchkReader;
using occ::qm::merge_atoms;
using occ::qm::merge_basis_sets;
using occ::qm::merge_molecular_orbitals;

void Energy::print() const {
    constexpr auto format_string = "{:<10s} {:10.6f}\n";
    fmt::print(format_string, "E_coul", coulomb);
    fmt::print(format_string, "E_ex", exchange);
    fmt::print(format_string, "E_nn", nuclear_repulsion);
    fmt::print(format_string, "E_en", nuclear_attraction);
    fmt::print(format_string, "E_kin", kinetic);
    fmt::print(format_string, "E_1e", core);
}

Wavefunction::Wavefunction(const FchkReader &fchk)
    : spinorbital_kind(fchk.spinorbital_kind()), num_alpha(fchk.num_alpha()),
      num_beta(fchk.num_beta()), num_electrons(fchk.num_electrons()),
      basis(fchk.basis_set()), nbf(occ::qm::nbf(basis)), atoms(fchk.atoms()) {
    energy.total = fchk.scf_energy();
    set_molecular_orbitals(fchk);
    compute_density_matrix();
}

Wavefunction::Wavefunction(const MoldenReader &molden)
    : spinorbital_kind(molden.spinorbital_kind()),
      num_alpha(molden.num_alpha()), num_beta(molden.num_beta()),
      num_electrons(molden.num_electrons()), basis(molden.basis_set()),
      nbf(molden.nbf()), atoms(molden.atoms()) {
    size_t rows, cols;
    nbf = occ::qm::nbf(basis);

    if (spinorbital_kind == SpinorbitalKind::General) {
        throw std::runtime_error(
            "Reading MOs from g09 unsupported for General spinorbitals");
    } else if (spinorbital_kind == SpinorbitalKind::Unrestricted) {
        std::tie(rows, cols) =
            occ::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(nbf);
        mo.C = Mat(rows, cols);
        mo.energies = Vec(rows);
        block::a(mo.C) = molden.alpha_mo_coefficients();
        block::b(mo.C) = molden.beta_mo_coefficients();
        block::a(mo.energies) = molden.alpha_mo_energies();
        block::b(mo.energies) = molden.beta_mo_energies();
        block::a(mo.C) = molden.convert_mo_coefficients_from_molden_convention(
            basis, block::a(mo.C));
        block::b(mo.C) = molden.convert_mo_coefficients_from_molden_convention(
            basis, block::b(mo.C));
    } else {
        mo.C = molden.alpha_mo_coefficients();
        mo.C = molden.convert_mo_coefficients_from_molden_convention(basis, mo.C);
        mo.energies = molden.alpha_mo_energies();
    }
    update_occupied_orbitals();
    compute_density_matrix();
}

Wavefunction::Wavefunction(const Wavefunction &wfn_a, const Wavefunction &wfn_b)
    : num_alpha(wfn_a.num_alpha + wfn_b.num_alpha),
      num_beta(wfn_a.num_beta + wfn_b.num_beta),
      basis(merge_basis_sets(wfn_a.basis, wfn_b.basis)),
      nbf(basis.nbf()), atoms(merge_atoms(wfn_a.atoms, wfn_b.atoms)) {
    spinorbital_kind = (wfn_a.is_restricted() && wfn_b.is_restricted())
                           ? SpinorbitalKind::Restricted
                           : SpinorbitalKind::Unrestricted;

    size_t rows, cols;
    if (is_restricted())
        std::tie(rows, cols) =
            matrix_dimensions<SpinorbitalKind::Restricted>(nbf);
    else
        std::tie(rows, cols) =
            matrix_dimensions<SpinorbitalKind::Unrestricted>(nbf);
    mo.C = Mat(rows, cols);
    mo.energies = occ::Vec(rows);
    // temporaries for merging orbitals
    Mat C_merged;
    occ::Vec energies_merged;
    occ::log::debug("Merging occupied orbitals, sorted by energy");
    if (wfn_a.is_restricted() && wfn_b.is_restricted()) {
        // merge occupied orbitals
        std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
            wfn_a.mo.C.leftCols(wfn_a.num_alpha),
            wfn_b.mo.C.leftCols(wfn_b.num_alpha),
            wfn_a.mo.energies.topRows(wfn_a.num_alpha),
            wfn_b.mo.energies.topRows(wfn_b.num_alpha));
        mo.C.leftCols(num_alpha) = C_merged;
        mo.energies.topRows(num_alpha) = energies_merged;

        // merge virtual orbitals
        size_t nv_a = wfn_a.mo.C.rows() - wfn_a.num_alpha,
               nv_b = wfn_b.mo.C.rows() - wfn_b.num_alpha;
        size_t nv_ab = nv_a + nv_b;

        occ::log::debug("Merging virtual orbitals, sorted by energy");
        std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
            wfn_a.mo.C.rightCols(nv_a), wfn_b.mo.C.rightCols(nv_b),
            wfn_a.mo.energies.bottomRows(nv_a),
            wfn_b.mo.energies.bottomRows(nv_b));
        mo.C.rightCols(nv_ab) = C_merged;
        mo.energies.bottomRows(nv_ab) = energies_merged;
    } else {
        if (wfn_a.is_restricted()) {
            { // alpha
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    wfn_a.mo.C.leftCols(wfn_a.num_alpha),
                    block::a(wfn_b.mo.C).leftCols(wfn_b.num_alpha),
                    wfn_a.mo.energies.topRows(wfn_a.num_alpha),
                    block::a(wfn_b.mo.energies).topRows(wfn_b.num_alpha));
                block::a(mo.C).leftCols(num_alpha) = C_merged;
                block::a(mo.energies).topRows(num_alpha) = energies_merged;

                // merge virtual orbitals
                size_t nv_a = wfn_a.mo.C.rows() - wfn_a.num_alpha,
                       nv_b = block::a(wfn_b.mo.C).rows() - wfn_b.num_alpha;
                size_t nv_ab = nv_a + nv_b;

                occ::log::debug("Merging virtual orbitals, sorted by energy");
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    wfn_a.mo.C.rightCols(nv_a), block::a(wfn_b.mo.C).rightCols(nv_b),
                    wfn_a.mo.energies.bottomRows(nv_a),
                    block::a(wfn_b.mo.energies).bottomRows(nv_b));
                block::a(mo.C).rightCols(nv_ab) = C_merged;
                block::a(mo.energies).bottomRows(nv_ab) = energies_merged;
            }
            { // beta
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    wfn_a.mo.C.leftCols(wfn_a.num_beta),
                    block::b(wfn_b.mo.C).leftCols(wfn_b.num_beta),
                    wfn_a.mo.energies.topRows(wfn_a.num_beta),
                    block::b(wfn_b.mo.energies).topRows(wfn_b.num_beta));
                block::b(mo.C).leftCols(num_beta) = C_merged;
                block::b(mo.energies).topRows(num_beta) = energies_merged;

                // merge virtual orbitals
                size_t nv_a = wfn_a.mo.C.rows() - wfn_a.num_beta,
                       nv_b = block::b(wfn_b.mo.C).rows() - wfn_b.num_beta;
                size_t nv_ab = nv_a + nv_b;

                occ::log::debug("Merging virtual orbitals, sorted by energy");
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    wfn_a.mo.C.rightCols(nv_a), block::b(wfn_b.mo.C).rightCols(nv_b),
                    wfn_a.mo.energies.bottomRows(nv_a),
                    block::b(wfn_b.mo.energies).bottomRows(nv_b));
                block::b(mo.C).rightCols(nv_ab) = C_merged;
                block::b(mo.energies).bottomRows(nv_ab) = energies_merged;
            }
        } else if (wfn_b.is_restricted()) {
            { // alpha
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    block::a(wfn_a.mo.C).leftCols(wfn_a.num_alpha),
                    wfn_b.mo.C.leftCols(wfn_b.num_alpha),
                    block::a(wfn_a.mo.energies).topRows(wfn_a.num_alpha),
                    wfn_b.mo.energies.topRows(wfn_b.num_alpha));
                block::a(mo.C).leftCols(num_alpha) = C_merged;
                block::a(mo.energies).topRows(num_alpha) = energies_merged;

                // merge virtual orbitals
                size_t nv_a = block::a(wfn_a.mo.C).rows() - wfn_a.num_alpha,
                       nv_b = wfn_b.mo.C.rows() - wfn_b.num_alpha;
                size_t nv_ab = nv_a + nv_b;

                occ::log::debug("Merging virtual orbitals, sorted by energy");
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    block::a(wfn_a.mo.C).rightCols(nv_a), wfn_b.mo.C.rightCols(nv_b),
                    wfn_a.mo.energies.bottomRows(nv_a),
                    wfn_b.mo.energies.bottomRows(nv_b));
                block::a(mo.C).rightCols(nv_ab) = C_merged;
                block::a(mo.energies).bottomRows(nv_ab) = energies_merged;
            }
            { // beta
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    block::b(wfn_a.mo.C).leftCols(wfn_a.num_beta),
                    wfn_b.mo.C.leftCols(wfn_b.num_beta),
                    block::b(wfn_a.mo.energies).topRows(wfn_a.num_beta),
                    wfn_b.mo.energies.topRows(wfn_b.num_beta));
                block::b(mo.C).leftCols(num_beta) = C_merged;
                block::b(mo.energies).topRows(num_beta) = energies_merged;

                // merge virtual orbitals
                size_t nv_a = block::b(wfn_a.mo.C).rows() - wfn_a.num_beta,
                       nv_b = wfn_b.mo.C.rows() - wfn_b.num_beta;
                size_t nv_ab = nv_a + nv_b;

                occ::log::debug("Merging virtual orbitals, sorted by energy");
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    block::b(wfn_a.mo.C).rightCols(nv_a), wfn_b.mo.C.rightCols(nv_b),
                    block::b(wfn_a.mo.energies).bottomRows(nv_a),
                    wfn_b.mo.energies.bottomRows(nv_b));
                block::b(mo.C).rightCols(nv_ab) = C_merged;
                block::b(mo.energies).bottomRows(nv_ab) = energies_merged;
            }
        } else {
            { // alpha
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    block::a(wfn_a.mo.C).leftCols(wfn_a.num_alpha),
                    block::a(wfn_b.mo.C).leftCols(wfn_b.num_alpha),
                    block::a(wfn_a.mo.energies).topRows(wfn_a.num_alpha),
                    block::a(wfn_b.mo.energies).topRows(wfn_b.num_alpha));
                block::a(mo.C).leftCols(num_alpha) = C_merged;
                block::a(mo.energies).topRows(num_alpha) = energies_merged;

                // merge virtual orbitals
                size_t nv_a = block::a(wfn_a.mo.C).rows() - wfn_a.num_alpha,
                       nv_b = block::a(wfn_b.mo.C).rows() - wfn_b.num_alpha;
                size_t nv_ab = nv_a + nv_b;

                occ::log::debug("Merging virtual orbitals, sorted by energy");
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    block::a(wfn_a.mo.C).rightCols(nv_a),
                    block::a(wfn_b.mo.C).rightCols(nv_b),
                    wfn_a.mo.energies.bottomRows(nv_a),
                    block::a(wfn_b.mo.energies).bottomRows(nv_b));
                block::a(mo.C).rightCols(nv_ab) = C_merged;
                block::a(mo.energies).bottomRows(nv_ab) = energies_merged;
            }
            { // beta
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    block::b(wfn_a.mo.C).leftCols(wfn_a.num_beta),
                    block::b(wfn_b.mo.C).leftCols(wfn_b.num_beta),
                    block::b(wfn_a.mo.energies).topRows(wfn_a.num_beta),
                    block::b(wfn_b.mo.energies).topRows(wfn_b.num_beta));
                block::b(mo.C).leftCols(num_beta) = C_merged;
                block::b(mo.energies).topRows(num_beta) = energies_merged;

                // merge virtual orbitals
                size_t nv_a = block::b(wfn_a.mo.C).rows() - wfn_a.num_beta,
                       nv_b = block::b(wfn_b.mo.C).rows() - wfn_b.num_beta;
                size_t nv_ab = nv_a + nv_b;

                occ::log::debug("Merging virtual orbitals, sorted by energy");
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    block::b(wfn_a.mo.C).rightCols(nv_a),
                    block::b(wfn_b.mo.C).rightCols(nv_b),
                    block::b(wfn_a.mo.energies).bottomRows(nv_a),
                    block::b(wfn_b.mo.energies).bottomRows(nv_b));
               block::b(mo.C).rightCols(nv_ab) = C_merged;
               block::b(mo.energies).bottomRows(nv_ab) = energies_merged;
            }
        }
    }
    update_occupied_orbitals();
}

void Wavefunction::update_occupied_orbitals() {
    if (mo.C.size() == 0) {
        return;
    }
    if (spinorbital_kind == SpinorbitalKind::General) {
        throw std::runtime_error(
            "Reading MOs from g09 unsupported for General spinorbitals");
    } else if (spinorbital_kind == SpinorbitalKind::Unrestricted) {
        mo.Cocc = occ::qm::orb::occupied_unrestricted(mo.C, num_alpha, num_beta);
    } else {
        mo.Cocc = occ::qm::orb::occupied_restricted(mo.C, num_alpha);
    }
}

void Wavefunction::set_molecular_orbitals(const FchkReader &fchk) {
    size_t rows, cols;
    nbf = occ::qm::nbf(basis);
    mo.kind = fchk.spinorbital_kind();
    if (spinorbital_kind == SpinorbitalKind::General) {
        throw std::runtime_error(
            "Reading MOs from g09 unsupported for General spinorbitals");
    } else if (spinorbital_kind == SpinorbitalKind::Unrestricted) {
        std::tie(rows, cols) =
            occ::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(nbf);
        mo.C = Mat(rows, cols);
        mo.energies = Vec(rows);
        block::a(mo.C) = fchk.alpha_mo_coefficients();
        block::b(mo.C) = fchk.beta_mo_coefficients();
        block::a(mo.energies) = fchk.alpha_mo_energies();
        block::b(mo.energies) = fchk.beta_mo_energies();
	if(!basis.is_pure()) {
	    block::a(mo.C) = occ::io::conversion::orb::from_gaussian_order_cartesian(
		basis, block::a(mo.C));
	    block::b(mo.C) = occ::io::conversion::orb::from_gaussian_order_cartesian(
		basis, block::b(mo.C));
	}
	else {
	    block::a(mo.C) = occ::io::conversion::orb::from_gaussian_order_spherical(
		basis, block::a(mo.C));
	    block::b(mo.C) = occ::io::conversion::orb::from_gaussian_order_spherical(
		basis, block::b(mo.C));
	}
    } else {
        mo.C = fchk.alpha_mo_coefficients();
        mo.energies = fchk.alpha_mo_energies();
	if(!basis.is_pure()) {
	    mo.C = occ::io::conversion::orb::from_gaussian_order_cartesian(basis, mo.C);
	}
	else {
	    mo.C = occ::io::conversion::orb::from_gaussian_order_spherical(basis, mo.C);
	}
    }
    update_occupied_orbitals();
}

void Wavefunction::compute_density_matrix() {
    if (spinorbital_kind == SpinorbitalKind::General) {
        throw std::runtime_error(
            "Reading MOs from g09 unsupported for General spinorbitals");
    } else if (spinorbital_kind == SpinorbitalKind::Unrestricted) {
        mo.D = occ::qm::orb::density_matrix_unrestricted(mo.Cocc, num_alpha,
                                                      num_beta);
    } else {
        mo.D = occ::qm::orb::density_matrix_restricted(mo.Cocc);
    }
}

void Wavefunction::symmetric_orthonormalize_molecular_orbitals(
    const Mat &overlap) {
    if (spinorbital_kind == SpinorbitalKind::Restricted) {
        mo.C = symmorthonormalize_molecular_orbitals(mo.C, overlap, num_alpha);
    } else {
        block::a(mo.C) = symmorthonormalize_molecular_orbitals(block::a(mo.C), overlap,
                                                          num_alpha);
        block::b(mo.C) =
            symmorthonormalize_molecular_orbitals(block::b(mo.C), overlap, num_beta);
    }
    update_occupied_orbitals();
}

Mat symmetrically_orthonormalize(const Mat &mat, const Mat &metric) {
    Mat X, X_invT;
    size_t n_cond;
    double x_condition_number, condition_number;
    double threshold = 1.0 / std::numeric_limits<double>::epsilon();
    Mat SS = mat.transpose() * metric * mat;
    std::tie(X, X_invT, n_cond, x_condition_number, condition_number) =
        occ::gensqrtinv(SS, true, threshold);
    return mat * X;
}

Mat symmorthonormalize_molecular_orbitals(const Mat &mos, const Mat &overlap,
                                          size_t n_occ) {
    Mat result(mos.rows(), mos.cols());
    size_t n_virt = mos.cols() - n_occ;
    Mat C_occ = mos.leftCols(n_occ);
    Mat C_virt = mos.rightCols(n_virt);
    result.leftCols(n_occ) = symmetrically_orthonormalize(C_occ, overlap);
    result.rightCols(n_virt) = symmetrically_orthonormalize(C_virt, overlap);
    return result;
}

void Wavefunction::apply_transformation(const occ::Mat3 &rot,
                                        const occ::Vec3 &trans) {
    apply_rotation(rot);
    apply_translation(trans);
}
void Wavefunction::apply_translation(const occ::Vec3 &trans) {
    basis.translate(trans);
    occ::core::translate_atoms(atoms, trans);
}

void Wavefunction::apply_rotation(const occ::Mat3 &rot) {
    mo.rotate(basis, rot);
    basis.rotate(rot);
    occ::core::rotate_atoms(atoms, rot);
    update_occupied_orbitals();
    compute_density_matrix();
}

occ::Mat3N Wavefunction::positions() const {
    occ::Mat3N pos(3, atoms.size());
    for (size_t i = 0; i < atoms.size(); i++) {
        pos(0, i) = atoms[i].x;
        pos(1, i) = atoms[i].y;
        pos(2, i) = atoms[i].z;
    }
    return pos;
}

occ::IVec Wavefunction::atomic_numbers() const {
    occ::IVec nums(atoms.size());
    for (size_t i = 0; i < atoms.size(); i++) {
        nums(i) = atoms[i].atomic_number;
    }
    return nums;
}

void Wavefunction::save(FchkWriter &fchk) {
    occ::timing::start(occ::timing::category::io);
    fchk.set_scalar("Number of atoms", atoms.size());
    fchk.set_scalar("Charge", charge());
    fchk.set_scalar("Multiplicity", multiplicity());
    fchk.set_scalar("SCF Energy", energy.total);
    fchk.set_scalar("Number of electrons", num_electrons);
    fchk.set_scalar("Number of alpha electrons", num_alpha);
    fchk.set_scalar("Number of beta electrons", num_beta);
    fchk.set_scalar("Number of basis functions", nbf);
    // nuclear charges
    occ::IVec nums = atomic_numbers();
    occ::Vec atomic_prop = nums.cast<double>();
    fchk.set_vector("Atomic numbers", nums);
    fchk.set_vector("Nuclear charges", atomic_prop);
    fchk.set_vector("Current cartesian coordinates", positions());
    // atomic weights
    for (Eigen::Index i = 0; i < atomic_prop.rows(); i++)
        atomic_prop(i) =
            static_cast<double>(occ::core::Element(nums(i)).mass());
    fchk.set_vector("Integer atomic weights",
                    atomic_prop.array().round().cast<int>());
    fchk.set_vector("Real atomic weights", atomic_prop);

    auto Cfchk = [&]() {
        if (basis.is_pure())
            return occ::io::conversion::orb::to_gaussian_order_spherical(basis, mo.C);
        return occ::io::conversion::orb::to_gaussian_order_cartesian(basis, mo.C);
    }();
    Mat Dfchk;

    std::vector<double> density_lower_triangle, spin_density_lower_triangle;

    if (spinorbital_kind == SpinorbitalKind::Unrestricted) {
        fchk.set_vector("Alpha Orbital Energies", block::a(mo.energies));
        fchk.set_vector("Alpha MO coefficients", block::a(Cfchk));
        fchk.set_vector("Beta Orbital Energies", block::b(mo.energies));
        fchk.set_vector("Beta MO coefficients", block::b(Cfchk));
        Mat occ_fchk =
            occ::qm::orb::occupied_unrestricted(Cfchk, num_alpha, num_beta);
        Dfchk = occ::qm::orb::density_matrix_unrestricted(occ_fchk, num_alpha,
                                                          num_beta);

        density_lower_triangle.reserve(nbf * (nbf - 1) / 2);
        spin_density_lower_triangle.reserve(nbf * (nbf - 1) / 2);
        auto da = block::a(Dfchk);
        auto db = block::b(Dfchk);
        for (Eigen::Index row = 0; row < nbf; row++) {
            for (Eigen::Index col = 0; col <= row; col++) {
                double va = da(row, col) * 2, vb = db(row, col) * 2;
                density_lower_triangle.push_back(va + vb);
                spin_density_lower_triangle.push_back(vb - va);
            }
        }
    } else {
        fchk.set_vector("Alpha Orbital Energies", mo.energies);
        fchk.set_vector("Alpha MO coefficients", Cfchk);
        Mat occ_fchk = occ::qm::orb::occupied_restricted(Cfchk, num_alpha);
        Dfchk = occ::qm::orb::density_matrix_restricted(occ_fchk);
        density_lower_triangle.reserve(nbf * (nbf - 1) / 2);
        for (Eigen::Index row = 0; row < nbf; row++) {
            for (Eigen::Index col = 0; col <= row; col++) {
                density_lower_triangle.push_back(Dfchk(row, col) * 2);
            }
        }
    }
    fchk.set_vector("Total SCF Density", density_lower_triangle);
    if (spin_density_lower_triangle.size() > 0)
        fchk.set_vector("Spin SCF Density", spin_density_lower_triangle);
    fchk.set_basis(basis);

    std::vector<int> shell2atom;
    shell2atom.reserve(basis.size());
    for (const auto &x : basis.shell2atom(atoms)) {
        shell2atom.push_back(x + 1);
    }
    fchk.set_vector("Shell to atom map", shell2atom);
    occ::timing::stop(occ::timing::category::io);
}

void Wavefunction::save_npz(const std::string &filename) {
    occ::timing::start(occ::timing::category::io);

    int scalar_i;
    double scalar_d;
    std::vector<size_t> scalar_shape{1}, vector_shape{2};

    auto set_scalar = [&](auto name, auto val, const std::string mode = "a") {
        enpy::save_npz(filename, name, &val, scalar_shape, mode);
    };
    set_scalar("n_atom", atoms.size(), "w");
    set_scalar("charge", charge());
    set_scalar("multiplicity", multiplicity());
    set_scalar("e_scf", energy.total);
    set_scalar("n_electron", num_electrons);
    set_scalar("n_alpha", num_alpha);
    set_scalar("n_beta", num_beta);
    set_scalar("n_bf", nbf);

    auto set_matrix = [&](auto name, auto val) {
        enpy::save_npz(filename, name, val, "a");
    };

    set_matrix("atomic_numbers", atomic_numbers());
    set_matrix("positions", positions());
    if (mo.D.rows())
        set_matrix("density_matrix", mo.D);
    if (mo.C.rows())
        set_matrix("mos", mo.C);
    if (mo.Cocc.rows())
        set_matrix("occupied_mos", mo.Cocc);
    if (mo.energies.rows())
        set_matrix("mo_energies", mo.energies);
    if (T.rows())
        set_matrix("kinetic_matrix", T);
    if (V.rows())
        set_matrix("nuclear_attraction_matrix", V);
    if (H.rows())
        set_matrix("core_hamiltonian", H);
    if (J.rows())
        set_matrix("coulomb_matrix", J);
    if (K.rows())
        set_matrix("exchange_matrix", K);

    occ::timing::stop(occ::timing::category::io);
}

} // namespace occ::qm
