#include <fmt/core.h>
#include <fmt/ostream.h>
#include <occ/core/element.h>
#include <occ/core/gensqrtinv.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/io/conversion.h>
#include <occ/io/fchkreader.h>
#include <occ/io/fchkwriter.h>
#include <occ/io/moldenreader.h>
#include <occ/qm/hf.h>
#include <occ/qm/merge.h>
#include <occ/qm/orb.h>
#include <occ/qm/partitioning.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/wavefunction.h>

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
    if (ecp != 0.0) {
        fmt::print(format_string, "E_ecp", ecp);
    }
}

Wavefunction::Wavefunction(const FchkReader &fchk)
    : spinorbital_kind(fchk.spinorbital_kind()), num_alpha(fchk.num_alpha()),
      num_beta(fchk.num_beta()), num_electrons(fchk.num_electrons()),
      basis(fchk.basis_set()), nbf(basis.nbf()), atoms(fchk.atoms()) {
    energy.total = fchk.scf_energy();
    set_molecular_orbitals(fchk);
    compute_density_matrix();
}

Wavefunction::Wavefunction(const OrcaJSONReader &json)
    : spinorbital_kind(json.spinorbital_kind()), num_alpha(json.num_alpha()),
      num_beta(json.num_beta()), num_electrons(json.num_electrons()),
      basis(json.basis_set()), nbf(basis.nbf()), atoms(json.atoms()) {
    energy.total = json.scf_energy();
    size_t rows, cols;

    mo.kind = spinorbital_kind;
    if (spinorbital_kind == SpinorbitalKind::General) {
        throw std::runtime_error(
            "Reading MOs from Orca json unsupported for General spinorbitals");
    } else if (spinorbital_kind == SpinorbitalKind::Unrestricted) {
        std::tie(rows, cols) =
            occ::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(nbf);
        mo.C = Mat(rows, cols);
        mo.energies = Vec(rows);
        block::a(mo.C) = json.alpha_mo_coefficients();
        block::b(mo.C) = json.beta_mo_coefficients();
        block::a(mo.energies) = json.alpha_mo_energies();
        block::b(mo.energies) = json.beta_mo_energies();
    } else {
        mo.C = json.alpha_mo_coefficients();
        mo.energies = json.alpha_mo_energies();
    }
    update_occupied_orbitals();
    compute_density_matrix();
}

Wavefunction::Wavefunction(const MoldenReader &molden)
    : spinorbital_kind(molden.spinorbital_kind()),
      num_alpha(molden.num_alpha()), num_beta(molden.num_beta()),
      num_electrons(molden.num_electrons()), basis(molden.basis_set()),
      nbf(molden.nbf()), atoms(molden.atoms()) {
    size_t rows, cols;
    nbf = basis.nbf();
    mo.kind = spinorbital_kind;

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
        mo.C =
            molden.convert_mo_coefficients_from_molden_convention(basis, mo.C);
        mo.energies = molden.alpha_mo_energies();
    }
    update_occupied_orbitals();
    compute_density_matrix();
}

Wavefunction::Wavefunction(const Wavefunction &wfn_a, const Wavefunction &wfn_b)
    : num_alpha(wfn_a.num_alpha + wfn_b.num_alpha),
      num_beta(wfn_a.num_beta + wfn_b.num_beta),
      basis(merge_basis_sets(wfn_a.basis, wfn_b.basis)), nbf(basis.nbf()),
      atoms(merge_atoms(wfn_a.atoms, wfn_b.atoms)) {
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
    // TODO refactor
    if (wfn_a.is_restricted() && wfn_b.is_restricted()) {
        mo.kind = SpinorbitalKind::Restricted;
        // merge occupied orbitals
        Vec occ_energies_merged;
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
        mo.kind = SpinorbitalKind::Unrestricted;
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
                    wfn_a.mo.C.rightCols(nv_a),
                    block::a(wfn_b.mo.C).rightCols(nv_b),
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
                    wfn_a.mo.C.rightCols(nv_a),
                    block::b(wfn_b.mo.C).rightCols(nv_b),
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
                    block::a(wfn_a.mo.C).rightCols(nv_a),
                    wfn_b.mo.C.rightCols(nv_b),
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
                    block::b(wfn_a.mo.C).rightCols(nv_a),
                    wfn_b.mo.C.rightCols(nv_b),
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

    if (wfn_a.have_xdm_parameters && wfn_b.have_xdm_parameters) {
        occ::log::debug("Merging XDM parameters");
        have_xdm_parameters = true;
        occ::log::debug("Merging XDM polarizabilities");
        xdm_polarizabilities = Vec(wfn_a.xdm_polarizabilities.rows() +
                                   wfn_b.xdm_polarizabilities.rows());
        xdm_polarizabilities << wfn_a.xdm_polarizabilities,
            wfn_b.xdm_polarizabilities;

        occ::log::debug("Merging XDM moments");
        xdm_moments = Mat(wfn_a.xdm_moments.rows(),
                          wfn_a.xdm_moments.cols() + wfn_b.xdm_moments.cols());
        xdm_moments.leftCols(wfn_a.xdm_moments.cols()) = wfn_a.xdm_moments;
        xdm_moments.rightCols(wfn_b.xdm_moments.cols()) = wfn_b.xdm_moments;

        occ::log::debug("Merging XDM volumes");
        xdm_volumes = Vec(wfn_a.xdm_volumes.rows() + wfn_b.xdm_volumes.rows());
        xdm_volumes << wfn_a.xdm_volumes, wfn_b.xdm_volumes;
        occ::log::debug("Merging XDM free volumes");
        xdm_free_volumes =
            Vec(wfn_a.xdm_free_volumes.rows() + wfn_b.xdm_free_volumes.rows());
        xdm_free_volumes << wfn_a.xdm_free_volumes, wfn_b.xdm_free_volumes;
        occ::log::debug("Finished merging");
    }
}

void Wavefunction::update_occupied_orbitals() {
    mo.n_ao = nbf;
    mo.n_alpha = num_alpha;
    mo.n_beta = num_beta;
    if (mo.C.size() == 0) {
        return;
    }
    if (spinorbital_kind == SpinorbitalKind::General) {
        throw std::runtime_error(
            "Reading MOs from g09 unsupported for General spinorbitals");
    } else if (spinorbital_kind == SpinorbitalKind::Unrestricted) {
        occ::log::debug("num alpha electrons = {}", num_alpha);
        occ::log::debug("num beta electrons = {}", num_beta);
        mo.Cocc =
            occ::qm::orb::occupied_unrestricted(mo.C, num_alpha, num_beta);
    } else {
        occ::log::debug("num alpha electrons = {}", num_alpha);
        occ::log::debug("num beta electrons = {}", num_alpha);
        mo.Cocc = occ::qm::orb::occupied_restricted(mo.C, num_alpha);
    }
}

void Wavefunction::set_molecular_orbitals(const FchkReader &fchk) {
    size_t rows, cols;
    nbf = basis.nbf();
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
    } else {
        mo.C = fchk.alpha_mo_coefficients();
        mo.energies = fchk.alpha_mo_energies();
    }
    mo = occ::io::conversion::orb::from_gaussian_order(basis, mo);
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
        block::a(mo.C) = symmorthonormalize_molecular_orbitals(
            block::a(mo.C), overlap, num_alpha);
        block::b(mo.C) = symmorthonormalize_molecular_orbitals(
            block::b(mo.C), overlap, num_beta);
    }
    update_occupied_orbitals();
}

Mat symmetrically_orthonormalize(const Mat &mat, const Mat &metric) {
    double threshold = 1.0 / std::numeric_limits<double>::epsilon();
    Mat SS = mat.transpose() * metric * mat;
    auto g = occ::core::gensqrtinv(SS, true, threshold);
    return mat * g.result;
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
    // check if identity matrix, otherwise do the rotation and
    // update the orbitals
    if (!rot.isIdentity(1e-6)) {
        mo.rotate(basis, rot);
        basis.rotate(rot);
        occ::core::rotate_atoms(atoms, rot);
        update_occupied_orbitals();
        compute_density_matrix();
    } else {
        occ::log::debug("Skipping rotation by identity matrix");
    }
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
    bool have_ecps = basis.have_ecps();
    occ::timing::start(occ::timing::category::io);
    fchk.set_scalar("Number of atoms", atoms.size());
    fchk.set_scalar("Charge", charge());
    fchk.set_scalar("Multiplicity", multiplicity());
    fchk.set_scalar("SCF Energy", energy.total);
    fchk.set_scalar("Number of electrons", num_electrons);
    fchk.set_scalar("Number of alpha electrons", num_alpha);
    fchk.set_scalar("Number of beta electrons", num_beta);
    fchk.set_scalar("Number of basis functions", nbf);
    fchk.set_scalar("Number of independent functions", nbf);
    fchk.set_scalar("Number of point charges in /Mol/", 0);
    fchk.set_scalar("Number of translation vectors", 0);
    fchk.set_scalar("Force Field", 0);

    // nuclear charges
    occ::IVec nums = atomic_numbers();
    occ::Vec atomic_prop = nums.cast<double>();
    if (have_ecps) {
        // set nuclear charges to include ecp
        const auto &ecp_electrons = basis.ecp_electrons();
        for (int i = 0; i < atoms.size(); i++) {
            atomic_prop(i) -= ecp_electrons[i];
        }
    }
    fchk.set_vector("Atomic numbers", nums);
    fchk.set_vector("Nuclear charges", atomic_prop);
    fchk.set_vector("Current cartesian coordinates", positions());
    fchk.set_vector("Int Atom Types", occ::IVec::Zero(nums.rows()));
    fchk.set_vector("MM Charges", occ::Vec::Zero(nums.rows()));
    fchk.set_vector("Atom residue num", occ::IVec::Zero(nums.rows()));

    // atomic weights
    for (Eigen::Index i = 0; i < atomic_prop.rows(); i++)
        atomic_prop(i) =
            static_cast<double>(occ::core::Element(nums(i)).mass());
    fchk.set_vector("Integer atomic weights",
                    atomic_prop.array().round().cast<int>());
    fchk.set_vector("Real atomic weights", atomic_prop);

    bool spherical = basis.is_pure();

    auto mo_fchk = occ::io::conversion::orb::to_gaussian_order(basis, mo);
    Mat Dfchk;

    {
        std::ofstream d("density_correct.txt");
        d << mo.D;
    }

    std::vector<double> density_lower_triangle, spin_density_lower_triangle;

    if (spinorbital_kind == SpinorbitalKind::Unrestricted) {
        fchk.set_vector("Alpha Orbital Energies", block::a(mo_fchk.energies));
        fchk.set_vector("Alpha MO coefficients", block::a(mo_fchk.C));
        fchk.set_vector("Beta Orbital Energies", block::b(mo_fchk.energies));
        fchk.set_vector("Beta MO coefficients", block::b(mo_fchk.C));
        Mat occ_fchk =
            occ::qm::orb::occupied_unrestricted(mo_fchk.C, num_alpha, num_beta);
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
        fchk.set_vector("Alpha Orbital Energies", mo_fchk.energies);
        fchk.set_vector("Alpha MO coefficients", mo_fchk.C);
        Mat occ_fchk = occ::qm::orb::occupied_restricted(mo_fchk.C, num_alpha);
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
    for (const auto &x : basis.shell_to_atom()) {
        shell2atom.push_back(x + 1);
    }
    fchk.set_vector("Shell to atom map", shell2atom);

    if (have_ecps) {
        // TODO finish ECP writing routines
        occ::log::debug("Writing ECP information to fchk\n");
        fchk.set_vector<int, double>("ECP-RNFroz", basis.ecp_electrons());
        std::vector<double> ecp_clp1;
        std::vector<double> ecp_clp2;
        std::vector<int> ecp_nlp;
        std::vector<double> ecp_zlp;
        std::vector<int> ecp_lmax(atoms.size(), 0);
        int ecp_max_length = 0;
        const auto &ecp_shell2atom = basis.ecp_shell_to_atom();
        int shell_index = 0;
        for (const auto &sh : basis.ecp_shells()) {
            int atom_idx = ecp_shell2atom[shell_index];
            ecp_max_length =
                std::max(static_cast<int>(sh.num_primitives()), ecp_max_length);
            ecp_lmax[atom_idx] =
                std::max(static_cast<int>(sh.l), ecp_lmax[atom_idx]);
            for (int i = 0; i < sh.num_primitives(); i++) {
                ecp_clp1.push_back(sh.contraction_coefficients(i, 0));
                ecp_zlp.push_back(sh.exponents(i));
                ecp_nlp.push_back(sh.ecp_r_exponents(i));
                ecp_clp2.push_back(0.0);
            }
            shell_index++;
        }
        fchk.set_scalar("ECP-MaxLECP", ecp_max_length);
        fchk.set_vector("ECP-LMax", ecp_lmax);
        fchk.set_vector("ECP-NLP", ecp_nlp);
        fchk.set_vector("ECP-CLP1", ecp_clp1);
        fchk.set_vector("ECP-CLP2", ecp_clp2);
        fchk.set_vector("ECP-ZLP", ecp_zlp);
    }

    // TODO fix this is wrong
    fchk.set_scalar("Virial ratio",
                    -(energy.nuclear_repulsion + energy.nuclear_attraction +
                      energy.coulomb + energy.exchange) /
                        energy.kinetic);
    fchk.set_scalar("SCF ratio", energy.total);
    fchk.set_scalar("Total ratio", energy.total);

    occ::timing::stop(occ::timing::category::io);
}

Vec Wavefunction::electric_potential(const Mat3N &points) const {
    HartreeFock hf(basis);
    Vec esp_e = hf.electronic_electric_potential_contribution(mo, points);
    Vec esp_n = hf.nuclear_electric_potential_contribution(points);
    return esp_e + esp_n;
}

Vec Wavefunction::mulliken_charges() const {

    HartreeFock hf(basis);
    Mat overlap = hf.compute_overlap_matrix();

    Vec charges = Vec::Zero(atoms.size());

    switch (spinorbital_kind) {
    case SpinorbitalKind::Unrestricted:
        charges =
            -2 * occ::qm::mulliken_partition<SpinorbitalKind::Unrestricted>(
                     basis, mo.D, overlap);
        break;
    case SpinorbitalKind::General:
        charges = -2 * occ::qm::mulliken_partition<SpinorbitalKind::General>(
                           basis, mo.D, overlap);
        break;
    default:
        charges = -2 * occ::qm::mulliken_partition<SpinorbitalKind::Restricted>(
                           basis, mo.D, overlap);
        break;
    }
    const auto &ecp_electrons = basis.ecp_electrons();
    for (size_t i = 0; i < atoms.size(); i++) {
        charges(i) += atoms[i].atomic_number - ecp_electrons[i];
    }
    return charges;
}

} // namespace occ::qm
