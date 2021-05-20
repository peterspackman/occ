#include <occ/core/logger.h>
#include <occ/qm/wavefunction.h>
#include <occ/qm/spinorbital.h>
#include <occ/io/conversion.h>
#include <occ/io/fchkreader.h>
#include <occ/io/fchkwriter.h>
#include <occ/io/moldenreader.h>
#include <occ/qm/merge.h>
#include <occ/qm/orb.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <occ/core/element.h>

namespace occ::qm {

using occ::qm::merge_molecular_orbitals;
using occ::qm::merge_basis_sets;
using occ::qm::merge_atoms;
using occ::io::FchkReader;

void Energy::print() const
{
    constexpr auto format_string = "{:<10s} {:10.6f}\n";
    fmt::print(format_string, "E_coul", coulomb);
    fmt::print(format_string, "E_ex", exchange);
    fmt::print(format_string, "E_nn", nuclear_repulsion);
    fmt::print(format_string, "E_en", nuclear_attraction);
    fmt::print(format_string, "E_kin", kinetic);
    fmt::print(format_string, "E_1e", core);
}

Wavefunction::Wavefunction(const FchkReader& fchk) :
    spinorbital_kind(fchk.spinorbital_kind()),
    num_alpha(fchk.num_alpha()),
    num_beta(fchk.num_beta()),
    num_electrons(fchk.num_electrons()),
    basis(fchk.basis_set()),
    nbf(occ::qm::nbf(basis)),
    atoms(fchk.atoms())
{
    set_molecular_orbitals(fchk);
    compute_density_matrix();
}


Wavefunction::Wavefunction(const MoldenReader& molden) :
    spinorbital_kind(molden.spinorbital_kind()),
    num_alpha(molden.num_alpha()),
    num_beta(molden.num_beta()),
    num_electrons(molden.num_electrons()),
    basis(molden.basis_set()),
    nbf(molden.nbf()),
    atoms(molden.atoms())
{
    size_t rows, cols;
    nbf = occ::qm::nbf(basis);

    if(spinorbital_kind == SpinorbitalKind::General) {
        throw std::runtime_error("Reading MOs from g09 unsupported for General spinorbitals");
    }
    else if(spinorbital_kind == SpinorbitalKind::Unrestricted) {
        std::tie(rows, cols) = occ::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(nbf);
        C = MatRM(rows, cols);
        mo_energies = Vec(rows);
        C.alpha() = molden.alpha_mo_coefficients();
        C.beta() = molden.beta_mo_coefficients();
        mo_energies.alpha() = molden.alpha_mo_energies();
        mo_energies.beta() = molden.beta_mo_energies();
        C.alpha() = molden.convert_mo_coefficients_from_molden_convention(basis, C.alpha());
        C.beta() = molden.convert_mo_coefficients_from_molden_convention(basis, C.beta());
    }
    else {
        C = molden.alpha_mo_coefficients();
        C = molden.convert_mo_coefficients_from_molden_convention(basis, C);
        mo_energies = molden.alpha_mo_energies();
    }
    update_occupied_orbitals();
    compute_density_matrix();
}

Wavefunction::Wavefunction(const Wavefunction &wfn_a, const Wavefunction &wfn_b) :
    num_alpha(wfn_a.num_alpha + wfn_b.num_alpha),
    num_beta(wfn_a.num_beta + wfn_b.num_beta),
    basis(merge_basis_sets(wfn_a.basis, wfn_b.basis)),
    nbf(wfn_a.nbf + wfn_b.nbf),
    atoms(merge_atoms(wfn_a.atoms, wfn_b.atoms))
{
    spinorbital_kind = (wfn_a.is_restricted() && wfn_b.is_restricted()) ? SpinorbitalKind::Restricted : SpinorbitalKind::Unrestricted;

    size_t rows, cols;
    if(is_restricted()) std::tie(rows, cols) = matrix_dimensions<SpinorbitalKind::Restricted>(nbf);
    else std::tie(rows, cols) = matrix_dimensions<SpinorbitalKind::Unrestricted>(nbf);
    C = MatRM(rows, cols);
    mo_energies = occ::Vec(rows);
    // temporaries for merging orbitals
    MatRM C_merged;
    occ::Vec energies_merged;
    occ::log::debug("Merging occupied orbitals, sorted by energy");
    if(wfn_a.is_restricted() && wfn_b.is_restricted())
    {
        // merge occupied orbitals
        std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
            wfn_a.C.leftCols(wfn_a.num_alpha), wfn_b.C.leftCols(wfn_b.num_alpha),
            wfn_a.mo_energies.topRows(wfn_a.num_alpha), wfn_b.mo_energies.topRows(wfn_b.num_alpha)
        );
        C.leftCols(num_alpha) = C_merged;
        mo_energies.topRows(num_alpha) = energies_merged;

        // merge virtual orbitals
        size_t nv_a = wfn_a.C.rows() - wfn_a.num_alpha, nv_b = wfn_b.C.rows() - wfn_b.num_alpha;
        size_t nv_ab = nv_a + nv_b;

        occ::log::debug("Merging virtual orbitals, sorted by energy");
        std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
            wfn_a.C.rightCols(nv_a), wfn_b.C.rightCols(nv_b),
            wfn_a.mo_energies.bottomRows(nv_a), wfn_b.mo_energies.bottomRows(nv_b));
        C.rightCols(nv_ab) = C_merged;
        mo_energies.bottomRows(nv_ab) = energies_merged;
    }
    else {
        if(wfn_a.is_restricted()) {
            { //alpha
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    wfn_a.C.leftCols(wfn_a.num_alpha), wfn_b.C.alpha().leftCols(wfn_b.num_alpha),
                    wfn_a.mo_energies.topRows(wfn_a.num_alpha), wfn_b.mo_energies.alpha().topRows(wfn_b.num_alpha)
                );
                C.alpha().leftCols(num_alpha) = C_merged;
                mo_energies.alpha().topRows(num_alpha) = energies_merged;

                // merge virtual orbitals
                size_t nv_a = wfn_a.C.rows() - wfn_a.num_alpha, nv_b = wfn_b.C.alpha().rows() - wfn_b.num_alpha;
                size_t nv_ab = nv_a + nv_b;

                occ::log::debug("Merging virtual orbitals, sorted by energy");
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    wfn_a.C.rightCols(nv_a), wfn_b.C.alpha().rightCols(nv_b),
                    wfn_a.mo_energies.bottomRows(nv_a), wfn_b.mo_energies.alpha().bottomRows(nv_b));
                C.alpha().rightCols(nv_ab) = C_merged;
                mo_energies.alpha().bottomRows(nv_ab) = energies_merged;
            }
            { //beta
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    wfn_a.C.leftCols(wfn_a.num_beta), wfn_b.C.beta().leftCols(wfn_b.num_beta),
                    wfn_a.mo_energies.topRows(wfn_a.num_beta), wfn_b.mo_energies.beta().topRows(wfn_b.num_beta)
                );
                C.beta().leftCols(num_beta) = C_merged;
                mo_energies.beta().topRows(num_beta) = energies_merged;

                // merge virtual orbitals
                size_t nv_a = wfn_a.C.rows() - wfn_a.num_beta, nv_b = wfn_b.C.beta().rows() - wfn_b.num_beta;
                size_t nv_ab = nv_a + nv_b;

                occ::log::debug("Merging virtual orbitals, sorted by energy");
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    wfn_a.C.rightCols(nv_a), wfn_b.C.beta().rightCols(nv_b),
                    wfn_a.mo_energies.bottomRows(nv_a), wfn_b.mo_energies.beta().bottomRows(nv_b));
                C.beta().rightCols(nv_ab) = C_merged;
                mo_energies.beta().bottomRows(nv_ab) = energies_merged;
            }
        }
        else if(wfn_b.is_restricted()) {
            { //alpha
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    wfn_a.C.alpha().leftCols(wfn_a.num_alpha), wfn_b.C.leftCols(wfn_b.num_alpha),
                    wfn_a.mo_energies.alpha().topRows(wfn_a.num_alpha), wfn_b.mo_energies.topRows(wfn_b.num_alpha)
                );
                C.alpha().leftCols(num_alpha) = C_merged;
                mo_energies.alpha().topRows(num_alpha) = energies_merged;

                // merge virtual orbitals
                size_t nv_a = wfn_a.C.alpha().rows() - wfn_a.num_alpha, nv_b = wfn_b.C.rows() - wfn_b.num_alpha;
                size_t nv_ab = nv_a + nv_b;

                occ::log::debug("Merging virtual orbitals, sorted by energy");
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    wfn_a.C.alpha().rightCols(nv_a), wfn_b.C.rightCols(nv_b),
                    wfn_a.mo_energies.bottomRows(nv_a), wfn_b.mo_energies.bottomRows(nv_b));
                C.alpha().rightCols(nv_ab) = C_merged;
                mo_energies.alpha().bottomRows(nv_ab) = energies_merged;
            }
            { //beta
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    wfn_a.C.beta().leftCols(wfn_a.num_beta), wfn_b.C.leftCols(wfn_b.num_beta),
                    wfn_a.mo_energies.beta().topRows(wfn_a.num_beta), wfn_b.mo_energies.topRows(wfn_b.num_beta)
                );
                C.beta().leftCols(num_beta) = C_merged;
                mo_energies.beta().topRows(num_beta) = energies_merged;

                // merge virtual orbitals
                size_t nv_a = wfn_a.C.beta().rows() - wfn_a.num_beta, nv_b = wfn_b.C.rows() - wfn_b.num_beta;
                size_t nv_ab = nv_a + nv_b;

                occ::log::debug("Merging virtual orbitals, sorted by energy");
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    wfn_a.C.beta().rightCols(nv_a), wfn_b.C.rightCols(nv_b),
                    wfn_a.mo_energies.beta().bottomRows(nv_a), wfn_b.mo_energies.bottomRows(nv_b));
                C.beta().rightCols(nv_ab) = C_merged;
                mo_energies.beta().bottomRows(nv_ab) = energies_merged;
            }
        }
        else {
            { //alpha
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    wfn_a.C.alpha().leftCols(wfn_a.num_alpha), wfn_b.C.alpha().leftCols(wfn_b.num_alpha),
                    wfn_a.mo_energies.alpha().topRows(wfn_a.num_alpha), wfn_b.mo_energies.alpha().topRows(wfn_b.num_alpha)
                );
                C.alpha().leftCols(num_alpha) = C_merged;
                mo_energies.alpha().topRows(num_alpha) = energies_merged;

                // merge virtual orbitals
                size_t nv_a = wfn_a.C.alpha().rows() - wfn_a.num_alpha, nv_b = wfn_b.C.alpha().rows() - wfn_b.num_alpha;
                size_t nv_ab = nv_a + nv_b;

                occ::log::debug("Merging virtual orbitals, sorted by energy");
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    wfn_a.C.alpha().rightCols(nv_a), wfn_b.C.alpha().rightCols(nv_b),
                    wfn_a.mo_energies.bottomRows(nv_a), wfn_b.mo_energies.alpha().bottomRows(nv_b));
                C.alpha().rightCols(nv_ab) = C_merged;
                mo_energies.alpha().bottomRows(nv_ab) = energies_merged;
            }
            { //beta
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    wfn_a.C.beta().leftCols(wfn_a.num_beta), wfn_b.C.beta().leftCols(wfn_b.num_beta),
                    wfn_a.mo_energies.beta().topRows(wfn_a.num_beta), wfn_b.mo_energies.beta().topRows(wfn_b.num_beta)
                );
                C.beta().leftCols(num_beta) = C_merged;
                mo_energies.beta().topRows(num_beta) = energies_merged;

                // merge virtual orbitals
                size_t nv_a = wfn_a.C.beta().rows() - wfn_a.num_beta, nv_b = wfn_b.C.beta().rows() - wfn_b.num_beta;
                size_t nv_ab = nv_a + nv_b;

                occ::log::debug("Merging virtual orbitals, sorted by energy");
                std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                    wfn_a.C.beta().rightCols(nv_a), wfn_b.C.beta().rightCols(nv_b),
                    wfn_a.mo_energies.beta().bottomRows(nv_a), wfn_b.mo_energies.beta().bottomRows(nv_b));
                C.beta().rightCols(nv_ab) = C_merged;
                mo_energies.beta().bottomRows(nv_ab) = energies_merged;
            }
        }
    }
    update_occupied_orbitals();
}


void Wavefunction::update_occupied_orbitals()
{
    if(C.size() == 0) { return; }
    if(spinorbital_kind == SpinorbitalKind::General) {
        throw std::runtime_error("Reading MOs from g09 unsupported for General spinorbitals");
    }
    else if(spinorbital_kind == SpinorbitalKind::Unrestricted) {
        C_occ = occ::qm::orb::occupied_unrestricted(C, num_alpha, num_beta);
    }
    else {
        C_occ = occ::qm::orb::occupied_restricted(C, num_alpha);
    }
}

void Wavefunction::set_molecular_orbitals(const FchkReader& fchk)
{
    size_t rows, cols;
    nbf = occ::qm::nbf(basis);

    if(spinorbital_kind == SpinorbitalKind::General) {
        throw std::runtime_error("Reading MOs from g09 unsupported for General spinorbitals");
    }
    else if(spinorbital_kind == SpinorbitalKind::Unrestricted) {
        std::tie(rows, cols) = occ::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(nbf);
        C = MatRM(rows, cols);
        mo_energies = Vec(rows);
        C.alpha() = fchk.alpha_mo_coefficients();
        C.beta() = fchk.beta_mo_coefficients();
        mo_energies.alpha() = fchk.alpha_mo_energies();
        mo_energies.beta() = fchk.beta_mo_energies();
        C.alpha() = occ::io::conversion::orb::from_gaussian(basis, C.alpha());
        C.beta() = occ::io::conversion::orb::from_gaussian(basis, C.beta());
    }
    else {
        C = fchk.alpha_mo_coefficients();
        mo_energies = fchk.alpha_mo_energies();
        C = occ::io::conversion::orb::from_gaussian(basis, C);
    }
    update_occupied_orbitals();
}

void Wavefunction::compute_density_matrix() {
    if(spinorbital_kind == SpinorbitalKind::General) {
        throw std::runtime_error("Reading MOs from g09 unsupported for General spinorbitals");
    }
    else if(spinorbital_kind == SpinorbitalKind::Unrestricted) {
        D = occ::qm::orb::density_matrix_unrestricted(C_occ, num_alpha, num_beta);
    }
    else {
        D = occ::qm::orb::density_matrix_restricted(C_occ);
    }
}

void Wavefunction::symmetric_orthonormalize_molecular_orbitals(const MatRM& overlap)
{
    if(spinorbital_kind == SpinorbitalKind::Restricted)
    {
        C = symmorthonormalize_molecular_orbitals(C, overlap, num_alpha);
    }
    else {
        C.alpha() = symmorthonormalize_molecular_orbitals(C.alpha(), overlap, num_alpha);
        C.beta() = symmorthonormalize_molecular_orbitals(C.beta(), overlap, num_beta);
    }
    update_occupied_orbitals();
}


MatRM symmetrically_orthonormalize(const MatRM& mat, const MatRM& metric)
{
    MatRM X, X_invT;
    size_t n_cond;
    double x_condition_number, condition_number;
    double threshold = 1.0 / std::numeric_limits<double>::epsilon();
    MatRM SS = mat.transpose() * metric * mat;
    std::tie(X, X_invT, n_cond, x_condition_number, condition_number) = occ::gensqrtinv(SS, true, threshold);
    return mat * X;
}

MatRM symmorthonormalize_molecular_orbitals(const MatRM& mos, const MatRM& overlap, size_t n_occ)
{
    MatRM result(mos.rows(), mos.cols());
    size_t n_virt = mos.cols() - n_occ;
    MatRM C_occ = mos.leftCols(n_occ);
    MatRM C_virt = mos.rightCols(n_virt);
    result.leftCols(n_occ) = symmetrically_orthonormalize(C_occ, overlap);
    result.rightCols(n_virt) = symmetrically_orthonormalize(C_virt, overlap);
    return result;
}


void Wavefunction::apply_transformation(const occ::Mat3& rot, const occ::Vec3& trans)
{
    apply_rotation(rot);
    apply_translation(trans);
}
void Wavefunction::apply_translation(const occ::Vec3& trans)
{
    basis.translate(trans);
    translate_atoms(atoms, trans);
}

void Wavefunction::apply_rotation(const occ::Mat3& rot)
{
    if(spinorbital_kind == SpinorbitalKind::Restricted)
    {
        MatRM rotated = rotate_molecular_orbitals(basis, rot, C);
        C.noalias() = rotated;
    }
    else {
        MatRM rotated = rotate_molecular_orbitals(basis, rot, C.alpha());
        C.alpha().noalias() = rotated;
        rotated = rotate_molecular_orbitals(basis, rot, C.beta());
        C.beta().noalias() = rotated;
    }
    basis.rotate(rot);
    rotate_atoms(atoms, rot);
    update_occupied_orbitals();
    compute_density_matrix();
}

occ::Mat3N Wavefunction::positions() const
{
    occ::Mat3N pos(3, atoms.size());
    for(size_t i = 0; i < atoms.size(); i++) {
        pos(0, i) = atoms[i].x;
        pos(1, i) = atoms[i].y;
        pos(2, i) = atoms[i].z;
    }
    return pos;
}

occ::IVec Wavefunction::atomic_numbers() const
{
    occ::IVec nums(atoms.size());
    for(size_t i = 0; i < atoms.size(); i++) {
        nums(i) = atoms[i].atomic_number;
    }
    return nums;
}

void Wavefunction::save(FchkWriter &fchk)
{
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
    for(Eigen::Index i = 0; i < atomic_prop.rows(); i++) atomic_prop(i) = static_cast<double>(occ::chem::Element(nums(i)).mass());
    fchk.set_vector("Integer atomic weights", atomic_prop.array().round().cast<int>());
    fchk.set_vector("Real atomic weights", atomic_prop);

    auto Cfchk = occ::io::conversion::orb::to_gaussian(basis, C);
    occ::MatRM Dfchk;

    std::vector<double> density_lower_triangle, spin_density_lower_triangle;

    if(spinorbital_kind == SpinorbitalKind::Unrestricted) {
        fchk.set_vector("Alpha Orbital Energies", mo_energies.alpha());
        fchk.set_vector("Alpha MO coefficients", Cfchk.alpha());
        fchk.set_vector("Beta Orbital Energies", mo_energies.beta());
        fchk.set_vector("Beta MO coefficients", Cfchk.beta());
        occ::MatRM occ_fchk = occ::qm::orb::occupied_unrestricted(Cfchk, num_alpha, num_beta);
        Dfchk = occ::qm::orb::density_matrix_unrestricted(occ_fchk, num_alpha, num_beta);

        density_lower_triangle.reserve(nbf * (nbf - 1) / 2);
        spin_density_lower_triangle.reserve(nbf * (nbf - 1) / 2);
        auto da = Dfchk.alpha();
        auto db = Dfchk.beta();
        for(Eigen::Index row = 0; row < nbf; row++) {
            for(Eigen::Index col = 0; col <= row; col++) {
                double va = da(row, col) * 2, vb = db(row, col) * 2;
                density_lower_triangle.push_back(va + vb);
                spin_density_lower_triangle.push_back(vb - va);
            }
        }
    }
    else {
        fchk.set_vector("Alpha Orbital Energies", mo_energies);
        fchk.set_vector("Alpha MO coefficients", Cfchk);
        occ::MatRM occ_fchk = occ::qm::orb::occupied_restricted(Cfchk, num_alpha);
        Dfchk = occ::qm::orb::density_matrix_restricted(occ_fchk);
        density_lower_triangle.reserve(nbf * (nbf - 1) / 2);
        for(Eigen::Index row = 0; row < nbf; row++) {
            for(Eigen::Index col = 0; col <= row; col++) {
                density_lower_triangle.push_back(Dfchk(row, col) * 2);
            }
        }
    }
    fchk.set_vector("Total SCF Density", density_lower_triangle);
    if(spin_density_lower_triangle.size() > 0) fchk.set_vector("Spin SCF Density", spin_density_lower_triangle);
    fchk.set_basis(basis);


    std::vector<int> shell2atom;
    shell2atom.reserve(basis.size());
    for(const auto& x: basis.shell2atom(atoms))
    {
        shell2atom.push_back(x + 1);
    }
    fchk.set_vector("Shell to atom map", shell2atom);
}

}