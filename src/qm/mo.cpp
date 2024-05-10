#include <fmt/core.h>
#include <occ/core/gensqrtinv.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/gto/gto.h>
#include <occ/gto/rotation.h>
#include <occ/qm/expectation.h>
#include <occ/qm/merge.h>
#include <occ/qm/mo.h>
#include <occ/qm/orb.h>

namespace occ::qm {

inline SpinorbitalKind compatible_kind(SpinorbitalKind a, SpinorbitalKind b) {
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto G = SpinorbitalKind::General;

    /*
     * Table should look something like this:
     *
     * 	 r u g
     * r R U G
     * u U U G
     * g G G G
     */

    if (a == b) // diagonal
        return a;
    if (a == R) // R U G row
        return b;
    if (b == R) // R U G col
        return a;
    if (a == G)
        return a; // G G G row
    return b;     // G G G col (should cover all cases
}

MolecularOrbitals::MolecularOrbitals(const MolecularOrbitals &mo1,
                                     const MolecularOrbitals &mo2)
    : kind(compatible_kind(mo1.kind, mo2.kind)),
      n_alpha(mo1.n_alpha + mo2.n_alpha),
      n_beta(mo1.n_beta + mo2.n_beta),
      n_ao(mo2.n_ao + mo1.n_ao) {
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto U = SpinorbitalKind::Unrestricted;

    auto [rows, cols] = matrix_dimensions(kind, n_ao);
    C = Mat(rows, cols);
    energies = Vec(rows);
    // temporaries for merging orbitals
    occ::log::debug("Merging occupied orbitals, sorted by energy");

    auto merge_mo_block =
        [](Eigen::Ref<Mat> dest, Eigen::Ref<Vec> dest_energies,
           Eigen::Ref<const Mat> coeffs1, Eigen::Ref<const Mat> coeffs2,
           Eigen::Ref<const Vec> energies1, Eigen::Ref<const Vec> energies2) {
            auto [C_merged, energies_merged] = merge_molecular_orbitals(
                coeffs1, coeffs2, energies1, energies2);
            occ::log::debug("MO occ merged shape {} {}", C_merged.rows(),
                            C_merged.cols());
            dest = C_merged;
            dest_energies = energies_merged;
        };

    // TODO refactor
    if (kind == R) {
        occ::log::debug(
            "Merging spin-restricted occupied orbitals, sorted by energy");
        merge_mo_block(C.leftCols(n_alpha), energies.topRows(n_alpha),
                       mo1.C.leftCols(mo1.n_alpha), mo2.C.leftCols(mo2.n_alpha),
                       mo1.energies.topRows(mo1.n_alpha),
                       mo2.energies.topRows(mo2.n_alpha));

        // merge virtual orbitals
        size_t nv_a = mo1.n_ao - mo1.n_alpha;
        size_t nv_b = mo2.n_ao - mo2.n_alpha;
        size_t nv_ab = nv_a + nv_b;
        if (nv_ab > 0) {
            occ::log::debug(
                "Merging spin-restricted virtual orbitals, sorted by energy");

            merge_mo_block(C.rightCols(nv_ab), energies.bottomRows(nv_ab),
                           mo1.C.rightCols(nv_a), mo2.C.rightCols(nv_b),
                           mo1.energies.bottomRows(nv_a),
                           mo2.energies.bottomRows(nv_b));
        }
    } else if (kind == U) {
        MolecularOrbitals mo1_u = mo1.as_kind(U);
        MolecularOrbitals mo2_u = mo2.as_kind(U);

        Eigen::Ref<Mat> alpha_coeffs = block::a(C);
        Eigen::Ref<Mat> beta_coeffs = block::b(C);
        Eigen::Ref<Vec> alpha_energies = block::a(energies);
        Eigen::Ref<Vec> beta_energies = block::b(energies);

        Eigen::Ref<const Mat> alpha_mo1 = block::a(mo1_u.C);
        Eigen::Ref<const Mat> alpha_mo1_energies = block::a(mo1_u.C);
        Eigen::Ref<const Mat> beta_mo1 = block::b(mo1_u.C);
        Eigen::Ref<const Mat> beta_mo1_energies = block::b(mo1_u.C);
        size_t mo1_na = mo1_u.n_alpha;
        size_t mo1_nb = mo1_u.n_beta;

        Eigen::Ref<const Mat> alpha_mo2 = block::a(mo2_u.C);
        Eigen::Ref<const Mat> alpha_mo2_energies = block::a(mo2_u.C);
        Eigen::Ref<const Mat> beta_mo2 = block::b(mo2_u.C);
        Eigen::Ref<const Mat> beta_mo2_energies = block::b(mo2_u.C);
        size_t mo2_na = mo2_u.n_alpha;
        size_t mo2_nb = mo2_u.n_beta;

        occ::log::debug("Merging alpha spin-unrestricted occupied orbitals, "
                        "sorted by energy");
        merge_mo_block(alpha_coeffs.leftCols(n_alpha),
                       alpha_energies.topRows(n_alpha),
                       alpha_mo1.leftCols(mo1_na), alpha_mo2.leftCols(mo2_na),
                       alpha_mo1_energies.topRows(mo1_na),
                       alpha_mo2_energies.topRows(mo2_na));
        // merge virtual orbitals
        size_t nv_a = mo1_u.n_ao - mo1_na;
        size_t nv_b = mo2_u.n_ao - mo2_na;
        size_t nv_ab = nv_a + nv_b;

        if (nv_ab > 0) {
            occ::log::debug("Merging alpha spin-unrestricted virtual orbitals, "
                            "sorted by energy");

            merge_mo_block(alpha_coeffs.rightCols(nv_ab),
                           alpha_energies.bottomRows(nv_ab),
                           alpha_mo1.rightCols(nv_a), alpha_mo2.rightCols(nv_b),
                           alpha_mo1_energies.bottomRows(nv_a),
                           alpha_mo2_energies.bottomRows(nv_b));
        }

        occ::log::debug("Merging beta spin-unrestricted occupied orbitals, "
                        "sorted by energy");
        merge_mo_block(beta_coeffs.leftCols(n_beta),
                       beta_energies.topRows(n_beta), beta_mo1.leftCols(mo1_nb),
                       beta_mo2.leftCols(mo2_nb),
                       beta_mo1_energies.topRows(mo1_nb),
                       beta_mo2_energies.topRows(mo2_nb));

        // merge virtual orbitals
        nv_a = mo1.n_ao - mo1_nb;
        nv_b = mo2.n_ao - mo2_nb;
        nv_ab = nv_a + nv_b;

        if (nv_ab > 0) {
            occ::log::debug("Merging beta spin-unrestricted virtual orbitals, "
                            "sorted by energy");

            merge_mo_block(beta_coeffs.rightCols(nv_ab),
                           beta_energies.bottomRows(nv_ab),
                           beta_mo1.rightCols(nv_a), beta_mo2.rightCols(nv_b),
                           beta_mo1_energies.bottomRows(nv_a),
                           beta_mo2_energies.bottomRows(nv_b));
        }
    } else {
        throw std::runtime_error(
            "Merging general spinorbitals not implemented");
    }
    update_occupied_orbitals();
    update_density_matrix();
}

void MolecularOrbitals::update(const Mat &ortho, const Mat &potential) {
    // solve F C = e S C by (conditioned) transformation to F' C' = e C',
    // where
    // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
    occ::timing::start(occ::timing::category::mo);
    switch (kind) {
    case SpinorbitalKind::Unrestricted: {
        Eigen::SelfAdjointEigenSolver<Mat> alpha_eig_solver(
            ortho.transpose() * block::a(potential) * ortho);
        Eigen::SelfAdjointEigenSolver<Mat> beta_eig_solver(
            ortho.transpose() * block::b(potential) * ortho);

        block::a(C) = ortho * alpha_eig_solver.eigenvectors();
        block::b(C) = ortho * beta_eig_solver.eigenvectors();

        block::a(energies) = alpha_eig_solver.eigenvalues();
        block::b(energies) = beta_eig_solver.eigenvalues();
        break;
    }
    case SpinorbitalKind::Restricted: {
	Mat tmp = ortho.transpose() * potential * ortho;
        Eigen::SelfAdjointEigenSolver<Mat> eig_solver(tmp);
        C = ortho * eig_solver.eigenvectors();
        energies = eig_solver.eigenvalues();
        break;
    }
    case SpinorbitalKind::General: {
        // same as restricted
        Eigen::SelfAdjointEigenSolver<Mat> eig_solver(ortho.transpose() *
                                                      potential * ortho);
        C = ortho * eig_solver.eigenvectors();
        energies = eig_solver.eigenvalues();
        break;
    }
    }
    update_occupied_orbitals();
    smearing.smear_orbitals(*this);
    update_density_matrix();

    occ::timing::stop(occ::timing::category::mo);
}

void MolecularOrbitals::update_occupied_orbitals() {
    if (C.size() == 0) {
        return;
    }
    occ::log::debug("Updating occupied orbitals, n_a = {}, n_b = {}", n_alpha, n_beta);
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;

    switch (kind) {
    case U:
        Cocc = occ::qm::orb::occupied_unrestricted(C, n_alpha, n_beta);
	occupation = Vec::Zero(n_ao * 2);
	occupation.topRows(n_alpha).setConstant(1.0);
	occupation.block(n_ao, 0, n_beta, 1).setConstant(1.0);
        break;
    case G:
        Cocc = occ::qm::orb::occupied_restricted(C, n_alpha);
	occupation = Vec::Zero(n_ao * 2);
	occupation.topRows(n_alpha).setConstant(1.0);
	occupation.block(n_ao, 0, n_beta, 1).setConstant(1.0);
        break;
    default:
        Cocc = occ::qm::orb::occupied_restricted(C, n_alpha);
	occupation = Vec::Zero(n_ao);
	occupation.topRows(n_alpha).setConstant(1.0);
        break;
    }
}

void MolecularOrbitals::update_occupied_orbitals_fractional() {
    if (C.size() == 0) {
        return;
    }
    occ::log::debug("Updating occupied orbitals, n_a = {}, n_b = {}", n_alpha, n_beta);
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;

    switch (kind) {
    case U:
        Cocc = occ::qm::orb::occupied_unrestricted_fractional(C, occupation);
        break;
    case G:
        Cocc = occ::qm::orb::occupied_restricted_fractional(C, occupation);
        break;
    default:
        Cocc = occ::qm::orb::occupied_restricted_fractional(C, occupation);
        break;
    }
}

void MolecularOrbitals::update_density_matrix() {
    occ::timing::start(occ::timing::category::la);
    switch (kind) {
    case SpinorbitalKind::Restricted:
        D = orb::density_matrix_restricted(Cocc);
        break;
    case SpinorbitalKind::Unrestricted:
        D = orb::density_matrix_unrestricted(Cocc, n_alpha, n_beta);
        break;
    case SpinorbitalKind::General:
        D = orb::density_matrix_general(Cocc);
        break;
    }
    occ::timing::stop(occ::timing::category::la);
}


Mat MolecularOrbitals::energy_weighted_density_matrix() const {
    occ::timing::start(occ::timing::category::la);
    switch (kind) {
    case SpinorbitalKind::Unrestricted:
	return orb::weighted_density_matrix_unrestricted(C, energies.array() * occupation.array());
	break;
    default:
	return orb::weighted_density_matrix_restricted(C, energies.array() * occupation.array());
        break;
    }
    occ::timing::stop(occ::timing::category::la);
}

void MolecularOrbitals::rotate(const AOBasis &basis, const Mat3 &rotation) {

    const auto shell2bf = basis.first_bf();
    occ::log::debug("Rotating {} MO coefficients, l max = {}",
                    basis.is_pure() ? "Spherical" : "Cartesian", basis.l_max());
    std::vector<Mat> rotation_matrices;
    if (basis.is_pure()) {
        rotation_matrices = occ::gto::spherical_gaussian_rotation_matrices(
            basis.l_max(), rotation);
    } else {
        rotation_matrices = occ::gto::cartesian_gaussian_rotation_matrices(
            basis.l_max(), rotation);
    }

    for (size_t s = 0; s < basis.size(); s++) {
        const auto &shell = basis[s];
        size_t bf_first = shell2bf[s];
        size_t shell_size = shell.size();
        int l = shell.l;
        Mat rot = rotation_matrices[l];
        if (kind == SpinorbitalKind::Restricted) {
            occ::log::trace("Restricted MO rotation");
            C.block(bf_first, 0, shell_size, C.cols()) =
                rot * C.block(bf_first, 0, shell_size, C.cols());
        } else {
            occ::log::trace("Unrestricted MO rotation");
            auto ca = block::a(C);
            auto cb = block::b(C);
            ca.block(bf_first, 0, shell_size, ca.cols()) =
                rot * ca.block(bf_first, 0, shell_size, ca.cols());
            cb.block(bf_first, 0, shell_size, cb.cols()) =
                rot * cb.block(bf_first, 0, shell_size, cb.cols());
        }
    }
}

void MolecularOrbitals::print() const {
    if (kind == SpinorbitalKind::Unrestricted) {
        int n_mo = energies.size() / 2;
        fmt::print("\nmolecular orbital energies\n");
        fmt::print("{0:3s}   {1:3s} {2:>16s}  {1:3s} {2:>16s}\n", "idx", "occ",
                   "energy");
        for (int i = 0; i < n_mo; i++) {
            auto s_a = i < n_alpha ? "a" : " ";
            auto s_b = i < n_beta ? "b" : " ";
            fmt::print("{:3d}   {:^3s} {:16.12f}  {:^3s} {:16.12f}\n", i, s_a,
                       energies(i), s_b, energies(n_ao + i));
        }
    } else {
        int n_mo = energies.size();
        fmt::print("\nmolecular orbital energies\n");
        fmt::print("{0:3s}   {1:3s} {2:>16s} {3:>16s}\n", "idx", "occ",
                   "energy", "norm");
        for (int i = 0; i < n_mo; i++) {
            auto s = i < n_alpha ? "ab" : " ";
            fmt::print("{:3d}   {:^3s} {:16.12f} {:16.12f}\n", i, s,
                       energies(i), C.col(i).sum());
        }
    }
}

void MolecularOrbitals::incorporate_norm(Eigen::Ref<const Vec> norms) {
    if (kind == SpinorbitalKind::Restricted) {
        for (int ao = 0; ao < norms.rows(); ao++) {
            fmt::print("Norm:{}\n", norms(ao));
            C.row(ao).array() /= norms(ao);
            Cocc.row(ao).array() /= norms(ao);
        }
    }
    update_density_matrix();
}

void MolecularOrbitals::to_cartesian(const AOBasis &bspure,
                                     const AOBasis &bscart) {
    const auto shell2bf_pure = bspure.first_bf();
    const auto shell2bf_cart = bscart.first_bf();
    occ::log::debug("Converting MO from spherical to Cartesian");

    // fail early if we already have the right number of AOs
    if (bspure.nbf() == bscart.nbf())
        return;

    Mat Cnew;
    if (kind == SpinorbitalKind::Restricted) {
        Cnew = Mat::Zero(bscart.nbf(), bscart.nbf());
    } else {
        auto [nrows, ncols] =
            occ::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(
                bscart.nbf());
        Cnew = Mat::Zero(nrows, ncols);
    }

    for (size_t s = 0; s < bspure.size(); s++) {
        const auto &shell_pure = bspure[s];
        const auto &shell_cart = bscart[s];
        size_t bf_first_pure = shell2bf_pure[s];
        size_t bf_first_cart = shell2bf_cart[s];
        size_t shell_size_pure = shell_pure.size();
        size_t shell_size_cart = shell_cart.size();
        int l = shell_pure.l;
        Mat T = occ::gto::spherical_to_cartesian_transformation_matrix(l);

        if (kind == SpinorbitalKind::Restricted) {
            occ::log::trace("Restricted MO transform spherical->Cartesian");
            Cnew.block(bf_first_cart, 0, shell_size_cart, Cnew.cols()) =
                T * C.block(bf_first_pure, 0, shell_size_pure, C.cols());
        } else {
            occ::log::trace("Unrestricted MO transform spherical->Cartesian");
            auto ca = block::a(C);
            auto cb = block::b(C);
            auto ca_new = block::a(Cnew);
            auto cb_new = block::b(Cnew);
            ca_new.block(bf_first_cart, 0, shell_size_cart, ca_new.cols()) =
                T * ca.block(bf_first_pure, 0, shell_size_pure, ca.cols());
            cb_new.block(bf_first_cart, 0, shell_size_cart, cb.cols()) =
                T * cb.block(bf_first_pure, 0, shell_size_pure, cb.cols());
        }
    }
    n_ao = bscart.nbf();
    C = Cnew;
    update_occupied_orbitals();
    update_density_matrix();
}

void MolecularOrbitals::to_spherical(const AOBasis &bscart,
                                     const AOBasis &bspure) {
    const auto shell2bf_pure = bspure.first_bf();
    const auto shell2bf_cart = bscart.first_bf();
    occ::log::debug("Converting MO from Cartesian to spherical (lossy)");

    // fail early if we already have the right number of AOs
    if (bspure.nbf() == bscart.nbf())
        return;

    Mat Cnew;
    if (kind == SpinorbitalKind::Restricted) {
        Cnew = Mat::Zero(bspure.nbf(), bspure.nbf());
    } else {
        auto [nrows, ncols] =
            occ::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(
                bspure.nbf());
        Cnew = Mat::Zero(nrows, ncols);
    }

    for (size_t s = 0; s < bspure.size(); s++) {
        const auto &shell_pure = bspure[s];
        const auto &shell_cart = bscart[s];
        size_t bf_first_pure = shell2bf_pure[s];
        size_t bf_first_cart = shell2bf_cart[s];
        size_t shell_size_pure = shell_pure.size();
        size_t shell_size_cart = shell_cart.size();
        int l = shell_pure.l;
        Mat T = occ::gto::cartesian_to_spherical_transformation_matrix(l);

        if (kind == SpinorbitalKind::Restricted) {
            occ::log::trace("Restricted MO transform Cartesian->spherical");
            Cnew.block(bf_first_pure, 0, shell_size_pure, Cnew.cols()) =
                T * C.block(bf_first_cart, 0, shell_size_cart, C.cols());
        } else {
            occ::log::trace("Unrestricted MO transform Cartesian->spherical");
            auto ca = block::a(C);
            auto cb = block::b(C);
            auto ca_new = block::a(Cnew);
            auto cb_new = block::b(Cnew);
            ca_new.block(bf_first_pure, 0, shell_size_pure, ca_new.cols()) =
                T * ca.block(bf_first_cart, 0, shell_size_cart, ca.cols());
            cb_new.block(bf_first_pure, 0, shell_size_pure, cb.cols()) =
                T * cb.block(bf_first_cart, 0, shell_size_cart, cb.cols());
        }
    }

    update_occupied_orbitals();
    update_density_matrix();
}

double MolecularOrbitals::expectation_value(const Mat &op) const {
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;
    switch (kind) {
    default:
        return occ::qm::expectation<R>(D, op);
    case U:
        return occ::qm::expectation<U>(D, op);
    case G:
        return occ::qm::expectation<G>(D, op);
    }
}

Mat symmetrically_orthonormalize_matrix(const Mat &mat, const Mat &metric) {
    double threshold = 1.0 / std::numeric_limits<double>::epsilon();
    Mat SS = mat.transpose() * metric * mat;
    auto g = occ::core::gensqrtinv(SS, true, threshold);
    return mat * g.result;
}

MolecularOrbitals MolecularOrbitals::as_kind(SpinorbitalKind new_kind) const {
    // do nothing if we don't have to
    if (new_kind == kind)
        return *this;
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;
    MolecularOrbitals result = *this;

    auto [rows, cols] = matrix_dimensions(new_kind, n_ao);
    result.C = Mat::Zero(rows, cols);
    result.energies = Vec::Zero(rows, cols);

    switch (new_kind) {
    default: {
        switch (kind) {
        case U: {
            result.C = 0.5 * (block::a(C) + block::b(C));
            result.energies = 0.5 * (block::a(energies) + block::b(energies));
            break;
        }
        case G: {
            // seems sensible
            result.C = 0.5 * (block::aa(C) + block::bb(C));
            result.energies = 0.5 * (block::aa(energies) + block::bb(energies));
            break;
        }
        default: // impossible
            throw std::runtime_error(
                "impossible state in MolecularOrbitals::as_kind");
        }
    }
    case U: {
        switch (kind) {
        case R: {
            block::a(result.C) = C;
            block::b(result.C) = C;
            block::a(result.energies) = energies;
            block::b(result.energies) = energies;
            break;
        }
        case G: {
            block::a(result.C) = block::aa(C);
            block::b(result.C) = block::bb(C);
            block::a(result.energies) = block::aa(energies);
            block::b(result.energies) = block::bb(energies);
            break;
        }
        default:
            throw std::runtime_error(
                "impossible state in MolecularOrbitals::as_kind");
        }
    }
    case G: {
        switch (kind) {
        case R: {
            block::aa(result.C) = C;
            block::ab(result.C) = C;
            block::ba(result.C) = C;
            block::bb(result.C) = C;
            block::aa(result.energies) = energies;
            block::ab(result.energies) = energies;
            block::ba(result.energies) = energies;
            block::bb(result.energies) = energies;
            break;
        }
        case U: {
            block::aa(result.C) = block::a(C);
            block::ab(result.C) = 0.5 * (block::a(C) + block::b(C));
            block::ba(result.C) = 0.5 * (block::a(C) + block::b(C));
            block::bb(result.C) = block::b(C);
            block::aa(result.energies) = block::a(energies);
            block::ab(result.energies) =
                0.5 * (block::a(energies) + block::b(energies));
            block::ba(result.energies) =
                0.5 * (block::a(energies) + block::b(energies));
            block::bb(result.energies) = block::b(energies);
            break;
        }
        default:
            throw std::runtime_error(
                "impossible state in MolecularOrbitals::as_kind");
        }
    }
    }
    result.update_occupied_orbitals();
    result.update_density_matrix();
    return result;
}

MolecularOrbitals MolecularOrbitals::symmetrically_orthonormalized(
    const Mat &overlap_matrix) const {
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;

    MolecularOrbitals result = *this;
    size_t n_occ = result.Cocc.cols();
    size_t n_virt = result.C.cols() - n_occ;

    switch (kind) {
    default: {
        result.C.leftCols(n_occ) =
            symmetrically_orthonormalize_matrix(result.Cocc, overlap_matrix);

        if (n_virt > 0) {
            Mat Cvirt = C.rightCols(n_virt);
            result.C.rightCols(n_virt) =
                symmetrically_orthonormalize_matrix(Cvirt, overlap_matrix);
        }
        break;
    }
    case U: {
        // alpha spin
        block::a(result.C).leftCols(n_occ) =
            symmetrically_orthonormalize_matrix(block::a(result.Cocc),
                                                overlap_matrix);
        // beta spin
        block::b(result.C).leftCols(n_occ) =
            symmetrically_orthonormalize_matrix(block::b(result.Cocc),
                                                overlap_matrix);

        if (n_virt > 0) {
            Mat Cvirt = result.C.rightCols(n_virt);

            block::a(result.C).rightCols(n_virt) =
                symmetrically_orthonormalize_matrix(block::a(Cvirt),
                                                    overlap_matrix);
            block::b(result.C).rightCols(n_virt) =
                symmetrically_orthonormalize_matrix(block::b(Cvirt),
                                                    overlap_matrix);
        }
        break;
    }
    case G: {
        throw std::runtime_error("Symmetric orthonormalization not implemented "
                                 "for General spinorbitals");
        break;
    }
    }
    result.update_occupied_orbitals();
    result.update_density_matrix();
    return result;
}

Mat MolecularOrbitals::density_matrix_single_mo(int mo_index) const {
    Mat Ctmp = Mat::Zero(C.rows(), C.cols());

    Mat result;

    Ctmp.col(mo_index) = C.col(mo_index);
    occ::timing::start(occ::timing::category::la);
    switch (kind) {
    case SpinorbitalKind::Restricted:
        result = orb::density_matrix_restricted(Ctmp);
        break;
    case SpinorbitalKind::Unrestricted:
        result = orb::density_matrix_unrestricted(Ctmp, Ctmp.cols(), Ctmp.cols());
        break;
    case SpinorbitalKind::General:
        result = orb::density_matrix_general(Ctmp);
        break;
    }
    occ::timing::stop(occ::timing::category::la);
    return result;
}

} // namespace occ::qm
