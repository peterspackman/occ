#include <fmt/core.h>
#include <occ/core/logger.h>
#include <occ/core/timings.h>
#include <occ/gto/gto.h>
#include <occ/qm/mo.h>

namespace occ::qm {

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

        Cocc = Mat::Zero(2 * n_ao, std::max(n_alpha, n_beta));
        Cocc.block(0, 0, n_ao, n_alpha) = block::a(C).leftCols(n_alpha);
        Cocc.block(n_ao, 0, n_ao, n_beta) = block::b(C).leftCols(n_beta);
        break;
    }
    case SpinorbitalKind::Restricted: {
        Eigen::SelfAdjointEigenSolver<Mat> eig_solver(ortho.transpose() *
                                                      potential * ortho);
        C = ortho * eig_solver.eigenvectors();
        energies = eig_solver.eigenvalues();
        Cocc = C.leftCols(n_alpha);
        break;
    }
    case SpinorbitalKind::General: {
        // same as restricted
        Eigen::SelfAdjointEigenSolver<Mat> eig_solver(ortho.transpose() *
                                                      potential * ortho);
        C = ortho * eig_solver.eigenvectors();
        energies = eig_solver.eigenvalues();
        Cocc = C.leftCols(n_alpha);
        break;
    }
    }
    occ::timing::stop(occ::timing::category::mo);
}

void MolecularOrbitals::update_density_matrix() {
    occ::timing::start(occ::timing::category::la);
    switch (kind) {
    case SpinorbitalKind::Restricted:
        D = Cocc * Cocc.transpose();
        break;
    case SpinorbitalKind::Unrestricted:
        block::a(D) = Cocc.block(0, 0, n_ao, n_alpha) *
                      Cocc.block(0, 0, n_ao, n_alpha).transpose();
        block::b(D) = Cocc.block(n_ao, 0, n_ao, n_beta) *
                      Cocc.block(n_ao, 0, n_ao, n_beta).transpose();
        D *= 0.5;
        break;
    case SpinorbitalKind::General:
        D = (Cocc * Cocc.transpose()) * 0.5;
        break;
    }
    occ::timing::stop(occ::timing::category::la);
}

void MolecularOrbitals::rotate(const AOBasis &basis, const Mat3 &rotation) {

    const auto shell2bf = basis.first_bf();
    occ::log::debug("Rotating MO coefficients");
    for (size_t s = 0; s < basis.size(); s++) {
        const auto &shell = basis[s];
        size_t bf_first = shell2bf[s];
        size_t shell_size = shell.size();
        int l = shell.l;
        bool pure = shell.is_pure();
        Mat rot;
        switch (l) {
        case 0:
            rot = occ::gto::cartesian_gaussian_rotation_matrix<0>(rotation);
            break;
        case 1:
            rot = occ::gto::cartesian_gaussian_rotation_matrix<1>(rotation);
            break;
        case 2:
            rot = occ::gto::cartesian_gaussian_rotation_matrix<2>(rotation);
            break;
        case 3:
            rot = occ::gto::cartesian_gaussian_rotation_matrix<3>(rotation);
            break;
        case 4:
            rot = occ::gto::cartesian_gaussian_rotation_matrix<4>(rotation);
            break;
        default:
            throw std::runtime_error(
                "MO rotation not implemented for angular momentum > 4");
        }
        if (pure) {
            Mat c = occ::gto::cartesian_to_spherical_transformation_matrix(l);
            Mat cinv =
                occ::gto::spherical_to_cartesian_transformation_matrix(l);
            rot = c * rot * cinv;
        }
        if (kind == SpinorbitalKind::Restricted) {
            occ::log::debug("Restricted MO rotation");
            C.block(bf_first, 0, shell_size, C.cols()) =
                rot * C.block(bf_first, 0, shell_size, C.cols());
        } else {
            occ::log::debug("Unrestricted MO rotation");
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

} // namespace occ::qm
