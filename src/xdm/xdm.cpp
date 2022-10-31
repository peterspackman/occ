#include <fmt/core.h>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>
#include <occ/xdm/becke_hole.h>
#include <occ/xdm/xdm.h>

using occ::Mat;
using occ::Mat3N;
using occ::Vec;
using occ::core::Element;

namespace occ::xdm {

double xdm_polarizability(int n, double v, double vfree) {
    return v * Element(n).polarizability() / vfree;
}

std::pair<double, Mat3N>
xdm_dispersion_energy(const std::vector<occ::core::Atom> &atoms,
                      const Mat &moments, const Vec &volume,
                      const Vec &volume_free, double alpha1 = 1.0,
                      double alpha2 = 1.0) {
    const size_t num_atoms = atoms.size();
    using occ::Vec3;
    Vec polarizabilities(num_atoms);
    for (int i = 0; i < num_atoms; i++) {
        polarizabilities(i) = xdm_polarizability(atoms[i].atomic_number,
                                                 volume(i), volume_free(i));
    }
    occ::log::debug("Volume\n{}\n", volume);
    occ::log::debug("Volume Free\n{}\n", volume_free);
    occ::log::debug("Polarizibility: \n{}\n", polarizabilities);
    Mat3N forces = Mat3N::Zero(3, num_atoms);
    double edisp = 0.0;
    for (int i = 0; i < num_atoms; i++) {
        Vec3 pi = {atoms[i].x, atoms[i].y, atoms[i].z};
        double pol_i = polarizabilities(i);
        for (int j = i; j < num_atoms; j++) {
            Vec3 pj = {atoms[j].x, atoms[j].y, atoms[j].z};
            Vec3 v_ij = pj - pi;
            double pol_j = polarizabilities(j);
            double factor =
                pol_i * pol_j / (moments(0, i) * pol_j + moments(0, j) * pol_i);
            double rij = v_ij.norm();

            double rij2 = rij * rij;
            double rij4 = rij2 * rij2;
            double rij6 = rij4 * rij2;
            double rij8 = rij4 * rij4;
            double rij10 = rij4 * rij6;

            double c6 = factor * moments(0, i) * moments(0, j);
            double c8 =
                1.5 * factor *
                (moments(0, i) * moments(1, j) + moments(1, i) * moments(0, j));
            double c10 = 2.0 * factor *
                             (moments(0, i) * moments(2, j) +
                              moments(2, i) * moments(0, j)) +
                         4.2 * factor * moments(1, i) * moments(1, j);
            double rc = (std::sqrt(c8 / c6) + std::sqrt(std::sqrt(c10 / c6)) +
                         std::sqrt(c10 / c8)) /
                        3;
            double rvdw = alpha1 * rc + alpha2 * occ::units::ANGSTROM_TO_BOHR;
            double rvdw2 = rvdw * rvdw;
            double rvdw4 = rvdw2 * rvdw2;
            double rvdw6 = rvdw4 * rvdw2;
            double rvdw8 = rvdw4 * rvdw4;
            double rvdw10 = rvdw4 * rvdw6;

            occ::log::debug("{:>3d} {:>3d} {:12.6f} {:12.6f} {:12.6f} {:12.6f} "
                            "{:12.6f} {:12.6f}\n",
                            i, j, rij, c6, c8, c10, rc, rvdw);
            if (rij < 1e-15)
                continue;
            edisp -= c6 / (rvdw6 + rij6) + c8 / (rvdw8 + rij8) +
                     c10 / (rvdw10 + rij10);

            double c6_com = 6.0 * c6 * rij4 / ((rvdw6 + rij6) * (rvdw6 + rij6));
            double c8_com = 8.0 * c8 * rij6 / ((rvdw8 + rij8) * (rvdw8 + rij8));
            double c10_com =
                10.0 * c10 * rij8 / ((rvdw10 + rij10) * (rvdw10 + rij10));
            forces.col(i) += (c6_com + c8_com + c10_com) * v_ij;
            forces.col(j) -= (c6_com + c8_com + c10_com) * v_ij;
        }
    }
    return {edisp, forces};
}

namespace impl {

void xdm_moment_kernel_restricted(
    Eigen::Ref<const Mat> r, Eigen::Ref<const Mat> rho,
    Eigen::Ref<const Vec> weights, Eigen::Ref<const Mat> rho_pro,
    Eigen::Ref<Vec> hirshfeld_charges, Eigen::Ref<Mat> volume,
    Eigen::Ref<Mat> moments, double &num_electrons,
    double &num_electrons_promol) {

    for (int j = 0; j < rho_pro.rows(); j++) {
        double protot = rho_pro.row(j).sum();
        if (protot < 1e-30)
            continue;
        // now it holds the weight function
        occ::RowVec hirshfeld_weights = rho_pro.row(j).array() / protot;
        hirshfeld_charges -=
            hirshfeld_weights.row(j).transpose() * 2 * rho(j, 0) * weights(j);
        num_electrons_promol += protot * weights(j);
        double lapl = rho(j, 4);
        double tau = 2 * rho(j, 5);
        double sigma = rho(j, 1) * rho(j, 1) + rho(j, 2) * rho(j, 2) +
                       rho(j, 3) * rho(j, 3);
        double dsigs = tau - 0.25 * sigma / std::max(rho(j, 0), 1e-30);
        double q = (lapl - 2 * dsigs) / 6.0;

        double bhole = occ::xdm::becke_hole_br89(rho(j, 0), q, 1.0);
        const auto &rja = r.row(j).array();
        occ::RowVec r_sub_b =
            (rja - bhole)
                .unaryExpr([](double x) { return std::max(x, 0.0); })
                .transpose();
        moments.row(0).array() += 2 * hirshfeld_weights.array() *
                                  (rja - r_sub_b.array()).pow(2) * rho(j, 0) *
                                  weights(j);
        moments.row(1).array() += 2 * hirshfeld_weights.array() *
                                  (rja.pow(2) - r_sub_b.array().pow(2)).pow(2) *
                                  rho(j, 0) * weights(j);
        moments.row(2).array() += 2 * hirshfeld_weights.array() *
                                  (rja.pow(3) - r_sub_b.array().pow(3)).pow(2) *
                                  rho(j, 0) * weights(j);
        num_electrons += 2 * rho(j, 0) * weights(j);
        volume.array() += (hirshfeld_weights.array() * r.row(j).array() *
                           r.row(j).array() * r.row(j).array())
                              .transpose()
                              .array() *
                          2 * rho(j, 0) * weights(j);
    }
}

void xdm_moment_kernel_unrestricted(
    Eigen::Ref<const Mat> r, Eigen::Ref<const Mat> rho,
    Eigen::Ref<const Vec> weights, Eigen::Ref<const Mat> rho_pro,
    Eigen::Ref<Vec> hirshfeld_charges, Eigen::Ref<Mat> volume,
    Eigen::Ref<Mat> moments, double &num_electrons,
    double &num_electrons_promol) {

    const auto &rho_a = occ::qm::block::a(rho);
    const auto &rho_b = occ::qm::block::b(rho);

    for (int j = 0; j < rho_pro.rows(); j++) {
        double protot = rho_pro.row(j).sum();
        if (protot < 1e-30)
            continue;
        // now it holds the weight function
        occ::RowVec hirshfeld_weights = rho_pro.row(j).array() / protot;
        hirshfeld_charges -=
            hirshfeld_weights.row(j).transpose() * 2 * rho(j, 0) * weights(j);
        num_electrons_promol += protot * weights(j);
        double lapl_a = rho_a(j, 4);
        double tau_a = 2 * rho_a(j, 5);
        double sigma_a = rho_a(j, 1) * rho_a(j, 1) + rho_a(j, 2) * rho_a(j, 2) +
                         rho_a(j, 3) * rho_a(j, 3);
        double dsigs_a = tau_a - 0.25 * sigma_a / std::max(rho_a(j, 0), 1e-30);
        double q_a = (lapl_a - 2 * dsigs_a) / 6.0;
        double bhole_a = occ::xdm::becke_hole_br89(rho_a(j, 0), q_a, 1.0);

        double lapl_b = rho_b(j, 4);
        double tau_b = 2 * rho_b(j, 5);
        double sigma_b = rho_b(j, 1) * rho_b(j, 1) + rho_b(j, 2) * rho_b(j, 2) +
                         rho_b(j, 3) * rho_b(j, 3);
        double dsigs_b = tau_b - 0.25 * sigma_b / std::max(rho_b(j, 0), 1e-30);
        double q_b = (lapl_b - 2 * dsigs_b) / 6.0;
        double bhole_b = occ::xdm::becke_hole_br89(rho_b(j, 0), q_b, 1.0);

        const auto &rja = r.row(j).array();
        occ::RowVec r_sub_ba =
            (rja - bhole_a)
                .unaryExpr([](double x) { return std::max(x, 0.0); })
                .transpose();
        occ::RowVec r_sub_bb =
            (rja - bhole_b)
                .unaryExpr([](double x) { return std::max(x, 0.0); })
                .transpose();

        moments.row(0).array() +=
            hirshfeld_weights.array() * weights(j) *
            ((rja - r_sub_ba.array()).pow(2) * rho_a(j, 0) +
             (rja - r_sub_bb.array()).pow(2) * rho_b(j, 0));

        moments.row(1).array() +=
            hirshfeld_weights.array() *
            (rja.pow(2) - r_sub_ba.array().pow(2)).pow(2) * rho_a(j, 0) *
            weights(j);
        moments.row(1).array() +=
            hirshfeld_weights.array() *
            (rja.pow(2) - r_sub_bb.array().pow(2)).pow(2) * rho_b(j, 0) *
            weights(j);

        moments.row(2).array() +=
            hirshfeld_weights.array() *
            (rja.pow(3) - r_sub_ba.array().pow(3)).pow(2) * rho_a(j, 0) *
            weights(j);
        moments.row(2).array() +=
            hirshfeld_weights.array() *
            (rja.pow(3) - r_sub_bb.array().pow(3)).pow(2) * rho_b(j, 0) *
            weights(j);

        num_electrons += (rho_a(j, 0) + rho_b(j, 0)) * weights(j);
        volume.array() += (hirshfeld_weights.array() * r.row(j).array() *
                           r.row(j).array() * r.row(j).array())
                              .transpose()
                              .array() *
                          (rho_a(j, 0) + rho_b(j, 0)) * weights(j);
    }
}

} // namespace impl

struct XDMResult {
    double energy{0.0};
};

XDM::XDM(const occ::qm::AOBasis &basis) : m_basis(basis), m_grid(basis) {
    for (int i = 0; i < basis.atoms().size(); i++) {
        m_atom_grids.push_back(m_grid.generate_partitioned_atom_grid(i));
    }
    size_t num_grid_points = std::accumulate(
        m_atom_grids.begin(), m_atom_grids.end(), 0.0,
        [&](double tot, const auto &grid) { return tot + grid.points.cols(); });
    m_slater_basis = occ::slater::slaterbasis_for_atoms(m_basis.atoms());
}

double XDM::energy(const occ::qm::MolecularOrbitals &mo) {
    populate_moments(mo);
    occ::log::debug("moments\n{}\n", m_moments);

    std::tie(m_energy, m_forces) = xdm_dispersion_energy(
        m_basis.atoms(), m_moments, m_volume, m_volume_free);
    return m_energy;
}

const Mat3N &XDM::forces(const occ::qm::MolecularOrbitals &mo) {
    populate_moments(mo);

    std::tie(m_energy, m_forces) = xdm_dispersion_energy(
        m_basis.atoms(), m_moments, m_volume, m_volume_free);
    return m_forces;
}

void XDM::populate_moments(const occ::qm::MolecularOrbitals &mo) {
    if (m_density_matrix.size() != 0 &&
        occ::util::all_close(mo.D, m_density_matrix)) {
        return;
    }
    m_density_matrix = mo.D;

    occ::gto::GTOValues gto_vals;
    const auto &atoms = m_basis.atoms();
    const size_t num_atoms = atoms.size();

    bool unrestricted = (mo.kind == occ::qm::SpinorbitalKind::Unrestricted);

    constexpr size_t BLOCKSIZE = 64;
    gto_vals.reserve(m_basis.nbf(), BLOCKSIZE, 2);
    Mat rho = Mat::Zero(BLOCKSIZE, occ::density::num_components(2));
    m_hirshfeld_charges = Vec::Zero(num_atoms);
    for (int i = 0; i < num_atoms; i++) {
        m_hirshfeld_charges(i) = static_cast<double>(atoms[i].atomic_number);
    }
    double num_electrons{0.0};
    double num_electrons_promol{0.0};
    m_moments = Mat::Zero(3, num_atoms);
    m_volume = Vec::Zero(num_atoms);
    m_volume_free = Vec::Zero(num_atoms);

    constexpr double density_tolerance = 1e-10;

    for (const auto &atom_grid : m_atom_grids) {
        const auto &atom_pts = atom_grid.points;
        const auto &atom_weights = atom_grid.weights;
        const size_t npt_total = atom_pts.cols();
        const size_t num_blocks = npt_total / BLOCKSIZE + 1;

        for (size_t block = 0; block < num_blocks; block++) {
            Eigen::Index l = block * BLOCKSIZE;
            Eigen::Index u = std::min(npt_total - 1, (block + 1) * BLOCKSIZE);
            Eigen::Index npt = u - l;
            if (npt <= 0)
                continue;
            Mat hirshfeld_weights = Mat::Zero(npt, num_atoms);
            Mat r = Mat::Zero(npt, num_atoms);
            const auto &pts_block = atom_pts.middleCols(l, npt);
            const auto &weights_block = atom_weights.segment(l, npt);
            occ::gto::evaluate_basis(m_basis, pts_block, gto_vals, 2);
            if (unrestricted) {
                occ::density::evaluate_density<
                    2, occ::qm::SpinorbitalKind::Unrestricted>(
                    m_density_matrix * 2, gto_vals, rho);
            } else {
                occ::density::evaluate_density<
                    2, occ::qm::SpinorbitalKind::Restricted>(m_density_matrix,
                                                             gto_vals, rho);
            }

            for (int i = 0; i < num_atoms; i++) {
                auto el = Element(i);
                const auto &sb = m_slater_basis[i];
                occ::Vec3 pos{atoms[i].x, atoms[i].y, atoms[i].z};
                r.col(i) = (pts_block.colwise() - pos).colwise().norm();
                const auto &ria = r.col(i).array();
                // currently the hirsfheld weights array just holds the free
                // atom density
                hirshfeld_weights.col(i) = sb.rho(r.col(i));
                m_volume_free(i) += (hirshfeld_weights.col(i).array() *
                                     weights_block.array() * ria * ria * ria)
                                        .sum();
            }
            if (unrestricted) {
                impl::xdm_moment_kernel_unrestricted(
                    r, rho, weights_block, hirshfeld_weights,
                    m_hirshfeld_charges, m_volume, m_moments, num_electrons,
                    num_electrons_promol);
            } else {
                impl::xdm_moment_kernel_restricted(
                    r, rho, weights_block, hirshfeld_weights,
                    m_hirshfeld_charges, m_volume, m_moments, num_electrons,
                    num_electrons_promol);
            }
        }
    }
}

} // namespace occ::xdm
