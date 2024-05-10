#include <fmt/core.h>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
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

double xdm_polarizability(int n, double v, double vfree, bool charged) {
    // TODO handle charged atoms!
    // Thakkar polarizabilities for default oxidation states
    // are in occ::interaction but is this enough?
    return v * Element(n).polarizability(charged) / vfree;
}

std::tuple<double, Mat3N, Mat3N>
xdm_dispersion_interaction_energy(const XDMAtomList &atom_info_a,
                                  const XDMAtomList &atom_info_b,
                                  const XDM::Parameters &params) {
    const auto &atoms_a = atom_info_a.atoms;
    const auto &moments_a = atom_info_a.moments;
    const auto &polarizabilities_a = atom_info_a.polarizabilities;

    const auto &atoms_b = atom_info_b.atoms;
    const auto &moments_b = atom_info_b.moments;
    const auto &polarizabilities_b = atom_info_b.polarizabilities;

    const size_t num_atoms_a = atoms_a.size();
    const size_t num_atoms_b = atoms_b.size();

    using occ::Vec3;

    Mat3N forces_a = Mat3N::Zero(3, num_atoms_a);
    Mat3N forces_b = Mat3N::Zero(3, num_atoms_b);
    double edisp = 0.0;
    for (int i = 0; i < num_atoms_a; i++) {
        Vec3 pi = {atoms_a[i].x, atoms_a[i].y, atoms_a[i].z};
        double pol_i = polarizabilities_a(i);

        for (int j = 0; j < num_atoms_b; j++) {
            Vec3 pj = {atoms_b[j].x, atoms_b[j].y, atoms_b[j].z};
            Vec3 v_ij = pj - pi;
            double pol_j = polarizabilities_b(j);
            double factor = pol_i * pol_j /
                            (moments_a(0, i) * pol_j + moments_b(0, j) * pol_i);
            double rij = v_ij.norm();

            double rij2 = rij * rij;
            double rij4 = rij2 * rij2;
            double rij6 = rij4 * rij2;
            double rij8 = rij4 * rij4;
            double rij10 = rij4 * rij6;

            double c6 = factor * moments_a(0, i) * moments_b(0, j);
            double c8 = 1.5 * factor *
                        (moments_a(0, i) * moments_b(1, j) +
                         moments_a(1, i) * moments_b(0, j));
            double c10 = 2.0 * factor *
                             (moments_a(0, i) * moments_b(2, j) +
                              moments_a(2, i) * moments_b(0, j)) +
                         4.2 * factor * moments_a(1, i) * moments_b(1, j);
            double rc = (std::sqrt(c8 / c6) + std::sqrt(std::sqrt(c10 / c6)) +
                         std::sqrt(c10 / c8)) /
                        3;
            double rvdw =
                params.a1 * rc + params.a2 * occ::units::ANGSTROM_TO_BOHR;
            double rvdw2 = rvdw * rvdw;
            double rvdw4 = rvdw2 * rvdw2;
            double rvdw6 = rvdw4 * rvdw2;
            double rvdw8 = rvdw4 * rvdw4;
            double rvdw10 = rvdw4 * rvdw6;

            occ::log::debug("{:>3d} {:>3d} {:12.6f} {:12.6f} {:12.6f} {:12.6f} "
                            "{:12.6f} {:12.6f}",
                            i, j, rij, c6, c8, c10, rc, rvdw);
            if (rij < 1e-15)
                continue;
            edisp -= c6 / (rvdw6 + rij6) + c8 / (rvdw8 + rij8) +
                     c10 / (rvdw10 + rij10);

            double c6_com = 6.0 * c6 * rij4 / ((rvdw6 + rij6) * (rvdw6 + rij6));
            double c8_com = 8.0 * c8 * rij6 / ((rvdw8 + rij8) * (rvdw8 + rij8));
            double c10_com =
                10.0 * c10 * rij8 / ((rvdw10 + rij10) * (rvdw10 + rij10));
            forces_a.col(i) += (c6_com + c8_com + c10_com) * v_ij;
            forces_b.col(j) -= (c6_com + c8_com + c10_com) * v_ij;
        }
    }
    return {edisp, forces_a, forces_b};
}

std::pair<double, Mat3N> xdm_dispersion_energy(const XDMAtomList &atom_info,
                                               const XDM::Parameters &params) {
    const auto &atoms = atom_info.atoms;
    const auto &volume = atom_info.volume;
    const auto &moments = atom_info.moments;
    const auto &volume_free = atom_info.volume_free;
    const auto &polarizabilities = atom_info.polarizabilities;
    const size_t num_atoms = atoms.size();
    using occ::Vec3;

    occ::log::debug("Volumes\n{}\n", volume);
    occ::log::debug("Volumes free\n{}\n", volume_free);
    occ::log::debug("Polarizabilities: \n{}\n", polarizabilities);
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
            double rvdw =
                params.a1 * rc + params.a2 * occ::units::ANGSTROM_TO_BOHR;
            double rvdw2 = rvdw * rvdw;
            double rvdw4 = rvdw2 * rvdw2;
            double rvdw6 = rvdw4 * rvdw2;
            double rvdw8 = rvdw4 * rvdw4;
            double rvdw10 = rvdw4 * rvdw6;

            occ::log::debug("{:>3d} {:>3d} {:12.6f} {:12.6f} {:12.6f} {:12.6f} "
                            "{:12.6f} {:12.6f}",
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
        double fac = 2 * rho(j, 0) * weights(j);
        hirshfeld_charges.array() -= hirshfeld_weights.array() * fac;
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
    Mat rho_tot = rho_a + rho_b;
    xdm_moment_kernel_restricted(r, rho_tot, weights, rho_pro,
                                 hirshfeld_charges, volume, moments,
                                 num_electrons, num_electrons_promol);
}

} // namespace impl

struct XDMResult {
    double energy{0.0};
};

XDM::XDM(const occ::qm::AOBasis &basis, int charge,
         const XDM::Parameters &params)
    : m_basis(basis), m_grid(basis), m_charge(charge), m_params(params) {
    for (int i = 0; i < basis.atoms().size(); i++) {
        m_atom_grids.push_back(m_grid.generate_partitioned_atom_grid(i));
    }
    size_t num_grid_points = std::accumulate(
        m_atom_grids.begin(), m_atom_grids.end(), 0.0,
        [&](double tot, const auto &grid) { return tot + grid.points.cols(); });
    occ::log::debug("{} total grid points used for XDM", num_grid_points);
    m_atomic_ion = (m_atom_grids.size() == 1) && (m_charge != 0);
    std::vector<int> ox = {};
    if (m_atomic_ion) {
        ox.push_back(m_charge);
    }
    m_slater_basis =
        occ::slater::slaterbasis_for_atoms(m_basis.atoms(), "thakkar", ox);
}

double XDM::energy(const occ::qm::MolecularOrbitals &mo) {
    occ::log::debug("MO has {} alpha electrons {} beta electrons\n", mo.n_alpha,
                    mo.n_beta);
    occ::timing::start(occ::timing::category::xdm);
    populate_moments(mo);
    populate_polarizabilities();
    occ::log::debug("moments\n{}\n", m_moments);

    std::tie(m_energy, m_forces) =
        xdm_dispersion_energy({m_basis.atoms(), m_polarizabilities, m_moments,
                               m_volume, m_volume_free},
                              m_params);
    occ::timing::stop(occ::timing::category::xdm);
    return m_energy;
}

const Mat3N &XDM::forces(const occ::qm::MolecularOrbitals &mo) {
    populate_moments(mo);
    populate_polarizabilities();

    std::tie(m_energy, m_forces) =
        xdm_dispersion_energy({m_basis.atoms(), m_polarizabilities, m_moments,
                               m_volume, m_volume_free},
                              m_params);
    return m_forces;
}

void XDM::populate_moments(const occ::qm::MolecularOrbitals &mo) {
    int nthreads = occ::parallel::get_num_threads();
    if (m_density_matrix.size() != 0 &&
        occ::util::all_close(mo.D, m_density_matrix)) {
        return;
    }
    m_density_matrix = mo.D;

    const auto &atoms = m_basis.atoms();
    const size_t num_atoms = atoms.size();

    bool unrestricted = (mo.kind == occ::qm::SpinorbitalKind::Unrestricted);
    occ::log::debug("XDM using {} wavefunction",
                    unrestricted ? "unrestricted" : "restricted");

    constexpr size_t BLOCKSIZE = 64;
    int num_rows_factor = 1;
    if (unrestricted)
        num_rows_factor = 2;

    //constexpr double density_tolerance = 1e-10;
    constexpr int max_derivative{2};
    std::vector<Vec> tl_hirshfeld_charges(nthreads, Vec::Zero(num_atoms));
    std::vector<Vec> tl_volumes(nthreads, Vec::Zero(num_atoms));
    std::vector<Vec> tl_free_volumes(nthreads, Vec::Zero(num_atoms));
    std::vector<Mat> tl_moments(nthreads, Mat::Zero(3, num_atoms));
    std::vector<double> tl_num_electrons(nthreads, 0.0);
    std::vector<double> tl_num_electrons_promol(nthreads, 0.0);

    for (const auto &atom_grid : m_atom_grids) {
        const auto &atom_pts = atom_grid.points;
        const auto &atom_weights = atom_grid.weights;
        const size_t npt_total = atom_pts.cols();
        const size_t num_blocks = npt_total / BLOCKSIZE + 1;

        auto lambda = [&](int thread_id) {
            occ::gto::GTOValues gto_vals;
            gto_vals.reserve(m_basis.nbf(), BLOCKSIZE, 2);
            auto &hirshfeld_charges = tl_hirshfeld_charges[thread_id];
            auto &volume = tl_volumes[thread_id];
            auto &volume_free = tl_free_volumes[thread_id];
            auto &moments = tl_moments[thread_id];
            auto &num_e = tl_num_electrons[thread_id];
            auto &num_e_promol = tl_num_electrons_promol[thread_id];
            for (size_t block = 0; block < num_blocks; block++) {
                if (block % nthreads != thread_id)
                    continue;
                Eigen::Index l = block * BLOCKSIZE;
                Eigen::Index u =
                    std::min(npt_total - 1, (block + 1) * BLOCKSIZE);
                Eigen::Index npt = u - l;
                if (npt <= 0)
                    continue;
                Mat hirshfeld_weights = Mat::Zero(npt, num_atoms);
                Mat r = Mat::Zero(npt, num_atoms);
                Mat rho =
                    Mat::Zero(num_rows_factor * npt,
                              occ::density::num_components(max_derivative));
                const auto &pts_block = atom_pts.middleCols(l, npt);
                const auto &weights_block = atom_weights.segment(l, npt);
                occ::gto::evaluate_basis(m_basis, pts_block, gto_vals,
                                         max_derivative);
                if (unrestricted) {
                    occ::density::evaluate_density<
                        max_derivative, occ::qm::SpinorbitalKind::Unrestricted>(
                        mo.D, gto_vals, rho);
                } else {
                    occ::density::evaluate_density<
                        max_derivative, occ::qm::SpinorbitalKind::Restricted>(
                        mo.D, gto_vals, rho);
                }

                for (int i = 0; i < num_atoms; i++) {
                    const auto &sb = m_slater_basis[i];
                    occ::Vec3 pos{atoms[i].x, atoms[i].y, atoms[i].z};
                    r.col(i) = (pts_block.colwise() - pos).colwise().norm();
                    const auto &ria = r.col(i).array();
                    // currently the hirsfheld weights array just holds the free
                    // atom density
                    hirshfeld_weights.col(i) = sb.rho(r.col(i));
                    volume_free(i) += (hirshfeld_weights.col(i).array() *
                                       weights_block.array() * ria * ria * ria)
                                          .sum();
                }
                if (unrestricted) {
                    impl::xdm_moment_kernel_unrestricted(
                        r, rho, weights_block, hirshfeld_weights,
                        hirshfeld_charges, volume, moments, num_e,
                        num_e_promol);
                } else {
                    impl::xdm_moment_kernel_restricted(
                        r, rho, weights_block, hirshfeld_weights,
                        hirshfeld_charges, volume, moments, num_e,
                        num_e_promol);
                }
            }
        };
        occ::parallel::parallel_do(lambda);
    }

    m_hirshfeld_charges = Vec::Zero(num_atoms);
    const auto &ecp_electrons = m_basis.ecp_electrons();
    for (int i = 0; i < num_atoms; i++) {
        m_hirshfeld_charges(i) = static_cast<double>(atoms[i].atomic_number);
        if (ecp_electrons.size() >= i) {
            m_hirshfeld_charges(i) -= ecp_electrons[i];
        }
    }
    double num_electrons{0.0};
    double num_electrons_promol{0.0};
    m_moments = Mat::Zero(3, num_atoms);
    m_volume = Vec::Zero(num_atoms);
    m_volume_free = Vec::Zero(num_atoms);

    for (size_t i = 0; i < nthreads; i++) {
        m_hirshfeld_charges += tl_hirshfeld_charges[i];
        m_volume += tl_volumes[i];
        m_volume_free += tl_free_volumes[i];
        m_moments += tl_moments[i];
        num_electrons += tl_num_electrons[i];
        num_electrons_promol += tl_num_electrons_promol[i];
    }

    occ::log::debug("Num electrons {:20.12f}, promolecule {:20.12f}\n",
                    num_electrons, num_electrons_promol);
    occ::log::debug("Hirshfeld charges:\n{}", m_hirshfeld_charges);
}

void XDM::populate_polarizabilities() {
    m_polarizabilities = Vec(m_volume.rows());
    const auto &atoms = m_basis.atoms();
    for (int i = 0; i < m_polarizabilities.rows(); i++) {
        m_polarizabilities(i) =
            xdm_polarizability(atoms[i].atomic_number, m_volume(i),
                               m_volume_free(i), m_atomic_ion);
    }
}

} // namespace occ::xdm
