#include <occ/mults/crystal_energy.h>
#include <occ/mults/strain_ad.h>
#include <occ/mults/ewald_sum.h>
#include <occ/mults/cartesian_hessian.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/core/timings.h>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <algorithm>
#include <array>
#include <cmath>

namespace occ::mults {

namespace {

Mat3 skew_symmetric(const Vec3& v) {
    Mat3 S;
    S <<      0.0, -v[2],  v[1],
           v[2],      0.0, -v[0],
          -v[1],   v[0],   0.0;
    return S;
}

const std::array<Mat3, 3>& so3_generators() {
    static const std::array<Mat3, 3> G = [] {
        std::array<Mat3, 3> out;
        out[0] << 0, 0, 0,  0, 0, -1,  0, 1, 0;
        out[1] << 0, 0, 1,  0, 0, 0,  -1, 0, 0;
        out[2] << 0, -1, 0,  1, 0, 0,  0, 0, 0;
        return out;
    }();
    return G;
}

// AD6/AD6R3 dual-number types and compute_ewald_explicit_strain_terms
// moved to strain_ad.h/cpp.
// TypedSelfBuckingham, williams_typed_self_params, bonded_neighbors,
// classify_williams_type moved to force_field_params.cpp.


void add_pair_strain_gradient(Vec6& strain_grad,
                              const Vec3& force_on_i,
                              const Vec3& disp_ij,
                              double weight) {
    const auto& B = voigt_basis_matrices();
    for (int a = 0; a < 6; ++a) {
        strain_grad[a] += weight * force_on_i.dot(B[a] * disp_ij);
    }
}

void accumulate_pair_strain_hessian_blocks(
    Mat6& H_ee,
    Mat& H_eq,
    int mol_i,
    int mol_j,
    const PairHessianResult& pair,
    double weight,
    const Vec3& pos_i,
    const Vec3& pos_j_image) {

    const auto& B = voigt_basis_matrices();
    const Vec3 disp = pos_j_image - pos_i;
    std::array<Vec3, 6> dA{};
    std::array<Vec3, 6> dB{};
    for (int a = 0; a < 6; ++a) {
        dA[a] = B[a] * pos_i;
        dB[a] = B[a] * pos_j_image;
    }

    const int ia = 6 * mol_i;
    const int ib = 6 * mol_j;
    const Mat3 H_rel =
        0.25 * (pair.H_posA_posA + pair.H_posB_posB -
                pair.H_posA_posB - pair.H_posA_posB.transpose());
    for (int a = 0; a < 6; ++a) {
        const Vec3& ba = dA[a];
        const Vec3& bb = dB[a];

        const Vec3 row_A_pos =
            pair.H_posA_posA.transpose() * ba + pair.H_posA_posB * bb;
        const Vec3 row_A_rot =
            pair.H_posA_rotA.transpose() * ba + pair.H_posB_rotA.transpose() * bb;
        const Vec3 row_B_pos =
            pair.H_posA_posB.transpose() * ba + pair.H_posB_posB.transpose() * bb;
        const Vec3 row_B_rot =
            pair.H_posA_rotB.transpose() * ba + pair.H_posB_rotB.transpose() * bb;

        H_eq.block<1, 3>(a, ia) += weight * row_A_pos.transpose();
        H_eq.block<1, 3>(a, ia + 3) += weight * row_A_rot.transpose();
        H_eq.block<1, 3>(a, ib) += weight * row_B_pos.transpose();
        H_eq.block<1, 3>(a, ib + 3) += weight * row_B_rot.transpose();

        for (int b = 0; b < 6; ++b) {
            const Vec3& ca = dA[b];
            const Vec3& cb = dB[b];
            const Vec3 dRa = bb - ba;
            const Vec3 dRb = cb - ca;
            const double val = dRa.dot(H_rel * dRb);
            H_ee(a, b) += weight * val;
        }
    }
}

bool is_canonical_explicit_pair(int i, int j, const IVec3& shift) {
    if (i < j) {
        return true;
    }
    if (i > j) {
        return false;
    }
    // For i == j keep only one of (+shift, -shift) to avoid mirrored duplicates.
    if (shift[0] != 0) return shift[0] > 0;
    if (shift[1] != 0) return shift[1] > 0;
    return shift[2] > 0;
}

void accumulate_pair_hessian_blocks(Mat& H,
                                    int mol_i,
                                    int mol_j,
                                    const PairHessianResult& pair,
                                    double weight) {
    const int ii = 6 * mol_i;
    const int jj = 6 * mol_j;
    auto add = [&](int r, int c, const Mat3& B) {
        H.block<3, 3>(r, c) += weight * B;
    };

    // Diagonal molecule blocks
    add(ii, ii, pair.H_posA_posA);
    add(ii, ii + 3, pair.H_posA_rotA);
    add(ii + 3, ii, pair.H_posA_rotA.transpose());
    add(ii + 3, ii + 3, pair.H_rotA_rotA);

    add(jj, jj, pair.H_posB_posB);
    add(jj, jj + 3, pair.H_posB_rotB);
    add(jj + 3, jj, pair.H_posB_rotB.transpose());
    add(jj + 3, jj + 3, pair.H_rotB_rotB);

    // Cross-molecule blocks
    add(ii, jj, pair.H_posA_posB);
    add(jj, ii, pair.H_posA_posB.transpose());

    add(ii, jj + 3, pair.H_posA_rotB);
    add(jj + 3, ii, pair.H_posA_rotB.transpose());

    add(jj, ii + 3, pair.H_posB_rotA);
    add(ii + 3, jj, pair.H_posB_rotA.transpose());

    add(ii + 3, jj + 3, pair.H_rotA_rotB);
    add(jj + 3, ii + 3, pair.H_rotA_rotB.transpose());
}

PairHessianResult short_range_site_pair_hessian(
    const Mat3& R_i,
    const Mat3& R_j,
    const Vec3& body_a,
    const Vec3& body_b,
    const Vec3& pos_a,
    const Vec3& pos_b,
    double dE_dr,
    double d2E_dr2) {

    PairHessianResult pair;

    const Vec3 r_ab = pos_b - pos_a;
    const double r = r_ab.norm();
    if (r < 1e-12) {
        return pair;
    }

    const Vec3 u = r_ab / r;
    const Mat3 I = Mat3::Identity();
    const Mat3 Hxx =
        (d2E_dr2 - dE_dr / r) * (u * u.transpose()) + (dE_dr / r) * I;

    // dE/dxA and dE/dxB
    const Vec3 gA = -dE_dr * u;
    const Vec3 gB = -gA;

    const Vec3 lever_a = R_i * body_a;
    const Vec3 lever_b = R_j * body_b;
    const Mat3 Jpsi_a = -skew_symmetric(lever_a);
    const Mat3 Jpsi_b = -skew_symmetric(lever_b);

    pair.H_posA_posA = Hxx;
    pair.H_posA_posB = -Hxx;
    pair.H_posB_posB = Hxx;

    pair.H_posA_rotA = Hxx * Jpsi_a;
    pair.H_posA_rotB = -Hxx * Jpsi_b;
    pair.H_posB_rotA = -Hxx * Jpsi_a;
    pair.H_posB_rotB = Hxx * Jpsi_b;

    pair.H_rotA_rotA = Jpsi_a.transpose() * Hxx * Jpsi_a;
    pair.H_rotA_rotB = Jpsi_a.transpose() * (-Hxx) * Jpsi_b;
    pair.H_rotB_rotB = Jpsi_b.transpose() * Hxx * Jpsi_b;

    // Exponential-map curvature at zero increment:
    // d2(exp([psi]x) v)/dpsi_k dpsi_l|_{psi=0} =
    // 0.5 * (G_k G_l + G_l G_k) v
    const auto& G = so3_generators();
    for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
            const Vec3 d2xa =
                0.5 * (G[k] * G[l] + G[l] * G[k]) * lever_a;
            const Vec3 d2xb =
                0.5 * (G[k] * G[l] + G[l] * G[k]) * lever_b;
            pair.H_rotA_rotA(k, l) += gA.dot(d2xa);
            pair.H_rotB_rotB(k, l) += gB.dot(d2xb);
        }
    }

    return pair;
}

} // namespace

const std::array<Mat3, 6>& voigt_basis_matrices() {
    static const std::array<Mat3, 6> B = [] {
        std::array<Mat3, 6> out{};
        out[0].setZero();
        out[0](0, 0) = 1.0;
        out[1].setZero();
        out[1](1, 1) = 1.0;
        out[2].setZero();
        out[2](2, 2) = 1.0;
        out[3].setZero();
        out[3](1, 2) = out[3](2, 1) = 0.5;
        out[4].setZero();
        out[4](0, 2) = out[4](2, 0) = 0.5;
        out[5].setZero();
        out[5](0, 1) = out[5](1, 0) = 0.5;
        return out;
    }();
    return B;
}

// ============================================================================
// MoleculeState
// ============================================================================

Mat3 MoleculeState::proper_rotation_matrix() const {
    double angle = angle_axis.norm();
    if (angle < 1e-12) {
        return Mat3::Identity();
    }
    Vec3 axis = angle_axis / angle;
    return Eigen::AngleAxisd(angle, axis).toRotationMatrix();
}

Mat3 MoleculeState::rotation_matrix() const {
    const int p = (parity < 0) ? -1 : 1;
    return static_cast<double>(p) * proper_rotation_matrix();
}

MoleculeState MoleculeState::from_rotation(const Vec3& pos, const Mat3& R) {
    MoleculeState state;
    state.position = pos;
    state.parity = 1;

    // Use exact orthogonal input directly (common in finite-difference
    // perturbations) to avoid introducing SVD gauge noise into the
    // angle-axis extraction. Reproject only when needed.
    Mat3 Q = R;
    const Mat3 I = Mat3::Identity();
    const double ortho_err = (R.transpose() * R - I).norm();
    const double det_err = std::abs(std::abs(R.determinant()) - 1.0);
    if (ortho_err > 1e-10 || det_err > 1e-10) {
        Eigen::JacobiSVD<Mat3> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Q = svd.matrixU() * svd.matrixV().transpose();
    }

    // Represent O(3) orientation as parity * SO(3) rotation.
    if (Q.determinant() < 0.0) {
        state.parity = -1;
        Q = -Q;
    }

    Eigen::AngleAxisd aa(Q);
    state.angle_axis = aa.angle() * aa.axis();
    return state;
}

// ============================================================================
// CrystalEnergyResult
// ============================================================================

Vec CrystalEnergyResult::pack_gradient() const {
    const int n = static_cast<int>(forces.size());
    Vec grad(6 * n);
    for (int i = 0; i < n; ++i) {
        grad.segment<3>(6 * i) = forces[i];
        grad.segment<3>(6 * i + 3) = torques[i];
    }
    return grad;
}

// Williams DE params, typed params, type labels, and type atomic numbers
// now in ForceFieldParams (force_field_params.cpp)

// ============================================================================
// CrystalEnergy Constructor
// ============================================================================

CrystalEnergy::CrystalEnergy(const crystal::Crystal& crystal,
                             std::vector<MultipoleSource> multipoles,
                             double cutoff_radius,
                             ForceFieldType ff,
                             bool use_cartesian,
                             bool use_ewald,
                             double ewald_accuracy,
                             double ewald_eta,
                             int ewald_kmax)
    : m_crystal(crystal)
    , m_multipoles(std::move(multipoles))
    , m_cutoff_radius(cutoff_radius)
    , m_force_field(ff)
    , m_use_cartesian(use_cartesian)
    , m_use_ewald(use_ewald)
    , m_ewald_accuracy(ewald_accuracy)
    , m_ewald_eta(ewald_eta)
    , m_ewald_kmax(ewald_kmax) {

    if (m_multipoles.empty()) {
        throw std::invalid_argument("CrystalEnergy: no multipoles provided");
    }

    build_neighbor_list();
    build_molecule_geometry();
    initialize_force_field();
}

CrystalEnergy::~CrystalEnergy() = default;
CrystalEnergy::CrystalEnergy(CrystalEnergy&&) noexcept = default;
CrystalEnergy& CrystalEnergy::operator=(CrystalEnergy&&) noexcept = default;

void CrystalEnergy::invalidate_ewald_params() {
    m_ewald_params_initialized = false;
    m_ewald_lattice_cache.reset();
}

void CrystalEnergy::ensure_ewald_params_initialized() const {
    if (m_ewald_params_initialized) {
        return;
    }

    double ewald_real_cutoff = m_use_com_elec_gate
        ? effective_electrostatic_com_cutoff()
        : effective_neighbor_pair_cutoff();
    const double elec_site_cutoff = effective_electrostatic_site_cutoff();
    if (elec_site_cutoff > 0.0) {
        ewald_real_cutoff = std::max(ewald_real_cutoff, elec_site_cutoff);
    }

    if (m_ewald_eta > 0.0) {
        m_ewald_alpha_fixed = m_ewald_eta;
    } else {
        const double x = std::sqrt(-std::log(m_ewald_accuracy));
        m_ewald_alpha_fixed = x / ewald_real_cutoff;
    }

    if (m_ewald_kmax > 0) {
        m_ewald_kmax_fixed = m_ewald_kmax;
    } else {
        const auto& uc = m_crystal.unit_cell();
        const double min_len = std::min({uc.a(), uc.b(), uc.c()});
        const double G_max =
            2.0 * m_ewald_alpha_fixed * std::sqrt(-std::log(m_ewald_accuracy));
        m_ewald_kmax_fixed = std::max(
            1, static_cast<int>(std::ceil(G_max * min_len / (2.0 * M_PI))));
    }

    m_ewald_params_initialized = true;
    occ::log::info("Ewald parameters fixed for this calculation: alpha={:.6f} /Ang, kmax={}",
                   m_ewald_alpha_fixed, m_ewald_kmax_fixed);
}

void CrystalEnergy::set_cutoff_radius(double cutoff) {
    if (cutoff <= 0.0) {
        throw std::invalid_argument("CrystalEnergy::set_cutoff_radius: cutoff must be > 0");
    }
    m_cutoff_radius = cutoff;
    invalidate_ewald_params();
    if (!m_explicit_neighbors) {
        build_neighbor_list();
    }
}

void CrystalEnergy::set_electrostatic_taper(double r_on, double r_off, int order) {
    if (r_off <= r_on) {
        throw std::invalid_argument("CrystalEnergy::set_electrostatic_taper: require r_off > r_on");
    }
    if (order != 3 && order != 5) {
        throw std::invalid_argument("CrystalEnergy::set_electrostatic_taper: order must be 3 or 5");
    }
    m_electrostatic_taper = {true, r_on, r_off, order};
    invalidate_ewald_params();
}

void CrystalEnergy::set_short_range_taper(double r_on, double r_off, int order) {
    if (r_off <= r_on) {
        throw std::invalid_argument("CrystalEnergy::set_short_range_taper: require r_off > r_on");
    }
    if (order != 3 && order != 5) {
        throw std::invalid_argument("CrystalEnergy::set_short_range_taper: order must be 3 or 5");
    }
    m_short_range_taper = {true, r_on, r_off, order};
    invalidate_ewald_params();
}

double CrystalEnergy::effective_electrostatic_com_cutoff() const {
    double cutoff = m_cutoff_radius;
    if (m_electrostatic_taper.is_valid()) {
        cutoff = std::max(cutoff, m_electrostatic_taper.r_off);
    }
    return cutoff;
}

double CrystalEnergy::effective_electrostatic_site_cutoff() const {
    double cutoff = m_elec_site_cutoff;
    if (m_electrostatic_taper.is_valid()) {
        cutoff = (cutoff > 0.0) ? std::max(cutoff, m_electrostatic_taper.r_off)
                                : m_electrostatic_taper.r_off;
    }
    return cutoff;
}

double CrystalEnergy::effective_buckingham_site_cutoff() const {
    double cutoff = (m_buck_site_cutoff > 0.0) ? m_buck_site_cutoff : m_cutoff_radius;
    if (m_short_range_taper.is_valid()) {
        cutoff = std::max(cutoff, m_short_range_taper.r_off);
    }
    return cutoff;
}

double CrystalEnergy::effective_neighbor_pair_cutoff() const {
    double cutoff = std::max(m_cutoff_radius, effective_buckingham_site_cutoff());
    if (m_use_com_elec_gate) {
        cutoff = std::max(cutoff, effective_electrostatic_com_cutoff());
    }
    return cutoff;
}

// ============================================================================
// Neighbor List Construction
// ============================================================================

void CrystalEnergy::build_neighbor_list() {
    m_neighbors.clear();

    auto dimers = m_crystal.symmetry_unique_dimers(effective_neighbor_pair_cutoff());

    const int num_unique = static_cast<int>(m_multipoles.size());

    for (size_t i = 0; i < dimers.molecule_neighbors.size(); ++i) {
        for (const auto& neighbor : dimers.molecule_neighbors[i]) {
            const auto& dimer = neighbor.dimer;

            // Get molecule indices
            int mol_i = static_cast<int>(i);
            // Get the asymmetric molecule index of the neighbor (molecule B in dimer)
            int mol_j = dimer.b().asymmetric_molecule_idx();

            // Validate indices
            if (mol_i < 0 || mol_i >= num_unique || mol_j < 0 || mol_j >= num_unique) {
                occ::log::debug("Skipping invalid neighbor pair: mol_i={}, mol_j={}, num_unique={}",
                               mol_i, mol_j, num_unique);
                continue;
            }

            // Compute cell shift from molecule B's position
            const auto& mol_a = dimer.a();
            const auto& mol_b = dimer.b();

            Vec3 center_a = mol_a.center_of_mass();
            Vec3 center_b = mol_b.center_of_mass();

            // Convert to fractional and find integer shift
            Vec3 frac_a = m_crystal.to_fractional(center_a);
            Vec3 frac_b = m_crystal.to_fractional(center_b);
            Vec3 diff = frac_b - frac_a;

            IVec3 cell_shift;
            cell_shift << static_cast<int>(std::round(diff[0])),
                         static_cast<int>(std::round(diff[1])),
                         static_cast<int>(std::round(diff[2]));

            // Weight: 0.5 for self-interaction, 1.0 otherwise
            double weight = (mol_i == mol_j && cell_shift == IVec3::Zero()) ? 0.5 : 1.0;

            // Skip true self (same molecule, no translation)
            if (mol_i == mol_j && cell_shift == IVec3::Zero()) {
                continue;
            }

            m_neighbors.push_back({mol_i, mol_j, cell_shift, weight});
        }
    }

    occ::log::debug("Built neighbor list with {} pairs", m_neighbors.size());
}

void CrystalEnergy::update_neighbors() {
    if (m_explicit_neighbors) {
        // Neighbor list was set externally (build_neighbor_list_from_positions
        // or set_neighbor_list). Don't rebuild via symmetry_unique_dimers as
        // that would corrupt an all-pairs list for Z UC molecules.
        occ::log::debug("Skipping neighbor list rebuild (explicit neighbor list)");
        return;
    }
    build_neighbor_list();
}

void CrystalEnergy::update_neighbors(const std::vector<MoleculeState>& states) {
    if (m_explicit_neighbors) {
        std::vector<Vec3> coms;
        coms.reserve(states.size());
        for (const auto& s : states) {
            coms.push_back(s.position);
        }
        build_neighbor_list_from_positions(coms, false, &states);
        return;
    }
    build_neighbor_list();
}

void CrystalEnergy::update_lattice(const crystal::Crystal& strained_crystal,
                                    std::vector<MoleculeState> new_states) {
    m_crystal = strained_crystal;
    m_initial_states = std::move(new_states);
    m_ewald_lattice_cache.reset();  // Invalidate: lattice changed
}

void CrystalEnergy::build_neighbor_list_from_positions(
        const std::vector<Vec3>& mol_coms, bool force_com_cutoff,
        const std::vector<MoleculeState>* orientation_states) {
    m_neighbors.clear();
    int n_mol = static_cast<int>(mol_coms.size());

    const auto& uc = m_crystal.unit_cell();

    // Use atom-based cutoff: include pair if any atom-atom distance < cutoff.
    // When force_com_cutoff=true, use COM distance (matches DMACRYS TBLCNT).
    // Need extra margin for COM-to-atom extent when determining cell search range.
    bool use_atom_cutoff = !m_geometry.empty() && !force_com_cutoff;
    double max_atom_extent = 0.0;
    // Pre-compute crystal-frame atom positions for each molecule
    std::vector<std::vector<Vec3>> crystal_atoms(n_mol);
    if (use_atom_cutoff) {
        const std::vector<MoleculeState>* orient_states = orientation_states;
        if (!orient_states ||
            static_cast<int>(orient_states->size()) != n_mol) {
            orient_states = &m_initial_states;
        }
        for (int m = 0; m < n_mol; ++m) {
            const auto& geom = m_geometry[m];
            Mat3 R = (orient_states &&
                      static_cast<int>(orient_states->size()) == n_mol)
                         ? (*orient_states)[m].rotation_matrix()
                         : Mat3::Identity();
            crystal_atoms[m].reserve(geom.atom_positions.size());
            for (const auto& bp : geom.atom_positions) {
                Vec3 cart = mol_coms[m] + R * bp;
                crystal_atoms[m].push_back(cart);
                double ext = (R * bp).norm();
                if (ext > max_atom_extent)
                    max_atom_extent = ext;
            }
        }
    }

    const double pair_cutoff = effective_neighbor_pair_cutoff();
    double search_radius = pair_cutoff + 2.0 * max_atom_extent;
    // Use anisotropic search extents from reciprocal lattice norms, matching
    // the DMACRYS TBLCNT-style bound N(i) ~ R * |b_i*| + 1.
    const Mat3 reciprocal = uc.reciprocal();
    IVec3 nmax = IVec3::Zero();
    for (int axis = 0; axis < 3; ++axis) {
        const double bnorm = reciprocal.col(axis).norm();
        int ni = static_cast<int>(std::ceil(search_radius * bnorm)) + 1;
        if (ni < 1) ni = 1;
        nmax[axis] = ni;
    }

    for (int i = 0; i < n_mol; ++i) {
        for (int j = 0; j < n_mol; ++j) {
            for (int nx = -nmax[0]; nx <= nmax[0]; ++nx) {
                for (int ny = -nmax[1]; ny <= nmax[1]; ++ny) {
                    for (int nz = -nmax[2]; nz <= nmax[2]; ++nz) {
                        if (i == j && nx == 0 && ny == 0 && nz == 0)
                            continue;

                        IVec3 shift(nx, ny, nz);
                        if (!is_canonical_explicit_pair(i, j, shift)) {
                            continue;
                        }
                        Vec3 trans = uc.to_cartesian(shift.cast<double>());

                        bool include = false;
                        if (use_atom_cutoff) {
                            // Check minimum atom-atom distance
                            for (const auto& ai : crystal_atoms[i]) {
                                for (const auto& aj : crystal_atoms[j]) {
                                    double d = (aj + trans - ai).norm();
                                    if (d < pair_cutoff) {
                                        include = true;
                                        break;
                                    }
                                }
                                if (include) break;
                            }
                        } else {
                            double dist = (mol_coms[j] + trans - mol_coms[i]).norm();
                            include = (dist < pair_cutoff);
                        }

                        if (include) {
                            // Canonical unique pair list (no mirrored duplicates).
                            double com_dist = (mol_coms[j] + trans - mol_coms[i]).norm();
                            m_neighbors.push_back({i, j, shift, 1.0, com_dist});
                        }
                    }
                }
            }
        }
    }

    m_explicit_neighbors = true;
    occ::log::debug("Built neighbor list from positions: {} pairs for {} molecules (atom-based cutoff: {})",
                    m_neighbors.size(), n_mol, use_atom_cutoff);
}

void CrystalEnergy::set_neighbor_list(const std::vector<NeighborPair>& neighbors) {
    m_neighbors = neighbors;
    m_explicit_neighbors = true;
}

void CrystalEnergy::set_molecule_geometry(std::vector<MoleculeGeometry> geometry) {
    m_geometry = std::move(geometry);
    if (m_force_field == ForceFieldType::BuckinghamDE) {
        assign_williams_atom_types();
    }
}

void CrystalEnergy::set_initial_states(std::vector<MoleculeState> states) {
    m_initial_states = std::move(states);
}

// ============================================================================
// Molecule Geometry
// ============================================================================

void CrystalEnergy::build_molecule_geometry() {
    m_geometry.clear();
    m_geometry.reserve(m_multipoles.size());

    const auto& unique_mols = m_crystal.symmetry_unique_molecules();

    for (size_t i = 0; i < m_multipoles.size() && i < unique_mols.size(); ++i) {
        const auto& mol = unique_mols[i];
        MoleculeGeometry geom;

        geom.center_of_mass = mol.center_of_mass();

        for (int j = 0; j < mol.size(); ++j) {
            geom.atomic_numbers.push_back(mol.atomic_numbers()(j));
            // Store in body frame (relative to COM)
            geom.atom_positions.push_back(mol.positions().col(j) - geom.center_of_mass);
        }
        geom.short_range_type_codes.assign(geom.atomic_numbers.size(), 0);

        m_geometry.push_back(std::move(geom));
    }
}

void CrystalEnergy::assign_williams_atom_types() {
    for (auto& geom : m_geometry) {
        const auto neighbors = ForceFieldParams::bonded_neighbors(
            geom.atomic_numbers, geom.atom_positions);
        geom.short_range_type_codes.resize(geom.atomic_numbers.size(), 0);
        for (size_t i = 0; i < geom.atomic_numbers.size(); ++i) {
            geom.short_range_type_codes[i] = ForceFieldParams::classify_williams_type(
                static_cast<int>(i), neighbors, geom.atomic_numbers);
        }
    }
    m_ff.set_use_short_range_typing(true);
}

// ============================================================================
// Force Field Initialization
// ============================================================================

void CrystalEnergy::initialize_force_field() {
    if (m_force_field == ForceFieldType::BuckinghamDE) {
        // Element-based params (used as fallback when typed params are missing)
        for (const auto& [key, p] : ForceFieldParams::williams_de_params()) {
            m_ff.set_buckingham(key.first, key.second, p);
        }
        // Type-code-based params (Williams atom typing)
        m_ff.set_typed_buckingham(ForceFieldParams::williams_typed_params());
        assign_williams_atom_types();
        m_ff.set_use_williams_atom_typing(true);
        m_ff.set_use_short_range_typing(true);
    }
}

// Force field parameter methods moved to force_field_params.cpp;
// CrystalEnergy methods are now inline wrappers in crystal_energy.h.

// ============================================================================
// Initial States
// ============================================================================

std::vector<MoleculeState> CrystalEnergy::initial_states() const {
    if (!m_initial_states.empty()) {
        return m_initial_states;
    }

    std::vector<MoleculeState> states;
    states.reserve(m_multipoles.size());

    const auto& unique_mols = m_crystal.symmetry_unique_molecules();

    for (size_t i = 0; i < m_multipoles.size() && i < unique_mols.size(); ++i) {
        MoleculeState state;
        state.position = unique_mols[i].center_of_mass();
        state.angle_axis = Vec3::Zero();  // Identity rotation
        states.push_back(state);
    }

    return states;
}

// ============================================================================
// Buckingham Site-Pair Masks
// ============================================================================

std::vector<std::vector<bool>> CrystalEnergy::compute_buckingham_site_masks(
    const std::vector<MoleculeState>& states) const {

    const double buck_cutoff = effective_buckingham_site_cutoff();

    std::vector<std::vector<bool>> masks(m_neighbors.size());
    for (size_t pair_idx = 0; pair_idx < m_neighbors.size(); ++pair_idx) {
        const auto& pair = m_neighbors[pair_idx];
        int mi = pair.mol_i;
        int mj = pair.mol_j;
        const auto& geom_i = m_geometry[mi];
        const auto& geom_j = m_geometry[mj];
        const size_t nA = geom_i.atom_positions.size();
        const size_t nB = geom_j.atom_positions.size();

        masks[pair_idx].resize(nA * nB, false);

        Mat3 R_i = states[mi].rotation_matrix();
        Mat3 R_j = states[mj].rotation_matrix();
        Vec3 cell_translation = m_crystal.unit_cell().to_cartesian(
            pair.cell_shift.cast<double>());

        for (size_t a = 0; a < nA; ++a) {
            Vec3 pos_a = states[mi].position + R_i * geom_i.atom_positions[a];
            for (size_t b = 0; b < nB; ++b) {
                Vec3 pos_b = states[mj].position + cell_translation +
                             R_j * geom_j.atom_positions[b];
                double r = (pos_b - pos_a).norm();
                if (r <= buck_cutoff && r >= 0.1) {
                    masks[pair_idx][a * nB + b] = true;
                }
            }
        }
    }

    return masks;
}

// ============================================================================
// Short-Range Pair Computation
// ============================================================================

void CrystalEnergy::compute_short_range_pair(
    int mol_i, int mol_j,
    const MoleculeState& state_i,
    const MoleculeState& state_j,
    const Vec3& translation,
    double weight,
    double& energy,
    Vec3& force_i, Vec3& force_j,
    Vec3& torque_i, Vec3& torque_j,
    int neighbor_idx,
    const MoleculeCache* cache_i,
    const MoleculeCache* cache_j,
    int* short_range_site_pairs) const {

    if (m_force_field == ForceFieldType::None) {
        return;
    }

    const auto& geom_i = m_geometry[mol_i];
    const auto& geom_j = m_geometry[mol_j];
    const size_t nA = geom_i.atom_positions.size();
    const size_t nB = geom_j.atom_positions.size();

    // Check if we have frozen site masks for this neighbor pair
    const bool use_frozen = (neighbor_idx >= 0 &&
                             static_cast<size_t>(neighbor_idx) < m_fixed_site_masks.size() &&
                             !m_fixed_site_masks[neighbor_idx].empty());

    // Use cached rotation matrices or compute them
    Mat3 R_i_storage, R_j_storage;
    const Mat3& R_i = cache_i ? cache_i->rotation : (R_i_storage = state_i.rotation_matrix());
    const Mat3& R_j = cache_j ? cache_j->rotation : (R_j_storage = state_j.rotation_matrix());

    // Loop over atom pairs
    for (size_t a = 0; a < nA; ++a) {
        int Z_a = geom_i.atomic_numbers[a];
        Vec3 pos_a = cache_i ? cache_i->lab_atom_positions[a]
                             : (state_i.position + R_i * geom_i.atom_positions[a]);

        for (size_t b = 0; b < nB; ++b) {
            // Apply frozen mask or distance cutoff
            if (use_frozen) {
                if (!m_fixed_site_masks[neighbor_idx][a * nB + b]) {
                    continue;
                }
            }

            int Z_b = geom_j.atomic_numbers[b];
            Vec3 pos_b;
            if (cache_j) {
                pos_b = cache_j->lab_atom_positions[b] + translation;
            } else {
                pos_b = state_j.position + translation + R_j * geom_j.atom_positions[b];
            }

            Vec3 r_ab = pos_b - pos_a;
            double r = r_ab.norm();

            if (!use_frozen) {
                double buck_cutoff = effective_buckingham_site_cutoff();
                if (r > buck_cutoff || r < 0.1) {
                    continue;
                }
            }

            ShortRangeInteraction::EnergyAndDerivatives sr;

            const bool has_type_codes =
                m_ff.use_short_range_typing() &&
                a < geom_i.short_range_type_codes.size() &&
                b < geom_j.short_range_type_codes.size() &&
                geom_i.short_range_type_codes[a] > 0 &&
                geom_j.short_range_type_codes[b] > 0;

            if (m_force_field == ForceFieldType::BuckinghamDE ||
                m_force_field == ForceFieldType::Custom) {
                BuckinghamParams params;
                if (has_type_codes) {
                    const int t1 = geom_i.short_range_type_codes[a];
                    const int t2 = geom_j.short_range_type_codes[b];
                    if (has_typed_buckingham_params(t1, t2)) {
                        params = get_buckingham_params_for_types(t1, t2);
                    } else {
                        params = get_buckingham_params(Z_a, Z_b);
                    }
                } else {
                    params = get_buckingham_params(Z_a, Z_b);
                }
                sr = ShortRangeInteraction::buckingham_all(r, params);
            } else if (m_force_field == ForceFieldType::LennardJones) {
                auto it = m_ff.lj_params().find({Z_a, Z_b});
                if (it == m_ff.lj_params().end()) {
                    continue;
                }
                sr = ShortRangeInteraction::lennard_jones_all(r, it->second);
            }

            if (m_short_range_taper.is_valid()) {
                auto sw = evaluate_cutoff_spline(r, m_short_range_taper);
                if (sw.value <= 0.0) {
                    continue;
                }
                const double e0 = sr.energy;
                const double d10 = sr.first_derivative;
                const double d20 = sr.second_derivative;
                sr.energy = sw.value * e0;
                sr.first_derivative = sw.value * d10 + e0 * sw.first_derivative;
                sr.second_derivative = sw.value * d20 +
                                       2.0 * sw.first_derivative * d10 +
                                       e0 * sw.second_derivative;
            }

            if (short_range_site_pairs) {
                ++(*short_range_site_pairs);
            }

            energy += weight * sr.energy;

            // Force from derivative
            Vec3 force_on_a = ShortRangeInteraction::derivative_to_force(sr.first_derivative, r_ab);

            // Weighted forces
            Vec3 wf = weight * force_on_a;
            force_i += wf;
            force_j -= wf;

            // Torque from lever arm
            Vec3 lever_a = R_i * geom_i.atom_positions[a];
            Vec3 lever_b = R_j * geom_j.atom_positions[b];

            // Torque in lab frame
            Vec3 torque_lab_a = lever_a.cross(wf);
            Vec3 torque_lab_b = lever_b.cross(-wf);

            // Lever-arm rotational gradient in lab frame:
            // torque_lab = lever × force = lever × (-dE/dr) = -dE/dψ_lab
            // so -torque_lab = +dE/dψ_lab
            torque_i -= torque_lab_a;
            torque_j -= torque_lab_b;

            // Anisotropic repulsion (additive to isotropic Buckingham)
            if (m_ff.has_any_aniso() && has_type_codes) {
                const int t1 = geom_i.short_range_type_codes[a];
                const int t2 = geom_j.short_range_type_codes[b];
                if (has_aniso_params(t1, t2)) {
                    const auto aniso_params = get_aniso_params(t1, t2);

                    // Get body-frame aniso axes and rotate to lab frame
                    Vec3 axis_a_lab = Vec3::Zero();
                    Vec3 axis_b_lab = Vec3::Zero();
                    if (a < geom_i.aniso_body_axes.size()) {
                        axis_a_lab = R_i * geom_i.aniso_body_axes[a];
                    }
                    if (b < geom_j.aniso_body_axes.size()) {
                        axis_b_lab = R_j * geom_j.aniso_body_axes[b];
                    }

                    auto aniso = ShortRangeInteraction::anisotropic_repulsion(
                        pos_a, pos_b, axis_a_lab, axis_b_lab, aniso_params);

                    // Apply taper if active
                    if (m_short_range_taper.is_valid()) {
                        auto sw = evaluate_cutoff_spline(r, m_short_range_taper);
                        if (sw.value > 0.0) {
                            // Taper: E_tapered = f(r) * E_aniso
                            // F_tapered = f(r)*F_aniso + f'(r)*(E_aniso/r)*r_ab  (chain rule)
                            const double r_inv = 1.0 / r;
                            const Vec3 taper_force_correction =
                                sw.first_derivative * aniso.energy * r_inv * r_ab;
                            energy += weight * sw.value * aniso.energy;

                            Vec3 aniso_fa = sw.value * aniso.force_A - taper_force_correction;
                            Vec3 aniso_fb = sw.value * aniso.force_B + taper_force_correction;
                            force_i += weight * aniso_fa;
                            force_j += weight * aniso_fb;

                            // Lever-arm torque from aniso forces
                            torque_i -= weight * lever_a.cross(aniso_fa);
                            torque_j -= weight * lever_b.cross(aniso_fb);

                            // Axis rotation torque
                            torque_i += weight * sw.value * aniso.torque_axis_A;
                            torque_j += weight * sw.value * aniso.torque_axis_B;
                        }
                    } else {
                        energy += weight * aniso.energy;
                        force_i += weight * aniso.force_A;
                        force_j += weight * aniso.force_B;

                        // Lever-arm torque from aniso forces
                        torque_i -= weight * lever_a.cross(aniso.force_A);
                        torque_j -= weight * lever_b.cross(aniso.force_B);

                        // Axis rotation torque (already dE/dψ_lab form)
                        torque_i += weight * aniso.torque_axis_A;
                        torque_j += weight * aniso.torque_axis_B;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Main Energy Computation
// ============================================================================

CrystalEnergyResult CrystalEnergy::compute(const std::vector<MoleculeState>& molecules) {
    occ::timing::StopWatch<6> sw;
    // 0: cache, 1: set_orientation, 2: cartesian(), 3: elec pairs, 4: SR pairs, 5: ewald

    const int N = static_cast<int>(molecules.size());
    CrystalEnergyResult result;
    result.forces.resize(N, Vec3::Zero());
    result.torques.resize(N, Vec3::Zero());
    result.molecule_energies.resize(N, 0.0);

    sw.start(0);
    // Precompute rotation matrices and lab-frame atom positions once
    std::vector<MoleculeCache> mol_cache(N);
    for (int i = 0; i < N; ++i) {
        mol_cache[i].rotation = molecules[i].rotation_matrix();
        if (!m_geometry.empty() && i < static_cast<int>(m_geometry.size())) {
            const auto& geom = m_geometry[i];
            mol_cache[i].lab_atom_positions.resize(geom.atom_positions.size());
            for (size_t a = 0; a < geom.atom_positions.size(); ++a) {
                mol_cache[i].lab_atom_positions[a] =
                    molecules[i].position + mol_cache[i].rotation * geom.atom_positions[a];
            }
        }
    }
    sw.stop(0);

    sw.start(1);
    // Update multipole orientations (modifies m_multipoles in place)
    for (int i = 0; i < N; ++i) {
        m_multipoles[i].set_orientation(
            mol_cache[i].rotation,
            molecules[i].position);
    }
    sw.stop(1);

    sw.start(2);
    // Pre-build all CartesianMolecules ONCE (expensive conversion happens here)
    std::vector<CartesianMolecule> cart_mols;
    cart_mols.reserve(N);
    for (int i = 0; i < N; ++i) {
        cart_mols.push_back(m_multipoles[i].cartesian());
    }
    sw.stop(2);

    // Loop over neighbor pairs
    const double elec_com_cutoff = effective_electrostatic_com_cutoff();
    const double elec_site_cutoff = effective_electrostatic_site_cutoff();
    const CutoffSpline* elec_taper =
        m_electrostatic_taper.is_valid() ? &m_electrostatic_taper : nullptr;
    for (size_t pair_idx = 0; pair_idx < m_neighbors.size(); ++pair_idx) {
        const auto& pair = m_neighbors[pair_idx];
        int i = pair.mol_i;
        int j = pair.mol_j;

        // Translation for molecule j
        Vec3 cell_translation = m_crystal.unit_cell().to_cartesian(
            pair.cell_shift.cast<double>());
        const Vec3 disp_ij =
            molecules[j].position + cell_translation - molecules[i].position;

        // Electrostatic interaction.
        // COM gate: skip electrostatics for pairs with COM distance > cutoff.
        // This matches DMACRYS TBLCNT which selects molecule pairs by COM distance.
        // Ewald correction compensates for qq+qμ+μμ regardless of pair selection.
        bool include_elec = m_use_cartesian;
        if (include_elec && m_use_com_elec_gate &&
            pair.com_distance > elec_com_cutoff) {
            include_elec = false;
        }
        if (include_elec) {
            sw.start(3);
            // Pass cell_translation as offset_B to avoid copying CartesianMolecule
            auto elec_result = compute_molecule_forces_torques(
                cart_mols[i],
                cart_mols[j],
                elec_site_cutoff,
                m_max_interaction_order,
                cell_translation,
                elec_taper);
            sw.stop(3);

            double e_elec = pair.weight * elec_result.energy;
            result.electrostatic_energy += e_elec;
            result.molecule_energies[i] += e_elec * 0.5;
            result.molecule_energies[j] += e_elec * 0.5;

            result.forces[i] += pair.weight * elec_result.force_A;
            result.forces[j] += pair.weight * elec_result.force_B;
            result.torques[i] += pair.weight * elec_result.grad_angle_axis_A;
            result.torques[j] += pair.weight * elec_result.grad_angle_axis_B;
            add_pair_strain_gradient(
                result.strain_gradient, elec_result.force_A, disp_ij, pair.weight);
        }

        // Short-range interaction
        double sr_energy = 0.0;
        Vec3 sr_force_i = Vec3::Zero();
        Vec3 sr_force_j = Vec3::Zero();
        Vec3 sr_torque_i = Vec3::Zero();
        Vec3 sr_torque_j = Vec3::Zero();

        sw.start(4);
        compute_short_range_pair(
            i, j,
            molecules[i], molecules[j],
            cell_translation,
            pair.weight,
            sr_energy,
            sr_force_i, sr_force_j,
            sr_torque_i, sr_torque_j,
            static_cast<int>(pair_idx),
            &mol_cache[i], &mol_cache[j]);
        sw.stop(4);

        result.repulsion_dispersion += sr_energy;
        result.molecule_energies[i] += sr_energy * 0.5;
        result.molecule_energies[j] += sr_energy * 0.5;

        result.forces[i] += sr_force_i;
        result.forces[j] += sr_force_j;
        result.torques[i] += sr_torque_i;
        result.torques[j] += sr_torque_j;
        add_pair_strain_gradient(result.strain_gradient, sr_force_i, disp_ij, 1.0);
    }

    // Ewald correction for charge-charge electrostatics
    if (m_use_ewald) {
        sw.start(5);
        auto ewald = compute_charge_ewald_correction(molecules, cart_mols);
        sw.stop(5);
        result.electrostatic_energy += ewald.energy;
        for (int i = 0; i < N; ++i) {
            result.forces[i] += ewald.forces[i];
            result.torques[i] += ewald.torques[i];
        }

        // Chain-rule strain contribution from Ewald site-position dependence
        // through molecule COM strain (orientation fixed in this model).
        {
            const auto& B = voigt_basis_matrices();
            for (int m = 0; m < N; ++m) {
                for (int a = 0; a < 6; ++a) {
                    result.strain_gradient[a] +=
                        (-ewald.forces[m]).dot(B[a] * molecules[m].position);
                }
            }
        }

        // Add explicit Ewald lattice/cell contribution to dE/dE at fixed
        // Cartesian site coordinates (image translations, reciprocal lattice,
        // and 1/V prefactor terms).
        auto ewald_sites = gather_ewald_sites(cart_mols, m_ewald_dipole);
        if (!ewald_sites.empty()) {
            auto mol_site_indices = build_mol_site_indices(cart_mols);
            EwaldParams params;
            ensure_ewald_params_initialized();
            params.alpha = m_ewald_alpha_fixed;
            params.kmax = m_ewald_kmax_fixed;
            params.include_dipole = m_ewald_dipole;
            if (!m_ewald_lattice_cache) {
                m_ewald_lattice_cache = std::make_unique<EwaldLatticeCache>(
                    build_ewald_lattice_cache(m_crystal.unit_cell(), params));
            }
            // Full Ewald strain derivatives in the rigid-body strain model:
            // site positions follow molecule COM affine strain (orientation fixed),
            // plus explicit lattice dependence (image vectors, reciprocal lattice, 1/V).
            auto ewald_strain = compute_ewald_explicit_strain_terms(
                ewald_sites, m_crystal.unit_cell(), m_neighbors, mol_site_indices,
                effective_electrostatic_com_cutoff(), m_use_com_elec_gate,
                elec_site_cutoff,
                params,
                m_electrostatic_taper.is_valid() ? &m_electrostatic_taper : nullptr,
                m_ewald_lattice_cache.get(),
                false);
            result.strain_gradient += ewald_strain.grad;
        }
    }

    result.total_energy = result.electrostatic_energy + result.repulsion_dispersion;

    occ::log::debug("compute() timing: cache={:.3f}ms orient={:.3f}ms "
                     "cartesian={:.3f}ms elec={:.3f}ms SR={:.3f}ms ewald={:.3f}ms",
                     sw.read(0)*1e3, sw.read(1)*1e3, sw.read(2)*1e3,
                     sw.read(3)*1e3, sw.read(4)*1e3, sw.read(5)*1e3);

    return result;
}

double CrystalEnergy::compute_energy(const std::vector<MoleculeState>& molecules) {
    // Simplified version without gradient computation
    // For now, just call compute() and return energy
    // Could be optimized later to skip gradient calculation
    return compute(molecules).total_energy;
}

// ============================================================================
// Hessian Computation - Analytical pairwise assembly
// ============================================================================

CrystalEnergyResultWithHessian CrystalEnergy::compute_with_hessian(
    const std::vector<MoleculeState>& molecules) {

    occ::timing::StopWatch<6> sw;
    // 0: cache/orient, 1: elec hessian pairs, 2: SR hessian pairs, 3: ewald, 4: ewald strain, 5: total

    sw.start(5);
    const int N = static_cast<int>(molecules.size());
    const int ndof = 6 * N;  // 3 translation + 3 rotation per molecule

    CrystalEnergyResultWithHessian result;
    result.forces.resize(N, Vec3::Zero());
    result.torques.resize(N, Vec3::Zero());
    result.molecule_energies.resize(N, 0.0);
    result.includes_ewald_terms = false;

    // Initialize Hessian (6 DOF per molecule: pos + lab-frame rotation increment)
    result.hessian = Mat::Zero(ndof, ndof);
    result.strain_hessian.setZero();
    result.strain_state_hessian = Mat::Zero(6, ndof);

    // Precompute rotation matrices and lab-frame atom positions
    sw.start(0);
    std::vector<MoleculeCache> mol_cache(N);
    for (int i = 0; i < N; ++i) {
        mol_cache[i].rotation = molecules[i].rotation_matrix();
        if (!m_geometry.empty() && i < static_cast<int>(m_geometry.size())) {
            const auto& geom = m_geometry[i];
            mol_cache[i].lab_atom_positions.resize(geom.atom_positions.size());
            for (size_t a = 0; a < geom.atom_positions.size(); ++a) {
                mol_cache[i].lab_atom_positions[a] =
                    molecules[i].position + mol_cache[i].rotation * geom.atom_positions[a];
            }
        }
    }

    // Update multipole orientations and build Cartesian molecules once.
    for (int i = 0; i < N; ++i) {
        m_multipoles[i].set_orientation(mol_cache[i].rotation, molecules[i].position);
    }
    std::vector<CartesianMolecule> cart_mols;
    cart_mols.reserve(N);
    for (int i = 0; i < N; ++i) {
        cart_mols.push_back(m_multipoles[i].cartesian());
    }

    // Precompute per-molecule Hessian data (rotation derivatives)
    // once per molecule instead of once per pair.
    std::vector<MoleculeHessianData> hess_data_A(N), hess_data_B(N);
    for (int i = 0; i < N; ++i) {
        hess_data_A[i] = build_molecule_hessian_data(cart_mols[i], true);
        hess_data_B[i] = build_molecule_hessian_data(cart_mols[i], false);
    }
    sw.stop(0);

    const double elec_com_cutoff = effective_electrostatic_com_cutoff();
    const double elec_site_cutoff = effective_electrostatic_site_cutoff();
    const CutoffSpline* elec_taper =
        m_electrostatic_taper.is_valid() ? &m_electrostatic_taper : nullptr;
    const double buck_cutoff = effective_buckingham_site_cutoff();

    for (size_t pair_idx = 0; pair_idx < m_neighbors.size(); ++pair_idx) {
        const auto& pair = m_neighbors[pair_idx];
        const int i = pair.mol_i;
        const int j = pair.mol_j;
        const Vec3 cell_translation = m_crystal.unit_cell().to_cartesian(
            pair.cell_shift.cast<double>());
        const Vec3 pos_i = molecules[i].position;
        const Vec3 pos_j_image = molecules[j].position + cell_translation;

        // Electrostatic pair Hessian (full Cartesian multipole rigid-body terms).
        bool include_elec = m_use_cartesian;
        if (include_elec && m_use_com_elec_gate && pair.com_distance > elec_com_cutoff) {
            include_elec = false;
        }
        if (include_elec) {
            sw.start(1);
            auto epair = compute_molecule_hessian_truncated(
                cart_mols[i], cart_mols[j],
                hess_data_A[i], hess_data_B[j],
                cell_translation,
                elec_site_cutoff,
                m_max_interaction_order,
                elec_taper,
                m_elec_taper_hessian);
            accumulate_pair_hessian_blocks(
                result.hessian, i, j, epair, pair.weight);
            accumulate_pair_strain_hessian_blocks(
                result.strain_hessian, result.strain_state_hessian,
                i, j, epair, pair.weight, pos_i, pos_j_image);

            // Accumulate energy and gradient from Hessian result
            double e_elec = pair.weight * epair.energy;
            result.electrostatic_energy += e_elec;
            result.molecule_energies[i] += e_elec * 0.5;
            result.molecule_energies[j] += e_elec * 0.5;
            result.forces[i] += pair.weight * epair.force_A;
            result.forces[j] += pair.weight * epair.force_B;
            result.torques[i] += pair.weight * epair.grad_angle_axis_A;
            result.torques[j] += pair.weight * epair.grad_angle_axis_B;
            const Vec3 disp_ij = pos_j_image - pos_i;
            add_pair_strain_gradient(
                result.strain_gradient, epair.force_A, disp_ij, pair.weight);
            sw.stop(1);
        }

        // Short-range Hessian (full rigid-body mapping for central site potentials).
        if (m_force_field == ForceFieldType::None) {
            continue;
        }

        sw.start(2);
        const auto& geom_i = m_geometry[i];
        const auto& geom_j = m_geometry[j];
        const size_t nA = geom_i.atom_positions.size();
        const size_t nB = geom_j.atom_positions.size();

        const bool use_frozen = (pair_idx < m_fixed_site_masks.size() &&
                                 !m_fixed_site_masks[pair_idx].empty());

        const Mat3& R_i = mol_cache[i].rotation;
        const Mat3& R_j = mol_cache[j].rotation;

        for (size_t a = 0; a < nA; ++a) {
            const int Z_a = geom_i.atomic_numbers[a];
            const Vec3 pos_a = mol_cache[i].lab_atom_positions[a];

            for (size_t b = 0; b < nB; ++b) {
                if (use_frozen && !m_fixed_site_masks[pair_idx][a * nB + b]) {
                    continue;
                }

                const int Z_b = geom_j.atomic_numbers[b];
                const Vec3 pos_b = mol_cache[j].lab_atom_positions[b] + cell_translation;
                const Vec3 r_ab = pos_b - pos_a;
                const double r = r_ab.norm();

                if (!use_frozen) {
                    if (r > buck_cutoff || r < 0.1) {
                        continue;
                    }
                }

                ShortRangeInteraction::EnergyAndDerivatives sr;
                if (m_force_field == ForceFieldType::BuckinghamDE ||
                    m_force_field == ForceFieldType::Custom) {
                    BuckinghamParams params;
                    const bool has_type_codes =
                        m_ff.use_short_range_typing() &&
                        a < geom_i.short_range_type_codes.size() &&
                        b < geom_j.short_range_type_codes.size() &&
                        geom_i.short_range_type_codes[a] > 0 &&
                        geom_j.short_range_type_codes[b] > 0;
                    if (has_type_codes) {
                        const int t1 = geom_i.short_range_type_codes[a];
                        const int t2 = geom_j.short_range_type_codes[b];
                        if (has_typed_buckingham_params(t1, t2)) {
                            params = get_buckingham_params_for_types(t1, t2);
                        } else {
                            params = get_buckingham_params(Z_a, Z_b);
                        }
                    } else {
                        params = get_buckingham_params(Z_a, Z_b);
                    }
                    sr = ShortRangeInteraction::buckingham_all(r, params);
                } else if (m_force_field == ForceFieldType::LennardJones) {
                    auto it = m_ff.lj_params().find({Z_a, Z_b});
                    if (it == m_ff.lj_params().end()) {
                        continue;
                    }
                    sr = ShortRangeInteraction::lennard_jones_all(r, it->second);
                }

                if (m_short_range_taper.is_valid()) {
                    auto sw = evaluate_cutoff_spline(r, m_short_range_taper);
                    if (sw.value <= 0.0) {
                        continue;
                    }
                    const double e0 = sr.energy;
                    const double d10 = sr.first_derivative;
                    const double d20 = sr.second_derivative;
                    sr.energy = sw.value * e0;
                    sr.first_derivative = sw.value * d10 + e0 * sw.first_derivative;
                    sr.second_derivative = sw.value * d20 +
                                           2.0 * sw.first_derivative * d10 +
                                           e0 * sw.second_derivative;
                }

                // Accumulate SR energy and gradient
                {
                    result.repulsion_dispersion += pair.weight * sr.energy;
                    result.molecule_energies[i] += pair.weight * sr.energy * 0.5;
                    result.molecule_energies[j] += pair.weight * sr.energy * 0.5;

                    Vec3 force_on_a = ShortRangeInteraction::derivative_to_force(
                        sr.first_derivative, r_ab);
                    Vec3 wf = pair.weight * force_on_a;
                    result.forces[i] += wf;
                    result.forces[j] -= wf;

                    Vec3 lever_a = R_i * geom_i.atom_positions[a];
                    Vec3 lever_b = R_j * geom_j.atom_positions[b];
                    result.torques[i] -= pair.weight * lever_a.cross(force_on_a);
                    result.torques[j] -= pair.weight * lever_b.cross(-force_on_a);

                    const Vec3 disp_ij = pos_j_image - pos_i;
                    add_pair_strain_gradient(
                        result.strain_gradient, wf, disp_ij, 1.0);
                }

                auto spair = short_range_site_pair_hessian(
                    R_i, R_j,
                    geom_i.atom_positions[a], geom_j.atom_positions[b],
                    pos_a, pos_b,
                    sr.first_derivative, sr.second_derivative);

                accumulate_pair_hessian_blocks(
                    result.hessian, i, j, spair, pair.weight);
                accumulate_pair_strain_hessian_blocks(
                    result.strain_hessian, result.strain_state_hessian,
                    i, j, spair, pair.weight, pos_i, pos_j_image);

                // Anisotropic repulsion Hessian (FD of energy for initial impl)
                if (m_ff.has_any_aniso()) {
                    const bool htc =
                        m_ff.use_short_range_typing() &&
                        a < geom_i.short_range_type_codes.size() &&
                        b < geom_j.short_range_type_codes.size() &&
                        geom_i.short_range_type_codes[a] > 0 &&
                        geom_j.short_range_type_codes[b] > 0;
                    if (htc) {
                        const int t1 = geom_i.short_range_type_codes[a];
                        const int t2 = geom_j.short_range_type_codes[b];
                        if (has_aniso_params(t1, t2)) {
                            const auto ap = get_aniso_params(t1, t2);
                            Vec3 ax_a_lab = Vec3::Zero();
                            Vec3 ax_b_lab = Vec3::Zero();
                            if (a < geom_i.aniso_body_axes.size())
                                ax_a_lab = R_i * geom_i.aniso_body_axes[a];
                            if (b < geom_j.aniso_body_axes.size())
                                ax_b_lab = R_j * geom_j.aniso_body_axes[b];

                            // FD Hessian of aniso energy wrt site positions
                            constexpr double h = 1e-5;
                            auto aniso_e = [&](const Vec3& pa, const Vec3& pb) {
                                auto res = ShortRangeInteraction::anisotropic_repulsion(
                                    pa, pb, ax_a_lab, ax_b_lab, ap);
                                if (m_short_range_taper.is_valid()) {
                                    double rr = (pb - pa).norm();
                                    auto sw = evaluate_cutoff_spline(rr, m_short_range_taper);
                                    return sw.value * res.energy;
                                }
                                return res.energy;
                            };

                            // Build 6x6 Hessian [posA(3), posB(3)]
                            Eigen::Matrix<double, 6, 6> H_aniso;
                            H_aniso.setZero();
                            Vec3 pa0 = pos_a, pb0 = pos_b;
                            for (int di = 0; di < 6; ++di) {
                                for (int dj = di; dj < 6; ++dj) {
                                    Vec3 pa_pp = pa0, pa_pm = pa0, pa_mp = pa0, pa_mm = pa0;
                                    Vec3 pb_pp = pb0, pb_pm = pb0, pb_mp = pb0, pb_mm = pb0;
                                    auto perturb = [&](Vec3& pa, Vec3& pb, int idx, double delta) {
                                        if (idx < 3) pa(idx) += delta;
                                        else pb(idx - 3) += delta;
                                    };
                                    perturb(pa_pp, pb_pp, di, +h);
                                    perturb(pa_pp, pb_pp, dj, +h);
                                    perturb(pa_pm, pb_pm, di, +h);
                                    perturb(pa_pm, pb_pm, dj, -h);
                                    perturb(pa_mp, pb_mp, di, -h);
                                    perturb(pa_mp, pb_mp, dj, +h);
                                    perturb(pa_mm, pb_mm, di, -h);
                                    perturb(pa_mm, pb_mm, dj, -h);
                                    double d2 = (aniso_e(pa_pp, pb_pp) - aniso_e(pa_pm, pb_pm)
                                                 - aniso_e(pa_mp, pb_mp) + aniso_e(pa_mm, pb_mm))
                                                / (4.0 * h * h);
                                    H_aniso(di, dj) = d2;
                                    H_aniso(dj, di) = d2;
                                }
                            }

                            // Map to rigid-body Hessian
                            PairHessianResult apair;
                            apair.H_posA_posA = H_aniso.block<3,3>(0, 0);
                            apair.H_posA_posB = H_aniso.block<3,3>(0, 3);
                            apair.H_posB_posB = H_aniso.block<3,3>(3, 3);

                            const Vec3 lever_a_h = R_i * geom_i.atom_positions[a];
                            const Vec3 lever_b_h = R_j * geom_j.atom_positions[b];
                            const Mat3 Jpsi_a = -skew_symmetric(lever_a_h);
                            const Mat3 Jpsi_b = -skew_symmetric(lever_b_h);

                            apair.H_posA_rotA = apair.H_posA_posA * Jpsi_a;
                            apair.H_posA_rotB = apair.H_posA_posB * Jpsi_b;
                            apair.H_posB_rotA = apair.H_posA_posB.transpose() * Jpsi_a;
                            apair.H_posB_rotB = apair.H_posB_posB * Jpsi_b;

                            apair.H_rotA_rotA = Jpsi_a.transpose() * apair.H_posA_posA * Jpsi_a;
                            apair.H_rotA_rotB = Jpsi_a.transpose() * apair.H_posA_posB * Jpsi_b;
                            apair.H_rotB_rotB = Jpsi_b.transpose() * apair.H_posB_posB * Jpsi_b;

                            // Exponential-map curvature terms
                            auto aniso_ref = ShortRangeInteraction::anisotropic_repulsion(
                                pos_a, pos_b, ax_a_lab, ax_b_lab, ap);
                            double e_aniso = aniso_ref.energy;

                            // Accumulate aniso energy and gradient
                            {
                                const Vec3 lever_a_an = R_i * geom_i.atom_positions[a];
                                const Vec3 lever_b_an = R_j * geom_j.atom_positions[b];
                                if (m_short_range_taper.is_valid()) {
                                    auto sw = evaluate_cutoff_spline(r, m_short_range_taper);
                                    if (sw.value > 0.0) {
                                        const double r_inv = 1.0 / r;
                                        const Vec3 taper_force_correction =
                                            sw.first_derivative * aniso_ref.energy * r_inv * r_ab;
                                        result.repulsion_dispersion += pair.weight * sw.value * aniso_ref.energy;
                                        result.molecule_energies[i] += pair.weight * sw.value * aniso_ref.energy * 0.5;
                                        result.molecule_energies[j] += pair.weight * sw.value * aniso_ref.energy * 0.5;

                                        Vec3 aniso_fa = sw.value * aniso_ref.force_A - taper_force_correction;
                                        Vec3 aniso_fb = sw.value * aniso_ref.force_B + taper_force_correction;
                                        result.forces[i] += pair.weight * aniso_fa;
                                        result.forces[j] += pair.weight * aniso_fb;
                                        result.torques[i] -= pair.weight * lever_a_an.cross(aniso_fa);
                                        result.torques[j] -= pair.weight * lever_b_an.cross(aniso_fb);
                                        result.torques[i] += pair.weight * sw.value * aniso_ref.torque_axis_A;
                                        result.torques[j] += pair.weight * sw.value * aniso_ref.torque_axis_B;

                                        const Vec3 disp_ij = pos_j_image - pos_i;
                                        add_pair_strain_gradient(
                                            result.strain_gradient,
                                            pair.weight * aniso_fa, disp_ij, 1.0);
                                    }
                                } else {
                                    result.repulsion_dispersion += pair.weight * aniso_ref.energy;
                                    result.molecule_energies[i] += pair.weight * aniso_ref.energy * 0.5;
                                    result.molecule_energies[j] += pair.weight * aniso_ref.energy * 0.5;
                                    result.forces[i] += pair.weight * aniso_ref.force_A;
                                    result.forces[j] += pair.weight * aniso_ref.force_B;
                                    result.torques[i] -= pair.weight * lever_a_an.cross(aniso_ref.force_A);
                                    result.torques[j] -= pair.weight * lever_b_an.cross(aniso_ref.force_B);
                                    result.torques[i] += pair.weight * aniso_ref.torque_axis_A;
                                    result.torques[j] += pair.weight * aniso_ref.torque_axis_B;

                                    const Vec3 disp_ij = pos_j_image - pos_i;
                                    add_pair_strain_gradient(
                                        result.strain_gradient,
                                        pair.weight * aniso_ref.force_A, disp_ij, 1.0);
                                }
                            }

                            if (m_short_range_taper.is_valid()) {
                                auto sw = evaluate_cutoff_spline(r, m_short_range_taper);
                                e_aniso *= sw.value;
                            }
                            const Vec3 gA = -aniso_ref.force_A;  // gradient
                            const Vec3 gB = -aniso_ref.force_B;
                            const auto& Ggen = so3_generators();
                            for (int k = 0; k < 3; ++k) {
                                for (int l = 0; l < 3; ++l) {
                                    const Vec3 d2xa = 0.5 * (Ggen[k]*Ggen[l] + Ggen[l]*Ggen[k]) * lever_a_h;
                                    const Vec3 d2xb = 0.5 * (Ggen[k]*Ggen[l] + Ggen[l]*Ggen[k]) * lever_b_h;
                                    apair.H_rotA_rotA(k, l) += gA.dot(d2xa);
                                    apair.H_rotB_rotB(k, l) += gB.dot(d2xb);
                                }
                            }

                            // TODO: axis rotation Hessian terms (aniso.torque_axis)
                            // For now, the FD position Hessian captures the dominant effect.

                            accumulate_pair_hessian_blocks(
                                result.hessian, i, j, apair, pair.weight);
                            accumulate_pair_strain_hessian_blocks(
                                result.strain_hessian, result.strain_state_hessian,
                                i, j, apair, pair.weight, pos_i, pos_j_image);
                        }
                    }
                }
            }
        }
        sw.stop(2);
    }

    if (m_use_ewald) {
        sw.start(3);
        auto ewald_sites = gather_ewald_sites(cart_mols, m_ewald_dipole);
        auto mol_site_indices = build_mol_site_indices(cart_mols);

        if (!ewald_sites.empty()) {
            EwaldParams params;
            ensure_ewald_params_initialized();
            params.alpha = m_ewald_alpha_fixed;
            params.kmax = m_ewald_kmax_fixed;
            params.include_dipole = m_ewald_dipole;

            if (!m_ewald_lattice_cache) {
                m_ewald_lattice_cache = std::make_unique<EwaldLatticeCache>(
                    build_ewald_lattice_cache(m_crystal.unit_cell(), params));
            }

            auto ewald = compute_ewald_correction_with_hessian(
                ewald_sites, m_crystal.unit_cell(), m_neighbors,
                mol_site_indices, effective_electrostatic_com_cutoff(),
                m_use_com_elec_gate, elec_site_cutoff, params,
                m_electrostatic_taper.is_valid() ? &m_electrostatic_taper : nullptr,
                m_ewald_lattice_cache.get());

            // Accumulate Ewald energy and gradient (replaces compute() Ewald section)
            const int ns = static_cast<int>(ewald_sites.size());
            result.electrostatic_energy += ewald.energy;
            for (int s = 0; s < ns; ++s) {
                const int m = ewald_sites[s].mol_index;
                const Vec3& f_kJ = ewald.site_forces[s];
                result.forces[m] += f_kJ;
                Vec3 lever = ewald_sites[s].position - molecules[m].position;
                Vec3 torque_lab = lever.cross(f_kJ);
                result.torques[m] -= torque_lab;
            }
            // Ewald strain: site-position dependence through COM strain
            {
                const auto& B_mats = voigt_basis_matrices();
                for (int s = 0; s < ns; ++s) {
                    const int m = ewald_sites[s].mol_index;
                    for (int a = 0; a < 6; ++a) {
                        result.strain_gradient[a] +=
                            (-ewald.site_forces[s]).dot(B_mats[a] * molecules[m].position);
                    }
                }
            }
            // Explicit Ewald lattice/cell strain gradient
            auto ewald_strain_grad = compute_ewald_explicit_strain_terms(
                ewald_sites, m_crystal.unit_cell(), m_neighbors, mol_site_indices,
                effective_electrostatic_com_cutoff(), m_use_com_elec_gate,
                elec_site_cutoff,
                params,
                m_electrostatic_taper.is_valid() ? &m_electrostatic_taper : nullptr,
                m_ewald_lattice_cache.get(),
                false);
            result.strain_gradient += ewald_strain_grad.grad;
            Mat J_state = Mat::Zero(3 * ns, ndof);
            Mat J_strain = Mat::Zero(3 * ns, 6);
            const auto& B = voigt_basis_matrices();

            for (int s = 0; s < ns; ++s) {
                const int m = ewald_sites[s].mol_index;
                const int row = 3 * s;
                const int col = 6 * m;
                J_state.block<3, 3>(row, col) = Mat3::Identity();

                const Vec3 lever = ewald_sites[s].position - molecules[m].position;
                const Mat3 J_rot = -skew_symmetric(lever);
                J_state.block<3, 3>(row, col + 3) = J_rot;

                // Cell strain acts on molecule COMs in this model; rigid internal
                // site offsets are not affinely deformed.
                for (int a = 0; a < 6; ++a) {
                    J_strain.block<3, 1>(row, a) = B[a] * molecules[m].position;
                }
            }

            result.hessian += J_state.transpose() * ewald.site_hessian * J_state;
            result.strain_state_hessian +=
                J_strain.transpose() * ewald.site_hessian * J_state;
            result.strain_hessian +=
                J_strain.transpose() * ewald.site_hessian * J_strain;

            auto ewald_strain = compute_ewald_explicit_strain_terms(
                ewald_sites, m_crystal.unit_cell(), m_neighbors, mol_site_indices,
                effective_electrostatic_com_cutoff(), m_use_com_elec_gate,
                elec_site_cutoff,
                params,
                m_electrostatic_taper.is_valid() ? &m_electrostatic_taper : nullptr,
                m_ewald_lattice_cache.get(),
                true);
            result.strain_hessian += ewald_strain.hess;
            if (ewald_strain.strain_site_mixed.rows() == 6 &&
                ewald_strain.strain_site_mixed.cols() == J_state.rows()) {
                result.strain_state_hessian +=
                    ewald_strain.strain_site_mixed * J_state;
                // Mixed explicit Ewald projection into d2E/dE^2.
                // strain_site_mixed stores d^2E / (dE_a dx_site), so projecting
                // through x_site(E) contributes M*J + (M*J)^T to W_ee.
                const Mat6 mixed_ee =
                    ewald_strain.strain_site_mixed * J_strain;
                result.strain_hessian += mixed_ee + mixed_ee.transpose();
            }

            // Exponential-map curvature term for rotation coordinates.
            const auto& G = so3_generators();
            for (int s = 0; s < ns; ++s) {
                const int m = ewald_sites[s].mol_index;
                const int rot = 6 * m + 3;
                const Vec3 lever = ewald_sites[s].position - molecules[m].position;
                const Vec3 g_site = -ewald.site_forces[s]; // gradient wrt site position
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        const Vec3 d2x =
                            0.5 * (G[k] * G[l] + G[l] * G[k]) * lever;
                        result.hessian(rot + k, rot + l) += g_site.dot(d2x);
                    }
                }
            }

            result.includes_ewald_terms = true;
        }
        sw.stop(3);
    }

    result.total_energy = result.electrostatic_energy + result.repulsion_dispersion;
    result.exact_for_model = can_compute_exact_hessian();

    // Symmetrize Hessian (handles numerical noise)
    result.hessian = 0.5 * (result.hessian + result.hessian.transpose());
    result.strain_hessian =
        0.5 * (result.strain_hessian + result.strain_hessian.transpose());

    sw.stop(5);
    occ::log::info("compute_with_hessian() timing: "
                   "cache={:.1f}ms elec_hess={:.1f}ms SR_hess={:.1f}ms "
                   "ewald={:.1f}ms total={:.1f}ms  ({} pairs)",
                   sw.read(0)*1e3, sw.read(1)*1e3, sw.read(2)*1e3,
                   sw.read(3)*1e3, sw.read(5)*1e3,
                   m_neighbors.size());

    return result;
}

bool CrystalEnergy::can_compute_exact_hessian() const {
    // The full chain rule taper Hessian is the correct d²(f·E)/dq²,
    // validated against finite differences (analytic W_ei matches FD
    // to ~0.2%). Always use analytic W_ei.
    return true;
}

// ============================================================================
// Pack Hessian for Reduced DOF
// ============================================================================

Mat CrystalEnergyResultWithHessian::pack_hessian(
    bool fix_first_translation, bool fix_first_rotation) const {

    int N = static_cast<int>(forces.size());
    int ndof_full = 6 * N;

    // Count reduced DOF
    int mol0_dof = 0;
    if (!fix_first_translation) mol0_dof += 3;
    if (!fix_first_rotation) mol0_dof += 3;
    int ndof_reduced = mol0_dof + 6 * (N - 1);

    if (ndof_reduced == ndof_full) {
        return hessian;
    }

    Mat H_reduced = Mat::Zero(ndof_reduced, ndof_reduced);

    // Build index mapping from reduced to full DOF
    std::vector<int> reduced_to_full;
    reduced_to_full.reserve(ndof_reduced);

    // Molecule 0
    if (!fix_first_translation) {
        reduced_to_full.push_back(0);
        reduced_to_full.push_back(1);
        reduced_to_full.push_back(2);
    }
    if (!fix_first_rotation) {
        reduced_to_full.push_back(3);
        reduced_to_full.push_back(4);
        reduced_to_full.push_back(5);
    }

    // Molecules 1..N-1
    for (int i = 1; i < N; ++i) {
        for (int c = 0; c < 6; ++c) {
            reduced_to_full.push_back(6 * i + c);
        }
    }

    // Extract submatrix
    for (int i = 0; i < ndof_reduced; ++i) {
        for (int j = 0; j < ndof_reduced; ++j) {
            H_reduced(i, j) = hessian(reduced_to_full[i], reduced_to_full[j]);
        }
    }

    return H_reduced;
}

int CrystalEnergy::max_multipole_rank() const {
    int max_rank = 0;
    for (const auto& mp : m_multipoles) {
        const auto& cart = mp.cartesian();
        for (const auto& site : cart.sites) {
            max_rank = std::max(max_rank, site.rank);
        }
    }
    return max_rank;
}

size_t CrystalEnergy::num_sites() const {
    size_t total = 0;
    for (const auto& mp : m_multipoles) {
        total += mp.cartesian().sites.size();
    }
    return total;
}

std::vector<PairEnergyDebug> CrystalEnergy::debug_pair_energies(const std::vector<MoleculeState>& molecules) {
    const int N = static_cast<int>(molecules.size());
    std::vector<PairEnergyDebug> out;
    out.reserve(m_neighbors.size());

    // Update multipole orientations
    for (int i = 0; i < N; ++i) {
        m_multipoles[i].set_orientation(
            molecules[i].rotation_matrix(),
            molecules[i].position);
    }

    // Pre-build Cartesian molecules
    std::vector<CartesianMolecule> cart_mols;
    cart_mols.reserve(N);
    for (int i = 0; i < N; ++i) {
        cart_mols.push_back(m_multipoles[i].cartesian());
    }

    for (const auto& pair : m_neighbors) {
        int i = pair.mol_i;
        int j = pair.mol_j;

        Vec3 cell_translation = m_crystal.unit_cell().to_cartesian(
            pair.cell_shift.cast<double>());

        PairEnergyDebug dbg;
        dbg.mol_i = i;
        dbg.mol_j = j;
        dbg.cell_shift = pair.cell_shift;
        dbg.weight = pair.weight;

        Vec3 com_i = molecules[i].position;
        Vec3 com_j = molecules[j].position + cell_translation;
        dbg.com_distance = (com_j - com_i).norm();

        bool include_elec = m_use_cartesian;
        if (include_elec && m_use_com_elec_gate &&
            pair.com_distance > effective_electrostatic_com_cutoff()) {
            include_elec = false;
        }
        if (include_elec) {
            auto elec = compute_molecule_forces_torques(
                cart_mols[i], cart_mols[j],
                effective_electrostatic_site_cutoff(),
                m_max_interaction_order,
                cell_translation,
                m_electrostatic_taper.is_valid() ? &m_electrostatic_taper : nullptr);
            dbg.electrostatic = pair.weight * elec.energy;
            dbg.total += dbg.electrostatic;
        }

        if (m_force_field != ForceFieldType::None) {
            double sr_energy = 0.0;
            int sr_pairs = 0;
            Vec3 dummyF, dummyF2, dummyT, dummyT2;
            compute_short_range_pair(i, j, molecules[i], molecules[j], cell_translation,
                                     pair.weight, sr_energy, dummyF, dummyF2, dummyT, dummyT2,
                                     -1, nullptr, nullptr, &sr_pairs);
            dbg.short_range = sr_energy;
            dbg.short_range_site_pairs = sr_pairs;
            dbg.total += sr_energy;
        }

        out.push_back(dbg);
    }

    return out;
}

std::vector<int> CrystalEnergy::neighbor_shell_histogram() const {
    std::vector<int> bins(5, 0);
    for (const auto& pair : m_neighbors) {
        Vec3 shift = m_crystal.unit_cell().to_cartesian(pair.cell_shift.cast<double>());
        Vec3 com_i = m_crystal.symmetry_unique_molecules()[pair.mol_i].center_of_mass();
        Vec3 com_j = m_crystal.symmetry_unique_molecules()[pair.mol_j].center_of_mass() + shift;
        double d = (com_j - com_i).norm();
        if (d < 3.0) bins[0]++; else if (d < 6.0) bins[1]++; else if (d < 10.0) bins[2]++; else if (d < 15.0) bins[3]++; else bins[4]++;
    }
    return bins;
}

// ============================================================================
// Charge-Charge Ewald Correction (thin wrapper around standalone engine)
// ============================================================================

CrystalEnergy::EwaldCorrectionResult CrystalEnergy::compute_charge_ewald_correction(
    const std::vector<MoleculeState>& molecules,
    const std::vector<CartesianMolecule>& cart_mols) const {

    const int N = static_cast<int>(molecules.size());
    EwaldCorrectionResult ewald_result;
    ewald_result.forces.resize(N, Vec3::Zero());
    ewald_result.torques.resize(N, Vec3::Zero());

    // Gather sites and indices from CartesianMolecules
    auto ewald_sites = gather_ewald_sites(cart_mols, m_ewald_dipole);
    auto mol_site_indices = build_mol_site_indices(cart_mols);

    if (ewald_sites.empty()) return ewald_result;

    // Select Ewald parameters
    EwaldParams params;
    ensure_ewald_params_initialized();
    params.alpha = m_ewald_alpha_fixed;
    params.kmax = m_ewald_kmax_fixed;
    double elec_site_cutoff = effective_electrostatic_site_cutoff();
    params.include_dipole = m_ewald_dipole;

    // Lazy-build lattice cache on first call (reused across evaluations
    // while the unit cell and Ewald params don't change).
    if (!m_ewald_lattice_cache) {
        m_ewald_lattice_cache = std::make_unique<EwaldLatticeCache>(
            build_ewald_lattice_cache(m_crystal.unit_cell(), params));
    }

    // Call standalone Ewald engine with cached lattice
    auto raw = compute_ewald_correction(
        ewald_sites, m_crystal.unit_cell(), m_neighbors,
        mol_site_indices, effective_electrostatic_com_cutoff(),
        m_use_com_elec_gate, elec_site_cutoff, params,
        m_electrostatic_taper.is_valid() ? &m_electrostatic_taper : nullptr,
        m_ewald_lattice_cache.get());

    ewald_result.energy = raw.energy;

    // Map per-site forces to rigid-body forces and torques
    for (size_t k = 0; k < ewald_sites.size(); ++k) {
        int m = ewald_sites[k].mol_index;
        const Vec3& f_kJ = raw.site_forces[k];

        ewald_result.forces[m] += f_kJ;

        // Lab-frame angular gradient from lever arm:
        // torque_lab = lever × force = -(lever × dE/dr) = -dE/dψ_lab
        // So -torque_lab = +dE/dψ_lab (consistent with grad_angle_axis convention)
        Vec3 lever = ewald_sites[k].position - molecules[m].position;
        Vec3 torque_lab = lever.cross(f_kJ);
        ewald_result.torques[m] -= torque_lab;
    }

    return ewald_result;
}

} // namespace occ::mults
