#include <occ/mults/crystal_optimizer.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/element.h>
#include <occ/core/units.h>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <set>

namespace occ::mults {

namespace {

Mat3 skew_symmetric(const Vec3& v) {
    Mat3 S;
    S <<      0, -v[2],  v[1],
           v[2],     0, -v[0],
          -v[1],  v[0],     0;
    return S;
}

Mat3 so3_left_jacobian_transpose(const Vec3& theta) {
    const double phi2 = theta.squaredNorm();
    const Mat3 I = Mat3::Identity();

    if (phi2 < 1e-16) {
        // J_L^T(theta) = I - 1/2 [theta]x + 1/6 [theta]x^2 + O(|theta|^3)
        const Mat3 K = skew_symmetric(theta);
        return I - 0.5 * K + (1.0 / 6.0) * (K * K);
    }

    const double phi = std::sqrt(phi2);
    const double a = (1.0 - std::cos(phi)) / phi2;
    const double b = (phi - std::sin(phi)) / (phi2 * phi);

    const Mat3 K = skew_symmetric(theta);
    return I - a * K + b * (K * K);
}

std::array<Mat3, 3> so3_left_jacobian_transpose_derivatives(const Vec3& theta) {
    std::array<Mat3, 3> dJ{};
    const Mat3 K = skew_symmetric(theta);
    const Mat3 K2 = K * K;

    std::array<Mat3, 3> E;
    E[0] = skew_symmetric(Vec3(1.0, 0.0, 0.0));
    E[1] = skew_symmetric(Vec3(0.0, 1.0, 0.0));
    E[2] = skew_symmetric(Vec3(0.0, 0.0, 1.0));

    const double phi2 = theta.squaredNorm();
    const double phi = std::sqrt(phi2);

    double a = 0.0;
    double b = 0.0;
    double da_dphi = 0.0;
    double db_dphi = 0.0;

    if (phi < 1e-6) {
        const double phi4 = phi2 * phi2;
        a = 0.5 - phi2 / 24.0 + phi4 / 720.0;
        b = 1.0 / 6.0 - phi2 / 120.0 + phi4 / 5040.0;
        da_dphi = -phi / 12.0 + (phi * phi2) / 180.0;
        db_dphi = -phi / 60.0 + (phi * phi2) / 1260.0;
    } else {
        a = (1.0 - std::cos(phi)) / phi2;
        b = (phi - std::sin(phi)) / (phi2 * phi);
        da_dphi = std::sin(phi) / phi2 - 2.0 * (1.0 - std::cos(phi)) / (phi2 * phi);
        db_dphi = (1.0 - std::cos(phi)) / (phi2 * phi) -
                  3.0 * (phi - std::sin(phi)) / (phi2 * phi2);
    }

    for (int l = 0; l < 3; ++l) {
        const double dphi_dtheta = (phi > 1e-14) ? theta[l] / phi : 0.0;
        const double da = da_dphi * dphi_dtheta;
        const double db = db_dphi * dphi_dtheta;
        dJ[l] = -da * K - a * E[l] + db * K2 + b * (E[l] * K + K * E[l]);
    }

    return dJ;
}

Vec3 apply_angle_axis_chain_rule(const Vec3& angle_axis,
                                 const Vec3& grad_psi_lab) {
    return so3_left_jacobian_transpose(angle_axis) * grad_psi_lab;
}

Vec6 cell_strain_first_derivative(const Vec6& vars) {
    (void)vars;
    return Vec6::Ones();
}

Vec6 cell_strain_second_derivative(const Vec6& vars) {
    (void)vars;
    return Vec6::Zero();
}

Mat3 voigt_basis_matrix(int j) {
    Mat3 M = Mat3::Zero();
    switch (j) {
    case 0: M(0, 0) = 1.0; break;
    case 1: M(1, 1) = 1.0; break;
    case 2: M(2, 2) = 1.0; break;
    case 3: M(1, 2) = M(2, 1) = 0.5; break;
    case 4: M(0, 2) = M(2, 0) = 0.5; break;
    case 5: M(0, 1) = M(1, 0) = 0.5; break;
    default: break;
    }
    return M;
}

double rotation_distance(const Vec3& aa_a, const Vec3& aa_b) {
    Mat3 R_a = MoleculeState{Vec3::Zero(), aa_a}.rotation_matrix();
    Mat3 R_b = MoleculeState{Vec3::Zero(), aa_b}.rotation_matrix();
    Mat3 R_rel = R_a * R_b.transpose();
    double tr = (R_rel.trace() - 1.0) * 0.5;
    tr = std::max(-1.0, std::min(1.0, tr));
    return std::acos(tr);
}

int monoclinic_shear_component(const crystal::UnitCell& uc) {
    constexpr double right_angle = occ::units::PI * 0.5;
    constexpr double angle_tol = 1e-3; // radians (~0.057 deg)
    std::array<double, 3> dev{
        std::abs(uc.alpha() - right_angle),
        std::abs(uc.beta() - right_angle),
        std::abs(uc.gamma() - right_angle)};
    const int imax = static_cast<int>(
        std::distance(dev.begin(), std::max_element(dev.begin(), dev.end())));
    if (dev[imax] < angle_tol) {
        // Default conventional setting if the monoclinic angle is near 90.
        return 4; // E5 (beta / xz shear)
    }
    return 3 + imax; // alpha->E4, beta->E5, gamma->E6
}

std::string active_voigt_components(const Vec6& mask) {
    std::string out;
    for (int i = 0; i < 6; ++i) {
        if (mask[i] > 0.5) {
            if (!out.empty()) out += ", ";
            out += "E" + std::to_string(i + 1);
        }
    }
    return out.empty() ? "none" : out;
}

int count_active_components(const Vec6& mask) {
    int n = 0;
    for (int i = 0; i < 6; ++i) {
        if (mask[i] > 0.5) n++;
    }
    return n;
}

Vec6 strain_gradient_to_stress_gpa(const Vec6& dE_dE, double volume_ang3) {
    Vec6 sigma = Vec6::Zero();
    if (volume_ang3 <= 1e-12) {
        return sigma;
    }
    // E = [eps_xx, eps_yy, eps_zz, 2*eps_yz, 2*eps_xz, 2*eps_xy]
    // dE/dE_i is in kJ/mol per unit cell.
    // sigma_diag = (1/V) dE/dE_diag
    // sigma_shear = (2/V) dE/dE_shear
    sigma[0] = dE_dE[0] / volume_ang3;
    sigma[1] = dE_dE[1] / volume_ang3;
    sigma[2] = dE_dE[2] / volume_ang3;
    sigma[3] = 2.0 * dE_dE[3] / volume_ang3;
    sigma[4] = 2.0 * dE_dE[4] / volume_ang3;
    sigma[5] = 2.0 * dE_dE[5] / volume_ang3;
    sigma *= occ::units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;
    return sigma;
}

struct PressureStrainTerms {
    double energy = 0.0; // kJ/mol per unit cell
    Vec6 gradient = Vec6::Zero(); // d(pV)/dE_i
    Mat6 hessian = Mat6::Zero();  // d2(pV)/dE_i dE_j
};

PressureStrainTerms pressure_strain_terms(double pressure_gpa,
                                          const Mat3& F,
                                          double reference_volume_ang3) {
    PressureStrainTerms out;
    if (std::abs(pressure_gpa) < 1e-16 || reference_volume_ang3 <= 1e-16) {
        return out;
    }

    const double p_kj_per_mol_per_ang3 =
        pressure_gpa / occ::units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;

    const double detF = F.determinant();
    const double volume = reference_volume_ang3 * detF;
    out.energy = p_kj_per_mol_per_ang3 * volume;

    const Mat3 Finv = F.inverse();
    std::array<Mat3, 6> G;
    for (int i = 0; i < 6; ++i) {
        G[i] = voigt_basis_matrix(i);
    }

    std::array<double, 6> trFiG{};
    for (int i = 0; i < 6; ++i) {
        trFiG[i] = (Finv * G[i]).trace();
        out.gradient[i] = p_kj_per_mol_per_ang3 * volume * trFiG[i];
    }

    for (int i = 0; i < 6; ++i) {
        for (int j = i; j < 6; ++j) {
            const double tr_mix = (Finv * G[i] * Finv * G[j]).trace();
            const double d2V =
                volume * (trFiG[i] * trFiG[j] - tr_mix);
            const double val = p_kj_per_mol_per_ang3 * d2V;
            out.hessian(i, j) = val;
            out.hessian(j, i) = val;
        }
    }

    return out;
}

} // namespace

// ============================================================================
// SymmetryMapping
// ============================================================================

int SymmetryMapping::total_dof() const {
    int dof = 0;
    for (const auto& info : independent) {
        dof += info.trans_dof + info.rot_dof;
    }
    return dof;
}

// ============================================================================
// build_symmetry_mapping
// ============================================================================

SymmetryMapping build_symmetry_mapping(const crystal::Crystal& crystal) {
    SymmetryMapping mapping;

    const auto& uc_mols = crystal.unit_cell_molecules();
    const auto& unique_mols = crystal.symmetry_unique_molecules();
    const auto& symops = crystal.symmetry_operations();
    const auto& unit_cell = crystal.unit_cell();

    mapping.num_uc_molecules = static_cast<int>(uc_mols.size());
    mapping.num_independent = static_cast<int>(unique_mols.size());
    mapping.uc_molecules.resize(mapping.num_uc_molecules);
    mapping.independent.resize(mapping.num_independent);

    // Initialize independent molecule info
    for (int k = 0; k < mapping.num_independent; ++k) {
        mapping.independent[k].uc_indices.clear();
    }

    // For each UC molecule, find its independent molecule and store the
    // Cartesian rotation from the Crystal's asymmetric_unit_transformation.
    // The translation (t_cart) is set to zero here and will be recomputed
    // by the optimizer from the actual initial states.
    for (int j = 0; j < mapping.num_uc_molecules; ++j) {
        const auto& uc_mol = uc_mols[j];
        int asym_idx = uc_mol.asymmetric_molecule_idx();

        if (asym_idx < 0 || asym_idx >= mapping.num_independent) {
            occ::log::warn("UC molecule {} has invalid asymmetric_molecule_idx {}",
                          j, asym_idx);
            asym_idx = 0;
        }

        mapping.uc_molecules[j].independent_idx = asym_idx;
        mapping.independent[asym_idx].uc_indices.push_back(j);

        // Get the Cartesian rotation from Crystal
        auto [R_cart, t_cart] = uc_mol.asymmetric_unit_transformation();
        mapping.uc_molecules[j].R_cart = R_cart;
        mapping.uc_molecules[j].t_cart = t_cart;
    }

    // Convert to relative transformations (relative to first UC image of each
    // independent molecule). This makes the mapping independent of the
    // initial positions — the translation will be recomputed from actual states.
    for (int k = 0; k < mapping.num_independent; ++k) {
        const auto& indices = mapping.independent[k].uc_indices;
        if (indices.empty()) continue;

        int ref_uc = indices[0];
        Mat3 R_ref = mapping.uc_molecules[ref_uc].R_cart;
        Vec3 t_ref = mapping.uc_molecules[ref_uc].t_cart;

        for (int uc_idx : indices) {
            auto& uc_info = mapping.uc_molecules[uc_idx];
            // R_rel = R_j * R_ref^{-1} (R_ref is orthogonal, so R_ref^{-1} = R_ref^T)
            Mat3 R_rel = uc_info.R_cart * R_ref.transpose();
            // t_rel: we can compute from original Crystal centroids
            // uc_centroid = R_j * asym_centroid + t_j
            // ref_centroid = R_ref * asym_centroid + t_ref
            // => uc_centroid = R_rel * ref_centroid + (t_j - R_rel * t_ref)
            Vec3 t_rel = uc_info.t_cart - R_rel * t_ref;
            uc_info.R_cart = R_rel;
            uc_info.t_cart = t_rel;
        }
    }

    // For each independent molecule, detect site symmetry and compute DOF.
    //
    // The site symmetry group (stabilizer) consists of space group operations
    // that map the molecule to itself. By orbit-stabilizer theorem:
    //   |stabilizer| = |G| / orbit_size
    // where orbit_size = number of UC images for this independent molecule.
    //
    // For a general position (Z'=1, orbit_size = |G|), |stabilizer| = 1 → 6 free DOF.
    // For a special position, |stabilizer| > 1 → reduced DOF.

    int num_symops = static_cast<int>(symops.size());

    for (int k = 0; k < mapping.num_independent; ++k) {
        auto& info = mapping.independent[k];
        const auto& indices = info.uc_indices;

        if (indices.empty()) {
            occ::log::warn("Independent molecule {} has no UC images", k);
            continue;
        }

        int orbit_size = static_cast<int>(indices.size());
        int stabilizer_size = num_symops / std::max(1, orbit_size);

        if (stabilizer_size <= 1) {
            // General position: trivial stabilizer → full 6 DOF
            info.trans_projector = Mat3::Identity();
            info.rot_projector = Mat3::Identity();
            info.trans_dof = 3;
            info.rot_dof = 3;
            info.trans_basis.resize(3, 3);
            info.trans_basis.setIdentity();
            info.rot_basis.resize(3, 3);
            info.rot_basis.setIdentity();
            occ::log::debug("Independent mol {}: {} UC images, general position, 6 DOF",
                           k, orbit_size);
            continue;
        }

        // Special position: find the actual stabilizer symops.
        // These are the space group ops that map the reference molecule's
        // fractional centroid back to itself (modulo lattice translations).
        int ref_uc = indices[0];
        Vec3 ref_centroid_cart = uc_mols[ref_uc].centroid();
        Vec3 ref_centroid_frac = crystal.to_fractional(ref_centroid_cart);

        std::vector<Mat3> site_symop_carts;
        for (const auto& symop : symops) {
            Vec3 transformed = symop.apply(ref_centroid_frac);
            // Check if transformed == ref_centroid_frac (mod 1)
            Vec3 diff = transformed - ref_centroid_frac;
            diff -= diff.array().round().matrix();
            if (diff.norm() < 1e-4) {
                // This is a site symmetry operation
                Mat3 R_cart = symop.cartesian_rotation(unit_cell);
                site_symop_carts.push_back(R_cart);
            }
        }

        occ::log::debug("Independent mol {}: {} UC images, {} site symops",
                        k, orbit_size, site_symop_carts.size());

        // Compute allowed translation subspace:
        // For each site symop R_S, translations must satisfy R_S * t = t,
        // i.e., (R_S - I) * t = 0. The allowed subspace is the intersection
        // of null(R_S - I) for all non-identity site symops.
        Mat3 trans_projector = Mat3::Identity();
        for (const auto& R_S : site_symop_carts) {
            Mat3 A = R_S - Mat3::Identity();
            if (A.norm() < 1e-10) continue;  // Skip identity

            Eigen::JacobiSVD<Mat3> svd(A, Eigen::ComputeFullV);
            Vec3 sv = svd.singularValues();
            Mat3 null_proj = Mat3::Zero();
            for (int i = 0; i < 3; ++i) {
                if (std::abs(sv[i]) < 1e-8) {
                    Vec3 v = svd.matrixV().col(i);
                    null_proj += v * v.transpose();
                }
            }
            // Intersect with current allowed space
            trans_projector = null_proj * trans_projector;
            Eigen::SelfAdjointEigenSolver<Mat3> eig(trans_projector);
            Mat3 new_proj = Mat3::Zero();
            for (int i = 0; i < 3; ++i) {
                if (eig.eigenvalues()[i] > 0.5) {
                    Vec3 v = eig.eigenvectors().col(i);
                    new_proj += v * v.transpose();
                }
            }
            trans_projector = new_proj;
        }

        // Compute allowed rotation subspace:
        // For pseudovector omega under R_S with det(R_S):
        //   det(R_S) * R_S * omega = omega
        // Constraint: (det(R_S) * R_S - I) * omega = 0
        Mat3 rot_projector = Mat3::Identity();
        for (const auto& R_S : site_symop_carts) {
            double det = R_S.determinant();
            Mat3 A = det * R_S - Mat3::Identity();
            if (A.norm() < 1e-10) continue;  // Skip identity

            Eigen::JacobiSVD<Mat3> svd(A, Eigen::ComputeFullV);
            Vec3 sv = svd.singularValues();
            Mat3 null_proj = Mat3::Zero();
            for (int i = 0; i < 3; ++i) {
                if (std::abs(sv[i]) < 1e-8) {
                    Vec3 v = svd.matrixV().col(i);
                    null_proj += v * v.transpose();
                }
            }
            rot_projector = null_proj * rot_projector;
            Eigen::SelfAdjointEigenSolver<Mat3> eig(rot_projector);
            Mat3 new_proj = Mat3::Zero();
            for (int i = 0; i < 3; ++i) {
                if (eig.eigenvalues()[i] > 0.5) {
                    Vec3 v = eig.eigenvectors().col(i);
                    new_proj += v * v.transpose();
                }
            }
            rot_projector = new_proj;
        }

        info.trans_projector = trans_projector;
        info.rot_projector = rot_projector;

        // Count DOF from projector rank
        {
            Eigen::SelfAdjointEigenSolver<Mat3> eig(trans_projector);
            info.trans_dof = 0;
            std::vector<Vec3> basis_vecs;
            for (int i = 0; i < 3; ++i) {
                if (eig.eigenvalues()[i] > 0.5) {
                    info.trans_dof++;
                    basis_vecs.push_back(eig.eigenvectors().col(i));
                }
            }
            info.trans_basis.resize(3, info.trans_dof);
            for (int i = 0; i < info.trans_dof; ++i) {
                info.trans_basis.col(i) = basis_vecs[i];
            }
        }
        {
            Eigen::SelfAdjointEigenSolver<Mat3> eig(rot_projector);
            info.rot_dof = 0;
            std::vector<Vec3> basis_vecs;
            for (int i = 0; i < 3; ++i) {
                if (eig.eigenvalues()[i] > 0.5) {
                    info.rot_dof++;
                    basis_vecs.push_back(eig.eigenvectors().col(i));
                }
            }
            info.rot_basis.resize(3, info.rot_dof);
            for (int i = 0; i < info.rot_dof; ++i) {
                info.rot_basis.col(i) = basis_vecs[i];
            }
        }

        occ::log::debug("Independent mol {}: {} UC images, {} trans DOF, {} rot DOF",
                        k, indices.size(), info.trans_dof, info.rot_dof);
    }

    occ::log::info("SymmetryMapping: Z'={}, Z={}, total DOF={}",
                   mapping.num_independent, mapping.num_uc_molecules,
                   mapping.total_dof());

    return mapping;
}

// ============================================================================
// generate_uc_states
// ============================================================================

std::vector<MoleculeState> generate_uc_states(
    const std::vector<MoleculeState>& independent_states,
    const SymmetryMapping& mapping) {

    std::vector<MoleculeState> uc_states(mapping.num_uc_molecules);

    for (int j = 0; j < mapping.num_uc_molecules; ++j) {
        const auto& uc_info = mapping.uc_molecules[j];
        int k = uc_info.independent_idx;
        const auto& indep_state = independent_states[k];

        // Position: R_j * pos_k + t_j
        uc_states[j].position = uc_info.R_cart * indep_state.position + uc_info.t_cart;

        // Rotation: compose full symop orientation with independent orientation.
        Mat3 R_mol = indep_state.rotation_matrix();
        Mat3 R_symop = uc_info.R_cart;
        Mat3 R_total = R_symop * R_mol;

        // Convert back to parity + angle-axis representation.
        uc_states[j] = MoleculeState::from_rotation(uc_states[j].position, R_total);
    }

    return uc_states;
}

// ============================================================================
// accumulate_gradients
// ============================================================================

void accumulate_gradients(
    const std::vector<Vec3>& uc_forces,
    const std::vector<Vec3>& uc_torques,
    const SymmetryMapping& mapping,
    std::vector<Vec3>& indep_forces,
    std::vector<Vec3>& indep_torques) {

    int Z_prime = mapping.num_independent;
    indep_forces.resize(Z_prime, Vec3::Zero());
    indep_torques.resize(Z_prime, Vec3::Zero());

    for (int k = 0; k < Z_prime; ++k) {
        Vec3 total_force = Vec3::Zero();
        Vec3 total_torque = Vec3::Zero();

        for (int uc_idx : mapping.independent[k].uc_indices) {
            const auto& uc_info = mapping.uc_molecules[uc_idx];
            double det = uc_info.R_cart.determinant();

            // Back-rotate force: R_j^T * f_j
            total_force += uc_info.R_cart.transpose() * uc_forces[uc_idx];

            // Back-rotate torque (pseudovector): (det * R_j)^T * tau_j
            total_torque += (det * uc_info.R_cart).transpose() * uc_torques[uc_idx];
        }

        // Apply site symmetry projection for translations.
        // Rotations are projected after SO(3) chain-rule conversion in objective().
        indep_forces[k] = mapping.independent[k].trans_projector * total_force;
        indep_torques[k] = total_torque;
    }
}

// ============================================================================
// Constructor
// ============================================================================

CrystalOptimizer::CrystalOptimizer(const crystal::Crystal& crystal,
                                   std::vector<MultipoleSource> multipoles,
                                   const CrystalOptimizerSettings& settings)
    : m_settings(settings)
    , m_reference_crystal(crystal)
    , m_energy(crystal, std::move(multipoles),
               settings.neighbor_radius,
               settings.force_field,
               settings.use_cartesian_engine,
               settings.use_ewald,
               settings.ewald_accuracy,
               settings.ewald_eta,
               settings.ewald_kmax) {

    m_energy.set_max_interaction_order(settings.max_interaction_order);
    m_num_molecules = m_energy.num_molecules();

    if (m_num_molecules == 0) {
        throw std::invalid_argument("CrystalOptimizer: no molecules in crystal");
    }

    // Initialize states from crystal geometry
    m_states = m_energy.initial_states();
    m_initial_states = m_states;
    update_neighbor_reference(m_states, Vec6::Zero());

    if (m_settings.use_symmetry) {
        // Build symmetry mapping to check if symmetry mode is applicable.
        // Symmetry mode requires the energy calculator to have all Z UC molecules
        // (e.g., from DMACRYS setup). When the energy calculator works with Z'
        // molecules (CIF path), it already handles symmetry internally via
        // symmetry_unique_dimers, so we fall back to legacy mode.
        auto mapping = build_symmetry_mapping(crystal);

        if (m_num_molecules != mapping.num_uc_molecules) {
            occ::log::info("CrystalOptimizer: energy calculator has {} molecules "
                           "(Z'={}, Z={}); using legacy mode (symmetry handled internally)",
                           m_num_molecules, mapping.num_independent,
                           mapping.num_uc_molecules);
            m_settings.use_symmetry = false;
        } else {
            m_symmetry_mapping = std::move(mapping);

            if (m_settings.fix_first_translation || m_settings.fix_first_rotation) {
                occ::log::warn(
                    "CrystalOptimizer: fix_first_translation/fix_first_rotation are "
                    "ignored in symmetry-reduced mode");
                m_settings.fix_first_translation = false;
                m_settings.fix_first_rotation = false;
            }

            // The independent state IS the reference UC molecule's state
            // (since we use relative transformations where the reference has R=I, t=0).
            m_independent_states.resize(m_symmetry_mapping.num_independent);
            for (int k = 0; k < m_symmetry_mapping.num_independent; ++k) {
                int ref_uc = m_symmetry_mapping.independent[k].uc_indices[0];
                m_independent_states[k] = m_states[ref_uc];
            }
            m_initial_independent_states = m_independent_states;

            // Compute DOF: sum of per-molecule DOF, with first molecule optionally fixed
            m_num_parameters = 0;
            for (int k = 0; k < m_symmetry_mapping.num_independent; ++k) {
                const auto& info = m_symmetry_mapping.independent[k];
                if (k == 0) {
                    if (!m_settings.fix_first_translation) m_num_parameters += info.trans_dof;
                    if (!m_settings.fix_first_rotation) m_num_parameters += info.rot_dof;
                } else {
                    m_num_parameters += info.trans_dof + info.rot_dof;
                }
            }

            occ::log::info("CrystalOptimizer (symmetry): Z'={}, Z={}, {} DOF (was {})",
                           m_symmetry_mapping.num_independent,
                           m_symmetry_mapping.num_uc_molecules,
                           m_num_parameters,
                           6 * m_num_molecules - 3);
        }
    }

    if (m_settings.use_symmetry &&
        m_settings.method == OptimizationMethod::TrustRegion) {
        occ::log::warn("CrystalOptimizer: TrustRegion with symmetry reduction "
                       "still lacks analytic reduced Hessians; using MSTMIN");
        m_settings.method = OptimizationMethod::MSTMIN;
    }

    if (m_settings.method == OptimizationMethod::TrustRegion &&
        m_settings.require_exact_hessian &&
        !m_energy.can_compute_exact_hessian()) {
        occ::log::warn("CrystalOptimizer: exact Hessian is unavailable for this model; "
                       "using MSTMIN");
        m_settings.method = OptimizationMethod::MSTMIN;
    }

    if (!m_settings.use_symmetry) {
        // Legacy mode: all Z molecules as independent
        int mol0_dof = 0;
        if (!m_settings.fix_first_translation) mol0_dof += 3;
        if (!m_settings.fix_first_rotation) mol0_dof += 3;

        m_num_parameters = mol0_dof + 6 * (m_num_molecules - 1);

        occ::log::debug("CrystalOptimizer: {} molecules, {} parameters",
                       m_num_molecules, m_num_parameters);
    }

    m_num_molecular_parameters = m_num_parameters;
    if (m_settings.optimize_cell) {
        initialize_cell_strain_mask();
        m_active_cell_components.clear();
        for (int j = 0; j < 6; ++j) {
            if (m_cell_strain_mask[j] > 0.5) {
                m_active_cell_components.push_back(j);
            }
        }
        m_num_cell_parameters = static_cast<int>(m_active_cell_components.size());
        m_num_parameters = m_num_molecular_parameters + m_num_cell_parameters;
        occ::log::info(
            "CrystalOptimizer: enabling variable cell ({} molecular + {} effective cell DOF)",
            m_num_molecular_parameters, m_num_cell_parameters);
        occ::log::info("CrystalOptimizer: active cell strains [{}]",
                       active_voigt_components(m_cell_strain_mask));
    } else {
        m_cell_strain_mask = Vec6::Zero();
        m_active_cell_components.clear();
        m_num_cell_parameters = 0;
        m_num_parameters = m_num_molecular_parameters;
    }

    // Log multipole info
    int lmax = m_energy.max_multipole_rank();
    size_t num_sites = m_energy.num_sites();
    size_t num_pairs = m_energy.num_neighbor_pairs();
    occ::log::info("  Multipoles: {} sites, lmax = {} ({})",
                   num_sites, lmax,
                   lmax == 0 ? "charge" :
                   lmax == 1 ? "dipole" :
                   lmax == 2 ? "quadrupole" :
                   lmax == 3 ? "octopole" :
                   lmax == 4 ? "hexadecapole" : "unknown");
    occ::log::info("  Neighbor pairs: {}", num_pairs);
}

// ============================================================================
// Symmetry-Mode Parameter Packing
// ============================================================================

Vec CrystalOptimizer::pack_symmetric_parameters(
    const std::vector<MoleculeState>& indep_states) const {

    Vec params(m_num_molecular_parameters);
    int offset = 0;

    for (int k = 0; k < m_symmetry_mapping.num_independent; ++k) {
        const auto& info = m_symmetry_mapping.independent[k];
        bool skip_trans = (k == 0 && m_settings.fix_first_translation);
        bool skip_rot = (k == 0 && m_settings.fix_first_rotation);

        // Pack translation DOF: project position onto allowed basis
        if (!skip_trans) {
            for (int d = 0; d < info.trans_dof; ++d) {
                params[offset++] = info.trans_basis.col(d).dot(indep_states[k].position);
            }
        }

        // Pack rotation DOF: project angle-axis onto allowed basis
        if (!skip_rot) {
            for (int d = 0; d < info.rot_dof; ++d) {
                params[offset++] = info.rot_basis.col(d).dot(indep_states[k].angle_axis);
            }
        }
    }

    return params;
}

std::vector<MoleculeState> CrystalOptimizer::unpack_symmetric_independent_parameters(
    const Vec& params) const {

    // Start from initial independent states to keep unpack deterministic.
    std::vector<MoleculeState> indep_states = m_initial_independent_states;
    int offset = 0;

    for (int k = 0; k < m_symmetry_mapping.num_independent; ++k) {
        const auto& info = m_symmetry_mapping.independent[k];
        bool skip_trans = (k == 0 && m_settings.fix_first_translation);
        bool skip_rot = (k == 0 && m_settings.fix_first_rotation);

        // Unpack translation from basis coefficients
        if (!skip_trans && info.trans_dof > 0) {
            // Free component from parameters
            Vec3 free_pos = Vec3::Zero();
            for (int d = 0; d < info.trans_dof; ++d) {
                free_pos += params[offset++] * info.trans_basis.col(d);
            }
            // Fixed component from initial state (null space of projector)
            Mat3 fixed_proj = Mat3::Identity() - info.trans_projector;
            indep_states[k].position =
                free_pos + fixed_proj * m_initial_independent_states[k].position;
        }
        // If skip_trans or trans_dof==0, keep position from m_independent_states

        // Unpack rotation from basis coefficients
        if (!skip_rot && info.rot_dof > 0) {
            Vec3 free_aa = Vec3::Zero();
            for (int d = 0; d < info.rot_dof; ++d) {
                free_aa += params[offset++] * info.rot_basis.col(d);
            }
            Mat3 fixed_proj = Mat3::Identity() - info.rot_projector;
            indep_states[k].angle_axis =
                free_aa + fixed_proj * m_initial_independent_states[k].angle_axis;
        }
        // If skip_rot or rot_dof==0, keep rotation from m_independent_states
    }

    return indep_states;
}

std::vector<MoleculeState> CrystalOptimizer::unpack_symmetric_parameters(
    const Vec& params) const {
    auto indep_states = unpack_symmetric_independent_parameters(params);
    // Generate all Z UC states from independent states
    return generate_uc_states(indep_states, m_symmetry_mapping);
}

Vec CrystalOptimizer::pack_symmetric_gradient(
    const std::vector<Vec3>& forces,
    const std::vector<Vec3>& torques) const {

    Vec gradient(m_num_molecular_parameters);
    int offset = 0;

    for (int k = 0; k < m_symmetry_mapping.num_independent; ++k) {
        const auto& info = m_symmetry_mapping.independent[k];
        bool skip_trans = (k == 0 && m_settings.fix_first_translation);
        bool skip_rot = (k == 0 && m_settings.fix_first_rotation);

        // Project force onto allowed translation basis
        // forces = -dE/dr, so gradient = -forces = +dE/dr
        if (!skip_trans) {
            for (int d = 0; d < info.trans_dof; ++d) {
                gradient[offset++] = -info.trans_basis.col(d).dot(forces[k]);
            }
        }

        // Project torque onto allowed rotation basis
        // torques = +dE/dθ, so gradient = +torques = +dE/dθ
        if (!skip_rot) {
            for (int d = 0; d < info.rot_dof; ++d) {
                gradient[offset++] = info.rot_basis.col(d).dot(torques[k]);
            }
        }
    }

    return gradient;
}

// ============================================================================
// Legacy Parameter Packing/Unpacking
// ============================================================================

Mat3 CrystalOptimizer::voigt_to_strain(const Vec6& voigt) {
    Mat3 eps = Mat3::Zero();
    eps(0, 0) = voigt[0];
    eps(1, 1) = voigt[1];
    eps(2, 2) = voigt[2];
    eps(1, 2) = eps(2, 1) = 0.5 * voigt[3];
    eps(0, 2) = eps(2, 0) = 0.5 * voigt[4];
    eps(0, 1) = eps(1, 0) = 0.5 * voigt[5];
    return eps;
}

Vec6 CrystalOptimizer::bounded_cell_strain(const Vec6& vars) const {
    return vars.cwiseProduct(m_cell_strain_mask);
}

Vec6 CrystalOptimizer::unpack_active_cell_parameters(const Vec& params) const {
    Vec6 full = Vec6::Zero();
    if (!m_settings.optimize_cell || m_num_cell_parameters == 0) {
        return full;
    }
    const int tail_offset = m_num_molecular_parameters;
    for (int k = 0; k < m_num_cell_parameters; ++k) {
        const int j = m_active_cell_components[k];
        full[j] = params[tail_offset + k];
    }
    return full;
}

Vec CrystalOptimizer::pack_active_cell_parameters(const Vec6& full_cell) const {
    Vec active(m_num_cell_parameters);
    for (int k = 0; k < m_num_cell_parameters; ++k) {
        const int j = m_active_cell_components[k];
        active[k] = full_cell[j];
    }
    return active;
}

Vec CrystalOptimizer::project_active_cell_gradient(const Vec6& full_grad) const {
    Vec active(m_num_cell_parameters);
    for (int k = 0; k < m_num_cell_parameters; ++k) {
        const int j = m_active_cell_components[k];
        active[k] = full_grad[j];
    }
    return active;
}

Mat CrystalOptimizer::project_active_cell_hessian(const Mat6& full_hessian) const {
    Mat active = Mat::Zero(m_num_cell_parameters, m_num_cell_parameters);
    for (int a = 0; a < m_num_cell_parameters; ++a) {
        const int ia = m_active_cell_components[a];
        for (int b = 0; b < m_num_cell_parameters; ++b) {
            const int ib = m_active_cell_components[b];
            active(a, b) = full_hessian(ia, ib);
        }
    }
    return active;
}

Mat CrystalOptimizer::project_active_cell_coupling(const Mat& full_coupling) const {
    Mat active = Mat::Zero(full_coupling.rows(), m_num_cell_parameters);
    for (int k = 0; k < m_num_cell_parameters; ++k) {
        const int j = m_active_cell_components[k];
        active.col(k) = full_coupling.col(j);
    }
    return active;
}

void CrystalOptimizer::initialize_cell_strain_mask() {
    m_cell_strain_mask = Vec6::Ones();
    if (!m_settings.optimize_cell || !m_settings.constrain_cell_strain_by_lattice) {
        return;
    }

    // Always allow principal stretches; constrain shear (angle-changing) terms.
    m_cell_strain_mask[3] = 0.0;
    m_cell_strain_mask[4] = 0.0;
    m_cell_strain_mask[5] = 0.0;

    const auto& uc = m_reference_crystal.unit_cell();
    const std::string type = uc.cell_type();

    if (type == "triclinic" || type == "rhombohedral") {
        m_cell_strain_mask[3] = 1.0;
        m_cell_strain_mask[4] = 1.0;
        m_cell_strain_mask[5] = 1.0;
        return;
    }

    if (type == "monoclinic") {
        const int shear_idx = monoclinic_shear_component(uc);
        m_cell_strain_mask[shear_idx] = 1.0;
        return;
    }

    // Orthorhombic/tetragonal/hexagonal/cubic: no shear DOF.
}

std::pair<crystal::Crystal, std::vector<MoleculeState>>
CrystalOptimizer::apply_cell_strain(const std::vector<MoleculeState>& base_states,
                                    const Vec6& cell_strain) const {
    Mat3 F = Mat3::Identity() + voigt_to_strain(cell_strain);
    Mat3 strained_direct = F * m_reference_crystal.unit_cell().direct();

    crystal::UnitCell strained_uc(strained_direct);
    crystal::Crystal strained_crystal(
        m_reference_crystal.asymmetric_unit(),
        m_reference_crystal.space_group(),
        strained_uc);

    std::vector<MoleculeState> strained_states = base_states;
    for (auto& s : strained_states) {
        s.position = F * s.position;
    }

    return {std::move(strained_crystal), std::move(strained_states)};
}

bool CrystalOptimizer::should_rebuild_neighbors(
    const std::vector<MoleculeState>& strained_states,
    const Vec6& cell_strain,
    bool force) const {

    if (force || !m_settings.adaptive_neighbor_rebuild || !m_have_neighbor_reference) {
        return true;
    }

    if (m_settings.neighbor_rebuild_interval > 0 &&
        (m_objective_eval_count % m_settings.neighbor_rebuild_interval) == 0) {
        return true;
    }

    double max_disp = 0.0;
    double max_rot = 0.0;
    const int n = static_cast<int>(strained_states.size());
    const int nref = static_cast<int>(m_neighbor_reference_states.size());
    const int ncmp = std::min(n, nref);
    for (int i = 0; i < ncmp; ++i) {
        max_disp = std::max(
            max_disp,
            (strained_states[i].position - m_neighbor_reference_states[i].position).norm());
        max_rot = std::max(
            max_rot,
            rotation_distance(strained_states[i].angle_axis,
                              m_neighbor_reference_states[i].angle_axis));
    }

    double max_cell = (cell_strain - m_neighbor_reference_cell).cwiseAbs().maxCoeff();

    return (max_disp > m_settings.neighbor_rebuild_displacement) ||
           (max_rot > m_settings.neighbor_rebuild_rotation) ||
           (max_cell > m_settings.neighbor_rebuild_cell_strain);
}

void CrystalOptimizer::update_neighbor_reference(
    const std::vector<MoleculeState>& strained_states,
    const Vec6& cell_strain) {
    m_neighbor_reference_states = strained_states;
    m_neighbor_reference_cell = cell_strain;
    m_have_neighbor_reference = true;
}

CrystalEnergyResult CrystalOptimizer::evaluate_model(
    const std::vector<MoleculeState>& base_states,
    const Vec6& cell_strain,
    bool rebuild_neighbors) {
    std::vector<MoleculeState> strained_states = base_states;
    if (m_settings.optimize_cell) {
        auto strained = apply_cell_strain(base_states, cell_strain);
        m_energy.update_lattice(strained.first, strained.second);
        strained_states = std::move(strained.second);
    }

    if (rebuild_neighbors &&
        should_rebuild_neighbors(strained_states, cell_strain, false)) {
        m_energy.update_neighbors(strained_states);
        update_neighbor_reference(strained_states, cell_strain);
    }

    return m_energy.compute(strained_states);
}

Vec CrystalOptimizer::pack_parameters(const std::vector<MoleculeState>& states) const {
    if (m_settings.use_symmetry) {
        Vec params = pack_symmetric_parameters(m_independent_states);
        if (!m_settings.optimize_cell) {
            return params;
        }
        Vec full(m_num_parameters);
        full.head(m_num_molecular_parameters) = params;
        full.tail(m_num_cell_parameters) =
            pack_active_cell_parameters(m_cell_strain);
        return full;
    }

    Vec params(m_num_molecular_parameters);
    int offset = 0;
    Mat3 F_inv = Mat3::Identity();
    if (m_settings.optimize_cell) {
        Mat3 F = Mat3::Identity() + voigt_to_strain(bounded_cell_strain(m_cell_strain));
        F_inv = F.inverse();
    }

    // Molecule 0: partial DOF based on settings
    if (!m_settings.fix_first_translation) {
        Vec3 pos = states[0].position;
        if (m_settings.optimize_cell) {
            pos = F_inv * pos;
        }
        params.segment<3>(offset) = pos;
        offset += 3;
    }
    if (!m_settings.fix_first_rotation) {
        params.segment<3>(offset) = states[0].angle_axis;
        offset += 3;
    }

    // Molecules 1..N-1: full 6 DOF
    for (int i = 1; i < m_num_molecules; ++i) {
        Vec3 pos = states[i].position;
        if (m_settings.optimize_cell) {
            pos = F_inv * pos;
        }
        params.segment<3>(offset) = pos;
        params.segment<3>(offset + 3) = states[i].angle_axis;
        offset += 6;
    }

    if (!m_settings.optimize_cell) {
        return params;
    }
    Vec full(m_num_parameters);
    full.head(m_num_molecular_parameters) = params;
    full.tail(m_num_cell_parameters) = pack_active_cell_parameters(m_cell_strain);
    return full;
}

std::vector<MoleculeState> CrystalOptimizer::unpack_parameters(const Vec& params) const {
    Vec molecular_params = m_settings.optimize_cell
                               ? params.head(m_num_molecular_parameters)
                               : params;

    if (m_settings.use_symmetry) {
        auto base_states = unpack_symmetric_parameters(molecular_params);
        if (!m_settings.optimize_cell) {
            return base_states;
        }
        Vec6 cell_vars = unpack_active_cell_parameters(params);
        return apply_cell_strain(base_states, bounded_cell_strain(cell_vars)).second;
    }

    // Start from initial states to keep unpack deterministic.
    std::vector<MoleculeState> states = m_initial_states;
    int offset = 0;

    // Molecule 0: partial DOF based on settings
    if (!m_settings.fix_first_translation) {
        states[0].position = molecular_params.segment<3>(offset);
        offset += 3;
    }
    if (!m_settings.fix_first_rotation) {
        states[0].angle_axis = molecular_params.segment<3>(offset);
        offset += 3;
    }

    // Molecules 1..N-1: full 6 DOF
    for (int i = 1; i < m_num_molecules; ++i) {
        states[i].position = molecular_params.segment<3>(offset);
        states[i].angle_axis = molecular_params.segment<3>(offset + 3);
        offset += 6;
    }

    if (!m_settings.optimize_cell) {
        return states;
    }
    Vec6 cell_vars = unpack_active_cell_parameters(params);
    return apply_cell_strain(states, bounded_cell_strain(cell_vars)).second;
}

Vec CrystalOptimizer::get_parameters() const {
    return pack_parameters(m_states);
}

void CrystalOptimizer::set_parameters(const Vec& params) {
    Vec molecular_params = m_settings.optimize_cell
                               ? params.head(m_num_molecular_parameters)
                               : params;

    std::vector<MoleculeState> base_states;
    if (m_settings.use_symmetry) {
        m_independent_states = unpack_symmetric_independent_parameters(molecular_params);
        base_states = generate_uc_states(m_independent_states, m_symmetry_mapping);
    } else {
        base_states = m_initial_states;
        int offset = 0;
        if (!m_settings.fix_first_translation) {
            base_states[0].position = molecular_params.segment<3>(offset);
            offset += 3;
        }
        if (!m_settings.fix_first_rotation) {
            base_states[0].angle_axis = molecular_params.segment<3>(offset);
            offset += 3;
        }
        for (int i = 1; i < m_num_molecules; ++i) {
            base_states[i].position = molecular_params.segment<3>(offset);
            base_states[i].angle_axis = molecular_params.segment<3>(offset + 3);
            offset += 6;
        }
    }

    if (m_settings.optimize_cell) {
        m_cell_strain = bounded_cell_strain(unpack_active_cell_parameters(params));
        Vec6 bounded_strain = m_cell_strain;
        auto [strained_crystal, strained_states] =
            apply_cell_strain(base_states, bounded_strain);
        m_states = std::move(strained_states);
        m_energy.update_lattice(strained_crystal, m_states);
        if (should_rebuild_neighbors(m_states, bounded_strain, false)) {
            m_energy.update_neighbors(m_states);
            update_neighbor_reference(m_states, bounded_strain);
        }
    } else {
        m_states = std::move(base_states);
        if (should_rebuild_neighbors(m_states, Vec6::Zero(), false)) {
            m_energy.update_neighbors(m_states);
            update_neighbor_reference(m_states, Vec6::Zero());
        }
    }
}

// ============================================================================
// Energy and Gradient
// ============================================================================

void CrystalOptimizer::reinitialize_states() {
    m_num_molecules = m_energy.num_molecules();
    m_states = m_energy.initial_states();
    m_initial_states = m_states;
    m_reference_crystal = m_energy.crystal();
    m_cell_strain.setZero();
    m_objective_eval_count = 0;
    update_neighbor_reference(m_states, Vec6::Zero());

    if (m_settings.use_symmetry) {
        // Re-initialize independent states from reference UC molecules
        m_independent_states.resize(m_symmetry_mapping.num_independent);
        for (int k = 0; k < m_symmetry_mapping.num_independent; ++k) {
            int ref_uc = m_symmetry_mapping.independent[k].uc_indices[0];
            if (ref_uc < static_cast<int>(m_states.size())) {
                m_independent_states[k] = m_states[ref_uc];
            }
        }
        m_initial_independent_states = m_independent_states;
    }
}

CrystalEnergyResult CrystalOptimizer::compute_energy_gradient() {
    return m_energy.compute(m_states);
}

double CrystalOptimizer::objective(const Vec& params, Vec& gradient,
                                  bool allow_neighbor_rebuild) {
    m_objective_eval_count++;

    Vec molecular_params = m_settings.optimize_cell
                               ? params.head(m_num_molecular_parameters)
                               : params;
    Vec6 cell_vars = Vec6::Zero();
    if (m_settings.optimize_cell) {
        cell_vars = unpack_active_cell_parameters(params);
    }
    Vec6 cell_params = bounded_cell_strain(cell_vars);

    std::vector<MoleculeState> base_states;
    std::vector<MoleculeState> indep_states;
    if (m_settings.use_symmetry) {
        indep_states = unpack_symmetric_independent_parameters(molecular_params);
        base_states = generate_uc_states(indep_states, m_symmetry_mapping);
    } else {
        base_states = m_initial_states;
        int offset = 0;
        if (!m_settings.fix_first_translation) {
            base_states[0].position = molecular_params.segment<3>(offset);
            offset += 3;
        }
        if (!m_settings.fix_first_rotation) {
            base_states[0].angle_axis = molecular_params.segment<3>(offset);
            offset += 3;
        }
        for (int i = 1; i < m_num_molecules; ++i) {
            base_states[i].position = molecular_params.segment<3>(offset);
            base_states[i].angle_axis = molecular_params.segment<3>(offset + 3);
            offset += 6;
        }
    }

    auto result = evaluate_model(base_states, cell_params, allow_neighbor_rebuild);
    double objective_energy = result.total_energy;
    Vec6 objective_strain_gradient = result.strain_gradient;
    m_last_eval_pv_energy = 0.0;
    {
        Mat3 F = Mat3::Identity();
        if (m_settings.optimize_cell) {
            F += voigt_to_strain(cell_params);
        }
        const auto pterms = pressure_strain_terms(
            m_settings.external_pressure_gpa,
            F,
            m_reference_crystal.unit_cell().volume());
        objective_energy += pterms.energy;
        m_last_eval_pv_energy = pterms.energy;
        if (m_settings.optimize_cell) {
            objective_strain_gradient += pterms.gradient;
        }
    }

    m_last_eval_max_force = 0.0;
    for (const auto& f : result.forces) {
        m_last_eval_max_force = std::max(m_last_eval_max_force, f.norm());
    }
    m_last_eval_max_torque = 0.0;
    for (const auto& t : result.torques) {
        m_last_eval_max_torque = std::max(m_last_eval_max_torque, t.norm());
    }
    m_last_eval_max_stress = 0.0;
    if (m_settings.optimize_cell) {
        const double V = m_energy.crystal().unit_cell().volume();
        const Vec6 sigma = strain_gradient_to_stress_gpa(objective_strain_gradient, V);
        m_last_eval_max_stress = sigma.cwiseAbs().maxCoeff();
    }

    if (m_settings.use_symmetry) {
        std::vector<Vec3> uc_forces = result.forces;
        if (m_settings.optimize_cell) {
            Mat3 F = Mat3::Identity() + voigt_to_strain(cell_params);
            for (auto& f : uc_forces) {
                f = F.transpose() * f;
            }
        }

        // Accumulate UC-level gradients dE/dpsi_lab to independent molecules.
        std::vector<Vec3> indep_forces, indep_torques;
        accumulate_gradients(uc_forces, result.torques, m_symmetry_mapping,
                           indep_forces, indep_torques);

        // Convert rotational gradients from dE/dpsi_lab to dE/dtheta and then
        // enforce site-symmetry rotational constraints.
        for (int k = 0; k < m_symmetry_mapping.num_independent; ++k) {
            Vec3 grad_theta = apply_angle_axis_chain_rule(
                indep_states[k].angle_axis, indep_torques[k]);
            indep_torques[k] =
                m_symmetry_mapping.independent[k].rot_projector * grad_theta;
        }

        // Pack into reduced gradient vector
        gradient = pack_symmetric_gradient(indep_forces, indep_torques);
    } else {
        Mat3 F = Mat3::Identity();
        if (m_settings.optimize_cell) {
            F += voigt_to_strain(cell_params);
        }

        // Legacy: pack gradient with same layout as parameters
        // base positions map to lab via r = F * p when optimize_cell=true.
        // forces = -dE/dr  → dE/dp = F^T * dE/dr = -F^T * force.
        // torques from CrystalEnergy are dE/dpsi_lab. Convert to dE/dtheta.
        gradient.resize(m_num_parameters);
        int offset = 0;

        // Molecule 0: partial DOF based on settings
        if (!m_settings.fix_first_translation) {
            Vec3 grad_pos = -result.forces[0];
            if (m_settings.optimize_cell) {
                grad_pos = F.transpose() * grad_pos;
            }
            gradient.segment<3>(offset) = grad_pos;
            offset += 3;
        }
        if (!m_settings.fix_first_rotation) {
            gradient.segment<3>(offset) = apply_angle_axis_chain_rule(
                base_states[0].angle_axis, result.torques[0]);
            offset += 3;
        }

        // Molecules 1..N-1: full 6 DOF
        for (int i = 1; i < m_num_molecules; ++i) {
            Vec3 grad_pos = -result.forces[i];
            if (m_settings.optimize_cell) {
                grad_pos = F.transpose() * grad_pos;
            }
            gradient.segment<3>(offset) = grad_pos;
            gradient.segment<3>(offset + 3) = apply_angle_axis_chain_rule(
                base_states[i].angle_axis, result.torques[i]);
            offset += 6;
        }
    }

    if (m_settings.optimize_cell) {
        // Cell gradient from CrystalEnergy includes pair-image virial terms.
        // Cell variables are direct Voigt strain components, so dE/du = dE/dE.
        const Vec6 dE_dE = objective_strain_gradient;
        const Vec6 dE_du_scale = cell_strain_first_derivative(cell_vars);
        Vec6 dE_du =
            dE_dE.cwiseProduct(dE_du_scale).cwiseProduct(m_cell_strain_mask);

        if (gradient.size() != m_num_parameters) {
            Vec expanded = Vec::Zero(m_num_parameters);
            expanded.head(gradient.size()) = gradient;
            gradient = std::move(expanded);
        }
        gradient.tail(m_num_cell_parameters) =
            project_active_cell_gradient(dE_du);
    }

    return objective_energy;
}

// ============================================================================
// Optimization
// ============================================================================

std::pair<double, Vec> CrystalOptimizer::objective_pair(const Vec& params,
                                                        bool allow_neighbor_rebuild) {
    Vec grad;
    double energy = objective(params, grad, allow_neighbor_rebuild);
    return {energy, grad};
}

Mat CrystalOptimizer::compute_hessian(const Vec& params) {
    if (m_settings.use_symmetry) {
        throw std::runtime_error(
            "compute_hessian: symmetry-reduced TrustRegion Hessian is not available");
    }

    Vec molecular_params = m_settings.optimize_cell
                               ? params.head(m_num_molecular_parameters)
                               : params;
    Vec6 cell_vars = Vec6::Zero();
    Vec6 cell_params = Vec6::Zero();
    Vec6 d1 = Vec6::Zero();
    Vec6 d2 = Vec6::Zero();
    if (m_settings.optimize_cell) {
        cell_vars = unpack_active_cell_parameters(params);
        cell_params = bounded_cell_strain(cell_vars);
        d1 = cell_strain_first_derivative(cell_vars).cwiseProduct(m_cell_strain_mask);
        d2 = cell_strain_second_derivative(cell_vars).cwiseProduct(m_cell_strain_mask);
    }

    // Rebuild base states from molecular parameter subset.
    std::vector<MoleculeState> base_states = m_initial_states;
    int p_offset = 0;
    if (!m_settings.fix_first_translation) {
        base_states[0].position = molecular_params.segment<3>(p_offset);
        p_offset += 3;
    }
    if (!m_settings.fix_first_rotation) {
        base_states[0].angle_axis = molecular_params.segment<3>(p_offset);
        p_offset += 3;
    }
    for (int i = 1; i < m_num_molecules; ++i) {
        base_states[i].position = molecular_params.segment<3>(p_offset);
        base_states[i].angle_axis = molecular_params.segment<3>(p_offset + 3);
        p_offset += 6;
    }

    std::vector<MoleculeState> strained_states = base_states;
    if (m_settings.optimize_cell) {
        auto strained = apply_cell_strain(base_states, cell_params);
        m_energy.update_lattice(strained.first, strained.second);
        if (!m_settings.freeze_neighbors_during_linesearch &&
            should_rebuild_neighbors(strained.second, cell_params, false)) {
            m_energy.update_neighbors(strained.second);
            update_neighbor_reference(strained.second, cell_params);
        }
        strained_states = std::move(strained.second);
    }
    auto eh = m_energy.compute_with_hessian(strained_states);
    if (m_settings.optimize_cell &&
        std::abs(m_settings.external_pressure_gpa) > 1e-16) {
        Mat3 F = Mat3::Identity() + voigt_to_strain(cell_params);
        const auto pterms = pressure_strain_terms(
            m_settings.external_pressure_gpa,
            F,
            m_reference_crystal.unit_cell().volume());
        eh.strain_gradient += pterms.gradient;
        eh.strain_hessian += pterms.hessian;
    }

    if (m_settings.require_exact_hessian && !eh.exact_for_model) {
        throw std::runtime_error(
            "compute_hessian: exact Hessian requested but unavailable for current model");
    }

    static bool warned_hessian_scope = false;
    if (!warned_hessian_scope && !eh.exact_for_model) {
        occ::log::warn("TrustRegion Hessian is approximate for this model");
        warned_hessian_scope = true;
    }

    const int N = static_cast<int>(strained_states.size());
    const int ns = 6 * N;
    const int nm = m_settings.optimize_cell
                       ? m_num_molecular_parameters
                       : m_num_parameters;

    Mat3 F = Mat3::Identity();
    std::array<Mat3, 6> dFdu;
    if (m_settings.optimize_cell) {
        F += voigt_to_strain(cell_params);
        for (int j = 0; j < 6; ++j) {
            const Mat3 Gj = voigt_basis_matrix(j);
            dFdu[j] = Gj * d1[j];
        }
    } else {
        for (int j = 0; j < 6; ++j) {
            dFdu[j].setZero();
        }
    }

    Mat Bm = Mat::Zero(ns, nm);
    std::vector<std::array<int, 3>> trans_cols(N), rot_cols(N);
    for (int i = 0; i < N; ++i) {
        trans_cols[i] = {-1, -1, -1};
        rot_cols[i] = {-1, -1, -1};
    }

    int qcol = 0;
    for (int i = 0; i < N; ++i) {
        const bool has_trans = (i != 0) || !m_settings.fix_first_translation;
        const bool has_rot = (i != 0) || !m_settings.fix_first_rotation;

        if (has_trans) {
            for (int k = 0; k < 3; ++k) {
                trans_cols[i][k] = qcol;
                Bm.block<3, 1>(6 * i, qcol) = F.col(k);
                qcol++;
            }
        }
        if (has_rot) {
            Mat3 J = so3_left_jacobian_transpose(base_states[i].angle_axis).transpose();
            for (int k = 0; k < 3; ++k) {
                rot_cols[i][k] = qcol;
                Bm.block<3, 1>(6 * i + 3, qcol) = J.col(k);
                qcol++;
            }
        }
    }

    Mat Hmm = Bm.transpose() * eh.hessian * Bm;
    Mat Hmu;
    Mat6 Huu = Mat6::Zero();
    if (m_settings.optimize_cell) {
        Hmu = Bm.transpose() * eh.strain_state_hessian.transpose();
        for (int j = 0; j < 6; ++j) {
            Hmu.col(j) *= d1[j];
        }
        Huu = eh.strain_hessian;
        Huu = d1.asDiagonal() * Huu * d1.asDiagonal();
    }

    // Curvature term from J_L^T(theta) in rotational gradients.
    for (int i = 0; i < N; ++i) {
        const Vec3 theta = base_states[i].angle_axis;
        const Vec3 gpsi = eh.torques[i];
        const auto dJt = so3_left_jacobian_transpose_derivatives(theta);
        for (int l = 0; l < 3; ++l) {
            if (rot_cols[i][l] < 0) continue;
            const Vec3 col = dJt[l] * gpsi;
            for (int k = 0; k < 3; ++k) {
                if (rot_cols[i][k] < 0) continue;
                Hmm(rot_cols[i][k], rot_cols[i][l]) += col[k];
            }
        }
    }

    if (!m_settings.optimize_cell) {
        return 0.5 * (Hmm + Hmm.transpose());
    }

    // Geometric mixed term from F(u) in translational gradients.
    for (int i = 0; i < N; ++i) {
        const Vec3 g_r = -eh.forces[i];
        for (int k = 0; k < 3; ++k) {
            int c = trans_cols[i][k];
            if (c < 0) continue;
            for (int j = 0; j < 6; ++j) {
                Hmu(c, j) += dFdu[j].col(k).dot(g_r);
            }
        }
    }

    // Second-derivative term from bounded strain variable map.
    for (int j = 0; j < 6; ++j) {
        Huu(j, j) += eh.strain_gradient[j] * d2[j];
    }

    Mat H = Mat::Zero(m_num_parameters, m_num_parameters);
    H.block(0, 0, nm, nm) = Hmm;
    Mat Hmu_active = project_active_cell_coupling(Hmu);
    Mat Huu_active = project_active_cell_hessian(Huu);
    H.block(0, nm, nm, m_num_cell_parameters) = Hmu_active;
    H.block(nm, 0, m_num_cell_parameters, nm) = Hmu_active.transpose();
    H.block(nm, nm, m_num_cell_parameters, m_num_cell_parameters) = Huu_active;
    return 0.5 * (H + H.transpose());
}

CrystalOptimizerResult CrystalOptimizer::optimize() {
    return optimize(nullptr);
}

CrystalOptimizerResult CrystalOptimizer::optimize(IterationCallback callback) {
    if (m_settings.method == OptimizationMethod::TrustRegion) {
        return optimize_trust_region(callback);
    } else if (m_settings.method == OptimizationMethod::LBFGS) {
        return optimize_lbfgs(callback);
    } else {
        return optimize_mstmin(callback);
    }
}

CrystalOptimizerResult CrystalOptimizer::optimize_trust_region(IterationCallback callback) {
    CrystalOptimizerResult result;

    // Handle edge case: no parameters to optimize
    if (m_num_parameters == 0) {
        occ::log::warn("No parameters to optimize (all DOF fixed)");
        auto energy_result = m_energy.compute(m_states);
        const double pv =
            (m_settings.external_pressure_gpa /
             occ::units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA) *
            m_energy.crystal().unit_cell().volume();
        result.pressure_volume_energy = pv / m_num_molecules;
        result.initial_energy =
            energy_result.total_energy / m_num_molecules + result.pressure_volume_energy;
        result.final_energy = result.initial_energy;
        result.electrostatic_energy = energy_result.electrostatic_energy / m_num_molecules;
        result.repulsion_dispersion_energy = energy_result.repulsion_dispersion / m_num_molecules;
        result.iterations = 0;
        result.function_evaluations = 1;
        result.converged = true;
        result.termination_reason = "No parameters to optimize";
        result.final_states = m_states;
        result.optimized_crystal = build_optimized_crystal();
        return result;
    }

    // Compute initial energy
    Vec params = get_parameters();
    Vec grad(m_num_parameters);

    occ::timing::StopWatch<> sw;
    sw.start();
    result.initial_energy = objective(params, grad);
    double init_time = sw.stop().count();

    const double gnorm0 = grad.norm();
    const double grms0 = gnorm0 / std::sqrt(static_cast<double>(std::max(1, m_num_parameters)));
    const double ginf0 = grad.cwiseAbs().maxCoeff();
    if (m_settings.optimize_cell) {
        occ::log::info(
            "Initial:    E = {:14.6f} kJ/mol   |g| = {:.3e}  (rms {:.3e}, max {:.3e}; max|F| {:.3e}, max|tau| {:.3e}, max|stress| {:.3e} GPa)  ({:.3f} s)",
            result.initial_energy / m_num_molecules, gnorm0, grms0, ginf0,
            m_last_eval_max_force, m_last_eval_max_torque, m_last_eval_max_stress, init_time);
    } else {
        occ::log::info(
            "Initial:    E = {:14.6f} kJ/mol   |g| = {:.3e}  (rms {:.3e}, max {:.3e}; max|F| {:.3e}, max|tau| {:.3e})  ({:.3f} s)",
            result.initial_energy / m_num_molecules, gnorm0, grms0, ginf0,
            m_last_eval_max_force, m_last_eval_max_torque, init_time);
    }
    std::fflush(stdout);

    // Open trajectory file if requested (callback will write frames)
    bool write_trajectory = !m_settings.trajectory_file.empty();
    if (write_trajectory) {
        m_trajectory_stream.open(m_settings.trajectory_file);
        if (m_trajectory_stream) {
            occ::log::info("Writing trajectory to {}", m_settings.trajectory_file);
        } else {
            occ::log::warn("Could not open trajectory file: {}", m_settings.trajectory_file);
            write_trajectory = false;
        }
    }

    // Setup Trust Region Newton
    TrustRegionSettings tr_settings;
    tr_settings.initial_radius = m_settings.trust_region_radius;
    // Scale tolerances by total DOF so user-provided values are per-DOF/per-molecule
    double grad_tol = m_settings.gradient_tolerance * std::sqrt(static_cast<double>(m_num_parameters));
    tr_settings.gradient_tol = grad_tol;
    // Ignore energy-based convergence to avoid premature stops with high gradients
    tr_settings.energy_tol = 0.0;
    tr_settings.max_iterations = m_settings.max_iterations;
    tr_settings.hessian_update_interval = m_settings.hessian_update_interval;
    tr_settings.verbose = false;  // We use our own callback for logging

    TrustRegion optimizer(tr_settings);

    int eval_count = 1;  // Already did one evaluation
    int hess_count = 0;

    // Objective function wrapper
    auto tr_objective = [this, &eval_count](const Vec& x) -> std::pair<double, Vec> {
        eval_count++;
        return objective_pair(x, !m_settings.freeze_neighbors_during_linesearch);
    };

    // Hessian function wrapper
    auto tr_hessian = [this, &hess_count](const Vec& x) -> Mat {
        hess_count++;
        return compute_hessian(x);
    };

    // Iteration callback for trajectory writing
    occ::timing::StopWatch<> iter_sw;
    iter_sw.start();

    auto tr_callback = [this, &callback, &eval_count, &iter_sw, write_trajectory](
        int iter, const Vec& x, double f, const Vec& g) -> bool {

        double iter_time = iter_sw.stop().count();
        iter_sw.start();

        // Update states for trajectory writing
        const_cast<CrystalOptimizer*>(this)->set_parameters(x);

        double gnorm = g.norm();
        double e_per_mol = f / m_num_molecules;
        if (m_settings.optimize_cell &&
            g.size() >= (m_num_molecular_parameters + m_num_cell_parameters)) {
            const double grms = gnorm / std::sqrt(static_cast<double>(std::max(1, m_num_parameters)));
            const double ginf = g.cwiseAbs().maxCoeff();
            const double gmol = g.head(m_num_molecular_parameters).norm();
            const double gcell = g.tail(m_num_cell_parameters).norm();
            occ::log::info(
                "Iter {:4d}:  E = {:14.6f} kJ/mol   |g| = {:.3e}  (rms {:.3e}, max {:.3e}; mol {:.3e}, cell {:.3e}; max|F| {:.3e}, max|tau| {:.3e}, max|stress| {:.3e} GPa)  ({:.3f} s, {} evals)",
                iter, e_per_mol, gnorm, grms, ginf, gmol, gcell,
                m_last_eval_max_force, m_last_eval_max_torque, m_last_eval_max_stress,
                iter_time, eval_count);
        } else {
            occ::log::info("Iter {:4d}:  E = {:14.6f} kJ/mol   |g| = {:.3e}  ({:.3f} s, {} evals)",
                           iter, e_per_mol, gnorm, iter_time, eval_count);
        }
        std::fflush(stdout);

        // Write trajectory frame
        if (write_trajectory && m_trajectory_stream) {
            write_trajectory_frame(m_trajectory_stream, iter, f);
        }

        if (callback) {
            return callback(iter, f, gnorm);
        }
        return true;
    };

    // Run optimization
    TrustRegionResult tr_result = optimizer.minimize(tr_objective, tr_hessian, params, tr_callback);

    double opt_time = iter_sw.stop().count();

    // Update states with final parameters
    set_parameters(tr_result.x);

    // Final energy evaluation for components
    auto final_result = m_energy.compute(m_states);
    const double pv =
        (m_settings.external_pressure_gpa /
         occ::units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA) *
        m_energy.crystal().unit_cell().volume();

    // Build result
    result.pressure_volume_energy = pv / m_num_molecules;
    result.final_energy =
        final_result.total_energy / m_num_molecules + result.pressure_volume_energy;
    result.electrostatic_energy = final_result.electrostatic_energy / m_num_molecules;
    result.repulsion_dispersion_energy = final_result.repulsion_dispersion / m_num_molecules;
    result.iterations = tr_result.iterations;
    result.function_evaluations = eval_count;
    result.converged = tr_result.converged;
    result.termination_reason = tr_result.termination_reason;
    result.final_states = m_states;
    result.optimized_crystal = build_optimized_crystal();
    result.initial_energy /= m_num_molecules;

    occ::log::info("\nTrust Region: {} iterations, {} function evals, {} Hessian evals ({:.1f} s)",
                   tr_result.iterations, eval_count, hess_count, opt_time);

    // Close trajectory file
    if (m_trajectory_stream.is_open()) {
        m_trajectory_stream.close();
    }

    return result;
}

CrystalOptimizerResult CrystalOptimizer::optimize_mstmin(IterationCallback callback) {
    CrystalOptimizerResult result;

    if (m_num_parameters == 0) {
        occ::log::warn("No parameters to optimize (all DOF fixed)");
        auto energy_result = m_energy.compute(m_states);
        const double pv =
            (m_settings.external_pressure_gpa /
             occ::units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA) *
            m_energy.crystal().unit_cell().volume();
        result.pressure_volume_energy = pv / m_num_molecules;
        result.initial_energy =
            energy_result.total_energy / m_num_molecules + result.pressure_volume_energy;
        result.final_energy = result.initial_energy;
        result.electrostatic_energy = energy_result.electrostatic_energy / m_num_molecules;
        result.repulsion_dispersion_energy = energy_result.repulsion_dispersion / m_num_molecules;
        result.iterations = 0;
        result.function_evaluations = 1;
        result.converged = true;
        result.termination_reason = "No parameters to optimize";
        result.final_states = m_states;
        result.optimized_crystal = build_optimized_crystal();
        return result;
    }

    Vec params = get_parameters();
    Vec grad(m_num_parameters);

    occ::timing::StopWatch<> sw;
    sw.start();
    result.initial_energy = objective(params, grad);
    double init_time = sw.stop().count();

    const double grad_tol =
        m_settings.gradient_tolerance * std::sqrt(static_cast<double>(m_num_parameters));

    const double gnorm0 = grad.norm();
    const double grms0 = gnorm0 / std::sqrt(static_cast<double>(std::max(1, m_num_parameters)));
    const double ginf0 = grad.cwiseAbs().maxCoeff();
    if (m_settings.optimize_cell) {
        occ::log::info(
            "Initial:    E = {:14.6f} kJ/mol   |g| = {:.3e}  (rms {:.3e}, max {:.3e}; max|F| {:.3e}, max|tau| {:.3e}, max|stress| {:.3e} GPa)  ({:.3f} s)",
            result.initial_energy / m_num_molecules, gnorm0, grms0, ginf0,
            m_last_eval_max_force, m_last_eval_max_torque, m_last_eval_max_stress, init_time);
    } else {
        occ::log::info(
            "Initial:    E = {:14.6f} kJ/mol   |g| = {:.3e}  (rms {:.3e}, max {:.3e}; max|F| {:.3e}, max|tau| {:.3e})  ({:.3f} s)",
            result.initial_energy / m_num_molecules, gnorm0, grms0, ginf0,
            m_last_eval_max_force, m_last_eval_max_torque, init_time);
    }
    std::fflush(stdout);

    bool write_trajectory = !m_settings.trajectory_file.empty();
    if (write_trajectory) {
        m_trajectory_stream.open(m_settings.trajectory_file);
        if (m_trajectory_stream) {
            occ::log::info("Writing trajectory to {}", m_settings.trajectory_file);
        } else {
            occ::log::warn("Could not open trajectory file: {}", m_settings.trajectory_file);
            write_trajectory = false;
        }
    }

    MSTMINSettings mst_settings;
    mst_settings.max_displacement = m_settings.max_displacement;
    mst_settings.max_updates = m_settings.max_hessian_updates;
    mst_settings.gradient_tol = grad_tol;
    mst_settings.step_tol = m_settings.mst_step_tolerance;
    mst_settings.energy_tol =
        m_settings.energy_tolerance * static_cast<double>(m_num_molecules);
    mst_settings.enforce_energy_decrease = m_settings.optimize_cell;
    if (mst_settings.enforce_energy_decrease) {
        // Cell optimization can introduce small numerical non-monotonicity
        // (neighbor updates, taper boundaries, Hessian resets). Allow tiny
        // uphill noise while still rejecting genuinely bad steps.
        mst_settings.energy_increase_tol_abs =
            std::max(1e-4 * static_cast<double>(m_num_molecules),
                     10.0 * mst_settings.energy_tol);
        mst_settings.energy_increase_tol_rel = 1e-8;
    }
    mst_settings.steepest_descent_on_positive_gd = false;
    mst_settings.max_line_search = m_settings.mst_max_line_search;
    mst_settings.max_line_search_restarts = m_settings.mst_max_line_search_restarts;
    mst_settings.max_function_evaluations = m_settings.mst_max_function_evaluations;

    MSTMIN optimizer(mst_settings);

    Vec param_scales = Vec::Ones(m_num_parameters);
    const double rot_scale = std::max(1e-6, m_settings.mst_rotation_scale);
    if (m_settings.use_symmetry) {
        int offset = 0;
        for (int k = 0; k < m_symmetry_mapping.num_independent; ++k) {
            const auto& info = m_symmetry_mapping.independent[k];
            const bool skip_trans = (k == 0 && m_settings.fix_first_translation);
            const bool skip_rot = (k == 0 && m_settings.fix_first_rotation);
            if (!skip_trans) {
                offset += info.trans_dof;
            }
            if (!skip_rot) {
                for (int d = 0; d < info.rot_dof; ++d) {
                    param_scales[offset++] = rot_scale;
                }
            }
        }
    } else {
        int offset = 0;
        if (!m_settings.fix_first_translation) {
            offset += 3;
        }
        if (!m_settings.fix_first_rotation) {
            for (int d = 0; d < 3; ++d) {
                param_scales[offset++] = rot_scale;
            }
        }
        for (int i = 1; i < m_num_molecules; ++i) {
            offset += 3; // translation
            for (int d = 0; d < 3; ++d) {
                param_scales[offset++] = rot_scale;
            }
        }
    }
    if (m_settings.optimize_cell && m_num_cell_parameters > 0) {
        const double cell_scale = std::max(1e-6, m_settings.mst_cell_scale);
        for (int j = 0; j < m_num_cell_parameters; ++j) {
            param_scales[m_num_molecular_parameters + j] = cell_scale;
        }
    }
    Vec scaled_params = params.cwiseQuotient(param_scales);
    Vec best_scaled_params = scaled_params;
    double best_energy = result.initial_energy;

    int eval_count = 1; // Already did one evaluation
    int eval_count_at_last_accept = eval_count;
    int next_stall_report_eval = eval_count +
        std::max(1, m_settings.mst_line_search_report_interval);
    const double stall_energy_tol =
        std::max(10.0 * mst_settings.energy_tol, 1e-4 * static_cast<double>(m_num_molecules));
    const double stall_grad_factor = 10.0;
    const int stall_iter_limit = 8;
    const int stall_eval_limit = std::max(
        200, 2 * std::max(1, m_settings.mst_line_search_report_interval));
    int stall_iter_count = 0;
    double last_significant_energy = result.initial_energy;
    int eval_at_last_significant_improvement = eval_count;
    double prev_iter_energy = result.initial_energy;
    bool stopped_for_stall = false;
    occ::timing::StopWatch<> iter_sw;
    iter_sw.start();

    auto mst_callback = [this, &callback, &eval_count, &iter_sw, write_trajectory, &param_scales,
                         &best_scaled_params, &best_energy,
                         &eval_count_at_last_accept, &next_stall_report_eval,
                         &stall_iter_count, stall_iter_limit, stall_eval_limit,
                         stall_energy_tol, stall_grad_factor, grad_tol,
                         &last_significant_energy, &eval_at_last_significant_improvement,
                         &prev_iter_energy, &stopped_for_stall](
        int iter, const Vec& x, double f, const Vec& g) -> bool {

        double iter_time = iter_sw.stop().count();
        iter_sw.start();

        if (f < best_energy) {
            best_energy = f;
            best_scaled_params = x;
        }
        eval_count_at_last_accept = eval_count;
        next_stall_report_eval = eval_count +
            std::max(1, m_settings.mst_line_search_report_interval);

        Vec phys_x = x.cwiseProduct(param_scales);
        Vec phys_g = g.cwiseQuotient(param_scales);
        const_cast<CrystalOptimizer*>(this)->set_parameters(phys_x);

        double gnorm = phys_g.norm();
        const double grms = gnorm / std::sqrt(static_cast<double>(std::max(1, m_num_parameters)));
        const double ginf = phys_g.cwiseAbs().maxCoeff();
        double e_per_mol = f / m_num_molecules;
        if (m_settings.optimize_cell &&
            phys_g.size() >= (m_num_molecular_parameters + m_num_cell_parameters)) {
            const double gmol = phys_g.head(m_num_molecular_parameters).norm();
            const double gcell = phys_g.tail(m_num_cell_parameters).norm();
            occ::log::info(
                "Iter {:4d}:  E = {:14.6f} kJ/mol   |g| = {:.3e}  (rms {:.3e}, max {:.3e}; mol {:.3e}, cell {:.3e}; max|F| {:.3e}, max|tau| {:.3e}, max|stress| {:.3e} GPa)  ({:.3f} s, {} evals)",
                iter, e_per_mol, gnorm, grms, ginf, gmol, gcell,
                m_last_eval_max_force, m_last_eval_max_torque, m_last_eval_max_stress,
                iter_time, eval_count);
        } else {
            occ::log::info(
                "Iter {:4d}:  E = {:14.6f} kJ/mol   |g| = {:.3e}  (rms {:.3e}, max {:.3e}; max|F| {:.3e}, max|tau| {:.3e}, max|stress| {:.3e} GPa)  ({:.3f} s, {} evals)",
                iter, e_per_mol, gnorm, grms, ginf,
                m_last_eval_max_force, m_last_eval_max_torque, m_last_eval_max_stress,
                iter_time, eval_count);
        }
        std::fflush(stdout);

        const double dE = std::abs(f - prev_iter_energy);
        if (dE < stall_energy_tol && gnorm > (stall_grad_factor * grad_tol)) {
            stall_iter_count++;
        } else {
            stall_iter_count = 0;
        }
        if ((last_significant_energy - f) > stall_energy_tol) {
            last_significant_energy = f;
            eval_at_last_significant_improvement = eval_count;
        }
        prev_iter_energy = f;
        if (stall_iter_count >= stall_iter_limit) {
            occ::log::warn(
                "MSTMIN stall detected: {} consecutive accepted iterations with "
                "|dE|<{:.3e} kJ/mol and |g|>{:.3e}; stopping",
                stall_iter_count, stall_energy_tol, stall_grad_factor * grad_tol);
            stopped_for_stall = true;
            return false;
        }
        if (gnorm > (stall_grad_factor * grad_tol) &&
            (eval_count - eval_at_last_significant_improvement) >= stall_eval_limit) {
            occ::log::warn(
                "MSTMIN stall detected: no significant energy decrease (>{:.3e} kJ/mol) "
                "for {} objective evaluations while |g|={:.3e}; stopping",
                stall_energy_tol, eval_count - eval_at_last_significant_improvement, gnorm);
            stopped_for_stall = true;
            return false;
        }

        if (write_trajectory && m_trajectory_stream) {
            write_trajectory_frame(m_trajectory_stream, iter, f);
        }

        if (callback) {
            return callback(iter, f, gnorm);
        }
        return true;
    };

    auto mst_objective = [this, &eval_count, &param_scales,
                          &eval_count_at_last_accept, &next_stall_report_eval](const Vec& x, Vec& g) -> double {
        Vec phys_x = x.cwiseProduct(param_scales);
        Vec phys_g(g.size());
        eval_count++;
        const int report_interval = std::max(0, m_settings.mst_line_search_report_interval);
        if (report_interval > 0 && eval_count >= next_stall_report_eval) {
            const int stalled_evals = eval_count - eval_count_at_last_accept;
            occ::log::info(
                "  MSTMIN line search: {} trial evaluations since last accepted step "
                "(total evals: {})",
                stalled_evals, eval_count);
            next_stall_report_eval += report_interval;
        }
        double f = objective(phys_x, phys_g,
                             !m_settings.freeze_neighbors_during_linesearch);
        g = phys_g.cwiseProduct(param_scales);
        return f;
    };

    MSTMINResult mst_result = optimizer.minimize(
        mst_objective, scaled_params, mst_callback, m_settings.max_iterations);

    if (stopped_for_stall) {
        mst_result.converged = false;
        mst_result.termination_reason =
            "Stalled: tiny accepted energy changes with large gradient";
    }

    Vec final_scaled_params = mst_result.x;
    bool used_best_iterate = false;
    if (!mst_result.converged && best_energy < mst_result.final_energy) {
        final_scaled_params = best_scaled_params;
        used_best_iterate = true;
        if (!mst_result.termination_reason.empty()) {
            mst_result.termination_reason += "; returned best accepted iterate";
        } else {
            mst_result.termination_reason = "Returned best accepted iterate";
        }
    }

    Vec final_params = final_scaled_params.cwiseProduct(param_scales);
    set_parameters(final_params);
    auto final_result = m_energy.compute(m_states);
    const double pv =
        (m_settings.external_pressure_gpa /
         occ::units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA) *
        m_energy.crystal().unit_cell().volume();

    if (used_best_iterate) {
        occ::log::warn(
            "MSTMIN fallback: returning best accepted iterate (E = {:.6f} kJ/mol) "
            "instead of failed endpoint (E = {:.6f} kJ/mol)",
            best_energy, mst_result.final_energy);
    }

    result.pressure_volume_energy = pv / m_num_molecules;
    result.final_energy =
        final_result.total_energy / m_num_molecules + result.pressure_volume_energy;
    result.electrostatic_energy = final_result.electrostatic_energy / m_num_molecules;
    result.repulsion_dispersion_energy = final_result.repulsion_dispersion / m_num_molecules;
    result.iterations = mst_result.iterations;
    result.function_evaluations = eval_count;
    result.converged = mst_result.converged;
    result.termination_reason = mst_result.termination_reason;
    result.final_states = m_states;
    result.optimized_crystal = build_optimized_crystal();
    result.initial_energy /= m_num_molecules;

    if (m_trajectory_stream.is_open()) {
        m_trajectory_stream.close();
    }

    return result;
}

CrystalOptimizerResult CrystalOptimizer::optimize_lbfgs(IterationCallback callback) {
    CrystalOptimizerResult result;

    // Handle edge case: no parameters to optimize
    if (m_num_parameters == 0) {
        occ::log::warn("No parameters to optimize (all DOF fixed)");
        auto energy_result = m_energy.compute(m_states);
        const double pv =
            (m_settings.external_pressure_gpa /
             occ::units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA) *
            m_energy.crystal().unit_cell().volume();
        result.pressure_volume_energy = pv / m_num_molecules;
        result.initial_energy =
            energy_result.total_energy / m_num_molecules + result.pressure_volume_energy;
        result.final_energy = result.initial_energy;
        result.electrostatic_energy = energy_result.electrostatic_energy / m_num_molecules;
        result.repulsion_dispersion_energy = energy_result.repulsion_dispersion / m_num_molecules;
        result.iterations = 0;
        result.function_evaluations = 1;
        result.converged = true;
        result.termination_reason = "No parameters to optimize";
        result.final_states = m_states;
        result.optimized_crystal = build_optimized_crystal();
        return result;
    }

    // Compute initial energy
    Vec params = get_parameters();
    Vec grad(m_num_parameters);

    occ::timing::StopWatch<> sw;
    sw.start();
    result.initial_energy = objective(params, grad);
    double init_time = sw.stop().count();

    // Scale gradient tolerance by DOF to interpret user tol as per-DOF RMS
    double grad_tol = m_settings.gradient_tolerance * std::sqrt(static_cast<double>(m_num_parameters));

    const double gnorm0 = grad.norm();
    const double grms0 = gnorm0 / std::sqrt(static_cast<double>(std::max(1, m_num_parameters)));
    const double ginf0 = grad.cwiseAbs().maxCoeff();
    if (m_settings.optimize_cell) {
        occ::log::info(
            "Initial:    E = {:14.6f} kJ/mol   |g| = {:.3e}  (rms {:.3e}, max {:.3e}; max|F| {:.3e}, max|tau| {:.3e}, max|stress| {:.3e} GPa)  ({:.3f} s)",
            result.initial_energy / m_num_molecules, gnorm0, grms0, ginf0,
            m_last_eval_max_force, m_last_eval_max_torque, m_last_eval_max_stress, init_time);
    } else {
        occ::log::info(
            "Initial:    E = {:14.6f} kJ/mol   |g| = {:.3e}  (rms {:.3e}, max {:.3e}; max|F| {:.3e}, max|tau| {:.3e})  ({:.3f} s)",
            result.initial_energy / m_num_molecules, gnorm0, grms0, ginf0,
            m_last_eval_max_force, m_last_eval_max_torque, init_time);
    }
    std::fflush(stdout);

    // Open trajectory file if requested (callback will write frames)
    bool write_trajectory = !m_settings.trajectory_file.empty();
    if (write_trajectory) {
        m_trajectory_stream.open(m_settings.trajectory_file);
        if (m_trajectory_stream) {
            occ::log::info("Writing trajectory to {}", m_settings.trajectory_file);
        } else {
            occ::log::warn("Could not open trajectory file: {}", m_settings.trajectory_file);
            write_trajectory = false;
        }
    }

    // Setup L-BFGS with simple backtracking line search
    LBFGSSettings lbfgs_settings;
    lbfgs_settings.memory = m_settings.lbfgs_memory;
    lbfgs_settings.gradient_tol = grad_tol;
    lbfgs_settings.energy_tol = m_settings.energy_tolerance * static_cast<double>(m_num_molecules);
    // Use simple backtracking (Armijo only) for fewer evaluations
    lbfgs_settings.backtracking_only = true;
    lbfgs_settings.backtrack_factor = 0.5;
    lbfgs_settings.max_linesearch = 20;
    // Initial step scaled by inverse gradient norm
    lbfgs_settings.initial_step = std::min(1.0, 1.0 / grad.norm());

    LBFGS optimizer(lbfgs_settings);

    // Run optimization
    int eval_count = 1;  // Already did one evaluation
    occ::timing::StopWatch<> iter_sw;
    iter_sw.start();

    auto lbfgs_callback = [this, &callback, &eval_count, &iter_sw, write_trajectory](
        int iter, const Vec& x, double f, const Vec& g) -> bool {

        double iter_time = iter_sw.stop().count();
        iter_sw.start();  // Reset for next iteration

        // Update states for trajectory writing
        const_cast<CrystalOptimizer*>(this)->set_parameters(x);

        double gnorm = g.norm();
        const double grms = gnorm / std::sqrt(static_cast<double>(std::max(1, m_num_parameters)));
        const double ginf = g.cwiseAbs().maxCoeff();
        double e_per_mol = f / m_num_molecules;
        if (m_settings.optimize_cell &&
            g.size() >= (m_num_molecular_parameters + m_num_cell_parameters)) {
            const double gmol = g.head(m_num_molecular_parameters).norm();
            const double gcell = g.tail(m_num_cell_parameters).norm();
            occ::log::info(
                "Iter {:4d}:  E = {:14.6f} kJ/mol   |g| = {:.3e}  (rms {:.3e}, max {:.3e}; mol {:.3e}, cell {:.3e}; max|F| {:.3e}, max|tau| {:.3e}, max|stress| {:.3e} GPa)  ({:.3f} s, {} evals)",
                iter, e_per_mol, gnorm, grms, ginf, gmol, gcell,
                m_last_eval_max_force, m_last_eval_max_torque, m_last_eval_max_stress,
                iter_time, eval_count);
        } else {
            occ::log::info(
                "Iter {:4d}:  E = {:14.6f} kJ/mol   |g| = {:.3e}  (rms {:.3e}, max {:.3e}; max|F| {:.3e}, max|tau| {:.3e}, max|stress| {:.3e} GPa)  ({:.3f} s, {} evals)",
                iter, e_per_mol, gnorm, grms, ginf,
                m_last_eval_max_force, m_last_eval_max_torque, m_last_eval_max_stress,
                iter_time, eval_count);
        }
        std::fflush(stdout);

        // Write trajectory frame
        if (write_trajectory && m_trajectory_stream) {
            write_trajectory_frame(m_trajectory_stream, iter, f);
        }

        if (callback) {
            return callback(iter, f, gnorm);
        }
        return true;
    };

    auto lbfgs_objective = [this, &eval_count](const Vec& x, Vec& g) -> double {
        eval_count++;
        return objective(x, g, !m_settings.freeze_neighbors_during_linesearch);
    };

    LBFGSResult lbfgs_result = optimizer.minimize(
        lbfgs_objective, params, lbfgs_callback, m_settings.max_iterations);

    // Update states with final parameters
    set_parameters(lbfgs_result.x);

    // Final energy evaluation for components
    auto final_result = m_energy.compute(m_states);
    const double pv =
        (m_settings.external_pressure_gpa /
         occ::units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA) *
        m_energy.crystal().unit_cell().volume();

    // Build result
    result.pressure_volume_energy = pv / m_num_molecules;
    result.final_energy =
        final_result.total_energy / m_num_molecules + result.pressure_volume_energy;
    result.electrostatic_energy = final_result.electrostatic_energy / m_num_molecules;
    result.repulsion_dispersion_energy = final_result.repulsion_dispersion / m_num_molecules;
    result.iterations = lbfgs_result.iterations;
    result.function_evaluations = eval_count;
    result.converged = lbfgs_result.converged;
    result.termination_reason = lbfgs_result.termination_reason;
    result.final_states = m_states;
    result.optimized_crystal = build_optimized_crystal();
    result.initial_energy /= m_num_molecules;

    // Close trajectory file
    if (m_trajectory_stream.is_open()) {
        m_trajectory_stream.close();
    }

    return result;
}

// ============================================================================
// Crystal Reconstruction
// ============================================================================

crystal::Crystal CrystalOptimizer::build_optimized_crystal() const {
    const auto& current = m_energy.crystal();
    const auto& ref = m_reference_crystal;
    const auto& unique_mols = ref.symmetry_unique_molecules();

    // Copy the asymmetric unit and modify positions
    crystal::AsymmetricUnit asym = ref.asymmetric_unit();

    // For each unique molecule, compute new atom positions
    for (size_t mol_idx = 0; mol_idx < unique_mols.size(); ++mol_idx) {
        int state_idx = static_cast<int>(mol_idx);
        if (m_settings.use_symmetry &&
            mol_idx < m_symmetry_mapping.independent.size() &&
            !m_symmetry_mapping.independent[mol_idx].uc_indices.empty()) {
            state_idx = m_symmetry_mapping.independent[mol_idx].uc_indices[0];
        }
        if (state_idx < 0 || state_idx >= static_cast<int>(m_states.size())) {
            continue;
        }

        const auto& mol = unique_mols[mol_idx];
        const auto& state = m_states[state_idx];
        const auto& asym_indices = mol.asymmetric_unit_idx();

        // Get the transformation from asymmetric unit to this molecule
        auto [asym_rot, asym_trans] = mol.asymmetric_unit_transformation();

        Vec3 original_com = mol.center_of_mass();
        Mat3 R = state.rotation_matrix();

        // Transform each atom
        for (int atom_idx = 0; atom_idx < mol.size(); ++atom_idx) {
            Vec3 original_pos = mol.positions().col(atom_idx);
            Vec3 body_pos = original_pos - original_com;
            Vec3 new_cart_pos = state.position + R * body_pos;

            // Convert new position to fractional
            Vec3 new_frac_pos = current.to_fractional(new_cart_pos);

            // Get asymmetric unit index for this atom
            int asym_idx = asym_indices(atom_idx);
            if (asym_idx >= 0 && asym_idx < asym.positions.cols()) {
                // Apply inverse of molecule's symop to get asymmetric unit position
                Vec3 asym_frac = asym_rot.transpose() * (new_frac_pos - asym_trans);
                asym.positions.col(asym_idx) = asym_frac;
            }
        }
    }

    // Construct new Crystal with modified asymmetric unit (fresh caches)
    return crystal::Crystal(asym, ref.space_group(), current.unit_cell());
}

// ============================================================================
// Gradient Check (for debugging)
// ============================================================================

bool CrystalOptimizer::check_gradient(const Vec& params, double tol) {
    if (m_num_parameters == 0) {
        occ::log::info("No parameters to check");
        return true;
    }

    const double h = 1e-5;
    Vec grad(m_num_parameters);
    double f0 = objective(params, grad);

    Vec fd_grad(m_num_parameters);
    Vec params_p = params;
    Vec params_m = params;

    for (int i = 0; i < m_num_parameters; ++i) {
        params_p[i] = params[i] + h;
        params_m[i] = params[i] - h;

        Vec dummy(m_num_parameters);
        double fp = objective(params_p, dummy);
        double fm = objective(params_m, dummy);

        fd_grad[i] = (fp - fm) / (2.0 * h);

        params_p[i] = params[i];
        params_m[i] = params[i];
    }

    double max_error = 0.0;
    for (int i = 0; i < m_num_parameters; ++i) {
        double error = std::abs(grad[i] - fd_grad[i]);
        double denom = std::max(1.0, std::abs(grad[i]));
        double rel_error = error / denom;
        max_error = std::max(max_error, rel_error);

        if (rel_error > tol) {
            occ::log::warn("Gradient check failed at index {}: analytical={:.6e}, fd={:.6e}, error={:.6e}",
                          i, grad[i], fd_grad[i], rel_error);
        }
    }

    occ::log::info("Gradient check: max relative error = {:.6e}", max_error);
    return max_error < tol;
}

// ============================================================================
// Trajectory Output
// ============================================================================

void CrystalOptimizer::write_trajectory_frame(std::ofstream& file, int iter, double energy) const {
    const auto& crystal = m_energy.crystal();
    const auto& geometry = m_energy.molecule_geometry();
    const auto& unit_cell = crystal.unit_cell();

    // Count total atoms
    const int nstates = std::min(static_cast<int>(m_states.size()),
                                 static_cast<int>(geometry.size()));
    int total_atoms = 0;
    for (int i = 0; i < nstates; ++i) {
        total_atoms += static_cast<int>(std::min(
            geometry[i].atomic_numbers.size(),
            geometry[i].atom_positions.size()));
    }

    Mat3 lattice = unit_cell.direct();

    // Write extended XYZ header
    file << total_atoms << "\n";
    file << fmt::format(
        "Lattice=\"{:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\" "
        "Properties=species:S:1:pos:R:3 "
        "energy={:.10f} "
        "iter={} "
        "pbc=\"T T T\"\n",
        lattice(0, 0), lattice(1, 0), lattice(2, 0),
        lattice(0, 1), lattice(1, 1), lattice(2, 1),
        lattice(0, 2), lattice(1, 2), lattice(2, 2),
        energy / m_num_molecules, iter);

    // Write transformed atomic positions directly from cached body-frame geometry.
    for (int i = 0; i < nstates; ++i) {
        const auto& geom = geometry[i];
        const auto& state = m_states[i];
        Mat3 R = state.rotation_matrix();
        const int nat = std::min(static_cast<int>(geom.atomic_numbers.size()),
                                 static_cast<int>(geom.atom_positions.size()));

        for (int a = 0; a < nat; ++a) {
            Vec3 new_pos = state.position + R * geom.atom_positions[a];
            std::string symbol = occ::core::Element(geom.atomic_numbers[a]).symbol();
            file << fmt::format("{:2s} {:16.10f} {:16.10f} {:16.10f}\n",
                               symbol, new_pos.x(), new_pos.y(), new_pos.z());
        }
    }
}

} // namespace occ::mults
