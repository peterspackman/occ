#include <CLI/App.hpp>
#include <filesystem>
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <fstream>
#include <fmt/os.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/driver/dma_driver.h>
#include <occ/dma/mult.h>
#include <occ/io/cifparser.h>
#include <occ/io/cifwriter.h>
#include <occ/io/structure_format.h>
#include <occ/main/monomer_wavefunctions.h>
#include <occ/main/occ_ropt.h>
#include <occ/mults/crystal_optimizer.h>
#include <occ/mults/crystal_energy_setup.h>
#include <occ/mults/multipole_source.h>

namespace fs = std::filesystem;
using occ::crystal::Crystal;
using occ::mults::CrystalEnergy;
using occ::mults::CrystalOptimizer;
using occ::mults::CrystalOptimizerSettings;
using occ::mults::ForceFieldType;
using occ::mults::MultipoleSource;
using occ::mults::MoleculeState;
using occ::mults::OptimizationMethod;

namespace {

struct ElasticTensorSummary {
    occ::Mat6 clamped_gpa = occ::Mat6::Zero();
    occ::Mat6 clamped_raw_gpa = occ::Mat6::Zero();
    std::optional<occ::Mat6> relaxed_gpa;
    std::optional<occ::Mat6> relaxed_raw_gpa;
    bool exact_for_model = false;
    int dropped_modes = 0;
    occ::Vec6 residual_stress_gpa = occ::Vec6::Zero();
    std::string relaxed_unavailable_reason;
};

occ::Vec6 strain_gradient_to_stress_gpa(const occ::Vec6& dE_dE,
                                        double volume_ang3) {
    occ::Vec6 sigma = occ::Vec6::Zero();
    if (volume_ang3 <= 1e-12) {
        return sigma;
    }
    sigma[0] = dE_dE[0] / volume_ang3;
    sigma[1] = dE_dE[1] / volume_ang3;
    sigma[2] = dE_dE[2] / volume_ang3;
    sigma[3] = 2.0 * dE_dE[3] / volume_ang3;
    sigma[4] = 2.0 * dE_dE[4] / volume_ang3;
    sigma[5] = 2.0 * dE_dE[5] / volume_ang3;
    sigma *= occ::units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;
    return sigma;
}

occ::Mat3 voigt_stress_to_tensor(const occ::Vec6& s) {
    occ::Mat3 sigma = occ::Mat3::Zero();
    sigma(0, 0) = s[0];
    sigma(1, 1) = s[1];
    sigma(2, 2) = s[2];
    sigma(1, 2) = sigma(2, 1) = s[3];
    sigma(0, 2) = sigma(2, 0) = s[4];
    sigma(0, 1) = sigma(1, 0) = s[5];
    return sigma;
}

const std::array<occ::Mat3, 6>& voigt_basis_matrices() {
    static const std::array<occ::Mat3, 6> B = [] {
        std::array<occ::Mat3, 6> out{};
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

occ::Mat6 finite_stress_correction_voigt_gpa(const occ::Vec6& stress_gpa) {
    const auto& B = voigt_basis_matrices();
    const occ::Mat3 sigma = voigt_stress_to_tensor(stress_gpa);

    occ::Mat6 correction = occ::Mat6::Zero();
    for (int a = 0; a < 6; ++a) {
        for (int b = 0; b < 6; ++b) {
            double val = 0.0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    for (int k = 0; k < 3; ++k) {
                        for (int l = 0; l < 3; ++l) {
                            const double A_ijkl = 0.5 * (
                                (i == k ? sigma(j, l) : 0.0) +
                                (j == k ? sigma(i, l) : 0.0) +
                                (i == l ? sigma(j, k) : 0.0) +
                                (j == l ? sigma(i, k) : 0.0) -
                                (k == l ? 2.0 * sigma(i, j) : 0.0));
                            val += B[a](i, j) * A_ijkl * B[b](k, l);
                        }
                    }
                }
            }
            correction(a, b) = val;
        }
    }
    return 0.5 * (correction + correction.transpose());
}

occ::Mat solve_symmetric_filtered(const occ::Mat& A, const occ::Mat& B,
                                  double rel_tol, double abs_tol,
                                  int* dropped_modes = nullptr) {
    Eigen::SelfAdjointEigenSolver<occ::Mat> es(A);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("Failed eigen-decomposition in relaxed elastic solve");
    }

    const occ::Vec evals = es.eigenvalues();
    const occ::Mat evecs = es.eigenvectors();
    const double max_abs_eval =
        (evals.size() > 0) ? evals.cwiseAbs().maxCoeff() : 0.0;
    const double thresh = std::max(abs_tol, rel_tol * max_abs_eval);

    occ::Vec inv = occ::Vec::Zero(evals.size());
    int dropped = 0;
    for (int i = 0; i < evals.size(); ++i) {
        if (std::abs(evals(i)) > thresh) {
            inv(i) = 1.0 / evals(i);
        } else {
            dropped++;
        }
    }
    if (dropped_modes) {
        *dropped_modes = dropped;
    }

    return (evecs * inv.asDiagonal() * evecs.transpose() * B).eval();
}

std::vector<int> reduced_internal_dof_indices(int n_mol) {
    std::vector<int> idx;
    idx.reserve(std::max(0, 6 * n_mol - 3));
    // Remove molecule-0 global translations; keep rotations.
    idx.push_back(3);
    idx.push_back(4);
    idx.push_back(5);
    for (int i = 1; i < n_mol; ++i) {
        for (int c = 0; c < 6; ++c) {
            idx.push_back(6 * i + c);
        }
    }
    return idx;
}

occ::Mat build_symmetry_internal_parameter_map(
    const occ::mults::SymmetryMapping& mapping,
    bool fix_first_translation,
    bool fix_first_rotation) {

    const int n_uc = mapping.num_uc_molecules;
    const int ndof_full = 6 * n_uc;

    int n_params = 0;
    for (int k = 0; k < mapping.num_independent; ++k) {
        const auto& info = mapping.independent[k];
        const bool skip_trans = (k == 0 && fix_first_translation);
        const bool skip_rot = (k == 0 && fix_first_rotation);
        if (!skip_trans) n_params += info.trans_dof;
        if (!skip_rot) n_params += info.rot_dof;
    }

    occ::Mat T = occ::Mat::Zero(ndof_full, n_params);
    int col = 0;
    for (int k = 0; k < mapping.num_independent; ++k) {
        const auto& info = mapping.independent[k];
        const bool skip_trans = (k == 0 && fix_first_translation);
        const bool skip_rot = (k == 0 && fix_first_rotation);

        if (!skip_trans) {
            for (int d = 0; d < info.trans_dof; ++d) {
                const occ::Vec3 v_indep = info.trans_basis.col(d);
                for (int uc_idx : info.uc_indices) {
                    const auto& uc = mapping.uc_molecules[uc_idx];
                    T.block<3, 1>(6 * uc_idx, col) = uc.R_cart * v_indep;
                }
                ++col;
            }
        }

        if (!skip_rot) {
            for (int d = 0; d < info.rot_dof; ++d) {
                const occ::Vec3 v_indep = info.rot_basis.col(d);
                for (int uc_idx : info.uc_indices) {
                    const auto& uc = mapping.uc_molecules[uc_idx];
                    const double det = uc.R_cart.determinant();
                    T.block<3, 1>(6 * uc_idx + 3, col) =
                        (det * uc.R_cart) * v_indep;
                }
                ++col;
            }
        }
    }

    return T;
}

ElasticTensorSummary compute_elastic_tensor_summary(
    CrystalOptimizer& optimizer,
    const std::vector<MoleculeState>& states,
    bool converged) {

    auto& energy = optimizer.energy_calculator();
    auto eh = energy.compute_with_hessian(states);
    const double V = energy.crystal().unit_cell().volume();
    if (V <= 1e-12) {
        throw std::runtime_error("Invalid unit-cell volume for elastic tensor");
    }

    ElasticTensorSummary out;
    out.exact_for_model = eh.exact_for_model;
    out.residual_stress_gpa =
        strain_gradient_to_stress_gpa(eh.strain_gradient, V);
    const occ::Mat6 stress_correction_gpa =
        finite_stress_correction_voigt_gpa(out.residual_stress_gpa);
    out.clamped_raw_gpa =
        (eh.strain_hessian / V) * occ::units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;
    out.clamped_gpa = out.clamped_raw_gpa + stress_correction_gpa;

    // Relaxed tensor requires a complete internal Hessian model and a
    // converged stationary point.
    if (!eh.exact_for_model) {
        out.relaxed_unavailable_reason =
            "second derivatives are not exact for current model";
        return out;
    }
    if (!converged) {
        out.relaxed_unavailable_reason =
            "structure is not converged; relaxed elastic constants are not reliable";
        return out;
    }
    occ::Mat W_ii;
    occ::Mat W_ei;
    if (optimizer.settings().use_symmetry) {
        const auto& mapping = optimizer.symmetry_mapping();
        const occ::Mat T = build_symmetry_internal_parameter_map(
            mapping,
            optimizer.settings().fix_first_translation,
            optimizer.settings().fix_first_rotation);
        if (T.cols() == 0) {
            out.relaxed_unavailable_reason =
                "no active internal symmetry parameters for relaxed elastic projection";
            return out;
        }
        if (eh.strain_state_hessian.cols() != T.rows()) {
            throw std::runtime_error(
                "Symmetry elastic projection: strain-state Hessian dimension mismatch");
        }
        const occ::Mat H_full = eh.pack_hessian(false, false);
        if (H_full.rows() != T.rows() || H_full.cols() != T.rows()) {
            throw std::runtime_error(
                "Symmetry elastic projection: internal Hessian dimension mismatch");
        }
        W_ii = T.transpose() * H_full * T;
        W_ei = eh.strain_state_hessian * T;
    } else {
        const int n_mol = static_cast<int>(states.size());
        W_ii = eh.pack_hessian(true, false);
        const auto reduced_to_full = reduced_internal_dof_indices(n_mol);
        if (static_cast<int>(reduced_to_full.size()) != W_ii.rows()) {
            throw std::runtime_error("Internal DOF mapping mismatch in elastic tensor");
        }
        W_ei = occ::Mat::Zero(6, W_ii.rows());
        for (int k = 0; k < W_ii.rows(); ++k) {
            W_ei.col(k) = eh.strain_state_hessian.col(reduced_to_full[k]);
        }
    }

    const occ::Mat X = solve_symmetric_filtered(
        W_ii, W_ei.transpose(), 1e-10, 1e-8, &out.dropped_modes);
    const occ::Mat6 correction = W_ei * X;
    const occ::Mat6 W_relaxed = eh.strain_hessian - correction;
    out.relaxed_raw_gpa =
        (W_relaxed / V) * occ::units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;
    out.relaxed_gpa = *out.relaxed_raw_gpa + stress_correction_gpa;

    return out;
}

void print_elastic_tensor_summary(const ElasticTensorSummary& summary,
                                  bool converged) {
    const auto print_mat6 = [](const occ::Mat6& C) {
        for (int i = 0; i < 6; ++i) {
            occ::log::info("  {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}",
                           C(i, 0), C(i, 1), C(i, 2), C(i, 3), C(i, 4), C(i, 5));
        }
    };

    occ::log::info("");
    occ::log::info("Elastic tensor at {} structure (GPa):",
                   converged ? "minimum/final" : "final (not converged)");
    occ::log::info("  Residual stress [s11,s22,s33,s23,s13,s12] (GPa): "
                   "[{:+.4f}, {:+.4f}, {:+.4f}, {:+.4f}, {:+.4f}, {:+.4f}]",
                   summary.residual_stress_gpa[0], summary.residual_stress_gpa[1],
                   summary.residual_stress_gpa[2], summary.residual_stress_gpa[3],
                   summary.residual_stress_gpa[4], summary.residual_stress_gpa[5]);
    occ::log::info("  Clamped elastic constants C_ij (raw, without stress correction):");
    print_mat6(summary.clamped_raw_gpa);
    occ::log::info("  Clamped elastic constants C_ij (stress-corrected):");
    print_mat6(summary.clamped_gpa);

    if (summary.relaxed_gpa.has_value()) {
        if (summary.relaxed_raw_gpa.has_value()) {
            occ::log::info("  Relaxed elastic constants C_ij (raw, without stress correction):");
            print_mat6(*summary.relaxed_raw_gpa);
        }
        occ::log::info("  Relaxed elastic constants C_ij (stress-corrected):");
        print_mat6(*summary.relaxed_gpa);
        if (summary.dropped_modes > 0) {
            occ::log::info(
                "  Relaxed solve: filtered {} near-null internal mode(s)",
                summary.dropped_modes);
        }
    } else {
        if (!summary.relaxed_unavailable_reason.empty()) {
            occ::log::warn("  Relaxed elastic tensor unavailable: {}",
                           summary.relaxed_unavailable_reason);
        } else {
            occ::log::warn("  Relaxed elastic tensor unavailable");
        }
    }
}

void write_elastic_tensor(const std::string &filepath,
                          const ElasticTensorSummary &summary) {
    auto ext = fs::path(filepath).extension().string();

    if (ext == ".json") {
        nlohmann::json j;
        auto mat_to_json = [](const occ::Mat6 &C) {
            nlohmann::json rows = nlohmann::json::array();
            for (int i = 0; i < 6; ++i) {
                nlohmann::json row = nlohmann::json::array();
                for (int k = 0; k < 6; ++k)
                    row.push_back(C(i, k));
                rows.push_back(row);
            }
            return rows;
        };
        j["clamped_gpa"] = mat_to_json(summary.clamped_gpa);
        j["clamped_raw_gpa"] = mat_to_json(summary.clamped_raw_gpa);
        if (summary.relaxed_gpa.has_value()) {
            j["relaxed_gpa"] = mat_to_json(*summary.relaxed_gpa);
        }
        if (summary.relaxed_raw_gpa.has_value()) {
            j["relaxed_raw_gpa"] = mat_to_json(*summary.relaxed_raw_gpa);
        }
        std::ofstream out(filepath);
        out << j.dump(2) << "\n";
        occ::log::info("Wrote elastic tensor (JSON) to {}", filepath);
    } else {
        // Plain text: print clamped then relaxed as 6x6 matrices
        std::ofstream out(filepath);
        out << "# Clamped elastic constants (GPa)\n";
        out << summary.clamped_gpa.format(
            Eigen::IOFormat(6, 0, "  ", "\n")) << "\n";
        if (summary.relaxed_gpa.has_value()) {
            out << "\n# Relaxed elastic constants (GPa)\n";
            out << summary.relaxed_gpa->format(
                Eigen::IOFormat(6, 0, "  ", "\n")) << "\n";
        }
        occ::log::info("Wrote elastic tensor to {}", filepath);
    }
}

const char* method_name(OptimizationMethod method) {
    switch (method) {
    case OptimizationMethod::MSTMIN:
        return "MSTMIN quasi-Newton";
    case OptimizationMethod::LBFGS:
        return "L-BFGS (quasi-Newton)";
    case OptimizationMethod::TrustRegion:
        return "Trust Region Newton (2nd order)";
    case OptimizationMethod::TrustRegionBFGS:
        return "Trust Region BFGS (quasi-Newton)";
    default:
        return "Unknown";
    }
}

Crystal read_crystal(const std::string &filename) {
    occ::io::CifParser parser;
    return parser.parse_crystal_from_file(filename).value();
}

void set_charges_and_multiplicities(const std::string &charge_string,
                                    const std::string &multiplicity_string,
                                    std::vector<occ::core::Molecule> &molecules) {
    if (!charge_string.empty()) {
        std::vector<int> charges;
        auto tokens = occ::util::tokenize(charge_string, ",");
        for (const auto &token : tokens) {
            charges.push_back(std::stoi(token));
        }
        if (charges.size() != molecules.size()) {
            throw std::runtime_error(
                fmt::format("Require {} charges to be specified, found {}",
                            molecules.size(), charges.size()));
        }
        for (size_t i = 0; i < charges.size(); i++) {
            occ::log::info("Setting net charge for molecule {} = {}", i, charges[i]);
            molecules[i].set_charge(charges[i]);
        }
    }

    if (!multiplicity_string.empty()) {
        std::vector<int> multiplicities;
        auto tokens = occ::util::tokenize(multiplicity_string, ",");
        for (const auto &token : tokens) {
            multiplicities.push_back(std::stoi(token));
        }
        if (multiplicities.size() != molecules.size()) {
            throw std::runtime_error(
                fmt::format("Require {} multiplicities to be specified, found {}",
                            molecules.size(), multiplicities.size()));
        }
        for (size_t i = 0; i < multiplicities.size(); i++) {
            occ::log::info("Setting multiplicity for molecule {} = {}", i,
                           multiplicities[i]);
            molecules[i].set_multiplicity(multiplicities[i]);
        }
    }
}

std::vector<MultipoleSource> compute_multipoles_for_crystal(
    const std::string &basename,
    Crystal &crystal,
    const std::string &model_name,
    bool spherical_basis) {

    auto molecules = crystal.symmetry_unique_molecules();
    occ::log::info("Computing multipoles for {} unique molecules", molecules.size());

    // Compute wavefunctions
    auto wavefunctions = occ::main::calculate_wavefunctions(
        basename, molecules, model_name, spherical_basis);

    std::vector<MultipoleSource> sources;
    sources.reserve(molecules.size());

    for (size_t i = 0; i < molecules.size(); ++i) {
        occ::log::info("Computing DMA for molecule {}", i);

        // Setup DMA config
        occ::driver::DMAConfig dma_config;
        dma_config.settings.max_rank = 4;  // Up to hexadecapole
        dma_config.settings.big_exponent = 4.0;

        // Run DMA directly on wavefunction
        occ::driver::DMADriver driver(dma_config);
        auto output = driver.run(wavefunctions[i]);

        // Convert DMA result to MultipoleSource
        // Each DMA site becomes a body-frame site
        std::vector<MultipoleSource::BodySite> body_sites;

        const auto &mol = molecules[i];
        occ::Vec3 com = mol.center_of_mass();

        for (size_t site_idx = 0; site_idx < output.result.multipoles.size(); ++site_idx) {
            MultipoleSource::BodySite site;
            site.multipole = output.result.multipoles[site_idx];

            // Position relative to COM (convert from bohr to angstrom)
            occ::Vec3 pos = output.sites.positions.col(site_idx) * occ::units::BOHR_TO_ANGSTROM;
            site.offset = pos - com;

            body_sites.push_back(site);
        }

        sources.emplace_back(std::move(body_sites));

        // Set initial orientation from crystal
        sources.back().set_orientation(occ::Mat3::Identity(), com);
    }

    return sources;
}

/// Expand Z' multipole sources to Z by applying symmetry operations.
/// Each UC molecule gets a copy of its corresponding unique molecule's
/// body-frame multipoles, oriented by the crystallographic symmetry operation.
std::vector<MultipoleSource> expand_to_unit_cell(
    const std::vector<MultipoleSource> &unique_sources,
    Crystal &crystal) {

    const auto &uc_mols = crystal.unit_cell_molecules();
    int Z = static_cast<int>(uc_mols.size());
    int Z_prime = static_cast<int>(unique_sources.size());

    if (Z == Z_prime) {
        return unique_sources;  // Already expanded (P1 or manually expanded)
    }

    std::vector<MultipoleSource> expanded;
    expanded.reserve(Z);

    for (int m = 0; m < Z; ++m) {
        const auto &uc_mol = uc_mols[m];
        int asym_idx = uc_mol.asymmetric_molecule_idx();
        if (asym_idx < 0 || asym_idx >= Z_prime) {
            occ::log::warn("UC molecule {} has invalid asymmetric_molecule_idx {}", m, asym_idx);
            asym_idx = 0;
        }

        // Copy body sites from the unique molecule
        const auto &body_sites = unique_sources[asym_idx].body_sites();

        // Get the full crystallographic orientation (proper or improper).
        auto [R_sym, t_sym] = uc_mol.asymmetric_unit_transformation();
        const double det = R_sym.determinant();
        occ::Vec3 com_m = uc_mol.center_of_mass();

        expanded.emplace_back(body_sites);
        expanded.back().set_orientation(R_sym, com_m);

        occ::log::debug("UC mol {}: asym_idx={}, det(R)={:.0f}, COM=({:.4f}, {:.4f}, {:.4f})",
                        m, asym_idx, det,
                        com_m[0], com_m[1], com_m[2]);
    }

    occ::log::info("Expanded {} unique multipole sources to {} UC molecules",
                   Z_prime, Z);
    return expanded;
}

struct NeighborInteractionRow {
    int center = -1;
    int neighbor = -1;
    occ::IVec3 shift = occ::IVec3::Zero();
    double nearest_distance = 0.0;
    double com_distance = 0.0;
    double electrostatic = 0.0;
    double short_range = 0.0;
    int short_range_site_pairs = 0;
    double total = 0.0;
};

double compute_nearest_contact_distance(const CrystalEnergy& energy,
                                       const std::vector<MoleculeState>& states,
                                       const occ::mults::PairEnergyDebug& pair) {
    const auto& geom = energy.molecule_geometry();
    if (geom.empty()) {
        return pair.com_distance;
    }

    const int gi = (geom.size() == 1) ? 0 : std::clamp(pair.mol_i, 0, static_cast<int>(geom.size()) - 1);
    const int gj = (geom.size() == 1) ? 0 : std::clamp(pair.mol_j, 0, static_cast<int>(geom.size()) - 1);
    const auto& geom_i = geom[gi];
    const auto& geom_j = geom[gj];
    if (geom_i.atom_positions.empty() || geom_j.atom_positions.empty()) {
        return pair.com_distance;
    }

    const occ::Mat3 R_i = states[pair.mol_i].rotation_matrix();
    const occ::Mat3 R_j = states[pair.mol_j].rotation_matrix();
    const occ::Vec3 t = energy.crystal().unit_cell().to_cartesian(
        pair.cell_shift.cast<double>());

    double r_min = std::numeric_limits<double>::infinity();
    for (const auto& a_body : geom_i.atom_positions) {
        const occ::Vec3 a = states[pair.mol_i].position + R_i * a_body;
        for (const auto& b_body : geom_j.atom_positions) {
            const occ::Vec3 b = states[pair.mol_j].position + t + R_j * b_body;
            r_min = std::min(r_min, (b - a).norm());
        }
    }

    return std::isfinite(r_min) ? r_min : pair.com_distance;
}

void print_neighbor_interaction_table(CrystalEnergy& energy,
                                      const std::vector<MoleculeState>& states) {
    if (states.empty()) {
        return;
    }

    constexpr int max_rows_per_molecule = 24;
    constexpr double nearest_shell_extra = 1.0; // Angstrom beyond the closest contact

    auto pairs = energy.debug_pair_energies(states);
    if (pairs.empty()) {
        occ::log::info("");
        occ::log::info("Nearest-neighbor dimer interaction summary: no neighbor pairs");
        return;
    }

    double pair_sum_elec = 0.0;
    double pair_sum_sr = 0.0;
    double pair_sum_total = 0.0;
    double pair_sum_sr_cross = 0.0;
    double pair_sum_sr_self = 0.0;
    long long pair_sum_sr_site_pairs = 0;
    for (const auto& p : pairs) {
        pair_sum_elec += p.electrostatic;
        pair_sum_sr += p.short_range;
        pair_sum_total += p.total;
        if (p.mol_i == p.mol_j) {
            pair_sum_sr_self += p.short_range;
        } else {
            pair_sum_sr_cross += p.short_range;
        }
        pair_sum_sr_site_pairs += static_cast<long long>(p.short_range_site_pairs);
    }

    const double inv_nmol = 1.0 / static_cast<double>(states.size());
    const double pair_elec_per_mol = pair_sum_elec * inv_nmol;
    const double pair_sr_per_mol = pair_sum_sr * inv_nmol;
    const double pair_total_per_mol = pair_sum_total * inv_nmol;

    // Re-evaluate once to expose full-model (including unresolved Ewald terms)
    // alongside pair-resolved real-space totals.
    const auto full = energy.compute(states);
    const double full_elec_per_mol = full.electrostatic_energy * inv_nmol;
    const double full_sr_per_mol = full.repulsion_dispersion * inv_nmol;
    const double full_total_per_mol = full.total_energy * inv_nmol;

    std::vector<std::vector<NeighborInteractionRow>> rows_by_center(states.size());
    for (const auto& p : pairs) {
        if (p.mol_i < 0 || p.mol_j < 0 ||
            p.mol_i >= static_cast<int>(states.size()) ||
            p.mol_j >= static_cast<int>(states.size())) {
            continue;
        }

        const double rn = compute_nearest_contact_distance(energy, states, p);
        NeighborInteractionRow a;
        a.center = p.mol_i;
        a.neighbor = p.mol_j;
        a.shift = p.cell_shift;
        a.nearest_distance = rn;
        a.com_distance = p.com_distance;
        a.electrostatic = p.electrostatic;
        a.short_range = p.short_range;
        a.short_range_site_pairs = p.short_range_site_pairs;
        a.total = p.total;
        rows_by_center[a.center].push_back(a);

        NeighborInteractionRow b;
        b.center = p.mol_j;
        b.neighbor = p.mol_i;
        b.shift = -p.cell_shift;
        b.nearest_distance = rn;
        b.com_distance = p.com_distance;
        b.electrostatic = p.electrostatic;
        b.short_range = p.short_range;
        b.short_range_site_pairs = p.short_range_site_pairs;
        b.total = p.total;
        rows_by_center[b.center].push_back(b);
    }

    occ::log::info("");
    occ::log::info("Nearest-neighbor dimer interaction summary (final structure):");
    if (energy.use_ewald()) {
        occ::log::info("  Note: pair electrostatics are real-space terms; reciprocal/self Ewald terms are not pair-resolved.");
    }
    occ::log::info("  Pair-resolved (real-space) per molecule: E_elec={:+.3f}  E_sr={:+.3f}  E_tot={:+.3f} kJ/mol",
                   pair_elec_per_mol, pair_sr_per_mol, pair_total_per_mol);
    occ::log::info("    SR decomposition per molecule: cross={:+.3f} self-image={:+.3f} kJ/mol",
                   pair_sum_sr_cross * inv_nmol, pair_sum_sr_self * inv_nmol);
    occ::log::info("  Pair-resolved short-range site pairs (unique, weighted-dimer sum): {}",
                   pair_sum_sr_site_pairs);
    if (energy.use_ewald()) {
        const double unresolved = full_total_per_mol - pair_total_per_mol;
        const double unresolved_elec = full_elec_per_mol - pair_elec_per_mol;
        occ::log::info("  Unresolved (mainly reciprocal/self Ewald) per molecule: E_elec={:+.3f}  E_tot={:+.3f} kJ/mol",
                       unresolved_elec, unresolved);
    }
    occ::log::info("  Full model per molecule: E_elec={:+.3f}  E_sr={:+.3f}  E_tot={:+.3f} kJ/mol",
                   full_elec_per_mol, full_sr_per_mol, full_total_per_mol);

    for (size_t center = 0; center < rows_by_center.size(); ++center) {
        auto& rows = rows_by_center[center];
        if (rows.empty()) {
            continue;
        }
        std::sort(rows.begin(), rows.end(),
                  [](const auto& lhs, const auto& rhs) {
                      if (lhs.nearest_distance != rhs.nearest_distance) {
                          return lhs.nearest_distance < rhs.nearest_distance;
                      }
                      return lhs.com_distance < rhs.com_distance;
                  });

        const double rn_cut = rows.front().nearest_distance + nearest_shell_extra;
        size_t n_shell = 0;
        while (n_shell < rows.size() && rows[n_shell].nearest_distance <= rn_cut) {
            ++n_shell;
        }
        if (n_shell == 0) {
            n_shell = std::min<size_t>(rows.size(), 1);
        }
        const size_t n_show = std::min<size_t>(n_shell, max_rows_per_molecule);

        double total_all = 0.0;
        long long sr_site_pairs_all = 0;
        for (const auto& r : rows) {
            total_all += r.total;
            sr_site_pairs_all += static_cast<long long>(r.short_range_site_pairs);
        }

        occ::log::info("  Molecule {}: {} nearest-shell dimers shown ({} total neighbors)",
                       center, n_show, rows.size());
        occ::log::info("    {:>7s} {:>7s} {:>13s} {:>6s} {:>9s} {:>9s} {:>9s}",
                       "Rn", "Rc", "Shift(hkl)", "Nbr", "E_elec", "E_sr", "E_tot");
        for (size_t i = 0; i < n_show; ++i) {
            const auto& r = rows[i];
            occ::log::info("    {:7.3f} {:7.3f} [{:2d} {:2d} {:2d}] {:6d} {:9.3f} {:9.3f} {:9.3f}",
                           r.nearest_distance, r.com_distance,
                           r.shift[0], r.shift[1], r.shift[2], r.neighbor,
                           r.electrostatic, r.short_range, r.total);
        }
        if (n_show < n_shell) {
            occ::log::info("    ... truncated nearest shell: showing {} of {} entries",
                           n_show, n_shell);
        }
        occ::log::info("    Molecule {} full interaction sum (all neighbors): {:+.3f} kJ/mol", center, total_all);
        occ::log::info("    Molecule {} lattice-energy share from pair terms (1/2 sum): {:+.3f} kJ/mol",
                       center, 0.5 * total_all);
        occ::log::info("    Molecule {} short-range site pairs (all neighbors): {}",
                       center, sr_site_pairs_all);
    }
}

} // anonymous namespace

namespace occ::main {

CLI::App *add_ropt_subcommand(CLI::App &app) {
    CLI::App *ropt = app.add_subcommand("ropt",
        "optimize rigid molecule crystal structure");

    auto config = std::make_shared<RoptSettings>();

    // Input/output
    ropt->add_option("crystal", config->crystal_filename,
                     "input crystal structure (CIF or structure JSON)")
        ->required();
    ropt->add_option("-o,--output", config->output_filename,
                     "output CIF filename (default: <input>_opt.cif)");
    ropt->add_option("--write-structure", config->structure_json,
                     "write structure JSON after optimization");
    ropt->add_option("--write-elastic", config->elastic_tensor_file,
                     "write elastic tensor to file (.json or .txt)");
    ropt->add_flag("--trajectory", config->write_trajectory,
                   "write trajectory to XYZ file");

    // DMA model (CIF path only)
    ropt->add_option("-m,--model", config->model_name,
                     "energy model for DMA calculation (default: ce-b3lyp)");
    ropt->add_option("--charges", config->charge_string,
                     "molecular charges (comma-separated)");
    ropt->add_option("--multiplicities", config->multiplicity_string,
                     "spin multiplicities (comma-separated)");

    // Optimization
    ropt->add_option("--optimizer", config->optimizer,
                     "optimization method: mstmin (default), lbfgs, trust-region");
    ropt->add_option("-r,--radius", config->neighbor_radius,
                     "neighbor cutoff in Angstrom (default: 20.0)");
    ropt->add_option("--gtol", config->gradient_tolerance,
                     "gradient convergence tolerance (default: 1e-3)");
    ropt->add_option("--etol", config->energy_tolerance,
                     "energy convergence tolerance (default: 1e-5)");
    ropt->add_option("--max-iter", config->max_iterations,
                     "maximum iterations (default: 200)");
    ropt->add_option("--max-order", config->max_interaction_order,
                     "max multipole interaction order lA+lB (default: 4)");

    // Cell optimization
    ropt->add_flag("--optimize-cell", config->optimize_cell,
                   "optimize unit cell (constrained by lattice symmetry)");
    ropt->add_option_function<double>(
        "--pressure",
        [config](double p) {
            config->has_external_pressure = true;
            config->external_pressure_gpa = p;
        },
        "external pressure in GPa (enthalpy objective)");
    ropt->add_flag("--no-elastic", [config](int64_t) {
        config->compute_elastic_tensor = false;
    }, "skip elastic tensor after cell optimization");

    // Electrostatics
    ropt->add_flag("--no-ewald", [config](int64_t) {
        config->use_ewald = false;
    }, "disable Ewald electrostatics");

    ropt->fallthrough();
    ropt->callback([config]() { run_ropt_subcommand(*config); });

    return ropt;
}

/// Prepared data for crystal optimization.
struct PreparedInputs {
    mults::CrystalEnergySetup setup;
    std::optional<occ::io::StructureInput> structure_input;
};

PreparedInputs prepare_from_json(const std::string &json_path) {
    occ::log::info("Loading structure JSON from {}", json_path);
    auto si = occ::io::read_structure_json(json_path);

    occ::log::info("  Title: {}", si.title);
    occ::log::info("  {} molecule types, {} independent molecules",
                   si.molecule_types.size(), si.molecules.size());
    occ::log::info("  {} Buckingham pairs", si.potentials.buckingham.size());

    if (si.reference.total != 0.0) {
        occ::log::info("  Reference energy: {:.6f} kJ/mol", si.reference.total);
    }
    if (si.settings.spline_min > 0.0) {
        occ::log::info("  SPLI settings: min={:.3f} max={:.3f} order={}",
                       si.settings.spline_min, si.settings.spline_max,
                       si.settings.spline_order);
    }

    auto setup = mults::from_structure_input(si);
    return PreparedInputs{std::move(setup), std::move(si)};
}

PreparedInputs prepare_from_dma(const RoptSettings &settings,
                                const std::string &filename,
                                const std::string &basename) {
    occ::log::info("Loading crystal from {}", filename);
    Crystal cryst = read_crystal(filename);

    auto molecules = cryst.symmetry_unique_molecules();
    occ::log::info("Crystal has {} symmetry-unique molecules",
                   molecules.size());

    set_charges_and_multiplicities(settings.charge_string,
                                   settings.multiplicity_string, molecules);

    occ::log::info("Computing distributed multipoles using {} model",
                   settings.model_name);
    auto unique_multipoles = compute_multipoles_for_crystal(
        basename, cryst, settings.model_name, false);

    auto multipoles = expand_to_unit_cell(unique_multipoles, cryst);

    auto setup = mults::from_crystal_and_multipoles(cryst, multipoles);
    return PreparedInputs{std::move(setup), std::nullopt};
}

void run_ropt_subcommand(const RoptSettings &settings) {
    std::string filename = settings.crystal_filename;
    std::string basename = fs::path(filename).stem().string();

    // Determine input source: structure JSON or CIF+DMA
    bool is_json = false;
    if (!filename.empty()) {
        auto ext = fs::path(filename).extension().string();
        if (ext == ".json") {
            try { is_json = occ::io::is_structure_format(filename); }
            catch (...) {}
        }
    }
    auto prepared = is_json
                        ? prepare_from_json(filename)
                        : prepare_from_dma(settings, filename, basename);

    // Build optimizer settings
    CrystalOptimizerSettings opt_settings;
    if (settings.optimizer == "lbfgs") {
        opt_settings.method = OptimizationMethod::LBFGS;
    } else if (settings.optimizer == "trust-region") {
        opt_settings.method = OptimizationMethod::TrustRegion;
    } else {
        opt_settings.method = OptimizationMethod::MSTMIN;
    }
    opt_settings.gradient_tolerance = settings.gradient_tolerance;
    opt_settings.energy_tolerance = settings.energy_tolerance;
    opt_settings.max_iterations = settings.max_iterations;
    opt_settings.neighbor_radius = std::max(settings.neighbor_radius,
                                            prepared.setup.cutoff_radius);
    opt_settings.force_field = prepared.setup.force_field;
    opt_settings.max_interaction_order = settings.max_interaction_order;
    opt_settings.optimize_cell = settings.optimize_cell;
    opt_settings.use_ewald = settings.use_ewald;
    opt_settings.ewald_accuracy = prepared.setup.ewald_accuracy;
    opt_settings.ewald_eta = prepared.setup.ewald_eta;
    opt_settings.ewald_kmax = prepared.setup.ewald_kmax;
    if (settings.has_external_pressure) {
        opt_settings.external_pressure_gpa = settings.external_pressure_gpa;
    } else if (prepared.structure_input &&
               std::abs(prepared.structure_input->settings.pressure_gpa) > 1e-16) {
        opt_settings.external_pressure_gpa =
            prepared.structure_input->settings.pressure_gpa;
    }
    if (settings.write_trajectory) {
        opt_settings.trajectory_file = basename + "_traj.xyz";
    }

    // Apply CLI settings to energy setup (CLI overrides JSON settings)
    prepared.setup.use_ewald = settings.use_ewald;
    prepared.setup.max_interaction_order = settings.max_interaction_order;
    prepared.setup.cutoff_radius = opt_settings.neighbor_radius;

    occ::log::info("Setting up crystal optimizer...");
    occ::log::info("  {} molecules, cutoff {:.1f} A, Ewald: {}",
                   prepared.setup.molecules.size(),
                   prepared.setup.cutoff_radius,
                   settings.use_ewald ? "on" : "off");

    // Keep a copy for structure JSON output (setup is moved into optimizer)
    auto saved_setup = prepared.setup;
    CrystalOptimizer optimizer(std::move(prepared.setup), opt_settings);

    if (optimizer.settings().method != opt_settings.method) {
        occ::log::warn("Optimizer method adjusted to {}",
                       method_name(optimizer.settings().method));
    }

    occ::log::info("  Neighbor pairs: {}",
                   optimizer.energy_calculator().neighbor_pairs().size());
    occ::log::info("  Multipoles: {} sites, lmax = {}",
                   optimizer.energy_calculator().num_sites(),
                   opt_settings.max_interaction_order);

    // Run optimization
    occ::log::info("\nStarting optimization...\n");
    occ::timing::start(occ::timing::optimization);
    auto result = optimizer.optimize();
    occ::timing::stop(occ::timing::optimization);

    // Report results
    occ::log::info("\n{:=<60s}", "");
    occ::log::info("Optimization completed");
    occ::log::info("{:=<60s}", "");
    occ::log::info("Converged: {}", result.converged ? "Yes" : "No");
    occ::log::info("Termination: {}", result.termination_reason);
    occ::log::info("Iterations: {}", result.iterations);
    occ::log::info("Function evaluations: {}", result.function_evaluations);
    occ::log::info("");
    occ::log::info("Initial energy:     {:12.4f} kJ/mol per molecule",
                   result.initial_energy);
    occ::log::info("Final energy:       {:12.4f} kJ/mol per molecule",
                   result.final_energy);
    occ::log::info("  Electrostatic:    {:12.4f} kJ/mol per molecule",
                   result.electrostatic_energy);
    occ::log::info("  Rep-dispersion:   {:12.4f} kJ/mol per molecule",
                   result.repulsion_dispersion_energy);
    if (std::abs(result.pressure_volume_energy) > 1e-12) {
        occ::log::info("  pV term:          {:12.4f} kJ/mol per molecule",
                       result.pressure_volume_energy);
        occ::log::info("  Static lattice E: {:12.4f} kJ/mol per molecule",
                       result.final_energy - result.pressure_volume_energy);
    }
    occ::log::info("Energy change:      {:12.4f} kJ/mol per molecule",
                   result.final_energy - result.initial_energy);

    // --- Per-molecule structural change summary ---
    {
        const auto& initial_states = optimizer.initial_states();
        const auto& final_states = result.final_states;
        const auto& geom = optimizer.energy_calculator().molecule_geometry();
        int n_mol = static_cast<int>(final_states.size());

        occ::log::info("");
        occ::log::info("Structural changes (initial -> final):");

        double total_rmsd_sq = 0.0;
        int total_atoms = 0;

        for (int m = 0; m < n_mol; ++m) {
            const auto& s0 = initial_states[m];
            const auto& s1 = final_states[m];

            // COM displacement
            occ::Vec3 dpos = s1.position - s0.position;
            double com_shift = dpos.norm();

            // Rotation change: R_delta = R1 * R0^T
            occ::Mat3 R0 = s0.rotation_matrix();
            occ::Mat3 R1 = s1.rotation_matrix();
            occ::Mat3 R_delta = R1 * R0.transpose();

            // Rotation angle from trace: cos(theta) = (tr(R_delta) - 1) / 2
            double trace = R_delta.trace();
            double cos_theta = std::clamp((trace - 1.0) / 2.0, -1.0, 1.0);
            double rot_angle_deg = std::acos(cos_theta) * 180.0 / M_PI;

            // Atom RMSD for this molecule (if geometry available)
            double mol_rmsd = 0.0;
            int n_atoms = 0;
            if (m < static_cast<int>(geom.size()) && !geom[m].atom_positions.empty()) {
                // Use body frame of molecule 0 for all (Z'=1: all same geometry)
                int geom_idx = (geom.size() == 1) ? 0 : m;
                if (geom_idx >= static_cast<int>(geom.size())) geom_idx = 0;
                const auto& body = geom[geom_idx];
                n_atoms = static_cast<int>(body.atom_positions.size());
                double sum_sq = 0.0;
                for (int a = 0; a < n_atoms; ++a) {
                    occ::Vec3 p0 = R0 * body.atom_positions[a] + s0.position;
                    occ::Vec3 p1 = R1 * body.atom_positions[a] + s1.position;
                    sum_sq += (p1 - p0).squaredNorm();
                }
                mol_rmsd = std::sqrt(sum_sq / n_atoms);
                total_rmsd_sq += sum_sq;
                total_atoms += n_atoms;
            }

            occ::log::info("  Mol {:2d}: COM shift {:7.4f} A, rot {:6.2f} deg, atom RMSD {:7.4f} A",
                           m, com_shift, rot_angle_deg, mol_rmsd);
        }

        if (total_atoms > 0) {
            double overall_rmsd = std::sqrt(total_rmsd_sq / total_atoms);
            occ::log::info("  Overall atom RMSD: {:.4f} A ({} atoms)",
                           overall_rmsd, total_atoms);
        }
    }

    // --- Unit-cell change summary ---
    if (settings.optimize_cell) {
        occ::log::info("");
        occ::log::info("Cell changes (initial -> final):");

        if (!result.optimized_crystal.has_value()) {
            occ::log::warn("  Unavailable (optimized crystal reconstruction failed)");
        } else {
            const auto& uc0 = optimizer.energy_calculator().crystal().unit_cell();
            const auto& uc1 = result.optimized_crystal->unit_cell();

            auto print_scalar_change = [](const char* label, double v0, double v1,
                                          const char* unit, bool percent) {
                const double dv = v1 - v0;
                if (percent && std::abs(v0) > 1e-12) {
                    const double rel = 100.0 * dv / v0;
                    occ::log::info("  {:>6s}: {:10.6f} -> {:10.6f} {}  (d={:+.6f}, {:+.4f}%)",
                                   label, v0, v1, unit, dv, rel);
                } else {
                    occ::log::info("  {:>6s}: {:10.6f} -> {:10.6f} {}  (d={:+.6f})",
                                   label, v0, v1, unit, dv);
                }
            };

            print_scalar_change("a", uc0.a(), uc1.a(), "A", true);
            print_scalar_change("b", uc0.b(), uc1.b(), "A", true);
            print_scalar_change("c", uc0.c(), uc1.c(), "A", true);
            print_scalar_change("alpha",
                                occ::units::degrees(uc0.alpha()),
                                occ::units::degrees(uc1.alpha()),
                                "deg", false);
            print_scalar_change("beta",
                                occ::units::degrees(uc0.beta()),
                                occ::units::degrees(uc1.beta()),
                                "deg", false);
            print_scalar_change("gamma",
                                occ::units::degrees(uc0.gamma()),
                                occ::units::degrees(uc1.gamma()),
                                "deg", false);
            print_scalar_change("Volume", uc0.volume(), uc1.volume(), "A^3", true);

            // Report symmetric small-strain in Voigt order [E1..E6]:
            // [exx, eyy, ezz, 2*eyz, 2*exz, 2*exy].
            const occ::Mat3 F = uc1.direct() * uc0.inverse();
            const occ::Mat3 eps = 0.5 * (F + F.transpose()) - occ::Mat3::Identity();
            const occ::Vec6 voigt((occ::Vec6() << eps(0, 0), eps(1, 1), eps(2, 2),
                                  2.0 * eps(1, 2), 2.0 * eps(0, 2), 2.0 * eps(0, 1))
                                     .finished());
            occ::log::info("  Strain Voigt [E1..E6]: [{:+.5e}, {:+.5e}, {:+.5e}, {:+.5e}, {:+.5e}, {:+.5e}]",
                           voigt[0], voigt[1], voigt[2], voigt[3], voigt[4], voigt[5]);
        }
    }

    if (settings.optimize_cell) {
        if (!settings.compute_elastic_tensor) {
            occ::log::info("Elastic tensor: skipped (--no-elastic)");
        } else {
            try {
                occ::timing::start(occ::timing::elastic_tensor);
                const auto elastic =
                    compute_elastic_tensor_summary(optimizer, result.final_states,
                                                   result.converged);
                occ::timing::stop(occ::timing::elastic_tensor);
                print_elastic_tensor_summary(elastic, result.converged);

                // Write elastic tensor to file if requested
                if (!settings.elastic_tensor_file.empty() &&
                    elastic.clamped_gpa.norm() > 0) {
                    write_elastic_tensor(settings.elastic_tensor_file, elastic);
                }
            } catch (const std::exception& e) {
                occ::log::warn("Elastic tensor evaluation failed: {}", e.what());
            }
        }
    }

    print_neighbor_interaction_table(optimizer.energy_calculator(), result.final_states);

    // Write output CIF
    std::string output_filename = settings.output_filename;
    if (output_filename.empty()) {
        output_filename = basename + "_opt.cif";
    }

    if (result.optimized_crystal.has_value()) {
        occ::log::info("\nWriting optimized structure to {}", output_filename);
        occ::io::CifWriter writer;
        writer.write(output_filename, result.optimized_crystal.value(),
                     basename + "_optimized");
    } else {
        occ::log::warn("Could not reconstruct optimized crystal structure");
    }

    // Write structure JSON if requested
    if (!settings.structure_json.empty()) {
        std::string struct_file = settings.structure_json;
        occ::log::info("Writing structure JSON to {}", struct_file);

        // Build a CrystalEnergySetup with final states
        auto final_setup = saved_setup;
        for (int m = 0; m < static_cast<int>(result.final_states.size()) &&
                        m < static_cast<int>(final_setup.molecules.size()); ++m) {
            final_setup.molecules[m].com = result.final_states[m].position;
            final_setup.molecules[m].angle_axis = result.final_states[m].angle_axis;
            final_setup.molecules[m].parity = result.final_states[m].parity;
        }

        auto si = mults::to_structure_input(final_setup, basename);
        if (settings.has_external_pressure) {
            si.settings.pressure_gpa = settings.external_pressure_gpa;
        }
        si.reference.total = result.final_energy;
        si.reference.components["electrostatic"] = result.electrostatic_energy;
        si.reference.components["buckingham"] = result.repulsion_dispersion_energy;

        try {
            occ::io::write_structure_json(struct_file, si);
        } catch (const std::exception &e) {
            occ::log::warn("Failed to write structure JSON: {}", e.what());
        }
    }

    occ::log::info("\nDone.");
}

} // namespace occ::main
