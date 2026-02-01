#include <occ/qm/auto_aux_basis.h>
#include <occ/qm/integral_engine.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <ankerl/unordered_dense.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

namespace occ::qm {

using gto::AOBasis;
using gto::Shell;
using occ::Vec3;
using occ::core::Atom;

namespace {

// Deduplication tolerances for similar exponents
constexpr double DEDUP_TOL_DEFAULT = 0.1;   // 10% relative difference
constexpr double DEDUP_TOL_MEDIUM = 0.2;    // 20% for large candidate sets
constexpr double DEDUP_TOL_LARGE = 0.3;     // 30% for very large sets
constexpr size_t CANDS_THRESHOLD_MEDIUM = 200;
constexpr size_t CANDS_THRESHOLD_LARGE = 500;

/// Candidate auxiliary function
struct AuxCandidate {
    int atom_idx;                 ///< Index of atom center
    double exponent;              ///< Effective exponent
    Vec3 coord;                   ///< Atom coordinates
};

/// Pivoted Cholesky decomposition for selecting linearly independent functions
std::vector<int> pivoted_cholesky(Mat& A, double threshold) {
    const int n = A.rows();
    std::vector<int> pivot_idx;
    Vec diag = A.diagonal();
    Mat L = Mat::Zero(n, n);
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);

    for (int k = 0; k < n; ++k) {
        int max_idx = k;
        double max_val = diag(k);
        for (int i = k + 1; i < n; ++i) {
            if (diag(i) > max_val) {
                max_val = diag(i);
                max_idx = i;
            }
        }

        if (max_val < threshold)
            break;

        if (max_idx != k) {
            std::swap(perm[k], perm[max_idx]);
            std::swap(diag(k), diag(max_idx));
            L.row(k).head(k).swap(L.row(max_idx).head(k));
        }

        pivot_idx.push_back(perm[k]);
        L(k, k) = std::sqrt(max_val);

        for (int i = k + 1; i < n; ++i) {
            int orig_i = perm[i];
            int orig_k = perm[k];
            double sum = 0.0;
            for (int j = 0; j < k; ++j) {
                sum += L(i, j) * L(k, j);
            }
            L(i, k) = (A(orig_i, orig_k) - sum) / L(k, k);

            double diag_sum = 0.0;
            for (int j = 0; j <= k; ++j) {
                diag_sum += L(i, j) * L(i, j);
            }
            diag(i) = std::max(0.0, A(orig_i, orig_i) - diag_sum);
        }
    }

    return pivot_idx;
}

/// Effective exponent for aux function, Eq. (16) of Lehtola JCTC 2021
double effective_exponent(int l_i, int l_j, int L, double alpha_sum) {
    int l_prod = l_i + l_j;
    if (L == l_prod)
        return alpha_sum;

    double scale = (std::tgamma(L + 2) * std::tgamma(l_prod + 1.5)) /
                   (std::tgamma(l_prod + 2) * std::tgamma(L + 1.5));
    return alpha_sum * scale * scale;
}

/// Deduplicate candidates within tolerance on same atom
std::vector<AuxCandidate> deduplicate_candidates(
    const std::vector<AuxCandidate>& cands, double tol = 0.1) {

    std::vector<AuxCandidate> unique;
    for (const auto& c : cands) {
        bool is_dup = false;
        for (const auto& u : unique) {
            if (c.atom_idx == u.atom_idx) {
                double exp_max = std::max(c.exponent, u.exponent);
                double exp_diff = std::abs(c.exponent - u.exponent);
                if (exp_diff / exp_max < tol) {
                    is_dup = true;
                    break;
                }
            }
        }
        if (!is_dup) {
            unique.push_back(c);
        }
    }
    return unique;
}

/// Create a shell with given parameters
Shell make_shell(int L, const Vec3& origin, double exponent) {
    std::vector<double> expo = {exponent};
    std::vector<std::vector<double>> contr = {{1.0}};
    std::array<double, 3> pos = {origin[0], origin[1], origin[2]};
    return Shell(L, expo, contr, pos);
}

} // anonymous namespace

AutoAuxResult generate_auto_aux(
    const AOBasis& basis,
    double threshold,
    std::optional<int> max_l)
{
    occ::timing::StopWatch<1> timer;
    timer.start();

    AutoAuxResult result;

    int max_l_ao = 0;
    for (size_t s = 0; s < basis.size(); ++s) {
        max_l_ao = std::max(max_l_ao, static_cast<int>(basis[s].l));
    }

    int effective_max_l = max_l.value_or(max_l_ao + 2);

    occ::log::debug("Auto auxiliary basis: {} shells, threshold={:.0e}, max_l={}",
                   basis.size(), threshold, effective_max_l);

    const auto& shell_to_atom = basis.shell_to_atom();
    const auto& atoms = basis.atoms();

    ankerl::unordered_dense::map<int, std::vector<std::pair<int, double>>> primitives_by_atom;
    ankerl::unordered_dense::map<int, Vec3> atom_coords;

    for (size_t s = 0; s < basis.size(); ++s) {
        const auto& shell = basis[s];
        int atom_idx = shell_to_atom[s];
        int l = shell.l;
        atom_coords[atom_idx] = shell.origin;

        for (size_t i = 0; i < shell.num_primitives(); ++i) {
            primitives_by_atom[atom_idx].emplace_back(l, shell.exponents[i]);
        }
    }

    // Generate candidate aux functions from AO products
    ankerl::unordered_dense::map<int, std::vector<AuxCandidate>> candidates_by_L;

    for (const auto& [atom_idx, prims] : primitives_by_atom) {
        const Vec3& coord = atom_coords[atom_idx];

        // Form products
        for (const auto& [l_i, exp_i] : prims) {
            for (const auto& [l_j, exp_j] : prims) {
                double alpha_sum = exp_i + exp_j;
                int L_min = std::abs(l_i - l_j);
                int L_max = l_i + l_j;

                for (int L = L_min; L <= L_max; ++L) {
                    if (L > effective_max_l) continue;

                    double alpha_eff = effective_exponent(l_i, l_j, L, alpha_sum);
                    candidates_by_L[L].push_back({atom_idx, alpha_eff, coord});
                }
            }
        }
    }

    for (auto& [L, cands] : candidates_by_L) {
        double tol = DEDUP_TOL_DEFAULT;
        if (cands.size() > CANDS_THRESHOLD_MEDIUM) tol = DEDUP_TOL_MEDIUM;
        if (cands.size() > CANDS_THRESHOLD_LARGE) tol = DEDUP_TOL_LARGE;
        cands = deduplicate_candidates(cands, tol);
        result.candidates_per_l[L] = static_cast<int>(cands.size());
    }

    // Per-L pivoted Cholesky on 2c Coulomb matrix
    ankerl::unordered_dense::map<int, std::vector<AuxCandidate>> selected_aux;

    for (auto& [L, cands] : candidates_by_L) {
        if (cands.empty()) continue;

        const size_t n_cands = cands.size();
        occ::log::debug("Auto aux basis: Processing L={} with {} candidates", L, n_cands);
        const int funcs_per_shell = 2 * L + 1;

        std::vector<Atom> temp_atoms;
        std::vector<Shell> temp_shells;
        temp_atoms.reserve(n_cands);
        temp_shells.reserve(n_cands);

        for (size_t i = 0; i < n_cands; ++i) {
            Atom atom;
            atom.atomic_number = 0;
            atom.x = cands[i].coord[0];
            atom.y = cands[i].coord[1];
            atom.z = cands[i].coord[2];
            temp_atoms.push_back(atom);

            temp_shells.push_back(make_shell(L, cands[i].coord, cands[i].exponent));
        }

        AOBasis temp_basis(temp_atoms, temp_shells);
        temp_basis.set_pure(false);

        IntegralEngine engine(temp_basis);
        Mat int2c = engine.one_electron_operator(cint::Operator::coulomb, false);

        // Average over m quantum numbers for L > 0
        const size_t n_funcs = static_cast<size_t>(int2c.rows());
        Mat int2c_shells;

        if (n_funcs != n_cands) {
            int2c_shells = Mat::Zero(n_cands, n_cands);
            for (size_t i = 0; i < n_cands; ++i) {
                for (size_t j = 0; j < n_cands; ++j) {
                    size_t i_s = i * funcs_per_shell;
                    size_t j_s = j * funcs_per_shell;
                    double trace = 0.0;
                    for (int m = 0; m < funcs_per_shell; ++m) {
                        trace += int2c(i_s + m, j_s + m);
                    }
                    int2c_shells(i, j) = trace / funcs_per_shell;
                }
            }
        } else {
            int2c_shells = int2c;
        }

        Vec diag = int2c_shells.diagonal();
        for (Eigen::Index i = 0; i < diag.size(); ++i) {
            diag(i) = std::sqrt(std::max(diag(i), 1e-16));
        }
        for (size_t i = 0; i < n_cands; ++i) {
            for (size_t j = 0; j < n_cands; ++j) {
                int2c_shells(i, j) /= (diag(i) * diag(j));
            }
        }

        // Presort by off-diagonal norm
        Vec offdiag_norms = int2c_shells.cwiseAbs().rowwise().sum();
        offdiag_norms -= int2c_shells.diagonal();

        std::vector<int> presort_idx(n_cands);
        std::iota(presort_idx.begin(), presort_idx.end(), 0);
        std::sort(presort_idx.begin(), presort_idx.end(),
                  [&](int a, int b) { return offdiag_norms(a) < offdiag_norms(b); });

        Mat int2c_sorted = Mat::Zero(n_cands, n_cands);
        for (size_t i = 0; i < n_cands; ++i) {
            for (size_t j = 0; j < n_cands; ++j) {
                int2c_sorted(i, j) = int2c_shells(presort_idx[i], presort_idx[j]);
            }
        }

        auto pivot_idx_sorted = pivoted_cholesky(int2c_sorted, threshold);

        for (int idx : pivot_idx_sorted) {
            int orig_idx = presort_idx[idx];
            if (orig_idx < static_cast<int>(n_cands)) {
                selected_aux[L].push_back(cands[orig_idx]);
            }
        }

        result.selected_per_l[L] = static_cast<int>(selected_aux[L].size());
    }

    // Group selected functions by atom and L
    ankerl::unordered_dense::map<int, ankerl::unordered_dense::map<int, std::set<double>>> aux_by_atom_L;

    for (const auto& [L, aux_list] : selected_aux) {
        for (const auto& aux : aux_list) {
            aux_by_atom_L[aux.atom_idx][L].insert(aux.exponent);
        }
    }

    std::vector<Shell> aux_shells;

    for (const auto& [atom_idx, L_exps] : aux_by_atom_L) {
        const Vec3& coord = atom_coords[atom_idx];

        for (const auto& [L, exps] : L_exps) {
            std::vector<double> sorted_exps(exps.rbegin(), exps.rend());
            std::vector<double> merged;
            for (double exp : sorted_exps) {
                bool is_dup = false;
                for (double m : merged) {
                    if (std::abs(exp - m) / std::max(exp, m) < DEDUP_TOL_DEFAULT) {
                        is_dup = true;
                        break;
                    }
                }
                if (!is_dup) {
                    merged.push_back(exp);
                }
            }

            for (double exp : merged) {
                aux_shells.push_back(make_shell(L, coord, exp));
            }
        }
    }

    AOBasis initial_aux(atoms, aux_shells, "auto-aux");
    initial_aux.set_pure(false);

    occ::log::debug("Auto auxiliary basis before final Cholesky: {} shells, {} functions",
                   initial_aux.size(), initial_aux.nbf());

    // Final molecular-level Cholesky to remove cross-atom/cross-L dependencies
    // (per-L Cholesky only handles one-center dependencies)
    const size_t n_shells = initial_aux.size();

    if (n_shells > 1) {
        occ::log::debug("Auto aux basis: performing final molecular Cholesky ({} shells)...",
                       n_shells);

        IntegralEngine full_engine(initial_aux);
        Mat V_full = full_engine.one_electron_operator(cint::Operator::coulomb, false);

        // Average over functions to get shell-level matrix
        Mat V_shells = Mat::Zero(n_shells, n_shells);
        const auto& shell_to_bf = initial_aux.first_bf();

        for (size_t i = 0; i < n_shells; ++i) {
            int nbf_i = initial_aux[i].size();
            int off_i = shell_to_bf[i];

            for (size_t j = 0; j < n_shells; ++j) {
                int nbf_j = initial_aux[j].size();
                int off_j = shell_to_bf[j];

                double sum = 0.0;
                for (int mi = 0; mi < nbf_i; ++mi) {
                    for (int mj = 0; mj < nbf_j; ++mj) {
                        sum += V_full(off_i + mi, off_j + mj);
                    }
                }
                V_shells(i, j) = sum / (nbf_i * nbf_j);
            }
        }

        Vec diag = V_shells.diagonal();
        for (Eigen::Index i = 0; i < diag.size(); ++i) {
            diag(i) = std::sqrt(std::max(diag(i), 1e-16));
        }
        for (size_t i = 0; i < n_shells; ++i) {
            for (size_t j = 0; j < n_shells; ++j) {
                V_shells(i, j) /= (diag(i) * diag(j));
            }
        }

        // Presort by off-diagonal norm
        Vec offdiag_norms = V_shells.cwiseAbs().rowwise().sum();
        offdiag_norms -= V_shells.diagonal();

        std::vector<int> presort_idx(n_shells);
        std::iota(presort_idx.begin(), presort_idx.end(), 0);
        std::sort(presort_idx.begin(), presort_idx.end(),
                  [&](int a, int b) { return offdiag_norms(a) < offdiag_norms(b); });

        Mat V_sorted = Mat::Zero(n_shells, n_shells);
        for (size_t i = 0; i < n_shells; ++i) {
            for (size_t j = 0; j < n_shells; ++j) {
                V_sorted(i, j) = V_shells(presort_idx[i], presort_idx[j]);
            }
        }

        auto pivot_idx_sorted = pivoted_cholesky(V_sorted, threshold);

        std::vector<Shell> final_shells;
        final_shells.reserve(pivot_idx_sorted.size());
        for (int idx : pivot_idx_sorted) {
            int orig_idx = presort_idx[idx];
            if (orig_idx >= 0 && orig_idx < static_cast<int>(n_shells)) {
                final_shells.push_back(aux_shells[orig_idx]);
            }
        }

        size_t removed = n_shells - final_shells.size();
        if (removed > 0) {
            occ::log::debug("Auto aux basis: final Cholesky removed {} linearly dependent shells",
                           removed);
        }

        result.aux_basis = AOBasis(atoms, final_shells, "auto-aux");
        result.aux_basis.set_pure(false);
    } else {
        result.aux_basis = std::move(initial_aux);
    }

    result.time_seconds = timer.stop().count();

    occ::log::debug("Auto auxiliary basis: {} functions in {:.3f}s",
                   result.aux_basis.nbf(), result.time_seconds);

    return result;
}

} // namespace occ::qm
