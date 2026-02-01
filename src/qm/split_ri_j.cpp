#include <occ/qm/split_ri_j.h>
#include <occ/qm/integral_engine.h>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/ints/boys.h>
#include <occ/ints/kernels.h>
#include <occ/ints/rints_specialized_3c.h>
#include <occ/gto/gto.h>
#include <Eigen/Cholesky>
#include <cmath>
#include <stdexcept>

namespace occ::qm {

using ints::Boys;
using ints::ncart;
using ints::nhermsum;
using ints::hermite_index;
using ints::BoysConstants;
using ints::BoysParamsDefault;
using ints::ShellPairData;
using ints::AuxShellData;
using ints::precompute_shell_pair_dispatch;
using ints::precompute_aux_shell_dispatch;
using ints::RIntsDynamic;
using ints::compute_r_ints_3c_dispatch;
using ints::fused_forward_dispatch;
using ints::fused_backward_dispatch;

// Number of spherical harmonics for angular momentum l
inline int nsph(int l) {
    return l == 0 ? 1 : 2 * l + 1;
}

// Build combined transformation matrix T_ab = kron(T_a, T_b)
// Transforms from Cartesian product basis to spherical product basis
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
build_cart_to_sph_transform(int la, int lb) {
    using MatRM = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // Get individual transformation matrices
    occ::Mat T_a = occ::gto::cartesian_to_spherical_transformation_matrix(la);
    occ::Mat T_b = occ::gto::cartesian_to_spherical_transformation_matrix(lb);

    int nsph_a = nsph(la);
    int nsph_b = nsph(lb);
    int ncart_a = ncart(la);
    int ncart_b = ncart(lb);

    // Build Kronecker product: T_ab[i*nsph_b + j, k*ncart_b + l] = T_a[i,k] * T_b[j,l]
    MatRM T_ab(nsph_a * nsph_b, ncart_a * ncart_b);

    for (int i = 0; i < nsph_a; ++i) {
        for (int j = 0; j < nsph_b; ++j) {
            for (int k = 0; k < ncart_a; ++k) {
                for (int l = 0; l < ncart_b; ++l) {
                    T_ab(i * nsph_b + j, k * ncart_b + l) =
                        static_cast<T>(T_a(i, k) * T_b(j, l));
                }
            }
        }
    }

    return T_ab;
}

// Transform E-matrices from Cartesian to spherical for shell pairs
template <typename T>
void transform_shell_pair_to_spherical(ShellPairData<T>& data, int la, int lb) {
    const int nab_cart = ncart(la) * ncart(lb);
    const int nab_sph = nsph(la) * nsph(lb);
    const int nherm = data.nherm();

    using MatRM = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // Build transformation matrix
    MatRM T_ab = build_cart_to_sph_transform<T>(la, lb);

    // Transform each primitive's E-matrix
    for (auto& prim : data.primitives) {
        Eigen::Map<MatRM> E_cart(prim.E_matrix.data(), nab_cart, nherm);

        // E_sph = T_ab @ E_cart
        MatRM E_sph = T_ab * E_cart;

        // Resize and copy
        prim.E_matrix.resize(nab_sph * nherm);
        prim.nab = nab_sph;
        Eigen::Map<MatRM>(prim.E_matrix.data(), nab_sph, nherm) = E_sph;
    }
}

// Transform E-matrices from Cartesian to spherical for aux shells
template <typename T>
void transform_aux_shell_to_spherical(AuxShellData<T>& data) {
    const int lc = data.lc;
    const int nc_cart = ncart(lc);
    const int nc_sph = nsph(lc);
    const int nherm = data.nherm();

    using MatRM = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // Get transformation matrix for single shell
    occ::Mat T_c = occ::gto::cartesian_to_spherical_transformation_matrix(lc);

    // Transform each primitive's E-matrix
    for (auto& prim : data.primitives) {
        Eigen::Map<MatRM> E_cart(prim.E_matrix.data(), nc_cart, nherm);

        // E_sph = T_c @ E_cart
        MatRM E_sph = T_c.cast<T>() * E_cart;

        // Resize and copy
        prim.E_matrix.resize(nc_sph * nherm);
        prim.nc = nc_sph;
        Eigen::Map<MatRM>(prim.E_matrix.data(), nc_sph, nherm) = E_sph;
    }
}

struct HermiteTUV {
    int8_t t, u, v;
};

inline std::vector<HermiteTUV> build_hermite_tuv_table(int L_max) {
    std::vector<HermiteTUV> table(nhermsum(L_max));
    for (int L = 0; L <= L_max; ++L) {
        for (int t = 0; t <= L; ++t) {
            for (int u = 0; u <= L - t; ++u) {
                int v = L - t - u;
                int h = hermite_index(t, u, v);
                table[h] = {static_cast<int8_t>(t), static_cast<int8_t>(u), static_cast<int8_t>(v)};
            }
        }
    }
    return table;
}

struct RIndexTable {
    int nherm_ab, nherm_c;
    std::vector<int> indices;
};

inline RIndexTable build_r_index_table(int L_ab, int lc, const std::vector<HermiteTUV>& tuv) {
    RIndexTable table;
    table.nherm_ab = nhermsum(L_ab);
    table.nherm_c = nhermsum(lc);
    table.indices.resize(table.nherm_ab * table.nherm_c);

    for (int h_ab = 0; h_ab < table.nherm_ab; ++h_ab) {
        auto [t_ab, u_ab, v_ab] = tuv[h_ab];
        for (int h_c = 0; h_c < table.nherm_c; ++h_c) {
            auto [t_c, u_c, v_c] = tuv[h_c];
            table.indices[h_ab * table.nherm_c + h_c] =
                hermite_index(t_ab + t_c, u_ab + u_c, v_ab + v_c);
        }
    }
    return table;
}

struct SplitRIJ::Impl {
    gto::AOBasis ao_basis;
    gto::AOBasis aux_basis;
    ShellPairList shellpair_list;  // Significant shell pairs from IntegralEngine
    Mat schwarz;                    // Schwarz screening matrix (nshells x nshells)
    Vec V_diag;                     // Diagonal of V matrix for aux shell screening
    Vec V_diag_sqrt_max;            // Max sqrt(V(r,r)) per aux shell for screening
    double screening_threshold = 1e-12;

    Eigen::LLT<Mat> V_LLt;
    Boys<> boys;

    // Flat list of significant shell pairs with precomputed data
    struct SignificantPair {
        size_t sa, sb;           // Shell indices
        size_t sp_data_idx;      // Index into shell_pair_data
    };
    std::vector<SignificantPair> significant_pairs;
    std::vector<ShellPairData<double>> shell_pair_data;
    std::vector<AuxShellData<double>> aux_shells;

    std::vector<HermiteTUV> hermite_tuv;
    std::vector<std::vector<RIndexTable>> r_index_tables;

    // Workspace - allocated once, reused
    struct Workspace {
        std::vector<double> X_all;      // X for all (shell_pair, ao_prim)
        std::vector<double> T_all;      // T for all (aux_shell, aux_prim)
        std::vector<double> Y_all;      // Y for all (aux_shell, aux_prim) - accumulated
        std::vector<double> U_all;      // U for all (shell_pair, ao_prim) - for aux-outer loop
        std::vector<double> D_block;
        std::vector<double> J_block;
        std::vector<double> D_max;      // Max |D| per significant shell pair
        std::vector<double> screen_val; // screen_val = D_max * Schwarz for each pair
        Vec g;
        Mat J;
    };
    mutable Workspace ws;

    // Indexing helpers
    std::vector<size_t> shell_pair_X_offset;  // Start of X data for each shell pair
    std::vector<size_t> aux_shell_T_offset;   // Start of T data for each aux shell
    std::vector<size_t> aux_shell_Y_offset;   // Start of Y data for each aux shell

    static double sph_factor(int l) {
        if (l == 0) return 0.28209479177387814;
        if (l == 1) return 0.48860251190291992;
        return 1.0;
    }

    void allocate_workspace() {
        const int max_la = ao_basis.l_max();
        const int max_lc = aux_basis.l_max();
        // Use spherical sizes for D/J blocks when basis is pure
        const int max_na = ao_basis.is_pure() ? nsph(max_la) : ncart(max_la);
        const int max_nab = max_na * max_na;
        const int max_nherm_ab = nhermsum(2 * max_la);
        const int max_nherm_c = nhermsum(max_lc);
        const size_t nbf = ao_basis.nbf();
        const size_t naux = aux_basis.nbf();

        // Calculate total X storage: sum over SIGNIFICANT shell pairs
        size_t total_X_size = 0;
        shell_pair_X_offset.resize(significant_pairs.size());
        for (size_t idx = 0; idx < significant_pairs.size(); ++idx) {
            shell_pair_X_offset[idx] = total_X_size;
            const auto& sp = shell_pair_data[significant_pairs[idx].sp_data_idx];
            total_X_size += sp.primitives.size() * sp.nherm();
        }

        // Calculate total T storage: sum over aux shells of (n_prims * nherm_c)
        size_t total_T_size = 0;
        aux_shell_T_offset.resize(aux_shells.size());
        aux_shell_Y_offset.resize(aux_shells.size());
        for (size_t sc = 0; sc < aux_shells.size(); ++sc) {
            aux_shell_T_offset[sc] = total_T_size;
            aux_shell_Y_offset[sc] = total_T_size;  // Same layout as T
            total_T_size += aux_shells[sc].primitives.size() * aux_shells[sc].nherm();
        }

        ws.X_all.resize(total_X_size);
        ws.T_all.resize(total_T_size);
        ws.Y_all.resize(total_T_size);  // Same size as T
        ws.U_all.resize(total_X_size);  // Same layout as X_all
        ws.D_block.resize(max_nab);
        ws.J_block.resize(max_nab);
        ws.D_max.resize(significant_pairs.size());
        ws.screen_val.resize(significant_pairs.size());
        ws.g.resize(naux);
        ws.J.resize(nbf, nbf);
    }

    void precompute_shell_data() {
        const auto& ao_shells_vec = ao_basis.shells();
        const auto& aux_shells_vec = aux_basis.shells();
        const size_t nao_shells = ao_shells_vec.size();
        const size_t naux_shells = aux_shells_vec.size();
        const bool ao_is_spherical = ao_basis.is_pure();

        // Build significant pairs from shellpair_list
        // shellpair_list[p] contains q indices where q <= p
        size_t num_significant = 0;
        for (size_t p = 0; p < shellpair_list.size(); ++p) {
            num_significant += shellpair_list[p].size();
        }

        significant_pairs.reserve(num_significant);
        shell_pair_data.reserve(num_significant);

        for (size_t sa = 0; sa < nao_shells; ++sa) {
            const auto& shell_a = ao_shells_vec[sa];
            for (size_t sb : shellpair_list[sa]) {
                const auto& shell_b = ao_shells_vec[sb];

                SignificantPair sig;
                sig.sa = sa;
                sig.sb = sb;
                sig.sp_data_idx = shell_pair_data.size();
                significant_pairs.push_back(sig);

                auto sp_data = precompute_shell_pair_dispatch<double>(
                    shell_a.l, shell_b.l,
                    shell_a.num_primitives(), shell_b.num_primitives(),
                    shell_a.exponents.data(), shell_b.exponents.data(),
                    shell_a.contraction_coefficients.col(0).data(),
                    shell_b.contraction_coefficients.col(0).data(),
                    shell_a.origin.data(), shell_b.origin.data());

                // Transform E-matrices to spherical if AO basis is pure
                if (ao_is_spherical) {
                    transform_shell_pair_to_spherical(sp_data, shell_a.l, shell_b.l);
                }

                shell_pair_data.push_back(std::move(sp_data));
            }
        }

        occ::log::debug("Split-RI-J: {} significant shell pairs ({})",
                        significant_pairs.size(),
                        ao_is_spherical ? "spherical" : "cartesian");

        const bool aux_is_spherical = aux_basis.is_pure();
        aux_shells.reserve(naux_shells);
        for (size_t sc = 0; sc < naux_shells; ++sc) {
            const auto& shell_c = aux_shells_vec[sc];
            auto aux_data = precompute_aux_shell_dispatch<double>(
                shell_c.l,
                shell_c.num_primitives(),
                shell_c.exponents.data(),
                shell_c.contraction_coefficients.col(0).data(),
                shell_c.origin.data());

            // Transform E-matrices to spherical if aux basis is pure
            if (aux_is_spherical) {
                transform_aux_shell_to_spherical(aux_data);
            }

            aux_shells.push_back(std::move(aux_data));
        }
    }

    void precompute_r_index_tables() {
        const int max_la = ao_basis.l_max();
        const int max_lc = aux_basis.l_max();
        const int max_L_ab = 2 * max_la;
        const int L_total_max = max_L_ab + max_lc;

        hermite_tuv = build_hermite_tuv_table(L_total_max);

        r_index_tables.resize(max_L_ab + 1);
        for (int L_ab = 0; L_ab <= max_L_ab; ++L_ab) {
            r_index_tables[L_ab].resize(max_lc + 1);
            for (int lc = 0; lc <= max_lc; ++lc) {
                r_index_tables[L_ab][lc] = build_r_index_table(L_ab, lc, hermite_tuv);
            }
        }
    }

    Impl(const gto::AOBasis& ao, const gto::AOBasis& aux, const ShellPairList& spl,
         const Mat& schwarz_in)
        : ao_basis(ao), aux_basis(aux), shellpair_list(spl), schwarz(schwarz_in) {

        occ::timing::start(occ::timing::category::df);
        IntegralEngine aux_engine(aux_basis);
        Mat V = aux_engine.one_electron_operator(cint::Operator::coulomb, false);
        occ::timing::stop(occ::timing::category::df);

        // Store diagonal for screening: sqrt(V(r,r))
        V_diag = V.diagonal().array().sqrt();

        occ::timing::start(occ::timing::category::la);
        V_LLt.compute(V);
        if (V_LLt.info() != Eigen::Success) {
            throw std::runtime_error("Split-RI-J: Cholesky decomposition of V failed - "
                                     "auxiliary basis may be linearly dependent");
        }
        occ::timing::stop(occ::timing::category::la);

        precompute_shell_data();
        precompute_r_index_tables();

        // Compute max sqrt(V(r,r)) per aux shell for screening
        const auto& aux_shells_vec = aux_basis.shells();
        const auto& aux_first_bf = aux_basis.first_bf();
        const size_t naux_shells = aux_shells_vec.size();
        V_diag_sqrt_max.resize(naux_shells);
        for (size_t sc = 0; sc < naux_shells; ++sc) {
            int bf_start = aux_first_bf[sc];
            int nc = aux_shells_vec[sc].size();
            V_diag_sqrt_max[sc] = V_diag.segment(bf_start, nc).maxCoeff();
        }

        allocate_workspace();
    }

    inline void build_R_pq(const RIntsDynamic<double>& R,
                           const RIndexTable& rtab,
                           double* R_pq) const {
        const int* idx = rtab.indices.data();
        const int n = rtab.nherm_ab * rtab.nherm_c;
        for (int i = 0; i < n; ++i) {
            R_pq[i] = R.data[idx[i]];
        }
    }

    Mat coulomb(const Mat& D) const {
        occ::timing::start(occ::timing::category::df);

        const int num_threads = occ::parallel::nthreads;

        const double* boys_table = boys.table();
        const auto& ao_shells_vec = ao_basis.shells();
        const auto& ao_first_bf = ao_basis.first_bf();
        const auto& aux_first_bf = aux_basis.first_bf();
        const size_t naux_shells = aux_shells.size();
        const size_t npairs = significant_pairs.size();
        const size_t nbf = ao_basis.nbf();
        const size_t naux = aux_basis.nbf();

        const double pi_1p5 = std::pow(BoysConstants<double>::pi, 1.5);

        using MatRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

        // Thread-local storage for accumulated results
        occ::parallel::thread_local_storage<std::vector<double>> Y_local(
            [this]() { return std::vector<double>(ws.Y_all.size(), 0.0); });
        occ::parallel::thread_local_storage<std::vector<double>> U_local(
            [this]() { return std::vector<double>(ws.U_all.size(), 0.0); });
        occ::parallel::thread_local_storage<Mat> J_local(
            [nbf]() { return Mat::Zero(nbf, nbf); });

        // Thread-local R buffer
        const size_t R_buf_size = nhermsum(2 * ao_basis.l_max()) * nhermsum(aux_basis.l_max());
        occ::parallel::thread_local_storage<std::vector<double>> R_pq_local(
            [R_buf_size]() { return std::vector<double>(R_buf_size); });

        // Thread-local D_block buffer
        const int max_la = ao_basis.l_max();
        const int max_nab = ao_basis.is_pure() ? nsph(max_la) * nsph(max_la)
                                                : ncart(max_la) * ncart(max_la);
        occ::parallel::thread_local_storage<std::vector<double>> D_block_local(
            [max_nab]() { return std::vector<double>(max_nab); });

        Vec g = Vec::Zero(naux);

        // Check if Schwarz screening is enabled
        const bool use_screening = schwarz.size() > 0;

        // ===== STEP 1: Compute X = E^T @ D for SIGNIFICANT shell pairs (Eq. 17) =====
        // Parallel over pairs - each pair writes to different region of X_all
        occ::parallel::parallel_for(size_t(0), npairs, [&](size_t idx) {
            auto& D_block = D_block_local.local();

            const auto& sig = significant_pairs[idx];
            const size_t sa = sig.sa;
            const size_t sb = sig.sb;
            const auto& sp = shell_pair_data[sig.sp_data_idx];

            const int bf_a = ao_first_bf[sa];
            const int bf_b = ao_first_bf[sb];
            const int na = ao_shells_vec[sa].size();
            const int nb = ao_shells_vec[sb].size();
            const int nab = sp.primitives[0].nab;
            const int nherm_ab = sp.nherm();

            // Extract D_block and compute D_max
            double d_max = 0.0;
            for (int a = 0; a < na; ++a) {
                for (int b = 0; b < nb; ++b) {
                    double D_val = D(bf_a + a, bf_b + b);
                    if (sa != sb) D_val *= 2.0;
                    D_block[a * nb + b] = D_val;
                    d_max = std::max(d_max, std::abs(D_val));
                }
            }
            ws.D_max[idx] = d_max;
            const double schwarz_ab = use_screening ? schwarz(sa, sb) : 1.0;
            ws.screen_val[idx] = d_max * schwarz_ab;
            Eigen::Map<const Eigen::VectorXd> D_vec(D_block.data(), nab);

            // Compute X for each primitive
            double* X_base = ws.X_all.data() + shell_pair_X_offset[idx];
            for (size_t i = 0; i < sp.primitives.size(); ++i) {
                const auto& prim = sp.primitives[i];
                Eigen::Map<const MatRM> E_ab(prim.E_matrix.data(), nab, nherm_ab);
                Eigen::Map<Eigen::VectorXd> X_i(X_base + i * nherm_ab, nherm_ab);
                X_i.noalias() = E_ab.transpose() * D_vec;
            }
        });

        // ===== STEP 2: Forward pass - parallel over pairs with thread-local Y =====
        occ::parallel::parallel_for(size_t(0), npairs, [&](size_t idx) {
            // Per-pair Schwarz screening (without aux factor for now)
            if (use_screening && ws.screen_val[idx] < screening_threshold) {
                return;
            }

            auto& Y_all = Y_local.local();
            auto& R_pq_buf = R_pq_local.local();

            const auto& sig = significant_pairs[idx];
            const size_t sa = sig.sa;
            const size_t sb = sig.sb;
            const auto& sp = shell_pair_data[sig.sp_data_idx];
            const int la = ao_shells_vec[sa].l;
            const int lb = ao_shells_vec[sb].l;
            const int nherm_ab = sp.nherm();
            const int L_ab = la + lb;

            double* X_base = ws.X_all.data() + shell_pair_X_offset[idx];

            for (size_t sc = 0; sc < naux_shells; ++sc) {
                const auto& aux = aux_shells[sc];
                const int nherm_c = aux.nherm();
                const int lc = aux.lc;
                const double sph_c = sph_factor(lc);
                const int L_total = L_ab + lc;

                // Per-aux Schwarz screening
                if (use_screening && ws.screen_val[idx] * V_diag_sqrt_max[sc] < screening_threshold) {
                    continue;
                }

                const auto& rtab = r_index_tables[L_ab][lc];
                double* Y_base = Y_all.data() + aux_shell_Y_offset[sc];

                for (size_t i = 0; i < sp.primitives.size(); ++i) {
                    const auto& ao_prim = sp.primitives[i];
                    const double p = ao_prim.p;
                    const double* X_i = X_base + i * nherm_ab;

                    for (size_t k = 0; k < aux.primitives.size(); ++k) {
                        const auto& aux_prim = aux.primitives[k];
                        const double gamma = aux_prim.gamma;
                        const double cc = aux_prim.coeff;
                        const double pq = p + gamma;
                        const double alpha = p * gamma / pq;

                        const double PCx = ao_prim.Px - aux.C[0];
                        const double PCy = ao_prim.Py - aux.C[1];
                        const double PCz = ao_prim.Pz - aux.C[2];

                        const double prefactor = ao_prim.prefactor * cc * sph_c *
                            pi_1p5 / (gamma * std::sqrt(pq));

                        double* Y_k = Y_base + k * nherm_c;

                        if (!fused_forward_dispatch<double, BoysParamsDefault>(
                                boys_table, L_ab, lc, alpha, PCx, PCy, PCz,
                                prefactor, X_i, Y_k)) {
                            double* R_pq = R_pq_buf.data();
                            RIntsDynamic<double> R;
                            compute_r_ints_3c_dispatch<double, BoysParamsDefault>(
                                boys_table, L_total, alpha, PCx, PCy, PCz, R);
                            build_R_pq(R, rtab, R_pq);

                            Eigen::Map<const Eigen::VectorXd> X_vec(X_i, nherm_ab);
                            Eigen::Map<const MatRM> R_mat(R_pq, nherm_ab, nherm_c);
                            Eigen::Map<Eigen::VectorXd> Y_vec(Y_k, nherm_c);
                            Y_vec.noalias() += prefactor * (R_mat.transpose() * X_vec);
                        }
                    }
                }
            }
        });

        // Combine thread-local Y arrays
        std::vector<double> Y_all(ws.Y_all.size(), 0.0);
        for (const auto& local_Y : Y_local) {
            for (size_t i = 0; i < Y_all.size(); ++i) {
                Y_all[i] += local_Y[i];
            }
        }

        // ===== STEP 2b: Transform Y to g (Eq. 19) =====
        for (size_t sc = 0; sc < naux_shells; ++sc) {
            const auto& aux = aux_shells[sc];
            const int nc = aux.primitives[0].nc;
            const int nherm_c = aux.nherm();
            const int bf_c = aux_first_bf[sc];

            double* Y_base = Y_all.data() + aux_shell_Y_offset[sc];
            for (size_t k = 0; k < aux.primitives.size(); ++k) {
                const auto& aux_prim = aux.primitives[k];
                Eigen::Map<const MatRM> E_c(aux_prim.E_matrix.data(), nc, nherm_c);
                Eigen::Map<const Eigen::VectorXd> Y_k(Y_base + k * nherm_c, nherm_c);
                g.segment(bf_c, nc).noalias() += E_c * Y_k;
            }
        }

        // ===== STEP 3: Solve d = V^-1 @ g (Eq. 12) =====
        Vec d = V_LLt.solve(g);

        // ===== STEP 4: Compute T = E^T @ d for ALL aux shells (Eq. 20) =====
        std::vector<double> T_all(ws.T_all.size());
        for (size_t sc = 0; sc < naux_shells; ++sc) {
            const auto& aux = aux_shells[sc];
            const int nc = aux.primitives[0].nc;
            const int nherm_c = aux.nherm();
            const int bf_c = aux_first_bf[sc];

            double* T_base = T_all.data() + aux_shell_T_offset[sc];
            for (size_t k = 0; k < aux.primitives.size(); ++k) {
                const auto& aux_prim = aux.primitives[k];
                Eigen::Map<const MatRM> E_c(aux_prim.E_matrix.data(), nc, nherm_c);
                Eigen::Map<Eigen::VectorXd> T_k(T_base + k * nherm_c, nherm_c);
                T_k.noalias() = E_c.transpose() * d.segment(bf_c, nc);
            }
        }

        // ===== STEP 5: Backward pass - parallel over pairs with thread-local U =====
        occ::parallel::parallel_for(size_t(0), npairs, [&](size_t idx) {
            auto& U_all = U_local.local();
            auto& R_pq_buf = R_pq_local.local();

            const auto& sig = significant_pairs[idx];
            const size_t sa = sig.sa;
            const size_t sb = sig.sb;
            const auto& sp = shell_pair_data[sig.sp_data_idx];
            const int la = ao_shells_vec[sa].l;
            const int lb = ao_shells_vec[sb].l;
            const int nherm_ab = sp.nherm();
            const int L_ab = la + lb;

            double* U_base = U_all.data() + shell_pair_X_offset[idx];

            for (size_t sc = 0; sc < naux_shells; ++sc) {
                const auto& aux = aux_shells[sc];
                const int nherm_c = aux.nherm();
                const int lc = aux.lc;
                const double sph_c = sph_factor(lc);
                const int L_total = L_ab + lc;

                const auto& rtab = r_index_tables[L_ab][lc];
                const double* T_base = T_all.data() + aux_shell_T_offset[sc];

                for (size_t i = 0; i < sp.primitives.size(); ++i) {
                    const auto& ao_prim = sp.primitives[i];
                    const double p = ao_prim.p;
                    double* U_i = U_base + i * nherm_ab;

                    for (size_t k = 0; k < aux.primitives.size(); ++k) {
                        const auto& aux_prim = aux.primitives[k];
                        const double gamma = aux_prim.gamma;
                        const double cc = aux_prim.coeff;
                        const double pq = p + gamma;
                        const double alpha = p * gamma / pq;

                        const double PCx = ao_prim.Px - aux.C[0];
                        const double PCy = ao_prim.Py - aux.C[1];
                        const double PCz = ao_prim.Pz - aux.C[2];

                        const double prefactor = ao_prim.prefactor * cc * sph_c *
                            pi_1p5 / (gamma * std::sqrt(pq));

                        const double* T_k = T_base + k * nherm_c;

                        if (!fused_backward_dispatch<double, BoysParamsDefault>(
                                boys_table, L_ab, lc, alpha, PCx, PCy, PCz,
                                prefactor, T_k, U_i)) {
                            double* R_pq = R_pq_buf.data();
                            RIntsDynamic<double> R;
                            compute_r_ints_3c_dispatch<double, BoysParamsDefault>(
                                boys_table, L_total, alpha, PCx, PCy, PCz, R);
                            build_R_pq(R, rtab, R_pq);

                            Eigen::Map<const MatRM> R_mat(R_pq, nherm_ab, nherm_c);
                            Eigen::Map<const Eigen::VectorXd> T_vec(T_k, nherm_c);
                            Eigen::Map<Eigen::VectorXd> U_vec(U_i, nherm_ab);
                            U_vec.noalias() += prefactor * (R_mat * T_vec);
                        }
                    }
                }
            }
        });

        // Combine thread-local U arrays
        std::vector<double> U_all(ws.U_all.size(), 0.0);
        for (const auto& local_U : U_local) {
            for (size_t i = 0; i < U_all.size(); ++i) {
                U_all[i] += local_U[i];
            }
        }

        // ===== STEP 6: Contract U_all with E_ab to get J - parallel with thread-local J =====
        occ::parallel::thread_local_storage<std::vector<double>> J_block_local(
            [max_nab]() { return std::vector<double>(max_nab); });

        occ::parallel::parallel_for(size_t(0), npairs, [&](size_t idx) {
            auto& J = J_local.local();
            auto& J_block = J_block_local.local();

            const auto& sig = significant_pairs[idx];
            const size_t sa = sig.sa;
            const size_t sb = sig.sb;
            const auto& sp = shell_pair_data[sig.sp_data_idx];

            const int bf_a = ao_first_bf[sa];
            const int bf_b = ao_first_bf[sb];
            const int na = ao_shells_vec[sa].size();
            const int nb = ao_shells_vec[sb].size();
            const int nab = sp.primitives[0].nab;
            const int nherm_ab = sp.nherm();

            std::fill(J_block.begin(), J_block.begin() + nab, 0.0);

            const double* U_base = U_all.data() + shell_pair_X_offset[idx];

            for (size_t i = 0; i < sp.primitives.size(); ++i) {
                const auto& ao_prim = sp.primitives[i];
                Eigen::Map<const MatRM> E_ab(ao_prim.E_matrix.data(), nab, nherm_ab);
                Eigen::Map<const Eigen::VectorXd> U_i(U_base + i * nherm_ab, nherm_ab);
                Eigen::Map<Eigen::VectorXd> J_vec(J_block.data(), nab);
                J_vec.noalias() += E_ab * U_i;
            }

            // Scatter J_block to J matrix
            for (int a = 0; a < na; ++a) {
                for (int b = 0; b < nb; ++b) {
                    J(bf_a + a, bf_b + b) += J_block[a * nb + b];
                    if (sa != sb) {
                        J(bf_b + b, bf_a + a) += J_block[a * nb + b];
                    }
                }
            }
        });

        // Combine thread-local J matrices
        Mat J_result = Mat::Zero(nbf, nbf);
        for (const auto& local_J : J_local) {
            J_result += local_J;
        }

        occ::timing::stop(occ::timing::category::df);
        return 2.0 * J_result;
    }
};

SplitRIJ::SplitRIJ(const gto::AOBasis& ao_basis, const gto::AOBasis& aux_basis,
                   const ShellPairList& shellpairs, const Mat& schwarz)
    : impl_(std::make_unique<Impl>(ao_basis, aux_basis, shellpairs, schwarz)) {}

SplitRIJ::~SplitRIJ() = default;

SplitRIJ::SplitRIJ(SplitRIJ&&) noexcept = default;
SplitRIJ& SplitRIJ::operator=(SplitRIJ&&) noexcept = default;

Mat SplitRIJ::coulomb(const MolecularOrbitals& mo) const {
    switch (mo.kind) {
    case SpinorbitalKind::Restricted:
        return coulomb_from_density(mo.D);
    case SpinorbitalKind::Unrestricted: {
        Mat D_total = qm::block::a(mo.D) + qm::block::b(mo.D);
        Mat J = coulomb_from_density(D_total);
        Mat result(2 * nbf(), nbf());
        qm::block::a(result) = J;
        qm::block::b(result) = J;
        return result;
    }
    case SpinorbitalKind::General: {
        Mat D_total = qm::block::aa(mo.D) + qm::block::bb(mo.D);
        Mat J = coulomb_from_density(D_total);
        Mat result = Mat::Zero(2 * nbf(), 2 * nbf());
        qm::block::aa(result) = J;
        qm::block::bb(result) = J;
        return result;
    }
    default:
        throw std::runtime_error("SplitRIJ::coulomb: unknown SpinorbitalKind");
    }
}

Mat SplitRIJ::coulomb_from_density(const Mat& D) const {
    return impl_->coulomb(D);
}

const gto::AOBasis& SplitRIJ::aux_basis() const {
    return impl_->aux_basis;
}

const gto::AOBasis& SplitRIJ::ao_basis() const {
    return impl_->ao_basis;
}

size_t SplitRIJ::naux() const {
    return impl_->aux_basis.nbf();
}

size_t SplitRIJ::nbf() const {
    return impl_->ao_basis.nbf();
}

} // namespace occ::qm
