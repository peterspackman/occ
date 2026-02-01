#include <occ/ints/esp.h>
#include <occ/ints/kernels.h>
#include <occ/ints/rints_specialized.h>
#include <occ/gto/gto.h>
#include <Eigen/Dense>
#include <stdexcept>

namespace occ::ints {

// Number of spherical harmonics for angular momentum l
inline int nsph(int l) {
    return l == 0 ? 1 : 2 * l + 1;
}

// Build combined transformation matrix T_ab = kron(T_a, T_b)
// Transforms from Cartesian product basis to spherical product basis
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
build_cart_to_sph_transform(int la, int lb) {
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // Get individual transformation matrices
    occ::Mat T_a = occ::gto::cartesian_to_spherical_transformation_matrix(la);
    occ::Mat T_b = occ::gto::cartesian_to_spherical_transformation_matrix(lb);

    int nsph_a = nsph(la);
    int nsph_b = nsph(lb);
    int ncart_a = ncart(la);
    int ncart_b = ncart(lb);

    // Build Kronecker product: T_ab[i*nsph_b + j, k*ncart_b + l] = T_a[i,k] * T_b[j,l]
    Mat T_ab(nsph_a * nsph_b, ncart_a * ncart_b);

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

// Type aliases matching ESPEvaluator
template <typename T>
using Mat3N = Eigen::Matrix<T, 3, Eigen::Dynamic>;
template <typename T>
using MatRM = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Type-erased shell pair storage
template <typename T>
struct ShellPairHolder {
    virtual ~ShellPairHolder() = default;
    virtual int la() const = 0;
    virtual int lb() const = 0;
    virtual int nab() const = 0;
    virtual int nherm() const = 0;
    virtual bool is_spherical() const = 0;
    virtual void evaluate(const Eigen::Ref<const Mat3N<T>>& C,
                          const T* boys_table,
                          Eigen::Ref<MatRM<T>> integrals,
                          Eigen::Ref<MatRM<T>> workspace) const = 0;
    virtual void evaluate_overlap(T* integrals) const = 0;

    // Screening support: pair center and extent based on most diffuse primitive
    virtual std::array<T, 3> pair_center() const = 0;
    virtual T pair_extent() const = 0;
};

// Concrete implementation for specific LA, LB (Cartesian)
template <typename T, int LA, int LB>
struct ShellPairHolderCartesian : ShellPairHolder<T> {
    ShellPairData<T> data;
    std::array<T, 3> m_pair_center;
    T m_pair_extent;

    ShellPairHolderCartesian(ShellPairData<T>&& d) : data(std::move(d)) {
        // Find the most diffuse primitive (smallest exponent p)
        // This determines the pair extent and effective center
        T p_min = std::numeric_limits<T>::max();
        m_pair_center = {T(0), T(0), T(0)};
        for (const auto& prim : data.primitives) {
            if (prim.p < p_min) {
                p_min = prim.p;
                m_pair_center = {prim.Px, prim.Py, prim.Pz};
            }
        }
        // Extent: radius where exp(-p * r^2) < threshold (1e-12)
        // r = sqrt(-ln(threshold) / p) = sqrt(27.6 / p)
        constexpr T log_threshold = T(27.631021115928547);  // -ln(1e-12)
        m_pair_extent = std::sqrt(log_threshold / p_min);
    }

    int la() const override { return LA; }
    int lb() const override { return LB; }
    int nab() const override { return ncart(LA) * ncart(LB); }
    int nherm() const override { return nhermsum(LA + LB); }
    bool is_spherical() const override { return false; }

    std::array<T, 3> pair_center() const override { return m_pair_center; }
    T pair_extent() const override { return m_pair_extent; }

    void evaluate(const Eigen::Ref<const Mat3N<T>>& C,
                  const T* boys_table,
                  Eigen::Ref<MatRM<T>> integrals,
                  Eigen::Ref<MatRM<T>> workspace) const override {
        constexpr int L = LA + LB;
        constexpr int nab_val = ncart(LA) * ncart(LB);
        constexpr int nherm_val = nhermsum(L);

        const int npts = static_cast<int>(C.cols());
        integrals.setZero();

        for (const auto& prim : data.primitives) {
            // Compute R-integrals for all points at once
            compute_r_ints_batch_direct<T, L, BoysParamsDefault>(
                boys_table, prim.p, npts,
                prim.Px, prim.Py, prim.Pz,
                C,
                workspace);

            Eigen::Map<const MatRM<T>> E_matrix(prim.E_matrix.data(), nab_val, nherm_val);
            integrals.noalias() += prim.prefactor * workspace * E_matrix.transpose();
        }
    }

    void evaluate_overlap(T* integrals) const override {
        constexpr int nab_val = ncart(LA) * ncart(LB);
        constexpr int nherm_val = nhermsum(LA + LB);

        // Zero the output
        for (int ab = 0; ab < nab_val; ++ab) {
            integrals[ab] = T(0);
        }

        // Overlap prefactor ratio to ESP prefactor:
        // (π/p)^{3/2} / (2π/p) = sqrt(π/p) / 2
        for (const auto& prim : data.primitives) {
            const T overlap_factor = std::sqrt(BoysConstants<T>::pi / prim.p) / T(2);
            const T overlap_prefactor = prim.prefactor * overlap_factor;

            // Overlap uses only the (0,0,0) Hermite component (index 0)
            for (int ab = 0; ab < nab_val; ++ab) {
                integrals[ab] += overlap_prefactor * prim.E_matrix[ab * nherm_val];
            }
        }
    }
};

// Concrete implementation for specific LA, LB (Spherical)
// E-matrices are pre-transformed to spherical basis
template <typename T, int LA, int LB>
struct ShellPairHolderSpherical : ShellPairHolder<T> {
    ShellPairData<T> data;  // E_matrix is already transformed to spherical
    int nab_sph;
    std::array<T, 3> m_pair_center;
    T m_pair_extent;

    ShellPairHolderSpherical(ShellPairData<T>&& d)
        : data(std::move(d)), nab_sph(nsph(LA) * nsph(LB)) {
        // Find the most diffuse primitive (smallest exponent p)
        T p_min = std::numeric_limits<T>::max();
        m_pair_center = {T(0), T(0), T(0)};
        for (const auto& prim : data.primitives) {
            if (prim.p < p_min) {
                p_min = prim.p;
                m_pair_center = {prim.Px, prim.Py, prim.Pz};
            }
        }
        // Extent: radius where exp(-p * r^2) < threshold (1e-12)
        constexpr T log_threshold = T(27.631021115928547);  // -ln(1e-12)
        m_pair_extent = std::sqrt(log_threshold / p_min);
    }

    int la() const override { return LA; }
    int lb() const override { return LB; }
    int nab() const override { return nab_sph; }
    int nherm() const override { return nhermsum(LA + LB); }
    bool is_spherical() const override { return true; }

    std::array<T, 3> pair_center() const override { return m_pair_center; }
    T pair_extent() const override { return m_pair_extent; }

    void evaluate(const Eigen::Ref<const Mat3N<T>>& C,
                  const T* boys_table,
                  Eigen::Ref<MatRM<T>> integrals,
                  Eigen::Ref<MatRM<T>> workspace) const override {
        constexpr int L = LA + LB;
        constexpr int nherm_val = nhermsum(L);

        const int npts = static_cast<int>(C.cols());
        integrals.setZero();

        for (const auto& prim : data.primitives) {
            // Compute R-integrals for all points at once
            compute_r_ints_batch_direct<T, L, BoysParamsDefault>(
                boys_table, prim.p, npts,
                prim.Px, prim.Py, prim.Pz,
                C,
                workspace);

            // E_matrix is already [nab_sph, nherm]
            Eigen::Map<const MatRM<T>> E_matrix(prim.E_matrix.data(), nab_sph, nherm_val);
            integrals.noalias() += prim.prefactor * workspace * E_matrix.transpose();
        }
    }

    void evaluate_overlap(T* integrals) const override {
        constexpr int nherm_val = nhermsum(LA + LB);

        // Zero the output
        for (int ab = 0; ab < nab_sph; ++ab) {
            integrals[ab] = T(0);
        }

        // Overlap prefactor ratio to ESP prefactor:
        // (π/p)^{3/2} / (2π/p) = sqrt(π/p) / 2
        for (const auto& prim : data.primitives) {
            const T overlap_factor = std::sqrt(BoysConstants<T>::pi / prim.p) / T(2);
            const T overlap_prefactor = prim.prefactor * overlap_factor;

            // Overlap uses only the (0,0,0) Hermite component (index 0)
            // E_matrix is already transformed to spherical [nab_sph, nherm]
            for (int ab = 0; ab < nab_sph; ++ab) {
                integrals[ab] += overlap_prefactor * prim.E_matrix[ab * nherm_val];
            }
        }
    }
};

// Transform E-matrices from Cartesian to spherical
template <typename T, int LA, int LB>
ShellPairData<T> transform_shell_pair_to_spherical(ShellPairData<T>&& cart_data) {
    constexpr int nab_cart = ncart(LA) * ncart(LB);
    const int nab_sph = nsph(LA) * nsph(LB);
    constexpr int nherm = nhermsum(LA + LB);

    using MatRM = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // Build transformation matrix
    MatRM T_ab = build_cart_to_sph_transform<T>(LA, LB);

    // Transform each primitive's E-matrix
    for (auto& prim : cart_data.primitives) {
        Eigen::Map<MatRM> E_cart(prim.E_matrix.data(), nab_cart, nherm);

        // E_sph = T_ab @ E_cart
        MatRM E_sph = T_ab * E_cart;

        // Resize and copy
        prim.E_matrix.resize(nab_sph * nherm);
        prim.nab = nab_sph;
        Eigen::Map<MatRM>(prim.E_matrix.data(), nab_sph, nherm) = E_sph;
    }

    return std::move(cart_data);
}

template <typename T>
struct ESPEvaluator<T>::Impl {
    const T* boys_table;
    std::vector<std::unique_ptr<ShellPairHolder<T>>> shell_pairs;
    std::vector<T> workspace;
    size_t workspace_capacity = 0;

    explicit Impl(const T* table) : boys_table(table) {}

    void ensure_workspace(size_t needed) {
        if (needed > workspace_capacity) {
            workspace.resize(needed);
            workspace_capacity = needed;
        }
    }
};

template <typename T>
ESPEvaluator<T>::ESPEvaluator(const T* boys_table)
    : impl_(std::make_unique<Impl>(boys_table)) {}

template <typename T>
ESPEvaluator<T>::~ESPEvaluator() = default;

template <typename T>
ESPEvaluator<T>::ESPEvaluator(ESPEvaluator&&) noexcept = default;

template <typename T>
ESPEvaluator<T>& ESPEvaluator<T>::operator=(ESPEvaluator&&) noexcept = default;

// Helper macros for dispatch
#define DISPATCH_CARTESIAN(LA, LB) \
    if (la == LA && lb == LB) { \
        auto data = precompute_shell_pair<T, LA, LB>( \
            na_prim, nb_prim, exponents_a, exponents_b, \
            coeffs_a, coeffs_b, A, B); \
        impl_->shell_pairs.push_back( \
            std::make_unique<ShellPairHolderCartesian<T, LA, LB>>(std::move(data))); \
        return impl_->shell_pairs.size() - 1; \
    }

#define DISPATCH_SPHERICAL(LA, LB) \
    if (la == LA && lb == LB) { \
        auto cart_data = precompute_shell_pair<T, LA, LB>( \
            na_prim, nb_prim, exponents_a, exponents_b, \
            coeffs_a, coeffs_b, A, B); \
        auto sph_data = transform_shell_pair_to_spherical<T, LA, LB>(std::move(cart_data)); \
        impl_->shell_pairs.push_back( \
            std::make_unique<ShellPairHolderSpherical<T, LA, LB>>(std::move(sph_data))); \
        return impl_->shell_pairs.size() - 1; \
    }

template <typename T>
size_t ESPEvaluator<T>::add_shell_pair(int la, int lb,
                                        int na_prim, int nb_prim,
                                        const T* exponents_a, const T* exponents_b,
                                        const T* coeffs_a, const T* coeffs_b,
                                        const T* A, const T* B) {
    // Default: Cartesian
    return add_shell_pair(la, lb, na_prim, nb_prim, exponents_a, exponents_b,
                          coeffs_a, coeffs_b, A, B, false);
}

template <typename T>
size_t ESPEvaluator<T>::add_shell_pair(int la, int lb,
                                        int na_prim, int nb_prim,
                                        const T* exponents_a, const T* exponents_b,
                                        const T* coeffs_a, const T* coeffs_b,
                                        const T* A, const T* B,
                                        bool spherical) {
    if (spherical) {
        DISPATCH_SPHERICAL(0, 0)
        DISPATCH_SPHERICAL(0, 1)
        DISPATCH_SPHERICAL(0, 2)
        DISPATCH_SPHERICAL(0, 3)
        DISPATCH_SPHERICAL(1, 0)
        DISPATCH_SPHERICAL(1, 1)
        DISPATCH_SPHERICAL(1, 2)
        DISPATCH_SPHERICAL(1, 3)
        DISPATCH_SPHERICAL(2, 0)
        DISPATCH_SPHERICAL(2, 1)
        DISPATCH_SPHERICAL(2, 2)
        DISPATCH_SPHERICAL(2, 3)
        DISPATCH_SPHERICAL(3, 0)
        DISPATCH_SPHERICAL(3, 1)
        DISPATCH_SPHERICAL(3, 2)
        DISPATCH_SPHERICAL(3, 3)
        DISPATCH_SPHERICAL(0, 4)
        DISPATCH_SPHERICAL(1, 4)
        DISPATCH_SPHERICAL(2, 4)
        DISPATCH_SPHERICAL(3, 4)
        DISPATCH_SPHERICAL(4, 0)
        DISPATCH_SPHERICAL(4, 1)
        DISPATCH_SPHERICAL(4, 2)
        DISPATCH_SPHERICAL(4, 3)
        DISPATCH_SPHERICAL(4, 4)
        DISPATCH_SPHERICAL(0, 5)
        DISPATCH_SPHERICAL(1, 5)
        DISPATCH_SPHERICAL(2, 5)
        DISPATCH_SPHERICAL(3, 5)
        DISPATCH_SPHERICAL(4, 5)
        DISPATCH_SPHERICAL(5, 0)
        DISPATCH_SPHERICAL(5, 1)
        DISPATCH_SPHERICAL(5, 2)
        DISPATCH_SPHERICAL(5, 3)
        DISPATCH_SPHERICAL(5, 4)
        DISPATCH_SPHERICAL(5, 5)
    } else {
        DISPATCH_CARTESIAN(0, 0)
        DISPATCH_CARTESIAN(0, 1)
        DISPATCH_CARTESIAN(0, 2)
        DISPATCH_CARTESIAN(0, 3)
        DISPATCH_CARTESIAN(1, 0)
        DISPATCH_CARTESIAN(1, 1)
        DISPATCH_CARTESIAN(1, 2)
        DISPATCH_CARTESIAN(1, 3)
        DISPATCH_CARTESIAN(2, 0)
        DISPATCH_CARTESIAN(2, 1)
        DISPATCH_CARTESIAN(2, 2)
        DISPATCH_CARTESIAN(2, 3)
        DISPATCH_CARTESIAN(3, 0)
        DISPATCH_CARTESIAN(3, 1)
        DISPATCH_CARTESIAN(3, 2)
        DISPATCH_CARTESIAN(3, 3)
        DISPATCH_CARTESIAN(0, 4)
        DISPATCH_CARTESIAN(1, 4)
        DISPATCH_CARTESIAN(2, 4)
        DISPATCH_CARTESIAN(3, 4)
        DISPATCH_CARTESIAN(4, 0)
        DISPATCH_CARTESIAN(4, 1)
        DISPATCH_CARTESIAN(4, 2)
        DISPATCH_CARTESIAN(4, 3)
        DISPATCH_CARTESIAN(4, 4)
        DISPATCH_CARTESIAN(0, 5)
        DISPATCH_CARTESIAN(1, 5)
        DISPATCH_CARTESIAN(2, 5)
        DISPATCH_CARTESIAN(3, 5)
        DISPATCH_CARTESIAN(4, 5)
        DISPATCH_CARTESIAN(5, 0)
        DISPATCH_CARTESIAN(5, 1)
        DISPATCH_CARTESIAN(5, 2)
        DISPATCH_CARTESIAN(5, 3)
        DISPATCH_CARTESIAN(5, 4)
        DISPATCH_CARTESIAN(5, 5)
    }

    throw std::runtime_error("Unsupported angular momentum combination: la=" +
                             std::to_string(la) + ", lb=" + std::to_string(lb));
}

#undef DISPATCH_CARTESIAN
#undef DISPATCH_SPHERICAL

template <typename T>
void ESPEvaluator<T>::evaluate(size_t shell_idx,
                                const Eigen::Ref<const Mat3N>& C,
                                Eigen::Ref<MatRM> integrals,
                                Eigen::Ref<MatRM> workspace) {
    if (shell_idx >= impl_->shell_pairs.size()) {
        throw std::out_of_range("Invalid shell pair index");
    }

    const auto& sp = impl_->shell_pairs[shell_idx];
    sp->evaluate(C, impl_->boys_table, integrals, workspace);
}

template <typename T>
size_t ESPEvaluator<T>::workspace_size(size_t shell_idx, int npts) const {
    if (shell_idx >= impl_->shell_pairs.size()) {
        throw std::out_of_range("Invalid shell pair index");
    }
    return static_cast<size_t>(npts) * impl_->shell_pairs[shell_idx]->nherm();
}

template <typename T>
void ESPEvaluator<T>::evaluate_overlap(size_t shell_idx, T* integrals) {
    if (shell_idx >= impl_->shell_pairs.size()) {
        throw std::out_of_range("Invalid shell pair index");
    }

    impl_->shell_pairs[shell_idx]->evaluate_overlap(integrals);
}

template <typename T>
int ESPEvaluator<T>::nab(size_t shell_idx) const {
    if (shell_idx >= impl_->shell_pairs.size()) {
        throw std::out_of_range("Invalid shell pair index");
    }
    return impl_->shell_pairs[shell_idx]->nab();
}

template <typename T>
int ESPEvaluator<T>::nherm(size_t shell_idx) const {
    if (shell_idx >= impl_->shell_pairs.size()) {
        throw std::out_of_range("Invalid shell pair index");
    }
    return impl_->shell_pairs[shell_idx]->nherm();
}

template <typename T>
std::pair<int, int> ESPEvaluator<T>::angular_momenta(size_t shell_idx) const {
    if (shell_idx >= impl_->shell_pairs.size()) {
        throw std::out_of_range("Invalid shell pair index");
    }
    const auto& sp = impl_->shell_pairs[shell_idx];
    return {sp->la(), sp->lb()};
}

template <typename T>
void ESPEvaluator<T>::clear() {
    impl_->shell_pairs.clear();
}

template <typename T>
size_t ESPEvaluator<T>::num_shell_pairs() const {
    return impl_->shell_pairs.size();
}

template <typename T>
bool ESPEvaluator<T>::is_spherical(size_t shell_idx) const {
    if (shell_idx >= impl_->shell_pairs.size()) {
        throw std::out_of_range("Invalid shell pair index");
    }
    return impl_->shell_pairs[shell_idx]->is_spherical();
}

template <typename T>
std::array<T, 3> ESPEvaluator<T>::pair_center(size_t shell_idx) const {
    if (shell_idx >= impl_->shell_pairs.size()) {
        throw std::out_of_range("Invalid shell pair index");
    }
    return impl_->shell_pairs[shell_idx]->pair_center();
}

template <typename T>
T ESPEvaluator<T>::pair_extent(size_t shell_idx) const {
    if (shell_idx >= impl_->shell_pairs.size()) {
        throw std::out_of_range("Invalid shell pair index");
    }
    return impl_->shell_pairs[shell_idx]->pair_extent();
}

// Explicit instantiations
template class ESPEvaluator<float>;
template class ESPEvaluator<double>;

} // namespace occ::ints
