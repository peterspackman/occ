#pragma once
#include <array>
#include <cmath>
#include <memory>

// GPU portability macros
#if defined(__CUDACC__) || defined(__HIPCC__)
#define OCC_GPU_ENABLED __host__ __device__
#define OCC_GPU_INLINE __forceinline__
#else
#define OCC_GPU_ENABLED
#define OCC_GPU_INLINE inline
#endif

namespace occ::ints {

template <typename T>
struct BoysConstants {
    static constexpr T sqrt_pi_over_4 = T(0.88622692545275801365);
    static constexpr T pi = T(3.14159265358979323846);
};

/// Asymptotic expansion for Boys function: F_m(T) ~ sqrt(pi/(4T)) * prod(i-0.5)/T
/// Valid for T > 30
template <typename T, int N>
OCC_GPU_ENABLED OCC_GPU_INLINE
void boys_asymptotic(T x, int m0, T* F) {
    const T one_over_x = T(1) / x;

#if defined(__CUDACC__)
    const T rsqrt_x = rsqrt(x);
#else
    const T rsqrt_x = std::sqrt(one_over_x);
#endif

    T Fm = BoysConstants<T>::sqrt_pi_over_4 * rsqrt_x;

    // Upward recursion to reach m0
    for (int i = 1; i <= m0; ++i) {
        Fm = Fm * (T(i) - T(0.5)) * one_over_x;
    }

    F[0] = Fm;

    // Continue upward for remaining values
    for (int i = 1; i < N; ++i) {
        Fm = Fm * (T(m0 + i) - T(0.5)) * one_over_x;
        F[i] = Fm;
    }
}

template <typename T>
OCC_GPU_ENABLED OCC_GPU_INLINE
T boys_asymptotic(T x, int m) {
    const T one_over_x = T(1) / x;

#if defined(__CUDACC__)
    T Fm = BoysConstants<T>::sqrt_pi_over_4 * rsqrt(x);
#else
    T Fm = BoysConstants<T>::sqrt_pi_over_4 * std::sqrt(one_over_x);
#endif

    for (int i = 1; i <= m; ++i) {
        Fm = Fm * (T(i) - T(0.5)) * one_over_x;
    }
    return Fm;
}

template <int Order_ = 7, int MaxM_ = 16, int MaxT_ = 36, int Segments_ = 252>
struct BoysTableParams {
    static constexpr int Order = Order_;
    static constexpr int MaxM = MaxM_;
    static constexpr int MaxT = MaxT_;
    static constexpr int Segments = Segments_;
    static constexpr double Delta = static_cast<double>(MaxT) / Segments;
    static constexpr int CoeffsPerSegment = (Order + 1) * MaxM;
    static constexpr int TableSize = CoeffsPerSegment * (Segments + 1);
};

using BoysParamsDefault = BoysTableParams<7, 16, 36, 252>;
using BoysParamsExtended = BoysTableParams<7, 32, 117, 819>;
using BoysParamsHighRes = BoysTableParams<7, 16, 36, 504>;

/// Interpolate F_m(x) from Chebyshev table
template <typename T, typename Params>
OCC_GPU_ENABLED OCC_GPU_INLINE
T boys_interpolate(const T* table, T x, int m) {
    constexpr int Order = Params::Order;
    constexpr int MaxM = Params::MaxM;
    constexpr T Delta = static_cast<T>(Params::Delta);

    const T x_over_delta = x / Delta;
    const int seg = static_cast<int>(x_over_delta);

    const T xd = x_over_delta - T(seg) - T(0.5);
    const T* p = table + seg * MaxM * (Order + 1) + m * (Order + 1);

    T result = p[Order];
    for (int k = Order - 1; k >= 0; --k) {
        result = result * xd + p[k];
    }
    return result;
}

template <typename T, int N, typename Params>
OCC_GPU_ENABLED OCC_GPU_INLINE
void boys_interpolate(const T* table, T x, int m0, T* F) {
    constexpr int Order = Params::Order;
    constexpr int MaxM = Params::MaxM;
    constexpr T Delta = static_cast<T>(Params::Delta);

    const T x_over_delta = x / Delta;
    const int seg = static_cast<int>(x_over_delta);
    const T xd = x_over_delta - T(seg) - T(0.5);

    const T* base = table + seg * MaxM * (Order + 1);

    for (int i = 0; i < N; ++i) {
        const T* p = base + (m0 + i) * (Order + 1);

        T result = p[Order];
        for (int k = Order - 1; k >= 0; --k) {
            result = result * xd + p[k];
        }
        F[i] = result;
    }
}

template <typename T, int N, typename Params>
OCC_GPU_ENABLED OCC_GPU_INLINE
void boys_evaluate(const T* table, T x, int m0, T* F) {
    if (x >= T(Params::MaxT)) {
        boys_asymptotic<T, N>(x, m0, F);
    } else {
        boys_interpolate<T, N, Params>(table, x, m0, F);
    }
}

template <typename T, typename Params>
OCC_GPU_ENABLED OCC_GPU_INLINE
T boys_evaluate(const T* table, T x, int m) {
    if (x >= T(Params::MaxT)) {
        return boys_asymptotic(x, m);
    }
    return boys_interpolate<T, Params>(table, x, m);
}

namespace detail {

inline double boys_maclaurin(double T, int m) {
    constexpr double epsilon = 1e-17;

    double denom = m + 0.5;
    double term = std::exp(-T) / (2.0 * denom);
    double old_term = 0.0;
    double sum = term;

    do {
        denom += 1.0;
        old_term = term;
        term = old_term * T / denom;
        sum += term;
    } while (term > sum * epsilon || old_term < term);

    return sum;
}

inline double boys_recursive(double T, int m) {
    if (T == 0.0) return 1.0 / (2 * m + 1);

    double sqrtT = std::sqrt(T);
    double Fm = BoysConstants<double>::sqrt_pi_over_4 / sqrtT * std::erf(sqrtT);

    double expT = std::exp(-T);
    double inv2T = 0.5 / T;
    for (int i = 0; i < m; ++i) {
        Fm = ((2 * i + 1) * Fm - expT) * inv2T;
    }

    return Fm;
}

inline double boys_reference(double T, int m) {
    if (T < m * 0.25 + 1) {
        return boys_maclaurin(T, m);
    }
    return boys_recursive(T, m);
}

} // namespace detail

template <typename Params>
class BoysTableGenerator {
public:
    static constexpr int Order = Params::Order;
    static constexpr int MaxM = Params::MaxM;
    static constexpr int Segments = Params::Segments;
    static constexpr double Delta = Params::Delta;
    static constexpr int TableSize = Params::TableSize;
    static constexpr int N = Order + 1;

    static std::unique_ptr<double[], void(*)(double*)> generate() {
        auto table = std::unique_ptr<double[], void(*)(double*)>(
            new (std::align_val_t{64}) double[TableSize],
            [](double* p) { operator delete[](p, std::align_val_t{64}); }
        );

        std::array<double, N> nodes;
        for (int k = 0; k < N; ++k) {
            nodes[k] = std::cos(((2 * k + 1) * M_PI) / (2.0 * N)) / 2.0;
        }

        std::array<std::array<double, N>, N> T_poly{};
        T_poly[0][0] = 1.0;
        if constexpr (N > 1) {
            T_poly[1][1] = 2.0;
        }
        for (int k = 2; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                if (j > 0) T_poly[k][j] += 4.0 * T_poly[k-1][j-1];
                T_poly[k][j] -= T_poly[k-2][j];
            }
        }

        std::array<std::array<double, N>, N> A{};
        for (int row = 0; row < N; ++row) {
            double x = nodes[row];
            double x2 = 2.0 * x;
            A[row][0] = 1.0;
            if constexpr (N > 1) {
                A[row][1] = x2;
            }
            for (int k = 2; k < N; ++k) {
                A[row][k] = 2.0 * x2 * A[row][k-1] - A[row][k-2];
            }
        }

        std::array<std::array<double, N>, N> LU = A;
        std::array<int, N> perm;
        for (int i = 0; i < N; ++i) perm[i] = i;

        for (int k = 0; k < N; ++k) {
            int max_row = k;
            double max_val = std::abs(LU[k][k]);
            for (int i = k + 1; i < N; ++i) {
                if (std::abs(LU[i][k]) > max_val) {
                    max_val = std::abs(LU[i][k]);
                    max_row = i;
                }
            }
            if (max_row != k) {
                std::swap(LU[k], LU[max_row]);
                std::swap(perm[k], perm[max_row]);
            }

            for (int i = k + 1; i < N; ++i) {
                LU[i][k] /= LU[k][k];
                for (int j = k + 1; j < N; ++j) {
                    LU[i][j] -= LU[i][k] * LU[k][j];
                }
            }
        }

        for (int seg = 0; seg < Segments; ++seg) {
            double a = seg * Delta;
            double b = a + Delta;

            for (int m = 0; m < MaxM; ++m) {
                std::array<double, N> f_values;
                for (int node = 0; node < N; ++node) {
                    double T = a + (nodes[node] + 0.5) * Delta;
                    f_values[node] = detail::boys_maclaurin(T, m);
                }

                std::array<double, N> b_perm;
                for (int i = 0; i < N; ++i) {
                    b_perm[i] = f_values[perm[i]];
                }

                std::array<double, N> y;
                for (int i = 0; i < N; ++i) {
                    y[i] = b_perm[i];
                    for (int j = 0; j < i; ++j) {
                        y[i] -= LU[i][j] * y[j];
                    }
                }

                std::array<double, N> cheb_coeffs;
                for (int i = N - 1; i >= 0; --i) {
                    cheb_coeffs[i] = y[i];
                    for (int j = i + 1; j < N; ++j) {
                        cheb_coeffs[i] -= LU[i][j] * cheb_coeffs[j];
                    }
                    cheb_coeffs[i] /= LU[i][i];
                }

                std::array<double, N> mono_coeffs{};
                for (int k = 0; k < N; ++k) {
                    for (int j = 0; j <= k; ++j) {
                        mono_coeffs[j] += cheb_coeffs[k] * T_poly[k][j];
                    }
                }

                size_t idx = seg * MaxM * N + m * N;
                for (int k = 0; k < N; ++k) {
                    table[idx + k] = mono_coeffs[k];
                }
            }
        }

        for (int k = 0; k < N * MaxM; ++k) {
            table[Segments * MaxM * N + k] = 0.0;
        }

        return table;
    }
};

template <typename Params = BoysParamsDefault>
class Boys {
public:
    using params_type = Params;
    static constexpr int Order = Params::Order;
    static constexpr int MaxM = Params::MaxM;
    static constexpr int MaxT = Params::MaxT;
    static constexpr int Segments = Params::Segments;
    static constexpr int TableSize = Params::TableSize;

    Boys() : m_table(BoysTableGenerator<Params>::generate()) {}

    const double* table() const { return m_table.get(); }
    double* table() { return m_table.get(); }

    static constexpr size_t table_size_bytes() { return TableSize * sizeof(double); }

    double compute(double x, int m) const {
        return boys_evaluate<double, Params>(m_table.get(), x, m);
    }

    template <int N>
    void compute(double x, int m0, double (&F)[N]) const {
        boys_evaluate<double, N, Params>(m_table.get(), x, m0, F);
    }

    template <int N>
    void compute(double x, double (&F)[N]) const {
        boys_evaluate<double, N, Params>(m_table.get(), x, 0, F);
    }

    static double reference(double x, int m) {
        return detail::boys_reference(x, m);
    }

private:
    std::unique_ptr<double[], void(*)(double*)> m_table;
};

using BoysDefault = Boys<BoysParamsDefault>;
using BoysExtended = Boys<BoysParamsExtended>;

inline const BoysDefault& boys() {
    static BoysDefault instance;
    return instance;
}

inline const BoysExtended& boys_extended() {
    static BoysExtended instance;
    return instance;
}

} // namespace occ::ints
