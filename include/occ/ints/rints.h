#pragma once
#include <occ/ints/boys.h>
#include <array>
#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define OCC_GPU_ENABLED __host__ __device__
#define OCC_GPU_INLINE __forceinline__
#else
#define OCC_GPU_ENABLED
#define OCC_GPU_INLINE inline
#endif

namespace occ::ints {

constexpr int nherm(int L) {
    return (L + 1) * (L + 2) / 2;
}

constexpr int nhermsum(int L) {
    return (L + 1) * (L + 2) * (L + 3) / 6;
}

OCC_GPU_ENABLED OCC_GPU_INLINE
constexpr int hermite_index(int t, int u, int v) {
    int L = t + u + v;
    int offset = nhermsum(L - 1);
    int uv = u + v;
    return offset + uv * (uv + 1) / 2 + v;
}

template <typename T, int L>
struct RInts {
    static constexpr int size = nhermsum(L);
    alignas(64) T data[size];

    OCC_GPU_ENABLED OCC_GPU_INLINE
    T& operator()(int t, int u, int v) {
        return data[hermite_index(t, u, v)];
    }

    OCC_GPU_ENABLED OCC_GPU_INLINE
    T operator()(int t, int u, int v) const {
        return data[hermite_index(t, u, v)];
    }
};

template <typename T, int L, typename BoysParams>
OCC_GPU_ENABLED
void compute_r_ints(const T* boys_table, T p, T PCx, T PCy, T PCz,
                    RInts<T, L>& R) {
    const T PC2 = PCx * PCx + PCy * PCy + PCz * PCz;
    const T Tp = p * PC2;
    const T neg_2p = -T(2) * p;

    T Fm[L + 1];
    boys_evaluate<T, L + 1, BoysParams>(boys_table, Tp, 0, Fm);

    constexpr int total_size = (L + 2) * nhermsum(L);
    T R_all[total_size];

    for (int i = 0; i < total_size; ++i) {
        R_all[i] = T(0);
    }

    auto R_m = [&](int m, int t, int u, int v) -> T& {
        return R_all[m * nhermsum(L) + hermite_index(t, u, v)];
    };

    T neg_2p_power = T(1);
    for (int m = 0; m <= L; ++m) {
        R_m(m, 0, 0, 0) = neg_2p_power * Fm[m];
        neg_2p_power *= neg_2p;
    }

    for (int t = 0; t < L; ++t) {
        for (int m = 0; m <= L - t - 1; ++m) {
            T val = PCx * R_m(m + 1, t, 0, 0);
            if (t > 0) {
                val += t * R_m(m + 1, t - 1, 0, 0);
            }
            R_m(m, t + 1, 0, 0) = val;
        }
    }

    for (int t = 0; t <= L; ++t) {
        for (int u = 0; u < L - t; ++u) {
            for (int m = 0; m <= L - t - u - 1; ++m) {
                T val = PCy * R_m(m + 1, t, u, 0);
                if (u > 0) {
                    val += u * R_m(m + 1, t, u - 1, 0);
                }
                R_m(m, t, u + 1, 0) = val;
            }
        }
    }

    for (int t = 0; t <= L; ++t) {
        for (int u = 0; u <= L - t; ++u) {
            for (int v = 0; v < L - t - u; ++v) {
                for (int m = 0; m <= L - t - u - v - 1; ++m) {
                    T val = PCz * R_m(m + 1, t, u, v);
                    if (v > 0) {
                        val += v * R_m(m + 1, t, u, v - 1);
                    }
                    R_m(m, t, u, v + 1) = val;
                }
            }
        }
    }

    for (int t = 0; t <= L; ++t) {
        for (int u = 0; u <= L - t; ++u) {
            for (int v = 0; v <= L - t - u; ++v) {
                R(t, u, v) = R_m(0, t, u, v);
            }
        }
    }
}

template <typename T, int L>
OCC_GPU_ENABLED
void compute_r_ints(const T* boys_table, T p, T PCx, T PCy, T PCz,
                    T* R_out) {
    RInts<T, L> R;
    compute_r_ints<T, L, BoysParamsDefault>(boys_table, p, PCx, PCy, PCz, R);
    for (int i = 0; i < RInts<T, L>::size; ++i) {
        R_out[i] = R.data[i];
    }
}

constexpr int RINTS_LMAX = 15;
constexpr int RINTS_MAX_SIZE = nhermsum(RINTS_LMAX);

template <typename T>
struct RIntsDynamic {
    alignas(64) T data[RINTS_MAX_SIZE];
    int L;

    OCC_GPU_ENABLED OCC_GPU_INLINE
    void set_L(int l) { L = l; }

    OCC_GPU_ENABLED OCC_GPU_INLINE
    T& operator()(int t, int u, int v) {
        return data[hermite_index(t, u, v)];
    }

    OCC_GPU_ENABLED OCC_GPU_INLINE
    T operator()(int t, int u, int v) const {
        return data[hermite_index(t, u, v)];
    }
};

template <typename T, typename BoysParams>
OCC_GPU_ENABLED
void compute_r_ints_dynamic(const T* boys_table, int L, T p,
                            T PCx, T PCy, T PCz, RIntsDynamic<T>& R) {
    R.set_L(L);

    const T PC2 = PCx * PCx + PCy * PCy + PCz * PCz;
    const T Tp = p * PC2;
    const T neg_2p = -T(2) * p;

    T Fm[RINTS_LMAX + 1];
    if (L <= 15) {
        boys_evaluate<T, 16, BoysParams>(boys_table, Tp, 0, Fm);
    } else {
        for (int m = 0; m <= L; ++m) {
            Fm[m] = detail::boys_reference(Tp, m);
        }
    }

    const int nsz = nhermsum(L);
    T R_all[(RINTS_LMAX + 2) * RINTS_MAX_SIZE];

    for (int i = 0; i < (L + 2) * nsz; ++i) {
        R_all[i] = T(0);
    }

    auto R_m = [&](int m, int t, int u, int v) -> T& {
        return R_all[m * nsz + hermite_index(t, u, v)];
    };

    T neg_2p_power = T(1);
    for (int m = 0; m <= L; ++m) {
        R_m(m, 0, 0, 0) = neg_2p_power * Fm[m];
        neg_2p_power *= neg_2p;
    }

    for (int t = 0; t < L; ++t) {
        for (int m = 0; m <= L - t - 1; ++m) {
            T val = PCx * R_m(m + 1, t, 0, 0);
            if (t > 0) val += t * R_m(m + 1, t - 1, 0, 0);
            R_m(m, t + 1, 0, 0) = val;
        }
    }

    for (int t = 0; t <= L; ++t) {
        for (int u = 0; u < L - t; ++u) {
            for (int m = 0; m <= L - t - u - 1; ++m) {
                T val = PCy * R_m(m + 1, t, u, 0);
                if (u > 0) val += u * R_m(m + 1, t, u - 1, 0);
                R_m(m, t, u + 1, 0) = val;
            }
        }
    }

    for (int t = 0; t <= L; ++t) {
        for (int u = 0; u <= L - t; ++u) {
            for (int v = 0; v < L - t - u; ++v) {
                for (int m = 0; m <= L - t - u - v - 1; ++m) {
                    T val = PCz * R_m(m + 1, t, u, v);
                    if (v > 0) val += v * R_m(m + 1, t, u, v - 1);
                    R_m(m, t, u, v + 1) = val;
                }
            }
        }
    }

    for (int t = 0; t <= L; ++t) {
        for (int u = 0; u <= L - t; ++u) {
            for (int v = 0; v <= L - t - u; ++v) {
                R(t, u, v) = R_m(0, t, u, v);
            }
        }
    }
}

} // namespace occ::ints
