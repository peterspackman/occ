#pragma once
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

constexpr int LMAX = 6;

constexpr int ncart(int L) {
    return (L + 1) * (L + 2) / 2;
}

constexpr int ncartsum(int L) {
    return (L + 1) * (L + 2) * (L + 3) / 6;
}

template <typename T, int A, int B>
struct ECoeffs1D {
    static constexpr int P = A + B;
    alignas(64) T data[(A + 1) * (B + 1) * (P + 1)];

    OCC_GPU_ENABLED OCC_GPU_INLINE
    T& operator()(int i, int j, int t) {
        return data[t + (P + 1) * (j + (B + 1) * i)];
    }

    OCC_GPU_ENABLED OCC_GPU_INLINE
    T operator()(int i, int j, int t) const {
        return data[t + (P + 1) * (j + (B + 1) * i)];
    }
};

template <typename T, int A, int B>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_e_coeffs_1d(T a, T b, T XAB, ECoeffs1D<T, A, B>& E) {
    constexpr int P = A + B;

    const T p = a + b;
    const T mu = a * b / p;
    const T inv_2p = T(0.5) / p;
    const T XPA = b / p * XAB;
    const T XPB = -a / p * XAB;
    const T E00_0 = std::exp(-mu * XAB * XAB);

    for (int i = 0; i <= A; ++i) {
        for (int j = 0; j <= B; ++j) {
            for (int t = 0; t <= P; ++t) {
                E(i, j, t) = T(0);
            }
        }
    }

    E(0, 0, 0) = E00_0;

    for (int j = 0; j < B; ++j) {
        for (int t = 0; t <= j; ++t) {
            T val = XPB * E(0, j, t);
            if (t > 0) val += inv_2p * E(0, j, t - 1);
            if (t < j) val += (t + 1) * E(0, j, t + 1);
            E(0, j + 1, t) += val;
        }
        E(0, j + 1, j + 1) = inv_2p * E(0, j, j);
    }

    for (int i = 0; i < A; ++i) {
        for (int j = 0; j <= B; ++j) {
            const int max_t = i + j;
            for (int t = 0; t <= max_t; ++t) {
                T val = XPA * E(i, j, t);
                if (t > 0) val += inv_2p * E(i, j, t - 1);
                if (t < max_t) val += (t + 1) * E(i, j, t + 1);
                E(i + 1, j, t) += val;
            }
            if (max_t >= 0) {
                E(i + 1, j, max_t + 1) = inv_2p * E(i, j, max_t);
            }
        }
    }
}

template <typename T, int LA, int LB>
struct ECoeffs3D {
    ECoeffs1D<T, LA, LB> x;
    ECoeffs1D<T, LA, LB> y;
    ECoeffs1D<T, LA, LB> z;

    OCC_GPU_ENABLED OCC_GPU_INLINE
    T operator()(int ax, int ay, int az,
                 int bx, int by, int bz,
                 int px, int py, int pz) const {
        return x(ax, bx, px) * y(ay, by, py) * z(az, bz, pz);
    }
};

template <typename T, int LA, int LB>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_e_coeffs_3d(T a, T b,
                         T Ax, T Ay, T Az,
                         T Bx, T By, T Bz,
                         ECoeffs3D<T, LA, LB>& E) {
    compute_e_coeffs_1d<T, LA, LB>(a, b, Bx - Ax, E.x);
    compute_e_coeffs_1d<T, LA, LB>(a, b, By - Ay, E.y);
    compute_e_coeffs_1d<T, LA, LB>(a, b, Bz - Az, E.z);
}

template <typename T, int LA, int LB>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_e_coeffs_3d(T a, T b,
                         const T* A,
                         const T* B,
                         ECoeffs3D<T, LA, LB>& E) {
    compute_e_coeffs_3d<T, LA, LB>(a, b, A[0], A[1], A[2], B[0], B[1], B[2], E);
}

constexpr int E1D_MAX_SIZE = (LMAX + 1) * (LMAX + 1) * (2 * LMAX + 1);

template <typename T>
struct ECoeffs1DDynamic {
    alignas(64) T data[E1D_MAX_SIZE];
    int la, lb, p;

    OCC_GPU_ENABLED OCC_GPU_INLINE
    void set_dims(int a, int b) {
        la = a;
        lb = b;
        p = a + b;
    }

    OCC_GPU_ENABLED OCC_GPU_INLINE
    T& operator()(int i, int j, int t) {
        return data[t + (p + 1) * (j + (lb + 1) * i)];
    }

    OCC_GPU_ENABLED OCC_GPU_INLINE
    T operator()(int i, int j, int t) const {
        return data[t + (p + 1) * (j + (lb + 1) * i)];
    }
};

template <typename T>
OCC_GPU_ENABLED
void compute_e_coeffs_1d_dynamic(int la, int lb, T a, T b, T XAB,
                                  ECoeffs1DDynamic<T>& E) {
    E.set_dims(la, lb);
    const int P = la + lb;

    const T p = a + b;
    const T mu = a * b / p;
    const T inv_2p = T(0.5) / p;
    const T XPA = b / p * XAB;
    const T XPB = -a / p * XAB;
    const T E00_0 = std::exp(-mu * XAB * XAB);

    for (int i = 0; i <= la; ++i) {
        for (int j = 0; j <= lb; ++j) {
            for (int t = 0; t <= P; ++t) {
                E(i, j, t) = T(0);
            }
        }
    }

    E(0, 0, 0) = E00_0;

    for (int j = 0; j < lb; ++j) {
        for (int t = 0; t <= j; ++t) {
            T val = XPB * E(0, j, t);
            if (t > 0) val += inv_2p * E(0, j, t - 1);
            if (t < j) val += (t + 1) * E(0, j, t + 1);
            E(0, j + 1, t) += val;
        }
        E(0, j + 1, j + 1) = inv_2p * E(0, j, j);
    }

    for (int i = 0; i < la; ++i) {
        for (int j = 0; j <= lb; ++j) {
            const int max_t = i + j;
            for (int t = 0; t <= max_t; ++t) {
                T val = XPA * E(i, j, t);
                if (t > 0) val += inv_2p * E(i, j, t - 1);
                if (t < max_t) val += (t + 1) * E(i, j, t + 1);
                E(i + 1, j, t) += val;
            }
            if (max_t >= 0) {
                E(i + 1, j, max_t + 1) = inv_2p * E(i, j, max_t);
            }
        }
    }
}

} // namespace occ::ints
