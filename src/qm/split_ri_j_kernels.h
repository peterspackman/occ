#pragma once
// Split RI-J inner kernels: Hermite Coulomb integrals R_{t+t'}(alpha, P - C)
// between one AO primitive pair and a chunk of NP auxiliary primitives of one
// angular-momentum class, contracted on the fly.
//
//   forward:  Y_k[t'] += sum_t S_k R_{t+t'} X[t]        (Y in Hermite aux space)
//   backward: U[t]    += sum_k sum_t' S_k R_{t+t'} T_k[t']
//
// with S_k = pref_ab * c_k * pi^1.5 / (gamma_k sqrt(p + gamma_k)). The aux
// primitives are laid out structure-of-arrays per class, so the Boys function,
// the Hermite recursion and both contractions vectorise across them. The
// Hermite degrees LA (pair side) and LC (aux side) are template parameters:
// the gradient uses the same kernels one degree higher on either side.
//
// On x86-64 each kernel is compiled twice, for the build's baseline and for
// AVX2/FMA (function-level target attribute), and the CPU picks one at run
// time; elsewhere once, for the build target.
#include <algorithm>
#include <array>
#include <cmath>
#include <occ/ints/boys_simd.h>
#include <occ/ints/rints.h>

namespace occ::qm::srij {

using ints::hermite_index;
using ints::nhermsum;

/// Auxiliary primitives per chunk.
inline constexpr int NP = 8;
/// Largest Hermite degrees with a kernel (AO l <= 4 and aux l <= 6, plus one
/// for the gradient), and the largest total the Boys table supports.
inline constexpr int MAX_LA = 9;
inline constexpr int MAX_LC = 7;
inline constexpr int MAX_L = 15;

/// One AO primitive pair as the kernels see it.
struct PrimPair {
  double p, Px, Py, Pz, pref;
};

/// One angular-momentum class of auxiliary primitives, structure-of-arrays
/// and padded to a multiple of NP (padding has cpref = 0). Hermite arrays of
/// the class are stored [t'][k] with row stride npad.
struct AuxView {
  const double *gamma, *cpref, *cx, *cy, *cz;
  int npad;
};

using FwdFn = void (*)(const double *boys, const PrimPair *pp, int nprim,
                       const double *X, int ldx, const AuxView &a, int kend,
                       double *Y);
using BwdFn = void (*)(const double *boys, const PrimPair *pp, int nprim,
                       const AuxView &a, int kend, const double *T, double *U,
                       int ldu);

/// Kernels for Hermite degrees (la, lc), nullptr if not compiled.
FwdFn forward_kernel(int la, int lc);
BwdFn backward_kernel(int la, int lc);

} // namespace occ::qm::srij
