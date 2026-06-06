#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/gto/shell.h>
#include <occ/numint/grid_settings.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/mo.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace occ::qm::cc {

using occ::gto::AOBasis;
using occ::qm::IntegralEngine;
using occ::qm::MolecularOrbitals;

/// Interpolation-point selection for ISDF.
enum class IsdfMethod {
  QR,      ///< pivoted QR on the pair-density collocation (accurate, O(N^4))
  Cholesky ///< pivoted Cholesky of the grid Gram with lazy columns (cheaper)
};

/// Regularisation of the (ill-conditioned) LS-THC metric inverse.
enum class ThcRegType {
  Eig,     ///< drop eigenvalues below reg*lambda_max (truncated pinv)
  Tikhonov ///< shift (S + reg*lambda_max I)^-1
};

/// Which functions' products the interpolation points must span.
enum class ThcSelectBasis {
  AO, ///< AO pair products: geometry-only / SCF-free (default)
  MO  ///< MO pair products (needs converged MOs for selection)
};

struct ThcOptions {
  // Cholesky is the default: it stops after `target` pivots (O(npts*nbf*P)),
  // whereas pivoted QR factors the full pair-collocation (O(nbf^4*npts)) and is
  // far slower for any non-tiny basis, at matching accuracy.
  IsdfMethod method{IsdfMethod::Cholesky};
  ThcSelectBasis select_basis{ThcSelectBasis::AO};
  int n_isdf{-1}; ///< absolute point count; <=0 -> use c_isdf / tol
  double c_isdf{
      6.0}; ///< count = round(c_isdf * n_select); ~6 = sub-mHa sweet spot
  double tol{1e-4}; ///< ISDF rank cut (fallback when neither count is set)
  // Candidate-grid density. The grid is only a pool of interpolation-point
  // candidates -- it needs spatial coverage, not DFT-integration accuracy -- so
  // it is kept deliberately coarse (selection cost scales linearly with npts).
  int grid_max_angular{110};
  double grid_radial_precision{1e-7};
  double reg{1e-10};
  ThcRegType reg_type{ThcRegType::Eig};
  size_t memory_budget{size_t(1) << 30}; ///< bytes, for the DF B-tensor build
};

/// THC factors for the spatial MOs: (pq|rs) ~ sum_PQ X(p,P) X(q,P) V(P,Q)
/// X(r,Q) X(s,Q).
struct ThcFactors {
  Mat X;         ///< (nmo x n_isdf): MO value at each interp. point
  Mat V;         ///< (n_isdf x n_isdf): fitted core
  int n_isdf{0}; ///< number of interpolation points selected
  double metric_condition{0.0}; ///< condition number of the LS-THC metric S
  int metric_n_kept{0}; ///< eigenvalues retained in the regularised inverse
};

/// Build THC factors (X, V) for the spatial MOs of `mo`.
///   basis     : AO basis of the molecule
///   aux_basis : density-fitting auxiliary basis (the LS-THC reference)
ThcFactors build_thc(const AOBasis &basis, const AOBasis &aux_basis,
                     const MolecularOrbitals &mo, const ThcOptions &opts = {});

/// As build_thc, but using a precomputed metric-folded DF reference tensor
/// `B` (nmo^2 x naux, row p*nmo+q). Lets a caller that already holds a
/// DFIntegrals reuse it instead of rebuilding the 3-center store.
ThcFactors build_thc_from_B(const AOBasis &basis, const MolecularOrbitals &mo,
                            const ThcOptions &opts, const Mat &B);

/// Cross-spin THC factors for unrestricted CCSD. One ISDF point set is selected
/// (AO-based, orbital-independent), then evaluated against both spins' MOs to
/// give Xa, Xb. Three cores are fitted against the shared DF reference: the
/// same-spin Vaa, Vbb and the cross core Vab (so (pq|RS) with p,q alpha and R,S
/// beta ~ sum_PQ Xa(p,P)Xa(q,P) Vab(P,Q) Xb(R,Q)Xb(S,Q)). `Ba`/`Bb` are the
/// per-spin metric-folded DF tensors build_b_tilde(Ca,Ca) /
/// build_b_tilde(Cb,Cb).
struct UThcFactors {
  Mat Xa, Xb;        ///< (nmoa x n_isdf), (nmob x n_isdf)
  Mat Vaa, Vbb, Vab; ///< (n_isdf x n_isdf) fitted cores
  int n_isdf{0};
};
UThcFactors build_uthc(const AOBasis &basis, const Mat &Ca, const Mat &Cb,
                       const Mat &Ba, const Mat &Bb, const ThcOptions &opts);

// --- lower-level pieces (exposed for reuse / testing) ----------------------

/// Indices of interpolation grid points spanning products of the columns of
/// `coll` (npts x nfunc). `target<=0` falls back to the `tol` rank cut.
std::vector<int> select_isdf_points(const Mat &coll, IsdfMethod method,
                                    int target, double tol);

/// Point-selection half of build_thc: select ISDF interpolation points (AO- or
/// MO-based per opts.select_basis, geometry-only) and return the MO collocation
/// X (nmo x n_isdf) = MO value at each selected point. The caller then fits the
/// core: fit_core over all pairs (CCSD), or fit_core_ov over the occ-virt block
/// (MP2).
Mat thc_select_collocation(const AOBasis &basis, const MolecularOrbitals &mo,
                           const ThcOptions &opts);

/// Regularised least-squares THC core fit. `B` is the metric-folded DF tensor
/// (nmo^2 x naux) with row p*nmo+q, so (pq|rs) = sum_A B(p*nmo+q,A)
/// B(r*nmo+s,A).
Mat fit_core(const Mat &X, const Mat &B, double reg, ThcRegType reg_type,
             double *condition_out = nullptr, int *n_kept_out = nullptr);

/// LS-THC core fit restricted to the occupied-virtual block -- the only
/// integrals MP2 needs. Fits V so (ia|jb) ~ sum_PQ Xo(i,P)Xv(a,P) V(P,Q)
/// Xo(j,Q)Xv(b,Q). `Xo` (o x P) / `Xv` (v x P) are the occ/virt rows of the THC
/// collocation; `B_ov` is the metric-folded DF tensor for the ov pairs
/// (o*v x naux, row i*v+a) = build_b_tilde(C_occ, C_virt). Metric is the ov
/// Gram S = (Xo^T Xo) o (Xv^T Xv). Much cheaper (o*v << nmo^2) and more
/// accurate for MP2 than the all-pairs fit_core.
Mat fit_core_ov(const Mat &Xo, const Mat &Xv, const Mat &B_ov, double reg,
                ThcRegType reg_type, double *condition_out = nullptr,
                int *n_kept_out = nullptr);

/// THC-reconstructed chemist integrals (pq|rs) as a dense nmo^4 tensor
/// (testing).
Eigen::Tensor<double, 4> reconstruct_eri(const Mat &X, const Mat &V);

/// Semidirect AO->MO transform (chemist (pq|rs)) for arbitrary MO coefficient
/// blocks. Blocks the left index by `budget` and uses
/// IntegralEngine::ao_direct_half_transform, so the nao^4 AO tensor is never
/// materialised. Returns G(L, q, r, s) with L over the columns of `C_L`.
Eigen::Tensor<double, 4> mo_eri_general(const IntegralEngine &engine,
                                        const Mat &C_L, const Mat &C_q,
                                        const Mat &C_r, const Mat &C_s,
                                        size_t budget = (size_t(1) << 28));

/// Relative Frobenius error of the THC-reconstructed MO integrals vs exact.
double reconstruction_error(const AOBasis &basis, const MolecularOrbitals &mo,
                            const Mat &X, const Mat &V);

} // namespace occ::qm::cc
