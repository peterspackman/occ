#pragma once
// Shared helpers for the analytic-derivative tests: central differences along
// one direction, displaced SCF solutions, and the test molecules they use.
//
// Most checks are directional: an analytic gradient g is compared through
// g . v with (E(x + hv) - E(x - hv)) / 2h, a Hessian H through H v with
// (g(x + hv) - g(x - hv)) / 2h, for a fixed generic unit vector v over all
// 3N coordinates. Two evaluations instead of 6N, and since every element of
// v is non-zero and unrelated to the geometry, no element can hide by
// symmetry. With |v| = 1 each component of H v carries no more than the
// per-element finite-difference error, so the elementwise tolerances carry
// over unchanged.
#include <algorithm>
#include <cmath>
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/gto/gto.h>
#include <occ/numint/grid_settings.h>
#include <occ/qm/gradients.h>
#include <occ/qm/expectation.h>
#include <occ/qm/scf.h>
#include <occ/qm/wavefunction.h>
#include <type_traits>
#include <vector>

namespace occ::test {

using Atoms = std::vector<occ::core::Atom>;

// Distorted water (bohr): no symmetry to hide a sign error. Also used as the
// non-degenerate doublet H2O+ for open-shell tests.
inline Atoms distorted_water() {
  return {{8, 0.03, -0.02, 0.21}, {1, 0.10, 1.45, -0.86}, {1, -0.05, -1.38, -0.92}};
}

// Distorted ammonia (bohr), no symmetry.
inline Atoms distorted_ammonia() {
  return {{7, 0.02, -0.01, 0.13},
          {1, 1.78, 0.05, -0.55},
          {1, -0.91, 1.60, -0.48},
          {1, -0.87, -1.55, -0.60}};
}

// A coarse DFT grid: cheap, and coarse enough that the grid response matters.
inline occ::numint::GridSettings coarse_grid() {
  occ::numint::GridSettings grid;
  grid.max_angular_points = 50;
  grid.radial_precision = 1e-4;
  return grid;
}

inline Vec flat(const Mat3N &g) { return Eigen::Map<const Vec>(g.data(), g.size()); }

// A fixed, generic unit vector of length n: deterministic, with no zero or
// repeated elements, and different for each seed.
inline Vec generic_direction(Eigen::Index n, int seed = 1) {
  Vec v(n);
  for (Eigen::Index i = 0; i < n; i++)
    v(i) = std::sin(1.3 + 2.39996 * (i + 1) * seed) +
           0.5 * std::cos(0.71 * (i + 1) + 1.7 * seed);
  return v.normalized();
}

// A generic unit direction over the 3N coordinates of `atoms`.
inline Vec generic_direction(const Atoms &atoms, int seed = 1) {
  return generic_direction(3 * Eigen::Index(atoms.size()), seed);
}

// `atoms` with every coordinate k moved by d * v(k).
inline Atoms displaced_along(Atoms atoms, const Vec &v, double d) {
  for (size_t a = 0; a < atoms.size(); a++) {
    atoms[a].x += d * v(3 * a);
    atoms[a].y += d * v(3 * a + 1);
    atoms[a].z += d * v(3 * a + 2);
  }
  return atoms;
}

// Central difference of f(atoms) along the direction v with step h; f may
// return a scalar or any Eigen expression.
template <typename F>
auto directional_difference(const Atoms &atoms, const Vec &v, double h, F &&f) {
  using R = std::decay_t<decltype(f(atoms))>;
  if constexpr (std::is_arithmetic_v<R>)
    return (f(displaced_along(atoms, v, h)) - f(displaced_along(atoms, v, -h))) /
           (2 * h);
  else
    return typename R::PlainObject(
        (f(displaced_along(atoms, v, h)) - f(displaced_along(atoms, v, -h))) /
        (2 * h));
}

// Largest violation of translational invariance: every row of a Hessian sums
// to zero over atoms, per Cartesian direction.
inline double translational_sum_error(const Mat &H) {
  const int natom = H.cols() / 3;
  double err = 0.0;
  for (int i = 0; i < H.rows(); i++)
    for (int x = 0; x < 3; x++) {
      double s = 0.0;
      for (int a = 0; a < natom; a++)
        s += H(i, 3 * a + x);
      err = std::max(err, std::abs(s));
    }
  return err;
}

inline double max_abs(const Mat &m) { return m.cwiseAbs().maxCoeff(); }

inline double asymmetry(const Mat &H) { return max_abs(H - H.transpose()); }

// `proc` re-built on its basis moved by d along v (same basis name and
// purity) and solved again, starting from `wfn`. Returns the SCF energy and,
// if `grad` is given, the analytic gradient there.
template <typename Proc>
double displaced_scf(Proc &proc, const occ::qm::Wavefunction &wfn,
                     const Vec &v, double d,
                     const occ::qm::SCFConvergenceSettings &convergence,
                     Mat3N *grad = nullptr) {
  const auto &basis = proc.aobasis();
  auto b = occ::gto::AOBasis::load(displaced_along(proc.atoms(), v, d),
                                   basis.name());
  b.set_pure(basis.is_pure());
  Proc p = proc.with_new_basis(b);
  occ::qm::SCF<Proc> scf(p, wfn.mo.kind);
  scf.set_charge_multiplicity(wfn.charge(), wfn.multiplicity());
  scf.set_initial_guess_from_wfn(wfn);
  scf.convergence_settings = convergence;
  const double e = scf.compute_scf_energy();
  if (grad)
    *grad = occ::qm::GradientEvaluator<Proc>(p)(scf.molecular_orbitals());
  return e;
}

// (g(x + hv) - g(x - hv)) / 2h for the analytic SCF gradient g, flattened:
// the finite-difference image of the Hessian along v.
template <typename Proc>
Vec gradient_difference_along(Proc &proc, const occ::qm::Wavefunction &wfn,
                              const Vec &v, double h,
                              const occ::qm::SCFConvergenceSettings &convergence) {
  Mat3N gp, gm;
  displaced_scf(proc, wfn, v, h, convergence, &gp);
  displaced_scf(proc, wfn, v, -h, convergence, &gm);
  return flat(gp - gm) / (2 * h);
}

// For a Hessian H that is the symmetric part of a gradient Jacobian J which
// is not itself symmetric (the COSX gradient is not an exact energy
// derivative), H v cannot be compared with J v. Instead, over two generic
// directions v1, v2: max over the pairs of |vi^T H vj - (vi.J vj + vj.J vi)/2|,
// which only involves the symmetric part of J.
template <typename Proc>
double symmetric_projection_error(Proc &proc, const occ::qm::Wavefunction &wfn,
                                  const Mat &H, double h,
                                  const occ::qm::SCFConvergenceSettings &convergence) {
  const Vec v[2] = {generic_direction(proc.atoms(), 1),
                    generic_direction(proc.atoms(), 2)};
  const Vec Jv[2] = {gradient_difference_along(proc, wfn, v[0], h, convergence),
                     gradient_difference_along(proc, wfn, v[1], h, convergence)};
  double err = 0.0;
  for (int i = 0; i < 2; i++)
    for (int j = i; j < 2; j++)
      err = std::max(err, std::abs(v[i].dot(H * v[j]) -
                                   0.5 * (v[i].dot(Jv[j]) + v[j].dot(Jv[i]))));
  return err;
}

inline occ::qm::SCFConvergenceSettings scf_convergence(double energy,
                                                       double commutator) {
  occ::qm::SCFConvergenceSettings c;
  c.energy_threshold = energy;
  c.commutator_threshold = commutator;
  return c;
}

// Electronic dipole <r> = 2 tr(D r) (per-spin D; for unrestricted both spin
// blocks), the field derivative of the energy for a field coupling F . r.
template <typename R>
Vec electronic_dipole(const occ::qm::MolecularOrbitals &mo, const R &r) {
  Vec mu(3);
  for (int x = 0; x < 3; x++)
    mu(x) = 2.0 * occ::qm::expectation(mo.kind, mo.D, Mat(r[x]));
  return mu;
}

} // namespace occ::test
