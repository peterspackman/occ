#include <cmath>
#include <occ/qm/cint_interface.h>
#include <occ/qm/integral_engine.h>
#include <occ/xtb/basis.h>
#include <occ/xtb/gamma.h>
#include <occ/xtb/gfn2_parameters.h>
#include <occ/xtb/h0.h>
#include <occ/xtb/multipole_ints.h>
#include <occ/xtb/periodic_integrals.h>
#include <stdexcept>

namespace occ::xtb {

namespace {

// Helper: translate a copy of the central-cell atoms by T (Bohr).
std::vector<core::Atom> translated_atoms(const std::vector<core::Atom> &atoms,
                                         const Vec3 &t) {
  std::vector<core::Atom> out = atoms;
  for (auto &a : out) {
    a.x += t.x();
    a.y += t.y();
    a.z += t.z();
  }
  return out;
}

bool is_central(const LatticeImage &im) {
  return im.hkl(0) == 0 && im.hkl(1) == 0 && im.hkl(2) == 0;
}

// Compute the (cell-0 row, cell-T col) block of an n-component one-electron
// operator on a merged 2-cell AO basis. Iterates only the cross-cell shell
// pairs (p in [0, nsh_central), q in [nsh_central, 2·nsh_central)) — half the
// work of the full symmetric kernel since the cell-0/cell-0 and cell-T/cell-T
// diagonal blocks are not needed.
//
// `n_components` is 1 for overlap, 3 for dipole (cint buffer order x, y, z),
// 9 for quadrupole (cint stores the 3×3 tensor row-major; caller picks the 6
// unique entries).
template <qm::cint::Operator op, gto::Shell::Kind kind>
std::vector<Mat>
cross_block_one_electron(const gto::AOBasis &merged_basis,
                          qm::cint::IntegralEnvironment &env,
                          int nsh_central, int nbf_central,
                          int n_components) {
  std::vector<Mat> result(n_components, Mat::Zero(nbf_central, nbf_central));
  const int nsh_total = static_cast<int>(merged_basis.size());
  const auto &first_bf = merged_basis.first_bf();

  qm::cint::Optimizer opt(env, op, 2);
  std::unique_ptr<double[]> buffer(new double[env.buffer_size_1e(op)]);

  for (int p = 0; p < nsh_central; ++p) {
    const int bf_p = first_bf[p];
    const int n_p = static_cast<int>(merged_basis[p].size());
    for (int q = nsh_central; q < nsh_total; ++q) {
      const int bf_q = first_bf[q] - nbf_central;  // offset into right block
      const int n_q = static_cast<int>(merged_basis[q].size());
      std::array<int, 2> idxs{p, q};
      auto dims = env.two_center_helper<op, kind>(idxs, opt.optimizer_ptr(),
                                                    buffer.get(), nullptr);
      if (dims[0] < 0) continue;
      const size_t block_size =
          static_cast<size_t>(dims[0]) * static_cast<size_t>(dims[1]);
      for (int n = 0; n < n_components; ++n) {
        Eigen::Map<const Mat> tmp(buffer.get() + n * block_size, dims[0],
                                    dims[1]);
        result[n].block(bf_p, bf_q, n_p, n_q) = tmp;
      }
    }
  }
  return result;
}

} // namespace

std::vector<Mat>
periodic_overlap_blocks(const PeriodicSystem &sys,
                        const Gfn2Parameters &params,
                        const std::vector<LatticeImage> &translations) {
  // Central-cell basis (used as the reference for nbf and as the T=0 block).
  auto central_basis = build_aobasis(sys.atoms, params);
  const int nbf = static_cast<int>(central_basis.nbf());

  // T=0: molecular overlap.
  qm::IntegralEngine central_engine(central_basis);
  Mat S0 = central_engine.one_electron_operator(
      qm::IntegralEngine::Op::overlap);

  std::vector<Mat> result;
  result.reserve(translations.size());
  for (const auto &im : translations) {
    if (is_central(im)) {
      result.push_back(S0);
      continue;
    }
    // Build a "two-cell" basis: central atoms + central atoms translated.
    // Cross-block-only: iterate only (cell-0 × cell-T) shell pairs (skip the
    // cell-0/cell-0 and cell-T/cell-T diagonal blocks we don't need).
    auto translated = translated_atoms(sys.atoms, im.t_bohr);
    auto translated_basis = build_aobasis(translated, params);
    gto::AOBasis merged = central_basis;
    merged.merge(translated_basis);
    qm::IntegralEngine engine(merged, qm::IntegralEngine::NoShellPairs{});
    auto &env = engine.env();
    const int nsh_central = static_cast<int>(central_basis.size());
    using Op = qm::cint::Operator;
    using SK = gto::Shell::Kind;
    std::vector<Mat> blocks =
        engine.is_spherical()
            ? cross_block_one_electron<Op::overlap, SK::Spherical>(
                  merged, env, nsh_central, nbf, 1)
            : cross_block_one_electron<Op::overlap, SK::Cartesian>(
                  merged, env, nsh_central, nbf, 1);
    result.push_back(std::move(blocks[0]));
  }
  return result;
}

std::vector<Mat>
periodic_h0_blocks(const PeriodicSystem &sys, const Gfn2Parameters &params,
                   const std::vector<LatticeImage> &translations,
                   const std::vector<Mat> &S_per_T, const Vec &cn) {
  if (S_per_T.size() != translations.size()) {
    throw std::runtime_error("periodic_h0_blocks: S/T size mismatch");
  }
  auto central_basis = build_aobasis(sys.atoms, params);
  auto central_shells = build_shell_table(sys.atoms, params);
  const int nbf = static_cast<int>(central_basis.nbf());

  // T=0: molecular H0.
  Mat H0_central = build_h0(sys.atoms, params, central_shells, central_basis,
                             S_per_T[0], cn);

  std::vector<Mat> result;
  result.reserve(translations.size());
  for (size_t ti = 0; ti < translations.size(); ++ti) {
    const auto &im = translations[ti];
    if (is_central(im)) {
      result.push_back(H0_central);
      continue;
    }
    // Two-cell: central + translated. By lattice translation symmetry the
    // CN at every cell is the same, so concatenate cn with itself.
    auto translated = translated_atoms(sys.atoms, im.t_bohr);
    std::vector<core::Atom> merged_atoms = sys.atoms;
    merged_atoms.insert(merged_atoms.end(), translated.begin(),
                        translated.end());

    auto merged_basis = central_basis;
    auto translated_basis = build_aobasis(translated, params);
    merged_basis.merge(translated_basis);

    auto merged_shells = build_shell_table(merged_atoms, params);
    Vec cn_merged(2 * cn.size());
    cn_merged.head(cn.size()) = cn;
    cn_merged.tail(cn.size()) = cn;

    // Recompute the merged overlap from blocks we already have:
    //   S_merged = [[S0,           S^T],
    //               [S^(-T) = S^T^T, S0]]
    Mat S_merged(2 * nbf, 2 * nbf);
    S_merged.block(0, 0, nbf, nbf) = S_per_T[0];
    S_merged.block(nbf, nbf, nbf, nbf) = S_per_T[0];
    S_merged.block(0, nbf, nbf, nbf) = S_per_T[ti];
    S_merged.block(nbf, 0, nbf, nbf) = S_per_T[ti].transpose();

    Mat H0_merged = build_h0(merged_atoms, params, merged_shells, merged_basis,
                              S_merged, cn_merged);
    result.push_back(H0_merged.block(0, nbf, nbf, nbf));
  }
  return result;
}

CMat bloch_sum(const std::vector<Mat> &M_per_T,
               const std::vector<LatticeImage> &translations, const Vec3 &k) {
  if (M_per_T.empty()) {
    throw std::runtime_error("bloch_sum: empty input");
  }
  CMat result = CMat::Zero(M_per_T[0].rows(), M_per_T[0].cols());
  for (size_t i = 0; i < M_per_T.size(); ++i) {
    const double phase = k.dot(translations[i].t_bohr);
    const std::complex<double> w(std::cos(phase), std::sin(phase));
    result.array() += w * M_per_T[i].cast<std::complex<double>>().array();
  }
  return result;
}

Mat bloch_sum_gamma(const std::vector<Mat> &M_per_T) {
  if (M_per_T.empty()) {
    throw std::runtime_error("bloch_sum_gamma: empty input");
  }
  Mat result = Mat::Zero(M_per_T[0].rows(), M_per_T[0].cols());
  for (const auto &m : M_per_T)
    result += m;
  return result;
}

void apply_traceless_quadrupole_transform(std::array<Mat, 6> &Q) {
  // Q layout {xx, xy, xz, yy, yz, zz} → diagonals at indices 0, 3, 5.
  // Per-AO-pair transform: tr = 0.5·(Q_xx + Q_yy + Q_zz);
  //                        Q_diag → 1.5·Q_diag - tr; Q_offdiag → 1.5·Q_offdiag.
  Mat tr = 0.5 * (Q[0] + Q[3] + Q[5]);
  for (int k = 0; k < 6; ++k) Q[k] *= 1.5;
  Q[0] -= tr;
  Q[3] -= tr;
  Q[5] -= tr;
}

PeriodicMultipoleAO build_periodic_multipole_ao(
    const PeriodicSystem &sys, const Gfn2Parameters &params,
    const std::vector<LatticeImage> &translations) {
  // Strategy: for each T image, compute the merged-basis (cell 0 + cell T)
  // multipole AO integrals at origin 0. Extract the (cell 0, cell T) block.
  // Apply two atom-centerings per T:
  //   Ket: D_T(μ, ν) - R_{A_μ} · S_T(μ, ν)            (origin at row atom A_μ)
  //   Bra: D_T(μ, ν) - (R_{A_ν} + T) · S_T(μ, ν)      (origin at imaged col atom)
  // Then Bloch sum each (just sum over T blocks).
  auto central_basis = build_aobasis(sys.atoms, params);
  const int nbf = static_cast<int>(central_basis.nbf());
  auto bf2at = central_basis.bf_to_atom();

  // Per-row/col atom positions for the centering shifts.
  Vec row_x(nbf), row_y(nbf), row_z(nbf);
  for (int p = 0; p < nbf; ++p) {
    const auto &a = sys.atoms[bf2at[p]];
    row_x(p) = a.x;
    row_y(p) = a.y;
    row_z(p) = a.z;
  }

  // Quadrupole component layout: 0=xx, 1=xy, 2=xz, 3=yy, 4=yz, 5=zz.
  PeriodicMultipoleAO out;
  out.S = Mat::Zero(nbf, nbf);
  out.D = MatTriple::Zero(nbf, nbf);
  out.D_ket = MatTriple::Zero(nbf, nbf);
  out.D_bra = MatTriple::Zero(nbf, nbf);
  for (int k = 0; k < 6; ++k) {
    out.Q[k] = Mat::Zero(nbf, nbf);
    out.Q_ket[k] = Mat::Zero(nbf, nbf);
    out.Q_bra[k] = Mat::Zero(nbf, nbf);
  }

  // Helper: apply row × Mat broadcast as `R_row · M(row, :)` per row.
  auto row_scale = [nbf](const Vec &r, const Mat &M) {
    Mat out_m(nbf, nbf);
    for (int p = 0; p < nbf; ++p) out_m.row(p) = r(p) * M.row(p);
    return out_m;
  };
  // Column-shift: (R_col + T_const) · M, broadcast per column. T_const is
  // baked into the column-side scalar (an extra additive vector per col).
  auto col_scale_with_offset = [nbf](const Vec &r, double offset,
                                       const Mat &M) {
    Mat out_m(nbf, nbf);
    for (int q = 0; q < nbf; ++q) out_m.col(q) = (r(q) + offset) * M.col(q);
    return out_m;
  };

  for (const auto &im : translations) {
    Mat S_T;
    MatTriple D_T;
    std::array<Mat, 6> Q_T;
    if (is_central(im)) {
      qm::IntegralEngine engine(central_basis, qm::IntegralEngine::NoShellPairs{});
      S_T = engine.one_electron_operator(qm::IntegralEngine::Op::overlap);
      D_T = dipole_ao_matrices(engine);
      Q_T = quadrupole_ao_matrices(engine);
    } else {
      auto translated = translated_atoms(sys.atoms, im.t_bohr);
      auto translated_basis = build_aobasis(translated, params);
      gto::AOBasis merged = central_basis;
      merged.merge(translated_basis);
      qm::IntegralEngine engine(merged, qm::IntegralEngine::NoShellPairs{});
      auto &env = engine.env();
      const int nsh_central = static_cast<int>(central_basis.size());
      // Cross-block-only: iterate only (cell-0 row × cell-T col) shell pairs.
      // The cell-0/cell-0 and cell-T/cell-T diagonal blocks would be ~50% of
      // the full kernel cost and we don't use them.
      auto sph = engine.is_spherical();
      using Op = qm::cint::Operator;
      using SK = gto::Shell::Kind;
      std::vector<Mat> S_blocks, D_blocks, Q_blocks;
      if (sph) {
        S_blocks = cross_block_one_electron<Op::overlap, SK::Spherical>(
            merged, env, nsh_central, nbf, 1);
        D_blocks = cross_block_one_electron<Op::dipole, SK::Spherical>(
            merged, env, nsh_central, nbf, 3);
        env.set_common_origin({0.0, 0.0, 0.0});
        Q_blocks = cross_block_one_electron<Op::quadrupole, SK::Spherical>(
            merged, env, nsh_central, nbf, 9);
      } else {
        S_blocks = cross_block_one_electron<Op::overlap, SK::Cartesian>(
            merged, env, nsh_central, nbf, 1);
        D_blocks = cross_block_one_electron<Op::dipole, SK::Cartesian>(
            merged, env, nsh_central, nbf, 3);
        env.set_common_origin({0.0, 0.0, 0.0});
        Q_blocks = cross_block_one_electron<Op::quadrupole, SK::Cartesian>(
            merged, env, nsh_central, nbf, 9);
      }
      S_T = std::move(S_blocks[0]);
      D_T.x = std::move(D_blocks[0]);
      D_T.y = std::move(D_blocks[1]);
      D_T.z = std::move(D_blocks[2]);
      // cint quadrupole layout (3×3 row-major): xx, xy, xz, yx, yy, yz, zx,
      // zy, zz. Pick the 6 unique upper-triangle entries (xx, xy, xz, yy,
      // yz, zz) — same convention as quadrupole_ao_matrices.
      Q_T[0] = std::move(Q_blocks[0]);  // xx
      Q_T[1] = std::move(Q_blocks[1]);  // xy
      Q_T[2] = std::move(Q_blocks[2]);  // xz
      Q_T[3] = std::move(Q_blocks[4]);  // yy
      Q_T[4] = std::move(Q_blocks[5]);  // yz
      Q_T[5] = std::move(Q_blocks[8]);  // zz
    }

    const double Tx = im.t_bohr.x();
    const double Ty = im.t_bohr.y();
    const double Tz = im.t_bohr.z();

    // Overlap (no centering).
    out.S += S_T;

    // Dipole at origin 0 (for H1 step).
    out.D.x += D_T.x;
    out.D.y += D_T.y;
    out.D.z += D_T.z;

    // Dipole — Ket: D - R_row · S, broadcast per row. Bra: D - (R_col + T) · S,
    // broadcast per column.
    out.D_ket.x += D_T.x - row_scale(row_x, S_T);
    out.D_ket.y += D_T.y - row_scale(row_y, S_T);
    out.D_ket.z += D_T.z - row_scale(row_z, S_T);

    out.D_bra.x += D_T.x - col_scale_with_offset(row_x, Tx, S_T);
    out.D_bra.y += D_T.y - col_scale_with_offset(row_y, Ty, S_T);
    out.D_bra.z += D_T.z - col_scale_with_offset(row_z, Tz, S_T);

    // Quadrupole — origin shift formulas. Q index order (xx, xy, xz, yy, yz, zz).
    // Index k → Cartesian pair (k0, k1):
    //   0: (x, x)  1: (x, y)  2: (x, z)  3: (y, y)  4: (y, z)  5: (z, z)
    const int k0[6] = {0, 0, 0, 1, 1, 2};
    const int k1[6] = {0, 1, 2, 1, 2, 2};
    const Mat *D_T_k[3] = {&D_T.x, &D_T.y, &D_T.z};
    const Vec *row_axis[3] = {&row_x, &row_y, &row_z};
    const double T_axis[3] = {Tx, Ty, Tz};

    for (int kk = 0; kk < 6; ++kk) {
      const int kk0 = k0[kk];
      const int kk1 = k1[kk];
      // Quadrupole at origin 0 (for H1 step).
      out.Q[kk] += Q_T[kk];
      // Ket: Q - R_row_k0 · D_l - R_row_k1 · D_k + R_row_k0 · R_row_k1 · S
      Mat ket = Q_T[kk];
      ket -= row_scale(*row_axis[kk0], *D_T_k[kk1]);
      ket -= row_scale(*row_axis[kk1], *D_T_k[kk0]);
      // Per-row R_k0 · R_k1 · S
      Mat rrs(nbf, nbf);
      for (int p = 0; p < nbf; ++p) {
        const double rk = (*row_axis[kk0])(p) * (*row_axis[kk1])(p);
        rrs.row(p) = rk * S_T.row(p);
      }
      ket += rrs;
      out.Q_ket[kk] += ket;

      // Bra: same formula but origin at (R_col + T). Per-column shift.
      Mat bra = Q_T[kk];
      // - (R_col_k0 + T_k0) · D_l (broadcast per column)
      bra -= col_scale_with_offset(*row_axis[kk0], T_axis[kk0], *D_T_k[kk1]);
      bra -= col_scale_with_offset(*row_axis[kk1], T_axis[kk1], *D_T_k[kk0]);
      // + (R_col_k0 + T_k0) · (R_col_k1 + T_k1) · S (broadcast per column)
      Mat rrs_col(nbf, nbf);
      for (int q = 0; q < nbf; ++q) {
        const double rkql = ((*row_axis[kk0])(q) + T_axis[kk0]) *
                             ((*row_axis[kk1])(q) + T_axis[kk1]);
        rrs_col.col(q) = rkql * S_T.col(q);
      }
      bra += rrs_col;
      out.Q_bra[kk] += bra;
    }
  }
  // Match tblite's traceless-Cartesian AO quadrupole convention so the H1
  // contribution `0.5·Q_AO·vq[A]` lines up with `add_vmp_to_h1` in tblite.
  apply_traceless_quadrupole_transform(out.Q_ket);
  apply_traceless_quadrupole_transform(out.Q_bra);
  return out;
}

PeriodicMultipoleAO
build_molecular_multipole_ao(const std::vector<core::Atom> &atoms,
                              const Gfn2Parameters &params) {
  // Molecular AO matrices with the per-row/per-col atomic origin shifts baked
  // in (Bra/Ket convention, matches tblite). D_ket/Q_ket are atom-of-row
  // centered, D_bra/Q_bra atom-of-col centered. Just one-shot — no lattice.
  auto basis = build_aobasis(atoms, params);
  const int nbf = static_cast<int>(basis.nbf());
  auto bf2at = basis.bf_to_atom();

  Vec row_x(nbf), row_y(nbf), row_z(nbf);
  for (int p = 0; p < nbf; ++p) {
    const auto &a = atoms[bf2at[p]];
    row_x(p) = a.x;
    row_y(p) = a.y;
    row_z(p) = a.z;
  }

  qm::IntegralEngine engine(basis);
  Mat S = engine.one_electron_operator(qm::IntegralEngine::Op::overlap);
  MatTriple D0 = dipole_ao_matrices(engine);
  std::array<Mat, 6> Q0 = quadrupole_ao_matrices(engine);

  PeriodicMultipoleAO out;
  out.S = S;
  out.D = D0;
  out.D_ket = MatTriple::Zero(nbf, nbf);
  out.D_bra = MatTriple::Zero(nbf, nbf);

  for (int p = 0; p < nbf; ++p) {
    out.D_ket.x.row(p) = D0.x.row(p) - row_x(p) * S.row(p);
    out.D_ket.y.row(p) = D0.y.row(p) - row_y(p) * S.row(p);
    out.D_ket.z.row(p) = D0.z.row(p) - row_z(p) * S.row(p);
  }
  for (int q = 0; q < nbf; ++q) {
    out.D_bra.x.col(q) = D0.x.col(q) - row_x(q) * S.col(q);
    out.D_bra.y.col(q) = D0.y.col(q) - row_y(q) * S.col(q);
    out.D_bra.z.col(q) = D0.z.col(q) - row_z(q) * S.col(q);
  }

  // Quadrupole: full origin-shift expansion
  //   Q'_kl = Q_kl - R_k·D_l - R_l·D_k + R_k·R_l·S
  const int k0[6] = {0, 0, 0, 1, 1, 2};
  const int k1[6] = {0, 1, 2, 1, 2, 2};
  const Mat *D0_k[3] = {&D0.x, &D0.y, &D0.z};
  const Vec *axis[3] = {&row_x, &row_y, &row_z};
  for (int kk = 0; kk < 6; ++kk) {
    out.Q[kk] = Q0[kk];
    out.Q_ket[kk] = Q0[kk];
    out.Q_bra[kk] = Q0[kk];
    for (int p = 0; p < nbf; ++p) {
      out.Q_ket[kk].row(p) -= (*axis[k0[kk]])(p) * D0_k[k1[kk]]->row(p);
      out.Q_ket[kk].row(p) -= (*axis[k1[kk]])(p) * D0_k[k0[kk]]->row(p);
      out.Q_ket[kk].row(p) +=
          (*axis[k0[kk]])(p) * (*axis[k1[kk]])(p) * S.row(p);
    }
    for (int q = 0; q < nbf; ++q) {
      out.Q_bra[kk].col(q) -= (*axis[k0[kk]])(q) * D0_k[k1[kk]]->col(q);
      out.Q_bra[kk].col(q) -= (*axis[k1[kk]])(q) * D0_k[k0[kk]]->col(q);
      out.Q_bra[kk].col(q) +=
          (*axis[k0[kk]])(q) * (*axis[k1[kk]])(q) * S.col(q);
    }
  }
  apply_traceless_quadrupole_transform(out.Q_ket);
  apply_traceless_quadrupole_transform(out.Q_bra);
  return out;
}

CGenSolveResult solve_generalized_hermitian(const CMat &H, const CMat &S,
                                              double s_eps) {
  // S = U · diag(s) · U^H; build X = U · diag(1/sqrt(s)).
  Eigen::SelfAdjointEigenSolver<CMat> es_S(S);
  if (es_S.info() != Eigen::Success) {
    throw std::runtime_error(
        "solve_generalized_hermitian: S diagonalization failed");
  }
  Vec s_evals = es_S.eigenvalues();
  CMat U = es_S.eigenvectors();
  const int n = static_cast<int>(s_evals.size());
  if (s_evals.minCoeff() <= s_eps) {
    throw std::runtime_error(
        "solve_generalized_hermitian: S near-singular (min eigenvalue = " +
        std::to_string(s_evals.minCoeff()) + ")");
  }
  CMat X(n, n);
  for (int j = 0; j < n; ++j) X.col(j) = U.col(j) / std::sqrt(s_evals(j));
  // Orthogonal Hermitian eigenproblem on H_orth = X^H · H · X.
  CMat H_orth = X.adjoint() * H * X;
  // Symmetrize residual numerical noise.
  H_orth = 0.5 * (H_orth + H_orth.adjoint().eval());
  Eigen::SelfAdjointEigenSolver<CMat> es_H(H_orth);
  if (es_H.info() != Eigen::Success) {
    throw std::runtime_error(
        "solve_generalized_hermitian: H_orth diagonalization failed");
  }
  CGenSolveResult r;
  r.eigenvalues = es_H.eigenvalues();
  r.eigenvectors = X * es_H.eigenvectors();
  return r;
}

} // namespace occ::xtb
