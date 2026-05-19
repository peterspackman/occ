#include <Eigen/Eigenvalues>
#include <cmath>
#include <occ/core/parallel.h>
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

// Fused cross-block evaluator for (overlap, dipole, quadrupole) — emits all
// 1+3+9 = 13 component blocks in a single shell-pair iteration. The cint
// optimizer + buffer for each operator is allocated once and reused; the
// per-pair loop body avoids redundant first_bf/dims lookups across the three
// operator calls. Quadrupole common origin is left to the caller (we do not
// reset it here since cint's set_common_origin mutates the env).
//
// Output layout:
//   results[0]              = S
//   results[1..3]           = D_x, D_y, D_z
//   results[4..12]          = Q_xx, Q_xy, Q_xz, Q_yx, Q_yy, Q_yz, Q_zx,
//                             Q_zy, Q_zz   (cint row-major 3×3 ordering)
template <gto::Shell::Kind kind>
std::array<Mat, 13>
cross_block_sdq(const gto::AOBasis &merged_basis,
                qm::cint::IntegralEnvironment &env, int nsh_central,
                int nbf_central) {
  using Op = qm::cint::Operator;
  std::array<Mat, 13> result;
  for (auto &m : result) m = Mat::Zero(nbf_central, nbf_central);
  const int nsh_total = static_cast<int>(merged_basis.size());
  const auto &first_bf = merged_basis.first_bf();

  qm::cint::Optimizer opt_s(env, Op::overlap, 2);
  qm::cint::Optimizer opt_d(env, Op::dipole, 2);
  qm::cint::Optimizer opt_q(env, Op::quadrupole, 2);
  std::unique_ptr<double[]> buf_s(new double[env.buffer_size_1e(Op::overlap)]);
  std::unique_ptr<double[]> buf_d(new double[env.buffer_size_1e(Op::dipole)]);
  std::unique_ptr<double[]> buf_q(
      new double[env.buffer_size_1e(Op::quadrupole)]);

  for (int p = 0; p < nsh_central; ++p) {
    const int bf_p = first_bf[p];
    const int n_p = static_cast<int>(merged_basis[p].size());
    for (int q = nsh_central; q < nsh_total; ++q) {
      const int bf_q = first_bf[q] - nbf_central;
      const int n_q = static_cast<int>(merged_basis[q].size());
      std::array<int, 2> idxs{p, q};

      auto dims_s = env.two_center_helper<Op::overlap, kind>(
          idxs, opt_s.optimizer_ptr(), buf_s.get(), nullptr);
      if (dims_s[0] < 0) continue;
      Eigen::Map<const Mat> S_pq(buf_s.get(), dims_s[0], dims_s[1]);
      result[0].block(bf_p, bf_q, n_p, n_q) = S_pq;

      auto dims_d = env.two_center_helper<Op::dipole, kind>(
          idxs, opt_d.optimizer_ptr(), buf_d.get(), nullptr);
      const size_t db = static_cast<size_t>(dims_d[0]) *
                        static_cast<size_t>(dims_d[1]);
      for (int n = 0; n < 3; ++n) {
        Eigen::Map<const Mat> D_pq(buf_d.get() + n * db, dims_d[0], dims_d[1]);
        result[1 + n].block(bf_p, bf_q, n_p, n_q) = D_pq;
      }

      auto dims_q = env.two_center_helper<Op::quadrupole, kind>(
          idxs, opt_q.optimizer_ptr(), buf_q.get(), nullptr);
      const size_t qb = static_cast<size_t>(dims_q[0]) *
                        static_cast<size_t>(dims_q[1]);
      for (int n = 0; n < 9; ++n) {
        Eigen::Map<const Mat> Q_pq(buf_q.get() + n * qb, dims_q[0], dims_q[1]);
        result[4 + n].block(bf_p, bf_q, n_p, n_q) = Q_pq;
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

  std::vector<Mat> result(translations.size());
  // Per-T iterations are independent — each builds its own merged basis +
  // IntegralEngine. Cint env is per-engine, so no shared mutable state.
  occ::parallel::parallel_for(size_t{0}, translations.size(), [&](size_t i) {
    const auto &im = translations[i];
    if (is_central(im)) {
      result[i] = S0;
      return;
    }
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
    result[i] = std::move(blocks[0]);
  });
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

  std::vector<Mat> result(translations.size());
  occ::parallel::parallel_for(size_t{0}, translations.size(), [&](size_t ti) {
    const auto &im = translations[ti];
    if (is_central(im)) {
      result[ti] = H0_central;
      return;
    }
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

    Mat S_merged(2 * nbf, 2 * nbf);
    S_merged.block(0, 0, nbf, nbf) = S_per_T[0];
    S_merged.block(nbf, nbf, nbf, nbf) = S_per_T[0];
    S_merged.block(0, nbf, nbf, nbf) = S_per_T[ti];
    S_merged.block(nbf, 0, nbf, nbf) = S_per_T[ti].transpose();

    Mat H0_merged = build_h0(merged_atoms, params, merged_shells, merged_basis,
                              S_merged, cn_merged);
    result[ti] = H0_merged.block(0, nbf, nbf, nbf);
  });
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

PeriodicMultipoleAOBlocks build_periodic_multipole_ao_blocks(
    const PeriodicSystem &sys, const Gfn2Parameters &params,
    const std::vector<LatticeImage> &translations) {
  // For each T image, compute the merged-basis (cell 0 + cell T) multipole AO
  // integrals at origin 0, extract the (cell 0, cell T) block, then apply two
  // atom-centerings per T:
  //   Ket: D_T(μ, ν) - R_{A_μ} · S_T(μ, ν)            (origin at row atom A_μ)
  //   Bra: D_T(μ, ν) - (R_{A_ν} + T) · S_T(μ, ν)      (origin at imaged col atom)
  // Each per-T block is stored separately so callers can Bloch-sum at any k.
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

  PeriodicMultipoleAOBlocks out;
  out.D_ket.resize(translations.size());
  out.D_bra.resize(translations.size());
  out.Q_ket.resize(translations.size());
  out.Q_bra.resize(translations.size());

  // Per-T iterations are independent — thread over translations. Each
  // iteration owns its merged basis + IntegralEngine + cint env.
  occ::parallel::parallel_for(size_t{0}, translations.size(), [&](size_t ti) {
    const auto &im = translations[ti];
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
      env.set_common_origin({0.0, 0.0, 0.0});
      using SK = gto::Shell::Kind;
      auto sdq = engine.is_spherical()
                     ? cross_block_sdq<SK::Spherical>(merged, env, nsh_central,
                                                        nbf)
                     : cross_block_sdq<SK::Cartesian>(merged, env, nsh_central,
                                                        nbf);
      S_T = std::move(sdq[0]);
      D_T.x = std::move(sdq[1]);
      D_T.y = std::move(sdq[2]);
      D_T.z = std::move(sdq[3]);
      // cint quadrupole layout (3×3 row-major within sdq[4..12]): xx, xy, xz,
      // yx, yy, yz, zx, zy, zz. Pick the 6 unique upper-triangle entries.
      Q_T[0] = std::move(sdq[4]);   // xx
      Q_T[1] = std::move(sdq[5]);   // xy
      Q_T[2] = std::move(sdq[6]);   // xz
      Q_T[3] = std::move(sdq[8]);   // yy
      Q_T[4] = std::move(sdq[9]);   // yz
      Q_T[5] = std::move(sdq[12]);  // zz
    }

    const double Tx = im.t_bohr.x();
    const double Ty = im.t_bohr.y();
    const double Tz = im.t_bohr.z();

    // Dipole Ket: out_x(p, q) = D_T.x(p, q) - R_row(p) · S_T(p, q). Same for
    // y, z. We allocate the destination (1 nbf×nbf alloc) and fill it in-place
    // from D_T to avoid the lambda-allocated temporaries the previous version
    // produced (each of which was an extra heap alloc per call).
    MatTriple ket_T{D_T.x, D_T.y, D_T.z};   // 3 allocs (copy from D_T)
    MatTriple bra_T{D_T.x, D_T.y, D_T.z};   // 3 allocs
    for (int p = 0; p < nbf; ++p) {
      ket_T.x.row(p) -= row_x(p) * S_T.row(p);
      ket_T.y.row(p) -= row_y(p) * S_T.row(p);
      ket_T.z.row(p) -= row_z(p) * S_T.row(p);
    }
    for (int q = 0; q < nbf; ++q) {
      bra_T.x.col(q) -= (row_x(q) + Tx) * S_T.col(q);
      bra_T.y.col(q) -= (row_y(q) + Ty) * S_T.col(q);
      bra_T.z.col(q) -= (row_z(q) + Tz) * S_T.col(q);
    }

    // Quadrupole layout (xx, xy, xz, yy, yz, zz) → Cartesian pair (k0, k1).
    const int k0[6] = {0, 0, 0, 1, 1, 2};
    const int k1[6] = {0, 1, 2, 1, 2, 2};
    const Mat *D_T_k[3] = {&D_T.x, &D_T.y, &D_T.z};
    const Vec *row_axis[3] = {&row_x, &row_y, &row_z};
    const double T_axis[3] = {Tx, Ty, Tz};

    std::array<Mat, 6> Q_ket_T, Q_bra_T;
    for (int kk = 0; kk < 6; ++kk) {
      const int kk0 = k0[kk];
      const int kk1 = k1[kk];
      // Ket: Q - R_row_k0 · D_l - R_row_k1 · D_k + R_row_k0 · R_row_k1 · S.
      // In-place row updates on a copy of Q_T[kk] — one alloc per (kk, ti),
      // none for intermediates.
      Q_ket_T[kk] = Q_T[kk];
      Mat &ket = Q_ket_T[kk];
      const Vec &ax0 = *row_axis[kk0];
      const Vec &ax1 = *row_axis[kk1];
      const Mat &Dl = *D_T_k[kk1];
      const Mat &Dk = *D_T_k[kk0];
      for (int p = 0; p < nbf; ++p) {
        const double r0 = ax0(p);
        const double r1 = ax1(p);
        ket.row(p) -= r0 * Dl.row(p);
        ket.row(p) -= r1 * Dk.row(p);
        ket.row(p) += (r0 * r1) * S_T.row(p);
      }

      // Bra: same with origin at (R_col + T). Per-column shift.
      Q_bra_T[kk] = Q_T[kk];
      Mat &bra = Q_bra_T[kk];
      for (int q = 0; q < nbf; ++q) {
        const double r0 = ax0(q) + T_axis[kk0];
        const double r1 = ax1(q) + T_axis[kk1];
        bra.col(q) -= r0 * Dl.col(q);
        bra.col(q) -= r1 * Dk.col(q);
        bra.col(q) += (r0 * r1) * S_T.col(q);
      }
    }
    // Apply traceless-Cartesian AO quadrupole convention per-T (linear, so
    // commutes with Bloch summation at any k).
    apply_traceless_quadrupole_transform(Q_ket_T);
    apply_traceless_quadrupole_transform(Q_bra_T);

    out.D_ket[ti] = std::move(ket_T);
    out.D_bra[ti] = std::move(bra_T);
    out.Q_ket[ti] = std::move(Q_ket_T);
    out.Q_bra[ti] = std::move(Q_bra_T);
  });
  return out;
}

PeriodicMultipoleAO build_periodic_multipole_ao(
    const PeriodicSystem &sys, const Gfn2Parameters &params,
    const std::vector<LatticeImage> &translations) {
  auto blocks = build_periodic_multipole_ao_blocks(sys, params, translations);
  if (blocks.D_ket.empty()) {
    throw std::runtime_error("build_periodic_multipole_ao: empty translations");
  }
  const Eigen::Index nbf = blocks.D_ket.front().x.rows();
  PeriodicMultipoleAO out;
  out.D_ket = MatTriple::Zero(nbf, nbf);
  out.D_bra = MatTriple::Zero(nbf, nbf);
  for (int k = 0; k < 6; ++k) {
    out.Q_ket[k] = Mat::Zero(nbf, nbf);
    out.Q_bra[k] = Mat::Zero(nbf, nbf);
  }
  for (size_t i = 0; i < blocks.D_ket.size(); ++i) {
    out.D_ket.x += blocks.D_ket[i].x;
    out.D_ket.y += blocks.D_ket[i].y;
    out.D_ket.z += blocks.D_ket[i].z;
    out.D_bra.x += blocks.D_bra[i].x;
    out.D_bra.y += blocks.D_bra[i].y;
    out.D_bra.z += blocks.D_bra[i].z;
    for (int k = 0; k < 6; ++k) {
      out.Q_ket[k] += blocks.Q_ket[i][k];
      out.Q_bra[k] += blocks.Q_bra[i][k];
    }
  }
  return out;
}

CMatTriple bloch_sum_triple(const std::vector<MatTriple> &per_T,
                             const std::vector<LatticeImage> &translations,
                             const Vec3 &k) {
  if (per_T.empty()) {
    throw std::runtime_error("bloch_sum_triple: empty input");
  }
  if (per_T.size() != translations.size()) {
    throw std::runtime_error("bloch_sum_triple: size mismatch with translations");
  }
  const Eigen::Index nbf = per_T.front().x.rows();
  CMatTriple result{CMat::Zero(nbf, nbf), CMat::Zero(nbf, nbf),
                    CMat::Zero(nbf, nbf)};
  for (size_t i = 0; i < per_T.size(); ++i) {
    const double phase = k.dot(translations[i].t_bohr);
    const std::complex<double> w(std::cos(phase), std::sin(phase));
    result.x.array() += w * per_T[i].x.cast<std::complex<double>>().array();
    result.y.array() += w * per_T[i].y.cast<std::complex<double>>().array();
    result.z.array() += w * per_T[i].z.cast<std::complex<double>>().array();
  }
  return result;
}

std::array<CMat, 6>
bloch_sum_array6(const std::vector<std::array<Mat, 6>> &per_T,
                 const std::vector<LatticeImage> &translations, const Vec3 &k) {
  if (per_T.empty()) {
    throw std::runtime_error("bloch_sum_array6: empty input");
  }
  if (per_T.size() != translations.size()) {
    throw std::runtime_error("bloch_sum_array6: size mismatch with translations");
  }
  const Eigen::Index nbf = per_T.front()[0].rows();
  std::array<CMat, 6> result;
  for (int c = 0; c < 6; ++c) result[c] = CMat::Zero(nbf, nbf);
  for (size_t i = 0; i < per_T.size(); ++i) {
    const double phase = k.dot(translations[i].t_bohr);
    const std::complex<double> w(std::cos(phase), std::sin(phase));
    for (int c = 0; c < 6; ++c) {
      result[c].array() += w * per_T[i][c].cast<std::complex<double>>().array();
    }
  }
  return result;
}

PeriodicMultipoleAO
center_multipole_ao(const std::vector<core::Atom> &atoms,
                    const std::vector<int> &bf_to_atom,
                    const Mat &S,
                    const MatTriple &D0,
                    const std::array<Mat, 6> &Q0) {
  const int nbf = static_cast<int>(S.rows());
  Vec row_x(nbf), row_y(nbf), row_z(nbf);
  for (int p = 0; p < nbf; ++p) {
    const auto &a = atoms[bf_to_atom[p]];
    row_x(p) = a.x;
    row_y(p) = a.y;
    row_z(p) = a.z;
  }

  PeriodicMultipoleAO out;
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

PeriodicMultipoleAO
build_molecular_multipole_ao(const std::vector<core::Atom> &atoms,
                              const Gfn2Parameters &params) {
  // Stand-alone wrapper: build basis + engine + raw AO multipole integrals,
  // then center them. Callers that already have an IntegralEngine (the SCC
  // and the analytical gradient) should compute D0/Q0/S themselves and call
  // `center_multipole_ao` directly to avoid the redundant rebuild here.
  auto basis = build_aobasis(atoms, params);
  qm::IntegralEngine engine(basis);
  Mat S = engine.one_electron_operator(qm::IntegralEngine::Op::overlap);
  MatTriple D0 = dipole_ao_matrices(engine);
  std::array<Mat, 6> Q0 = quadrupole_ao_matrices(engine);
  return center_multipole_ao(atoms, basis.bf_to_atom(), S, D0, Q0);
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
