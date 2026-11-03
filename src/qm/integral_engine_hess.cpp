// Analytic second-derivative building blocks, contracted inside the integral
// loops (no nbf x nbf x 9 intermediates):
//
//  - one_electron_hessian:  sum P h^{xy} - sum Q S^{xy}            (3N x 3N)
//  - jk_hessian:            sum Gamma (mn|ls)^{xy}                 (3N x 3N)
//  - jk_derivative_fock:    skeleton two-electron Fock derivatives (3N mats)
//  - one_electron_derivative_matrices: S^{(x)} and h^{(x)}         (3N mats)
//
// Throughout, a derivative "with respect to centre c" of an integral is the
// derivative with respect to the nucleus carrying shell c, i.e. minus the
// libcint nabla on that function; second derivatives carry (-1)^2 = +1.
// Translational invariance (the integral depends only on differences of its
// centres) supplies every block involving the centre group we choose not to
// evaluate, which is also how operator centres (nuclei in the nuclear
// attraction) are handled.

#include "detail/kernel_traits.h"
#include "detail/libcint_derivatives.h"
#include "detail/quartet_derivatives.h"
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/qm/integral_engine.h>

namespace occ::qm {

using ShellPairList = std::vector<std::vector<size_t>>;
using IntEnv = cint::IntegralEnvironment;
using ShellKind = Shell::Kind;
using Op = cint::Operator;

using detail::add_block;
using detail::call;
using detail::Intor;
using detail::OptHolder;
using detail::unique_pairs;
using detail::centre_first;
using detail::quartet_centres;

// ---------------------------------------------------------------------------
// One-electron skeleton Hessian
// ---------------------------------------------------------------------------

Mat IntegralEngine::one_electron_hessian(const Mat &P, const Mat &Q) const {
  occ::timing::start(occ::timing::category::ints1e);
  const bool sph = is_spherical();
  const auto &basis = m_aobasis;
  const auto &atoms = basis.atoms();
  const int natom = static_cast<int>(atoms.size());
  const auto &first_bf = basis.first_bf();
  const auto &shell_atom = basis.shell_to_atom();
  const auto pairs = unique_pairs(m_shellpairs);

  Intor *ipipovlp = sph ? libcint::int1e_ipipovlp_sph : libcint::int1e_ipipovlp_cart;
  Intor *ipipkin = sph ? libcint::int1e_ipipkin_sph : libcint::int1e_ipipkin_cart;
  Intor *ipiprinv = sph ? libcint::int1e_ipiprinv_sph : libcint::int1e_ipiprinv_cart;
  Intor *iprinvip = sph ? libcint::int1e_iprinvip_sph : libcint::int1e_iprinvip_cart;

  const size_t bufsize = 9 * static_cast<size_t>(m_env.buffer_size_1e());
  struct ThreadData {
    IntEnv env;
    std::vector<double> b0, b1, b2;
    Mat H;
  };
  occ::parallel::thread_local_storage<ThreadData> tls([&]() {
    return ThreadData{m_env, std::vector<double>(bufsize),
                      std::vector<double>(bufsize),
                      std::vector<double>(bufsize),
                      Mat::Zero(3 * natom, 3 * natom)};
  });

  // sum_{mu in p, nu in q} X_{mu nu} buf[comp](mu, nu), comp = 3a + b
  auto contract2 = [&](const Mat &X, int p, int q, const double *buf,
                       bool transposed) {
    const int np = basis[p].size(), nq = basis[q].size();
    const int bp = first_bf[p], bq = first_bf[q];
    Mat3 out;
    const size_t n = static_cast<size_t>(np) * nq;
    for (int c = 0; c < 9; c++) {
      Eigen::Map<const Mat> blk(buf + c * n, np, nq);
      out(c / 3, c % 3) =
          transposed ? X.block(bq, bp, nq, np).transpose().cwiseProduct(blk).sum()
                     : X.block(bp, bq, np, nq).cwiseProduct(blk).sum();
    }
    return out;
  };

  // Tasks 0..natom-1: nuclear attraction of nucleus C. Task natom: S and T.
  occ::parallel::parallel_for(size_t(0), size_t(natom + 1), [&](size_t task) {
    auto &t = tls.local();
    if (static_cast<int>(task) == natom) {
      // Two-centre terms: f(R_M - R_N), so MM = NN = -MN = -NM.
      for (const auto &[p, q] : pairs) {
        const int M = shell_atom[p], N = shell_atom[q];
        if (M == N)
          continue;
        int shls[2] = {p, q};
        const double f = (p == q) ? 1.0 : 2.0;
        Mat3 B = Mat3::Zero();
        if (call(ipipkin, t.env, t.b0.data(), shls))
          B += contract2(P, p, q, t.b0.data(), false);
        if (call(ipipovlp, t.env, t.b1.data(), shls))
          B -= contract2(Q, p, q, t.b1.data(), false);
        B *= f;
        add_block(t.H, M, M, B);
        add_block(t.H, N, N, B);
        add_block(t.H, M, N, -B);
        add_block(t.H, N, M, -B.transpose());
      }
      return;
    }
    const int C = static_cast<int>(task);
    // The operator in V carries the ECP-reduced charge.
    const double Z = effective_nuclear_charge(C);
    t.env.set_rinv_origin({atoms[C].x, atoms[C].y, atoms[C].z});
    for (const auto &[p, q] : pairs) {
      const int M = shell_atom[p], N = shell_atom[q];
      if (M == C && N == C)
        continue;
      const double f = -Z * ((p == q) ? 1.0 : 2.0);
      int pq[2] = {p, q}, qp[2] = {q, p};
      Mat3 MM = Mat3::Zero(), NN = Mat3::Zero(), MN = Mat3::Zero();
      if (call(ipiprinv, t.env, t.b0.data(), pq))
        MM = f * contract2(P, p, q, t.b0.data(), false);
      if (call(ipiprinv, t.env, t.b1.data(), qp))
        NN = f * contract2(P, q, p, t.b1.data(), false);
      if (call(iprinvip, t.env, t.b2.data(), pq))
        MN = f * contract2(P, p, q, t.b2.data(), false);
      detail::add_operator_centre_blocks(t.H, M, N, C, MM, NN, MN);
    }
  });

  Mat H = Mat::Zero(3 * natom, 3 * natom);
  for (const auto &t : tls)
    H += t.H;
  H += effective_core_potential_hessian(P);
  occ::timing::stop(occ::timing::category::ints1e);
  return H;
}

// ---------------------------------------------------------------------------
// One-electron derivative matrices
// ---------------------------------------------------------------------------

void IntegralEngine::one_electron_derivative_matrices(
    std::vector<Mat> &S1, std::vector<Mat> &H1) const {
  const auto &basis = m_aobasis;
  const auto &atoms = basis.atoms();
  const size_t natom = atoms.size();
  const auto nbf = basis.nbf();
  const auto &first_bf = basis.first_bf();
  const auto &atom_to_shell = basis.atom_to_shell();

  // (nabla mu | O | nu) with the derivative on the bra, every mu.
  const MatTriple Sg = one_electron_operator_grad(Op::overlap);
  const MatTriple Tg = one_electron_operator_grad(Op::kinetic);
  const MatTriple Vg = one_electron_operator_grad(Op::nuclear);

  S1.assign(3 * natom, Mat());
  H1.assign(3 * natom, Mat());
  for (size_t A = 0; A < natom; A++) {
    // Operator (Hellmann-Feynman) part: d/dA of -Z_A/|r - A| moves the
    // operator, = -Z_A [(nabla mu|rinv|nu) + transpose].
    MatTriple rinv = rinv_operator_grad_atom(A);
    const double Z = effective_nuclear_charge(A);
    for (int x = 0; x < 3; x++) {
      Mat Ms = Mat::Zero(nbf, nbf);
      Mat Mh = -Z * rinv[x];
      for (int s : atom_to_shell[A]) {
        const int b0 = first_bf[s], n = basis[s].size();
        Ms.middleRows(b0, n) -= Sg[x].middleRows(b0, n);
        Mh.middleRows(b0, n) -= Tg[x].middleRows(b0, n) + Vg[x].middleRows(b0, n);
      }
      S1[3 * A + x] = Ms + Ms.transpose();
      H1[3 * A + x] = Mh + Mh.transpose();
    }
  }
  add_effective_core_potential_derivatives(H1);
}

// ---------------------------------------------------------------------------
// Two-electron skeleton Hessian
// ---------------------------------------------------------------------------

template <ShellKind kind>
Mat jk_hessian_kernel(IntEnv &env, const AOBasis &basis,
                      const ShellPairList &shellpairs,
                      const MolecularOrbitals &mo, double jf, double kf,
                      double precision, const Mat &Schwarz) {
  constexpr bool sph = kind == ShellKind::Spherical;
  const int natom = static_cast<int>(basis.atoms().size());
  const auto &first_bf = basis.first_bf();
  const auto &shell_atom = basis.shell_to_atom();
  const auto pairs = unique_pairs(shellpairs);
  const size_t npair = pairs.size();
  const auto tpd = detail::two_particle_density<kind>(basis, mo);
  const auto fill_gamma = tpd.unrestricted ? detail::fill_gamma<true>
                                           : detail::fill_gamma<false>;

  Intor *ipip1 = sph ? libcint::int2e_ipip1_sph : libcint::int2e_ipip1_cart;
  Intor *ipvip1 = sph ? libcint::int2e_ipvip1_sph : libcint::int2e_ipvip1_cart;
  Intor *ip1ip2 = sph ? libcint::int2e_ip1ip2_sph : libcint::int2e_ip1ip2_cart;

  const size_t nmax4 = env.buffer_size_2e(0);
  struct ThreadData {
    std::vector<double> buf, gamma;
    Mat H;
    OptHolder o_ipip1, o_ipvip1, o_ip1ip2;
  };
  occ::parallel::thread_local_storage<ThreadData> tls([&]() {
    ThreadData t{std::vector<double>(9 * nmax4), std::vector<double>(nmax4),
                 Mat::Zero(3 * natom, 3 * natom)};
    libcint::int2e_ipip1_optimizer(&t.o_ipip1.ptr, env.atom_data_ptr(),
                                   env.num_atoms(), env.basis_data_ptr(),
                                   env.num_basis(), env.env_data_ptr());
    libcint::int2e_ipvip1_optimizer(&t.o_ipvip1.ptr, env.atom_data_ptr(),
                                    env.num_atoms(), env.basis_data_ptr(),
                                    env.num_basis(), env.env_data_ptr());
    libcint::int2e_ip1ip2_optimizer(&t.o_ip1ip2.ptr, env.atom_data_ptr(),
                                    env.num_atoms(), env.basis_data_ptr(),
                                    env.num_basis(), env.env_data_ptr());
    return t;
  });

  auto process_row = [&](size_t ij) {
    auto &t = tls.local();
    const auto [p, q] = pairs[ij];
    for (size_t kl = 0; kl <= ij; kl++) {
      const auto [r, s] = pairs[kl];
      const int sh[4] = {p, q, r, s};
      const auto qc = quartet_centres(shell_atom[p], shell_atom[q],
                                      shell_atom[r], shell_atom[s]);
      if (qc.trivial)
        continue;
      if (detail::screened_out(tpd, Schwarz, p, q, r, s, jf, kf, precision))
        continue;
      const int nn[4] = {static_cast<int>(basis[p].size()),
                         static_cast<int>(basis[q].size()),
                         static_cast<int>(basis[r].size()),
                         static_cast<int>(basis[s].size())};
      const double deg = ((p == q) ? 1.0 : 2.0) * ((r == s) ? 1.0 : 2.0) *
                         ((ij == kl) ? 1.0 : 2.0);
      fill_gamma(t.gamma.data(), tpd, first_bf[p], first_bf[q], first_bf[r],
                 first_bf[s], nn, deg, jf, kf);

      // Blocks d2/dc_i dc_j for needed centres (i <= j).
      Mat3 B[4][4];
      for (int a = 0; a < qc.nneeded; a++) {
        for (int b = a; b < qc.nneeded; b++) {
          const int ci = qc.needed[a], cj = qc.needed[b];
          Intor *f;
          libcint::CINTOpt *opt;
          int perm[4];
          if (ci == cj) {
            f = ipip1;
            opt = t.o_ipip1.ptr;
            centre_first(ci, perm);
          } else if ((ci < 2) == (cj < 2)) {
            // same side: (nabla ci nabla cj | other pair)
            f = ipvip1;
            opt = t.o_ipvip1.ptr;
            const int other0 = ci < 2 ? 2 : 0;
            perm[0] = ci;
            perm[1] = cj;
            perm[2] = other0;
            perm[3] = other0 + 1;
          } else {
            // (nabla ci partner | nabla cj partner); ci < cj so ci is bra
            f = ip1ip2;
            opt = t.o_ip1ip2.ptr;
            perm[0] = ci;
            perm[1] = ci ^ 1;
            perm[2] = cj;
            perm[3] = cj ^ 1;
          }
          int shls[4] = {sh[perm[0]], sh[perm[1]], sh[perm[2]], sh[perm[3]]};
          double out[9];
          if (call(f, env, t.buf.data(), shls, opt)) {
            detail::contract4<9>(t.gamma.data(), t.buf.data(), nn, perm, out);
          } else {
            for (double &o : out)
              o = 0.0;
          }
          Mat3 blk;
          for (int c = 0; c < 9; c++)
            blk(c / 3, c % 3) = out[c];
          B[a][b] = blk;
          if (a != b)
            B[b][a] = blk.transpose();
        }
      }
      // Accumulate needed-centre blocks, and the omitted group by
      // translational invariance: d/dO = -sum_needed d/dc.
      Mat3 OO = Mat3::Zero();
      for (int a = 0; a < qc.nneeded; a++) {
        Mat3 rowsum = Mat3::Zero();
        for (int b = 0; b < qc.nneeded; b++) {
          add_block(t.H, qc.atoms[qc.needed[a]], qc.atoms[qc.needed[b]],
                    B[a][b]);
          rowsum += B[a][b];
        }
        add_block(t.H, qc.atoms[qc.needed[a]], qc.omit, -rowsum);
        add_block(t.H, qc.omit, qc.atoms[qc.needed[a]], -rowsum.transpose());
        OO += rowsum;
      }
      add_block(t.H, qc.omit, qc.omit, OO);
    }
  };

  occ::parallel::parallel_for(size_t(0), npair, [&](size_t k) {
    process_row(npair - 1 - k);
  });

  Mat H = Mat::Zero(3 * natom, 3 * natom);
  for (const auto &t : tls)
    H += t.H;
  return H;
}

Mat IntegralEngine::jk_hessian(const MolecularOrbitals &mo, double jf,
                               double kf, const Mat &Schwarz) const {
  if (mo.kind == SpinorbitalKind::General)
    throw std::runtime_error("jk_hessian: general spinorbitals not supported");
  occ::timing::start(occ::timing::category::fock);
  Mat H = is_spherical()
              ? jk_hessian_kernel<ShellKind::Spherical>(
                    m_env, m_aobasis, m_shellpairs, mo, jf, kf, m_precision,
                    Schwarz)
              : jk_hessian_kernel<ShellKind::Cartesian>(
                    m_env, m_aobasis, m_shellpairs, mo, jf, kf, m_precision,
                    Schwarz);
  occ::timing::stop(occ::timing::category::fock);
  return H;
}

// ---------------------------------------------------------------------------
// Skeleton two-electron Fock derivatives
// ---------------------------------------------------------------------------

template <ShellKind kind>
std::vector<Mat>
jk_derivative_fock_kernel(IntEnv &env, const AOBasis &basis,
                          const ShellPairList &shellpairs,
                          const MolecularOrbitals &mo, double jf, double kf,
                          double precision, const Mat &Schwarz) {
  constexpr bool sph = kind == ShellKind::Spherical;
  const int natom = static_cast<int>(basis.atoms().size());
  const auto nbf = basis.nbf();
  const auto &first_bf = basis.first_bf();
  const auto &shell_atom = basis.shell_to_atom();
  const auto pairs = unique_pairs(shellpairs);
  const size_t npair = pairs.size();
  const auto tpd = detail::two_particle_density<kind>(basis, mo);
  // The Fock derivative is linear in the density, so screen on the density
  // alone: the bound is |T| * |P| with T the derivative integral.
  const Mat &Pt = tpd.Pt;
  // One Fock derivative per spin, with exchange from that spin's density.
  const int nspin = tpd.unrestricted ? 2 : 1;
  const Mat *Pspin[2] = {&tpd.Pa, &tpd.Pb};
  const bool do_k = kf != 0.0;

  Intor *ip1 = sph ? libcint::int2e_ip1_sph : libcint::int2e_ip1_cart;
  const size_t nmax4 = env.buffer_size_2e(0);

  struct ThreadData {
    std::vector<double> buf;
    std::vector<double> dc; // per needed centre, 3 comps, canonical layout
    std::vector<double> T;  // per-atom derivative tensor, 3 comps
    std::vector<std::vector<Mat>> F; // per spin
    OptHolder opt;
  };
  occ::parallel::thread_local_storage<ThreadData> tls([&]() {
    ThreadData t{std::vector<double>(3 * nmax4),
                 std::vector<double>(4 * 3 * nmax4),
                 std::vector<double>(3 * nmax4),
                 std::vector<std::vector<Mat>>(
                     nspin, std::vector<Mat>(3 * natom, Mat::Zero(nbf, nbf)))};
    libcint::int2e_ip1_optimizer(&t.opt.ptr, env.atom_data_ptr(),
                                 env.num_atoms(), env.basis_data_ptr(),
                                 env.num_basis(), env.env_data_ptr());
    return t;
  });

  auto process_row = [&](size_t ij) {
    auto &t = tls.local();
    const auto [p, q] = pairs[ij];
    for (size_t kl = 0; kl <= ij; kl++) {
      const auto [r, s] = pairs[kl];
      const int sh[4] = {p, q, r, s};
      const auto qc = quartet_centres(shell_atom[p], shell_atom[q],
                                      shell_atom[r], shell_atom[s]);
      if (qc.trivial)
        continue;
      if (Schwarz.size() > 0) {
        double bound = std::abs(jf) * std::max(tpd.Nt(p, q), tpd.Nt(r, s));
        if (do_k)
          bound = std::max(bound, std::abs(kf) * detail::max_of(tpd.Ns(p, r), tpd.Ns(q, s),
                                                        tpd.Ns(p, s), tpd.Ns(q, r)));
        if (bound * Schwarz(p, q) * Schwarz(r, s) < precision)
          continue;
      }
      const int nn[4] = {static_cast<int>(basis[p].size()),
                         static_cast<int>(basis[q].size()),
                         static_cast<int>(basis[r].size()),
                         static_cast<int>(basis[s].size())};
      const size_t nelem = static_cast<size_t>(nn[0]) * nn[1] * nn[2] * nn[3];
      const size_t cstride[4] = {1, static_cast<size_t>(nn[0]),
                                 static_cast<size_t>(nn[0]) * nn[1],
                                 static_cast<size_t>(nn[0]) * nn[1] * nn[2]};

      // Per needed centre: d/dc (pq|rs) in canonical layout, 3 components.
      bool any = false;
      for (int a = 0; a < qc.nneeded; a++) {
        const int c = qc.needed[a];
        int perm[4];
        centre_first(c, perm);
        int shls[4] = {sh[perm[0]], sh[perm[1]], sh[perm[2]], sh[perm[3]]};
        double *dst = t.dc.data() + a * 3 * nelem;
        if (!call(ip1, env, t.buf.data(), shls, t.opt.ptr)) {
          std::fill(dst, dst + 3 * nelem, 0.0);
          continue;
        }
        any = true;
        const int n0 = nn[perm[0]], n1 = nn[perm[1]], n2 = nn[perm[2]],
                  n3 = nn[perm[3]];
        const size_t s0 = cstride[perm[0]], s1 = cstride[perm[1]],
                     s2 = cstride[perm[2]], s3 = cstride[perm[3]];
        for (int x = 0; x < 3; x++) {
          const double *src = t.buf.data() + x * nelem;
          double *d = dst + x * nelem;
          size_t n = 0;
          for (int i3 = 0; i3 < n3; i3++)
            for (int i2 = 0; i2 < n2; i2++)
              for (int i1 = 0; i1 < n1; i1++) {
                const size_t base = i3 * s3 + i2 * s2 + i1 * s1;
                for (int i0 = 0; i0 < n0; i0++, n++)
                  d[base + i0 * s0] = -src[n]; // d/dnucleus = -nabla
              }
        }
      }
      if (!any)
        continue;

      const double deg = ((p == q) ? 1.0 : 2.0) * ((r == s) ? 1.0 : 2.0) *
                         ((ij == kl) ? 1.0 : 2.0);
      const int bp = first_bf[p], bq = first_bf[q], br = first_bf[r],
                bs = first_bf[s];

      // Distinct atoms of this quartet, each with its derivative tensor.
      int distinct[4];
      int ndistinct = 0;
      for (int c = 0; c < 4; c++) {
        bool seen = false;
        for (int k = 0; k < ndistinct; k++)
          seen |= distinct[k] == qc.atoms[c];
        if (!seen)
          distinct[ndistinct++] = qc.atoms[c];
      }
      for (int k = 0; k < ndistinct; k++) {
        const int A = distinct[k];
        double *T = t.T.data();
        std::fill(T, T + 3 * nelem, 0.0);
        for (int a = 0; a < qc.nneeded; a++) {
          const int c = qc.needed[a];
          const double sign = (A == qc.omit) ? -1.0
                              : (qc.atoms[c] == A) ? 1.0
                                                   : 0.0;
          if (sign == 0.0)
            continue;
          const double *src = t.dc.data() + a * 3 * nelem;
          for (size_t n = 0; n < 3 * nelem; n++)
            T[n] += sign * src[n];
        }
        for (int x = 0; x < 3; x++)
          for (int sp = 0; sp < nspin; sp++) {
            Mat &F = t.F[sp][3 * A + x];
            const Mat &Ps = *Pspin[sp];
            const double *Tx = T + x * nelem;
            for (int d = 0, n = 0; d < nn[3]; d++) {
              const int si = bs + d;
              for (int c = 0; c < nn[2]; c++) {
                const int la = br + c;
                for (int b = 0; b < nn[1]; b++) {
                  const int nu = bq + b;
                  for (int a = 0; a < nn[0]; a++, n++) {
                    const int mu = bp + a;
                    const double v = Tx[n] * deg;
                    if (v == 0.0)
                      continue;
                    F(mu, nu) += jf * Pt(la, si) * v;
                    F(la, si) += jf * Pt(mu, nu) * v;
                    if (do_k) {
                      const double ck = 0.5 * kf * v;
                      F(mu, la) -= ck * Ps(nu, si);
                      F(nu, si) -= ck * Ps(mu, la);
                      F(mu, si) -= ck * Ps(nu, la);
                      F(nu, la) -= ck * Ps(mu, si);
                    }
                  }
                }
              }
            }
          }
      }
    }
  };

  occ::parallel::parallel_for(size_t(0), npair, [&](size_t k) {
    process_row(npair - 1 - k);
  });

  // Unrestricted: alpha rows over beta rows.
  std::vector<Mat> result(3 * natom, Mat::Zero(nspin * nbf, nbf));
  for (const auto &t : tls)
    for (int sp = 0; sp < nspin; sp++)
      for (int k = 0; k < 3 * natom; k++)
        result[k].middleRows(sp * nbf, nbf) += t.F[sp][k];
  for (auto &F : result)
    for (int sp = 0; sp < nspin; sp++) {
      auto Fs = F.middleRows(sp * nbf, nbf);
      Fs = 0.25 * (Fs + Fs.transpose()).eval();
    }
  return result;
}

std::vector<Mat> IntegralEngine::jk_derivative_fock(const MolecularOrbitals &mo,
                                                    double jf, double kf,
                                                    const Mat &Schwarz) const {
  if (mo.kind == SpinorbitalKind::General)
    throw std::runtime_error(
        "jk_derivative_fock: general spinorbitals not supported");
  occ::timing::start(occ::timing::category::fock);
  auto result =
      is_spherical()
          ? jk_derivative_fock_kernel<ShellKind::Spherical>(
                m_env, m_aobasis, m_shellpairs, mo, jf, kf, m_precision, Schwarz)
          : jk_derivative_fock_kernel<ShellKind::Cartesian>(
                m_env, m_aobasis, m_shellpairs, mo, jf, kf, m_precision,
                Schwarz);
  occ::timing::stop(occ::timing::category::fock);
  return result;
}

} // namespace occ::qm
