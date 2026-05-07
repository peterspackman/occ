#include <cmath>
#include <occ/xtb/anisotropic.h>
#include <occ/xtb/camm.h>
#include <occ/xtb/gfn2_parameters.h>
#include <occ/xtb/multipole_damping.h>

namespace occ::xtb {

namespace {

// Map (k, l) Cartesian indices to xtb's qp flat index (xx, xy, yy, xz, yz, zz):
//   idx[k][l] (0-based)
constexpr int kl_to_qp[3][3] = {
    {0, 1, 3},
    {1, 2, 4},
    {3, 4, 5},
};

} // namespace

AnisotropicEnergy
anisotropic_energy(const std::vector<core::Atom> &atoms, const Vec &q,
                   const CammMoments &m, const DampedCoulomb &damped,
                   const Gfn2Parameters &params) {
  const int nat = static_cast<int>(atoms.size());

  // 1. On-site polarization: dipKernel * |μ|² + quadKernel * Σ Q²
  double epol = 0.0;
  for (int i = 0; i < nat; ++i) {
    const auto *e = params.element(atoms[i].atomic_number);
    double dd = 0.0;
    for (int k = 0; k < 3; ++k)
      dd += m.dipm(k, i) * m.dipm(k, i);
    double qq = 0.0;
    for (int k = 0; k < 3; ++k) {
      for (int l = 0; l < 3; ++l) {
        const int idx = kl_to_qp[k][l];
        qq += m.qp(idx, i) * m.qp(idx, i);
      }
    }
    epol += e->dip_kernel * dd + e->quad_kernel * qq;
  }

  // 2. Pair interactions: charge-dipole, charge-quadrupole, dipole-dipole.
  double e01 = 0.0, e02 = 0.0, e11 = 0.0;
  for (int i = 0; i < nat; ++i) {
    const double q_i = q(i);
    const double xi = atoms[i].x, yi = atoms[i].y, zi = atoms[i].z;
    for (int j = 0; j < i; ++j) {
      const double q_j = q(j);
      const double rx = atoms[j].x - xi;
      const double ry = atoms[j].y - yi;
      const double rz = atoms[j].z - zi;
      const double r2 = rx * rx + ry * ry + rz * rz;
      const double rij[3] = {rx, ry, rz};

      // Charge-dipole: tblite convention is `e_qd_v1 = Σ dipm[i]·(R_i-R_j)·q_j·g3`
      // over ordered (i, j ≠ i). With our pair loop (j < i) and `rij = R_j - R_i`,
      // we negate to match tblite directly.
      double ed = 0.0, eq = 0.0, edd = 0.0;
      for (int k = 0; k < 3; ++k) {
        ed += q_j * m.dipm(k, i) * rij[k];
        ed -= m.dipm(k, j) * q_i * rij[k];
        for (int l = 0; l < 3; ++l) {
          const double tt = rij[l] * rij[k];
          const double tt3 = 3.0 * tt;
          const int idx = kl_to_qp[k][l];
          eq += q_j * m.qp(idx, i) * tt;
          eq += m.qp(idx, j) * q_i * tt;
          edd -= m.dipm(k, j) * m.dipm(l, i) * tt3;
        }
        edd += m.dipm(k, j) * m.dipm(k, i) * r2;
      }
      e01 += ed * damped.gab3(j, i);
      e02 += eq * damped.gab5(j, i);
      e11 += edd * damped.gab5(j, i);
    }
  }

  AnisotropicEnergy out;
  out.aes = e01 + e02 + e11;
  out.polariz = epol;
  return out;
}

namespace {

// Closed-form derivatives of the damped multipole kernels.
//   g3(R) = damp3 / R³,    damp3 = 1 / (1 + 6 (R_co/R)^k3)
//   g3'(R) = damp3·[k3·(1 − damp3) − 3] / R⁴
// Same shape for g5 with k5 / R⁶.
struct PairDamp {
  double g3;
  double g5;
  double g3prime;  // dg3 / dR
  double g5prime;  // dg5 / dR
};

inline PairDamp pair_damp(double R, double R_co, double k3, double k5) {
  const double rinv = 1.0 / R;
  const double rcoinvr = R_co * rinv;
  const double damp3 = 1.0 / (1.0 + 6.0 * std::pow(rcoinvr, k3));
  const double damp5 = 1.0 / (1.0 + 6.0 * std::pow(rcoinvr, k5));
  const double r3inv = rinv * rinv * rinv;
  const double r5inv = r3inv * rinv * rinv;
  PairDamp p;
  p.g3 = damp3 * r3inv;
  p.g5 = damp5 * r5inv;
  p.g3prime = damp3 * (k3 * (1.0 - damp3) - 3.0) * r3inv * rinv;
  p.g5prime = damp5 * (k5 * (1.0 - damp5) - 5.0) * r5inv * rinv;
  return p;
}

} // namespace

Mat3N
anisotropic_pair_gradient(const std::vector<core::Atom> &atoms, const Vec &q,
                          const CammMoments &m, const Vec &mp_radii,
                          const Gfn2Parameters &params) {
  const int nat = static_cast<int>(atoms.size());
  const auto &g = params.globals();
  const double k3 = g.aesdmp3;
  const double k5 = g.aesdmp5;

  Mat3N grad = Mat3N::Zero(3, nat);

  // Pair loop mirrors `anisotropic_energy`'s ordering: j < i with
  // rij = R_j − R_i. ∂rij/∂R_iα = −δ_α (atom i moves rij in the −α
  // direction), ∂rij/∂R_jα = +δ_α. Newton III gives an antisymmetric
  // pair contribution (atom j picks up minus what atom i picks up).
  for (int i = 0; i < nat; ++i) {
    const double q_i = q(i);
    const double xi = atoms[i].x, yi = atoms[i].y, zi = atoms[i].z;
    for (int j = 0; j < i; ++j) {
      const double q_j = q(j);
      const double rx = atoms[j].x - xi;
      const double ry = atoms[j].y - yi;
      const double rz = atoms[j].z - zi;
      const double r2 = rx * rx + ry * ry + rz * rz;
      const double R = std::sqrt(r2);
      const double rij[3] = {rx, ry, rz};
      const double R_co = 0.5 * (mp_radii(i) + mp_radii(j));
      const PairDamp d = pair_damp(R, R_co, k3, k5);

      // -- Energy parts (frozen multipoles) --
      // ed   = (q_j μ_i − q_i μ_j) · rij
      // eq   = q_j (rij^T Q_i rij) + q_i (rij^T Q_j rij)
      // edd  = (μ_i·μ_j) R² − 3 (μ_i·rij)(μ_j·rij)
      double ed = 0.0, eq = 0.0, edd = 0.0;
      double mu_i_dot_rij = 0.0, mu_j_dot_rij = 0.0;
      double mu_i_dot_mu_j = 0.0;
      // Q · rij for each atom: stored kl_to_qp index map:
      // {{0,1,3},{1,2,4},{3,4,5}} → flat (xx, xy, yy, xz, yz, zz).
      double Qi_rij[3] = {0.0, 0.0, 0.0};
      double Qj_rij[3] = {0.0, 0.0, 0.0};
      for (int k = 0; k < 3; ++k) {
        const double mui_k = m.dipm(k, i);
        const double muj_k = m.dipm(k, j);
        mu_i_dot_rij += mui_k * rij[k];
        mu_j_dot_rij += muj_k * rij[k];
        mu_i_dot_mu_j += mui_k * muj_k;
        ed += q_j * mui_k * rij[k] - q_i * muj_k * rij[k];
        for (int l = 0; l < 3; ++l) {
          const int idx = kl_to_qp[k][l];
          const double tt = rij[l] * rij[k];
          eq += q_j * m.qp(idx, i) * tt + q_i * m.qp(idx, j) * tt;
          edd -= 3.0 * m.dipm(k, j) * m.dipm(l, i) * tt;
          Qi_rij[k] += m.qp(idx, i) * rij[l];
          Qj_rij[k] += m.qp(idx, j) * rij[l];
        }
        edd += m.dipm(k, j) * m.dipm(k, i) * r2;
      }

      // -- Energy at this pair (for the kernel-derivative term) --
      const double E_g3_factor = ed;            // multiplied by g3
      const double E_g5_factor = eq + edd;      // multiplied by g5

      // ∂R/∂R_iα = −rij[α] / R, so ∂g3/∂R_iα = −g3'(R) · rij[α] / R, etc.
      const double dg3_factor = -d.g3prime / R; // multiply by rij[α] for Δgrad_i
      const double dg5_factor = -d.g5prime / R;

      // -- ∂(ed, eq, edd)/∂R_iα at frozen multipoles --
      // ∂ed/∂R_iα   = −(q_j μ_i[α] − q_i μ_j[α])
      // ∂eq/∂R_iα   = −2 (q_j Q_i + q_i Q_j) · rij)[α]
      // ∂edd/∂R_iα  = −2 (μ_i·μ_j) rij[α]
      //              + 3 [μ_i[α] (μ_j·rij) + μ_j[α] (μ_i·rij)]
      // (We then multiply ∂ed by g3, ∂eq + ∂edd by g5.)
      double grad_i[3];
      for (int a = 0; a < 3; ++a) {
        const double dEd_da =
            -(q_j * m.dipm(a, i) - q_i * m.dipm(a, j));
        const double dEq_da = -2.0 * (q_j * Qi_rij[a] + q_i * Qj_rij[a]);
        const double dEdd_da = -2.0 * mu_i_dot_mu_j * rij[a]
                                + 3.0 * (m.dipm(a, i) * mu_j_dot_rij +
                                          m.dipm(a, j) * mu_i_dot_rij);

        // Chain rule total: explicit-rij + kernel-R chains.
        grad_i[a] = dEd_da * d.g3 + (dEq_da + dEdd_da) * d.g5
                  + E_g3_factor * dg3_factor * rij[a]
                  + E_g5_factor * dg5_factor * rij[a];
      }

      // Distribute to atoms i and j.
      for (int a = 0; a < 3; ++a) {
        grad(a, i) += grad_i[a];
        grad(a, j) -= grad_i[a]; // Newton's third law
      }
    }
  }
  return grad;
}

namespace {

// xtb's qpint flat index for (k, l), 0-based, in (xx, yy, zz, xy, xz, yz)
// order. Diagonals: xx=0, yy=1, zz=2. Off-diagonals (k != l): xy=3, xz=4,
// yz=5.
constexpr int qpint_idx(int k, int l) {
  if (k == l) return k;
  // k+l+1 in 0-based:
  //   (0,1)=1+1=2 → +1 = 3 (xy)? Actually in 1-based xtb: ki = l1+l2+1.
  // In 0-based it's k+l+1 → (0,1)=2 (xy), (0,2)=3 (xz), (1,2)=4 (yz). Wait
  // we want xy=3, xz=4, yz=5 (0-based). Let me just enumerate manually.
  if ((k == 0 && l == 1) || (k == 1 && l == 0)) return 3;
  if ((k == 0 && l == 2) || (k == 2 && l == 0)) return 4;
  if ((k == 1 && l == 2) || (k == 2 && l == 1)) return 5;
  return -1;
}

// xtb's qp (CAMM storage) flat index for (k, l), 0-based:
// (xx, xy, yy, xz, yz, zz).
constexpr int qp_idx(int k, int l) {
  // From kl_to_qp earlier in the file: {{0,1,3},{1,2,4},{3,4,5}}
  static_assert(true);
  constexpr int t[3][3] = {{0, 1, 3}, {1, 2, 4}, {3, 4, 5}};
  return t[k][l];
}

// Mapping from xtb's qpint order (xx, yy, zz, xy, xz, yz) → my Q array order
// (xx, xy, xz, yy, yz, zz).  q_idx_from_qpint[i] gives the index into my Q.
constexpr int q_from_qpint[6] = {0, 3, 5, 1, 2, 4};

} // namespace

AnisotropicPotentials
anisotropic_potentials(const std::vector<core::Atom> &atoms, const Vec &q,
                       const CammMoments &m, const DampedCoulomb &damped,
                       const Gfn2Parameters &params) {
  const int nat = static_cast<int>(atoms.size());

  AnisotropicPotentials out;
  out.vs = Vec::Zero(nat);
  out.vd = Mat3N::Zero(3, nat);
  out.vq = Mat::Zero(6, nat);

  // Pair contributions (all j, including self via the self-interaction
  // limit, mirrors xtb's setvsdq).
  for (int i = 0; i < nat; ++i) {
    const double rai[3] = {atoms[i].x, atoms[i].y, atoms[i].z};
    double stmp = 0.0;
    double dtmp[3] = {0.0, 0.0, 0.0};
    double qtmp[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (int j = 0; j < nat; ++j) {
      const double g3 = damped.gab3(j, i);
      const double g5 = damped.gab5(j, i);
      const double rbj[3] = {atoms[j].x, atoms[j].y, atoms[j].z};
      const double dra[3] = {rai[0] - rbj[0], rai[1] - rbj[1], rai[2] - rbj[2]};

      double r2a = 0.0, r2ab = 0.0;
      double t1a = 0.0, t2a = 0.0, t3a = 0.0;
      for (int l1 = 0; l1 < 3; ++l1) {
        r2a += rai[l1] * rai[l1];
        r2ab += dra[l1] * dra[l1];
        t1a += rai[l1] * dra[l1];
        t2a += m.dipm(l1, j) * dra[l1];
        t3a += rai[l1] * m.dipm(l1, j);
      }

      double dum5a = 0.0;
      for (int l1 = 0; l1 < 3; ++l1) {
        for (int l2 = 0; l2 < 3; ++l2) {
          dum5a -= m.qp(qp_idx(l1, l2), j) * dra[l1] * dra[l2];
          dum5a -= 1.5 * q(j) * dra[l1] * dra[l2] * rai[l1] * rai[l2];
          if (l2 >= l1) continue;
          // Off-diagonal qpint slot — ki in xtb is l1+l2+1 (1-based) which
          // is xy/xz/yz for the three off-diag (l1>l2) pairs.
          const int ki = qpint_idx(l1, l2);
          qtmp[ki] -= 3.0 * q(j) * g5 * dra[l2] * dra[l1];
        }
        // Diagonal qpint slot — index l1 (xx, yy, zz)
        qtmp[l1] -= 1.5 * q(j) * g5 * dra[l1] * dra[l1];
      }

      const double dum3a = -t1a * q(j) - t2a;
      dum5a += t3a * r2ab - 3.0 * t1a * t2a + 0.5 * q(j) * r2a * r2ab;
      stmp += dum5a * g5 + dum3a * g3;

      for (int l1 = 0; l1 < 3; ++l1) {
        const double dd3 = dra[l1] * q(j);
        const double dd5 = 3.0 * dra[l1] * t2a -
                           r2ab * m.dipm(l1, j) -
                           q(j) * r2ab * rai[l1] +
                           3.0 * q(j) * dra[l1] * t1a;
        dtmp[l1] += dd3 * g3 + dd5 * g5;
        qtmp[l1] += 0.5 * r2ab * q(j) * g5;
      }
    }

    // CT (on-site polarization) terms
    const auto *e = params.element(atoms[i].atomic_number);
    const double qs1 = 2.0 * e->dip_kernel;
    const double qs2 = 6.0 * e->quad_kernel;
    double t3a = 0.0, t2a = 0.0;
    for (int l1 = 0; l1 < 3; ++l1) {
      t3a += rai[l1] * m.dipm(l1, i) * qs1;
      dtmp[l1] -= qs1 * m.dipm(l1, i);
      for (int l2 = 0; l2 < l1; ++l2) {
        const int ll = qp_idx(l1, l2);
        const int ki = qpint_idx(l1, l2);
        qtmp[ki] -= m.qp(ll, i) * qs2;
        t3a -= rai[l1] * rai[l2] * m.qp(ll, i) * qs2;
        dtmp[l1] += rai[l2] * m.qp(ll, i) * qs2;
        dtmp[l2] += rai[l1] * m.qp(ll, i) * qs2;
      }
      const int ll_diag = qp_idx(l1, l1);
      qtmp[l1] -= m.qp(ll_diag, i) * qs2 * 0.5;
      t3a -= rai[l1] * rai[l1] * m.qp(ll_diag, i) * qs2 * 0.5;
      dtmp[l1] += rai[l1] * m.qp(ll_diag, i) * qs2;
      t2a += m.qp(ll_diag, i);
    }
    stmp += t3a;
    t2a *= e->quad_kernel;
    for (int l1 = 0; l1 < 3; ++l1) {
      qtmp[l1] += t2a;
      dtmp[l1] -= 2.0 * rai[l1] * t2a;
      stmp += t2a * rai[l1] * rai[l1];
    }

    out.vs(i) = stmp;
    for (int l1 = 0; l1 < 3; ++l1) out.vd(l1, i) = dtmp[l1];
    for (int l1 = 0; l1 < 6; ++l1) out.vq(l1, i) = qtmp[l1];
  }

  return out;
}

void apply_anisotropic_h1_periodic(
    Mat &H, const Mat &S,
    const MatTriple &D_ket, const MatTriple &D_bra,
    const std::array<Mat, 6> &Q_ket, const std::array<Mat, 6> &Q_bra,
    const std::vector<int> &bf_to_atom,
    const AnisotropicPotentials &pot) {
  // Periodic AO pair (μ at cell 0, ν at image T): split the H1 contribution
  // so the Ket integral (atom-of-μ-centered) pairs with vd of atom-of-μ, and
  // the Bra integral (atom-of-ν-image-centered) pairs with vd of atom-of-ν.
  // Symmetric averaging (D × (v_μ + v_ν)/2) only works when AOs sit at the
  // same absolute origin — true for molecular but not in general for periodic.
  const int nbf = static_cast<int>(H.rows());
  for (int mu = 0; mu < nbf; ++mu) {
    const int ii = bf_to_atom[mu];
    for (int nu = 0; nu < nbf; ++nu) {
      const int jj = bf_to_atom[nu];
      double eh1 = 0.5 * S(mu, nu) * (pot.vs(ii) + pot.vs(jj));
      // Dipole: each side gets its own atom-centered AO matrix × that atom's
      // potential. (Sum the two halves, no extra 1/2 factor — the original
      // (1/2)(v_μ + v_ν) was averaging the two atomic-frame views, here we
      // assemble them directly.)
      eh1 += 0.5 * (D_ket.x(mu, nu) * pot.vd(0, ii) +
                    D_ket.y(mu, nu) * pot.vd(1, ii) +
                    D_ket.z(mu, nu) * pot.vd(2, ii));
      eh1 += 0.5 * (D_bra.x(mu, nu) * pot.vd(0, jj) +
                    D_bra.y(mu, nu) * pot.vd(1, jj) +
                    D_bra.z(mu, nu) * pot.vd(2, jj));
      // Quadrupole — same split. vq is in qpint order; Q_ket/Q_bra in
      // {xx, xy, xz, yy, yz, zz} order; remap via q_from_qpint.
      for (int l = 0; l < 6; ++l) {
        eh1 += 0.5 * Q_ket[q_from_qpint[l]](mu, nu) * pot.vq(l, ii);
        eh1 += 0.5 * Q_bra[q_from_qpint[l]](mu, nu) * pot.vq(l, jj);
      }
      // Sign convention matches tblite scf/potential.f90 add_vmp_to_h1 +
      // add_vao_to_h1: H1 -= 0.5·integral·potential.
      H(mu, nu) -= eh1;
    }
  }
}

void apply_anisotropic_h1_kpoint(
    CMat &H, const CMat &S,
    const CMatTriple &D_ket, const CMatTriple &D_bra,
    const std::array<CMat, 6> &Q_ket, const std::array<CMat, 6> &Q_bra,
    const std::vector<int> &bf_to_atom,
    const AnisotropicPotentials &pot) {
  // Same per-AO-pair formula as the real Γ-only path, with complex Bloch-summed
  // AO matrices at this k. vs/vd/vq are real (atom-resolved), so the H1 update
  // contributes complex values whose Hermitian symmetry is inherited from the
  // (D_bra(k) = D_ket(k)^H, Q_bra(k) = Q_ket(k)^H) relation.
  const Eigen::Index nbf = H.rows();
  for (Eigen::Index mu = 0; mu < nbf; ++mu) {
    const int ii = bf_to_atom[mu];
    for (Eigen::Index nu = 0; nu < nbf; ++nu) {
      const int jj = bf_to_atom[nu];
      std::complex<double> eh1 =
          0.5 * S(mu, nu) * (pot.vs(ii) + pot.vs(jj));
      eh1 += 0.5 * (D_ket.x(mu, nu) * pot.vd(0, ii) +
                    D_ket.y(mu, nu) * pot.vd(1, ii) +
                    D_ket.z(mu, nu) * pot.vd(2, ii));
      eh1 += 0.5 * (D_bra.x(mu, nu) * pot.vd(0, jj) +
                    D_bra.y(mu, nu) * pot.vd(1, jj) +
                    D_bra.z(mu, nu) * pot.vd(2, jj));
      for (int l = 0; l < 6; ++l) {
        eh1 += 0.5 * Q_ket[q_from_qpint[l]](mu, nu) * pot.vq(l, ii);
        eh1 += 0.5 * Q_bra[q_from_qpint[l]](mu, nu) * pot.vq(l, jj);
      }
      H(mu, nu) -= eh1;
    }
  }
}

} // namespace occ::xtb
