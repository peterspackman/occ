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
//   g3(R)       = damp3 / R³,                  damp3 = 1 / (1 + 6 (R_co/R)^k3)
//   ∂g3/∂R     = damp3·[k3·(1 − damp3) − 3] / R⁴
//   ∂g3/∂R_co  = −damp3·k3·(1 − damp3) / (R_co · R³)
// Same shapes for g5 with k5 / R⁶.
struct PairDamp {
  double g3;
  double g5;
  double g3prime;       // ∂g3 / ∂R
  double g5prime;       // ∂g5 / ∂R
  double dg3_drco;      // ∂g3 / ∂R_co
  double dg5_drco;      // ∂g5 / ∂R_co
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
  const double rco_inv = 1.0 / R_co;
  p.dg3_drco = -damp3 * k3 * (1.0 - damp3) * rco_inv * r3inv;
  p.dg5_drco = -damp5 * k5 * (1.0 - damp5) * rco_inv * r5inv;
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

AnisotropicPairGradient
anisotropic_pair_gradient_with_dcn(const std::vector<core::Atom> &atoms,
                                    const Vec &q, const CammMoments &m,
                                    const Vec &mp_radii,
                                    const Vec &dmp_radii_dcn,
                                    const Gfn2Parameters &params) {
  const int nat = static_cast<int>(atoms.size());
  const auto &g = params.globals();
  const double k3 = g.aesdmp3;
  const double k5 = g.aesdmp5;

  AnisotropicPairGradient out;
  out.grad_explicit = Mat3N::Zero(3, nat);
  out.dE_dcn = Vec::Zero(nat);

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

      double ed = 0.0, eq = 0.0, edd = 0.0;
      double mu_i_dot_rij = 0.0, mu_j_dot_rij = 0.0, mu_i_dot_mu_j = 0.0;
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

      // Explicit-R gradient (same as `anisotropic_pair_gradient`).
      const double E_g3_factor = ed;
      const double E_g5_factor = eq + edd;
      const double dg3_factor = -d.g3prime / R;
      const double dg5_factor = -d.g5prime / R;
      double grad_i[3];
      for (int a = 0; a < 3; ++a) {
        const double dEd_da =
            -(q_j * m.dipm(a, i) - q_i * m.dipm(a, j));
        const double dEq_da = -2.0 * (q_j * Qi_rij[a] + q_i * Qj_rij[a]);
        const double dEdd_da = -2.0 * mu_i_dot_mu_j * rij[a]
                                + 3.0 * (m.dipm(a, i) * mu_j_dot_rij +
                                          m.dipm(a, j) * mu_i_dot_rij);
        grad_i[a] = dEd_da * d.g3 + (dEq_da + dEdd_da) * d.g5
                  + E_g3_factor * dg3_factor * rij[a]
                  + E_g5_factor * dg5_factor * rij[a];
      }
      for (int a = 0; a < 3; ++a) {
        out.grad_explicit(a, i) += grad_i[a];
        out.grad_explicit(a, j) -= grad_i[a];
      }

      // ∂E_pair/∂R_co contribution. R_co = ½(mp_radii(i) + mp_radii(j)),
      // so ∂R_co/∂CN_A = ½ δ_iA · dmp_radii(i)/dCN_i + ½ δ_jA · dmp_radii(j)/dCN_j.
      const double dE_pair_drco =
          E_g3_factor * d.dg3_drco + E_g5_factor * d.dg5_drco;
      out.dE_dcn(i) += 0.5 * dmp_radii_dcn(i) * dE_pair_drco;
      out.dE_dcn(j) += 0.5 * dmp_radii_dcn(j) * dE_pair_drco;
    }
  }
  return out;
}

namespace {

// Mapping from xtb's qpint order (xx, yy, zz, xy, xz, yz) → my Q array order
// (xx, xy, xz, yy, yz, zz).  q_idx_from_qpint[i] gives the index into my Q.
constexpr int q_from_qpint[6] = {0, 3, 5, 1, 2, 4};

} // namespace

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

namespace {

// Per-AO scalar U(ν) := Σ_α vd_α(A_ν) · R_{A_ν, α}.  Used by the centering
// chain term R_{A_ν,α} · ∂S/∂R in the dipole-Pulay gradient.
inline Vec compute_U_per_bf(const std::vector<core::Atom> &atoms,
                            const std::vector<int> &bf_to_atom,
                            const Mat3N &vd) {
  const int nbf = static_cast<int>(bf_to_atom.size());
  Vec U(nbf);
  for (int nu = 0; nu < nbf; ++nu) {
    const int A = bf_to_atom[nu];
    U(nu) = vd(0, A) * atoms[A].x + vd(1, A) * atoms[A].y +
            vd(2, A) * atoms[A].z;
  }
  return U;
}

} // namespace

Mat3N anisotropic_density_pulay_gradient(
    const std::vector<core::Atom> &atoms,
    const std::vector<int> &bf_to_atom,
    const Mat &P, const Mat &S,
    const MatTriple &D_origin0,
    const std::array<Mat, 6> &Q_origin0,
    const std::array<MatTriple, 3> &irp,
    const std::array<MatTriple, 6> &irrp,
    const MatTriple &ovlp_grad,
    const AnisotropicPotentials &pot) {
  // Mulliken partition (matches `compute_camm_moments_periodic`):
  //   μ_A_α   = -Σ_{ν ∈ A, μ} P_μν · D_bra_α(μ, ν)
  //   Q_A_qp[l] = -Σ_{ν ∈ A, μ} P_μν · Q_bra_traceless[remap(l)](μ, ν)
  //
  // where D_bra and Q_bra are the per-atom (column-side) centered AO
  // matrices (centering origin = R_{A_ν}).  Energy chain through density
  // contributes
  //   Σ_A vd_α(A) ∂μ_A_α/∂R_C |_P  +  Σ_A Σ_l vq_l(A) ∂Q_A_l/∂R_C |_P
  // = -Σ μν P_μν · vd_α(A_ν) · ∂D_bra_α(μ,ν)/∂R_C
  //   -Σ μν P_μν · vq_l(A_ν) · ∂Q_bra_traceless[remap(l)](μ,ν)/∂R_C
  //
  // Per ordered AO pair (μ, ν) with A_μ ≠ A_ν the contribution distributes
  // to atoms A_μ and A_ν via the IBP / centering chain rule (same-atom pairs
  // give 0 by translation invariance).
  const int N = static_cast<int>(atoms.size());
  const int nbf = static_cast<int>(P.rows());
  Mat3N g = Mat3N::Zero(3, N);

  Vec U = compute_U_per_bf(atoms, bf_to_atom, pot.vd);

  // Quadrupole helpers: precompute the six independent vq components
  // already paired with the corresponding Q-storage index (k in
  // {xx, xy, xz, yy, yz, zz}). vq is in qpint order (xx, yy, zz, xy, xz, yz);
  // the bridge map q_to_qpint[k] gives the qpint slot for storage slot k.
  constexpr int q_to_qpint[6] = {0, 3, 4, 1, 5, 2};
  // (α, β) pair for each q-storage slot k.
  constexpr int kab[6][2] = {
      {0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2},
  };
  // Build per-atom 3×3 vq_full(α, β) tensor (symmetric, qpint layout
  // collapsed). Used for fast dot-products in the inner loop.
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> dummy33;
  (void)dummy33;
  std::array<Eigen::Matrix3d, 1> _; // placeholder to silence unused warnings
  (void)_;

  // Precompute traceless-Cartesian q-storage of Q_origin0 for the bra-side
  // chain through ∂D_β(O=0)/∂R; the algebraic structure δ_αγ D_β + δ_βγ D_α
  // doesn't lift trivially to traceless form, so we handle it via raw 6-vector
  // accumulation per pair followed by a linear traceless projection at the
  // contraction step.

  for (int mu = 0; mu < nbf; ++mu) {
    const int Aμ = bf_to_atom[mu];
    for (int nu = 0; nu < nbf; ++nu) {
      const int Aν = bf_to_atom[nu];
      if (Aμ == Aν)
        continue;

      const double Pμν = P(mu, nu);
      const double Sμν = S(mu, nu);

      // Per-pair coordinates of the col-side atom (where centering origin
      // sits for D_bra and Q_bra).
      const double Rν[3] = {atoms[Aν].x, atoms[Aν].y, atoms[Aν].z};

      // Cache D_origin0(μ, ν) as a 3-vector for the (α, β) loops.
      const double D0[3] = {D_origin0.x(mu, nu), D_origin0.y(mu, nu),
                             D_origin0.z(mu, nu)};

      // Cache Q_origin0(μ, ν) as a 6-vector in q-storage order.
      const double Q0[6] = {Q_origin0[0](mu, nu), Q_origin0[1](mu, nu),
                             Q_origin0[2](mu, nu), Q_origin0[3](mu, nu),
                             Q_origin0[4](mu, nu), Q_origin0[5](mu, nu)};

      for (int gamma = 0; gamma < 3; ++gamma) {
        // ∂S(μ,ν)/∂R_(A_μ)γ = -⟨∂_γ φ_μ | φ_ν⟩ = -ovlp_grad.γ(μ,ν).
        // ∂S(μ,ν)/∂R_(A_ν)γ = -⟨∂_γ φ_ν | φ_μ⟩ = -ovlp_grad.γ(ν,μ).
        const double dS_mu = -((gamma == 0) ? ovlp_grad.x(mu, nu)
                                : (gamma == 1) ? ovlp_grad.y(mu, nu)
                                                : ovlp_grad.z(mu, nu));
        const double dS_nu = -((gamma == 0) ? ovlp_grad.x(nu, mu)
                                : (gamma == 1) ? ovlp_grad.y(nu, mu)
                                                : ovlp_grad.z(nu, mu));

        // -- DIPOLE chain ----------------------------------------------
        // ∂D_bra_α/∂R_(A_μ)γ = δ_αγ Sμν + irp[α].γ(μ,ν) − R_{A_ν,α} · dS_mu
        // ∂D_bra_α/∂R_(A_ν)γ = -irp[α].γ(μ,ν) − δ_αγ Sμν − R_{A_ν,α} · dS_nu
        //
        // g_C(γ) += -P_μν · vd_α(A_ν) · ∂D_bra_α/∂R_Cγ.
        double dip_mu = pot.vd(gamma, Aν) * Sμν - U(nu) * dS_mu;
        double dip_nu = -pot.vd(gamma, Aν) * Sμν - U(nu) * dS_nu;
        // Cache irp[α].γ(μ, ν) for the (α = 0, 1, 2) indices — shared by
        // the dipole and quadrupole chains (the latter sees ∂D_α/∂R via
        // ∂(R · D)).
        const double irp_g[3] = {
            ((gamma == 0) ? irp[0].x : (gamma == 1) ? irp[0].y : irp[0].z)(mu, nu),
            ((gamma == 0) ? irp[1].x : (gamma == 1) ? irp[1].y : irp[1].z)(mu, nu),
            ((gamma == 0) ? irp[2].x : (gamma == 1) ? irp[2].y : irp[2].z)(mu, nu),
        };
        for (int alpha = 0; alpha < 3; ++alpha) {
          dip_mu += pot.vd(alpha, Aν) * irp_g[alpha];
          dip_nu -= pot.vd(alpha, Aν) * irp_g[alpha];
        }
        g(gamma, Aμ) -= Pμν * dip_mu;
        g(gamma, Aν) -= Pμν * dip_nu;

        // -- QUADRUPOLE chain ------------------------------------------
        // Q_bra_(αβ) = Q_(αβ)(O=0) − R_{A_ν,α} D_β − R_{A_ν,β} D_α
        //              + R_{A_ν,α} R_{A_ν,β} S
        //
        // Build raw (non-traceless) 6-vector ∂Q_bra/∂R_Cγ (q-storage layout
        // {xx, xy, xz, yy, yz, zz}) for both atoms C ∈ {A_μ, A_ν}, then
        // traceless-project (1.5×k − 0.5·δ_αβ tr) and contract with vq.
        //
        // Bra-side ∂D_β/∂R_(A_μ)γ = δ_βγ Sμν + irp[β].γ(μ,ν).
        // Ket-side ∂D_β/∂R_(A_ν)γ = -irp[β].γ(μ,ν).
        // ∂Q(O=0)/∂R: bra = irrp[k].γ; ket = -irrp[k].γ.
        const double irrp_g_k[6] = {
            ((gamma == 0) ? irrp[0].x : (gamma == 1) ? irrp[0].y : irrp[0].z)(mu, nu),
            ((gamma == 0) ? irrp[1].x : (gamma == 1) ? irrp[1].y : irrp[1].z)(mu, nu),
            ((gamma == 0) ? irrp[2].x : (gamma == 1) ? irrp[2].y : irrp[2].z)(mu, nu),
            ((gamma == 0) ? irrp[3].x : (gamma == 1) ? irrp[3].y : irrp[3].z)(mu, nu),
            ((gamma == 0) ? irrp[4].x : (gamma == 1) ? irrp[4].y : irrp[4].z)(mu, nu),
            ((gamma == 0) ? irrp[5].x : (gamma == 1) ? irrp[5].y : irrp[5].z)(mu, nu),
        };

        double dQbra_mu[6] = {0, 0, 0, 0, 0, 0};
        double dQbra_nu[6] = {0, 0, 0, 0, 0, 0};
        for (int k = 0; k < 6; ++k) {
          const int alpha = kab[k][0];
          const int beta = kab[k][1];

          // ∂Q(O=0)/∂R_Cγ.  IBP on ⟨φ_μ | r_α r_β | φ_ν⟩ at the bra side gives
          //   ⟨-∂_γ φ_μ | r_α r_β | φ_ν⟩ = +irrp[k][γ] + δ_αγ D_β + δ_βγ D_α
          // (matches the predictions verified in the irrp FD test).
          dQbra_mu[k] += irrp_g_k[k];
          if (alpha == gamma)
            dQbra_mu[k] += D0[beta];
          if (beta == gamma)
            dQbra_mu[k] += D0[alpha];
          // Ket side: ⟨φ_μ | r_α r_β | -∂_γ φ_ν⟩ = -irrp[k][γ] (no extra δ-D).
          dQbra_nu[k] += -irrp_g_k[k];

          // -R_{A_ν,α} · ∂D_β(O=0)/∂R_Cγ -R_{A_ν,β} · ∂D_α(O=0)/∂R_Cγ
          // Bra-side (C = A_μ): ∂D/∂R_(A_μ)γ = δ_*γ S + irp[*].γ
          double dDb_mu_β = (beta == gamma ? Sμν : 0.0) + irp_g[beta];
          double dDb_mu_α = (alpha == gamma ? Sμν : 0.0) + irp_g[alpha];
          dQbra_mu[k] += -Rν[alpha] * dDb_mu_β - Rν[beta] * dDb_mu_α;
          // Ket-side (C = A_ν): ∂D/∂R_(A_ν)γ = -irp[*].γ
          dQbra_nu[k] += -Rν[alpha] * (-irp_g[beta]) -
                          Rν[beta] * (-irp_g[alpha]);

          // Explicit ∂R_{A_ν}/∂R_(A_ν)γ chain (only A_ν side):
          //   −δ_αγ D_β − δ_βγ D_α
          //   +(δ_αγ R_β + δ_βγ R_α) · S
          if (alpha == gamma) {
            dQbra_nu[k] += -D0[beta] + Rν[beta] * Sμν;
          }
          if (beta == gamma) {
            dQbra_nu[k] += -D0[alpha] + Rν[alpha] * Sμν;
          }

          // R_{A_ν,α} R_{A_ν,β} · ∂S/∂R_Cγ
          dQbra_mu[k] += Rν[alpha] * Rν[beta] * dS_mu;
          dQbra_nu[k] += Rν[alpha] * Rν[beta] * dS_nu;

          (void)Q0; // Q0 itself doesn't appear in the derivative — only ∂Q.
        }

        // Linear traceless-Cartesian projection on the 6-vector
        // (q-storage {xx, xy, xz, yy, yz, zz}):
        //   Q'_xx = 1.5·Q_xx - 0.5·(Q_xx + Q_yy + Q_zz)
        //   Q'_yy = 1.5·Q_yy - 0.5·(... )
        //   Q'_zz = 1.5·Q_zz - 0.5·(... )
        //   Q'_off = 1.5·Q_off
        auto traceless6 = [](const double in[6], double out[6]) {
          const double trace = in[0] + in[3] + in[5];
          out[0] = 1.5 * in[0] - 0.5 * trace;
          out[3] = 1.5 * in[3] - 0.5 * trace;
          out[5] = 1.5 * in[5] - 0.5 * trace;
          out[1] = 1.5 * in[1];
          out[2] = 1.5 * in[2];
          out[4] = 1.5 * in[4];
        };
        double dQbra_t_mu[6], dQbra_t_nu[6];
        traceless6(dQbra_mu, dQbra_t_mu);
        traceless6(dQbra_nu, dQbra_t_nu);

        // Contract with vq (qpint order). Sum over k = 0..5 q-storage slots,
        // mapping into qpint via q_to_qpint[k]. Subtraction sign comes from
        //   g_C += Σ_l vq_l ∂Q_l/∂R_C  with  Q_l = -Σ μν P_μν · Q_bra_l,
        // i.e.  g_C -= Σ_l vq_l · Pμν · ∂Q_bra_l/∂R_C.
        double quad_mu = 0.0, quad_nu = 0.0;
        for (int k = 0; k < 6; ++k) {
          const double vqk = pot.vq(q_to_qpint[k], Aν);
          quad_mu += vqk * dQbra_t_mu[k];
          quad_nu += vqk * dQbra_t_nu[k];
        }
        g(gamma, Aμ) -= Pμν * quad_mu;
        g(gamma, Aν) -= Pμν * quad_nu;
      }
    }
  }

  return g;
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
