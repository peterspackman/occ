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

      // Charge-dipole convention: `e_qd = Σ dipm[i]·(R_i-R_j)·q_j·g3` over
      // ordered (i, j ≠ i). With our pair loop (j < i) and `rij = R_j - R_i`,
      // we negate so the unordered-pair sum matches the ordered-pair form.
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

AnisotropicPairGradient
anisotropic_pair_gradient_with_dcn(const std::vector<core::Atom> &atoms,
                                    const Vec &q, const CammMoments &m,
                                    const Vec &mp_radii,
                                    const Vec &dmp_radii_dcn,
                                    const Gfn2Parameters &params) {
  // Pair loop mirrors `anisotropic_energy`'s ordering: j < i with
  // rij = R_j − R_i. ∂rij/∂R_iα = −δ_α, ∂rij/∂R_jα = +δ_α (Newton III).
  //
  // Energy pieces (per-pair, frozen multipoles):
  //   ed  = (q_j μ_i − q_i μ_j) · rij                 (× g3)
  //   eq  = q_j (rij^T Q_i rij) + q_i (rij^T Q_j rij) (× g5)
  //   edd = (μ_i·μ_j) R² − 3 (μ_i·rij)(μ_j·rij)       (× g5)
  //
  // The "explicit" gradient (∂ at frozen mp_radii) sums the rij and kernel-R
  // chains. The dE/dCN piece is the kernel-R_co chain through mp_radii(CN_A);
  // for `dmp_radii_dcn = 0` the result reduces to the frozen-radii gradient.
  const int nat = static_cast<int>(atoms.size());
  const auto &g = params.globals();
  const double k3 = g.aesdmp3;
  const double k5 = g.aesdmp5;
  const bool has_dcn = (dmp_radii_dcn.size() == nat);

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

      const double E_g3_factor = ed;          // E_pair contribution × g3
      const double E_g5_factor = eq + edd;    // × g5
      const double dg3_factor = -d.g3prime / R;
      const double dg5_factor = -d.g5prime / R;
      for (int a = 0; a < 3; ++a) {
        const double dEd_da = -(q_j * m.dipm(a, i) - q_i * m.dipm(a, j));
        const double dEq_da = -2.0 * (q_j * Qi_rij[a] + q_i * Qj_rij[a]);
        const double dEdd_da = -2.0 * mu_i_dot_mu_j * rij[a]
                                + 3.0 * (m.dipm(a, i) * mu_j_dot_rij +
                                          m.dipm(a, j) * mu_i_dot_rij);
        const double grad_a = dEd_da * d.g3 + (dEq_da + dEdd_da) * d.g5
                            + E_g3_factor * dg3_factor * rij[a]
                            + E_g5_factor * dg5_factor * rij[a];
        out.grad_explicit(a, i) += grad_a;
        out.grad_explicit(a, j) -= grad_a;
      }

      if (has_dcn) {
        // R_co = ½ (mp_radii(i) + mp_radii(j)), so
        //   ∂R_co/∂CN_A = ½ (δ_iA · dmp(i)/dCN_i + δ_jA · dmp(j)/dCN_j)
        const double dE_pair_drco =
            E_g3_factor * d.dg3_drco + E_g5_factor * d.dg5_drco;
        out.dE_dcn(i) += 0.5 * dmp_radii_dcn(i) * dE_pair_drco;
        out.dE_dcn(j) += 0.5 * dmp_radii_dcn(j) * dE_pair_drco;
      }
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
      // Sign convention: H1 -= 0.5·integral·potential (combined dipole and
      // quadrupole AO contributions on this pair).
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

// Linear traceless-Cartesian projection on a 6-vector in q-storage layout
// {xx, xy, xz, yy, yz, zz}:
//   Q'_xx = 1.5·Q_xx - 0.5·(Q_xx + Q_yy + Q_zz)  (and yy, zz analogously)
//   Q'_off = 1.5·Q_off  (xy, xz, yz)
namespace {
inline void traceless_q6(const double in[6], double out[6]) {
  const double trace = in[0] + in[3] + in[5];
  out[0] = 1.5 * in[0] - 0.5 * trace;
  out[3] = 1.5 * in[3] - 0.5 * trace;
  out[5] = 1.5 * in[5] - 0.5 * trace;
  out[1] = 1.5 * in[1];
  out[2] = 1.5 * in[2];
  out[4] = 1.5 * in[4];
}
} // namespace

Mat3N anisotropic_density_pulay_gradient(
    const std::vector<core::Atom> &atoms,
    const std::vector<int> &bf_to_atom,
    const Mat &P, const Mat &S,
    const MatTriple &D_origin0,
    const std::array<MatTriple, 3> &irp,
    const std::array<MatTriple, 6> &irrp,
    const MatTriple &ovlp_grad,
    const AnisotropicPotentials &pot) {
  // Mulliken partition (matches `compute_camm_moments_periodic`):
  //   μ_A_α   = -Σ_{ν ∈ A, μ} P_μν · D_bra_α(μ, ν)
  //   Q_A_qp[l] = -Σ_{ν ∈ A, μ} P_μν · Q_bra_traceless[remap(l)](μ, ν)
  //
  // where D_bra and Q_bra are the per-atom (column-side) centered AO
  // matrices (centering origin = R_{A_ν}). Energy chain through density:
  //   Σ_A vd_α(A) ∂μ_A_α/∂R_C |_P  +  Σ_A Σ_l vq_l(A) ∂Q_A_l/∂R_C |_P
  // = -Σ μν P_μν · vd_α(A_ν) · ∂D_bra_α(μ,ν)/∂R_C
  //   -Σ μν P_μν · vq_l(A_ν) · ∂Q_bra_traceless[remap(l)](μ,ν)/∂R_C
  //
  // Per ordered AO pair (μ, ν) with A_μ ≠ A_ν the contribution distributes
  // to atoms A_μ and A_ν via the IBP / centering chain rule (same-atom pairs
  // give 0 by translation invariance). The bra-side accumulator builds the
  // raw (non-traceless) 6-vector then projects to traceless before the
  // final vq contraction (the projection is linear, so ordering is fine).

  // q-storage k → qpint l (vq's layout): xx → xx, xy → xy, xz → xz,
  // yy → yy, yz → yz, zz → zz, but in different slot orders.
  constexpr int q_to_qpint[6] = {0, 3, 4, 1, 5, 2};
  // (α, β) pair for each q-storage slot k.
  constexpr int kab[6][2] = {
      {0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2},
  };

  const int N = static_cast<int>(atoms.size());
  const int nbf = static_cast<int>(P.rows());
  Mat3N g = Mat3N::Zero(3, N);
  // Per-AO scalar U(ν) := Σ_α vd_α(A_ν) · R_{A_ν, α} for the centering chain
  // R_{A_ν,α} · ∂S/∂R term in the dipole-Pulay branch.
  const Vec U = compute_U_per_bf(atoms, bf_to_atom, pot.vd);

  for (int mu = 0; mu < nbf; ++mu) {
    const int Aμ = bf_to_atom[mu];
    for (int nu = 0; nu < nbf; ++nu) {
      const int Aν = bf_to_atom[nu];
      if (Aμ == Aν)
        continue;

      const double Pμν = P(mu, nu);
      const double Sμν = S(mu, nu);
      const double Rν[3] = {atoms[Aν].x, atoms[Aν].y, atoms[Aν].z};
      const double D0[3] = {D_origin0.x(mu, nu), D_origin0.y(mu, nu),
                             D_origin0.z(mu, nu)};

      for (int gamma = 0; gamma < 3; ++gamma) {
        // ∂S(μ,ν)/∂R_(A_μ)γ = -⟨∂_γ φ_μ | φ_ν⟩ = -ovlp_grad[γ](μ,ν).
        // ∂S(μ,ν)/∂R_(A_ν)γ = -⟨∂_γ φ_ν | φ_μ⟩ = -ovlp_grad[γ](ν,μ).
        const Mat &dS_g = ovlp_grad[gamma];
        const double dS_mu = -dS_g(mu, nu);
        const double dS_nu = -dS_g(nu, mu);

        // -- DIPOLE chain ----------------------------------------------
        // ∂D_bra_α/∂R_(A_μ)γ = δ_αγ Sμν + irp[α][γ](μ,ν) − R_{A_ν,α}·dS_mu
        // ∂D_bra_α/∂R_(A_ν)γ = -irp[α][γ](μ,ν) − δ_αγ Sμν − R_{A_ν,α}·dS_nu
        //
        // g_C(γ) += -P_μν · vd_α(A_ν) · ∂D_bra_α/∂R_Cγ.
        // The δ_αγ piece collapses to vd[γ]·S; the R_{A_ν,α}·∂S piece
        // collapses to U(ν)·∂S after summing α.
        const double irp_g[3] = {
            irp[0][gamma](mu, nu),
            irp[1][gamma](mu, nu),
            irp[2][gamma](mu, nu),
        };
        double dip_mu = pot.vd(gamma, Aν) * Sμν - U(nu) * dS_mu;
        double dip_nu = -pot.vd(gamma, Aν) * Sμν - U(nu) * dS_nu;
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
        // Build raw (non-traceless) 6-vector ∂Q_bra/∂R_Cγ for both atoms
        // C ∈ {A_μ, A_ν}, then traceless-project and contract with vq.
        //
        // Bra-side ∂D_β/∂R_(A_μ)γ = δ_βγ Sμν + irp[β][γ].
        // Ket-side ∂D_β/∂R_(A_ν)γ = -irp[β][γ].
        // ∂Q(O=0)/∂R: bra = irrp[k][γ] + δ_αγ D_β + δ_βγ D_α; ket = -irrp.
        double dQbra_mu[6], dQbra_nu[6];
        for (int k = 0; k < 6; ++k) {
          const int alpha = kab[k][0];
          const int beta = kab[k][1];
          const double irrp_kg = irrp[k][gamma](mu, nu);
          const double dDb_mu_β = (beta == gamma ? Sμν : 0.0) + irp_g[beta];
          const double dDb_mu_α = (alpha == gamma ? Sμν : 0.0) + irp_g[alpha];

          // Bra-side (C = A_μ).
          double m_acc = irrp_kg
                       - Rν[alpha] * dDb_mu_β
                       - Rν[beta]  * dDb_mu_α
                       + Rν[alpha] * Rν[beta] * dS_mu;
          if (alpha == gamma) m_acc += D0[beta];
          if (beta == gamma)  m_acc += D0[alpha];
          dQbra_mu[k] = m_acc;

          // Ket-side (C = A_ν), with the explicit ∂R_{A_ν,α}/∂R chain.
          double n_acc = -irrp_kg
                       + Rν[alpha] * irp_g[beta]
                       + Rν[beta]  * irp_g[alpha]
                       + Rν[alpha] * Rν[beta] * dS_nu;
          if (alpha == gamma) n_acc += -D0[beta]  + Rν[beta]  * Sμν;
          if (beta == gamma)  n_acc += -D0[alpha] + Rν[alpha] * Sμν;
          dQbra_nu[k] = n_acc;
        }
        double dQbra_t_mu[6], dQbra_t_nu[6];
        traceless_q6(dQbra_mu, dQbra_t_mu);
        traceless_q6(dQbra_nu, dQbra_t_nu);

        // g_C(γ) += -P_μν · Σ_l vq_l(A_ν) · ∂Q_bra_traceless_l/∂R_Cγ.
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
