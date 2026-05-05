#include <cmath>
#include <occ/xtb/anisotropic.h>
#include <occ/xtb/camm.h>
#include <occ/xtb/gfn2_parameters.h>
#include <occ/xtb/multipole_ewald.h>
#include <occ/xtb/periodic_gamma.h>
#include <stdexcept>

namespace occ::xtb {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSqrtPi = 1.7724538509055160273;

// Quadrupole storage order (matches CammMoments::qp / OCC's kl_to_qp):
//   0 = xx, 1 = xy, 2 = yy, 3 = xz, 4 = yz, 5 = zz
// (Same order as tblite's dir kernel amat_sq indices 1..6.)
constexpr int qp_xx = 0, qp_xy = 1, qp_yy = 2, qp_xz = 3, qp_yz = 4, qp_zz = 5;

// Same enumeration helper used in periodic_gamma.cpp.
std::vector<Vec3> enumerate_g_vectors(const Mat3 &reciprocal_bohr,
                                       double recip_cutoff) {
  const Vec3 b1 = reciprocal_bohr.col(0);
  const Vec3 b2 = reciprocal_bohr.col(1);
  const Vec3 b3 = reciprocal_bohr.col(2);
  auto bound = [&](const Vec3 &b) {
    return static_cast<int>(std::ceil(recip_cutoff / b.norm())) + 1;
  };
  const int n1 = bound(b1);
  const int n2 = bound(b2);
  const int n3 = bound(b3);
  std::vector<Vec3> out;
  out.reserve(static_cast<size_t>((2 * n1 + 1) * (2 * n2 + 1) * (2 * n3 + 1)));
  const double cutoff2 = recip_cutoff * recip_cutoff;
  for (int i = -n1; i <= n1; ++i) {
    for (int j = -n2; j <= n2; ++j) {
      for (int k = -n3; k <= n3; ++k) {
        Vec3 G = i * b1 + j * b2 + k * b3;
        const double g2 = G.squaredNorm();
        if (g2 < 1e-20 || g2 > cutoff2) continue;
        out.push_back(G);
      }
    }
  }
  return out;
}

} // namespace

MultipolePairTensors
build_multipole_ewald_tensors(const PeriodicSystem &sys, const Vec &mp_radii,
                              const Gfn2Parameters &params, double tol,
                              double alpha_user) {
  const int n = sys.num_atoms();
  const auto &g = params.globals();
  const double kdmp3 = g.aesdmp3;
  const double kdmp5 = g.aesdmp5;

  const double V = sys.volume();
  const double alpha = (alpha_user > 0.0) ? alpha_user
                                          : kSqrtPi / std::cbrt(V);
  const double x = std::sqrt(-std::log(tol));
  const double real_cutoff = x / alpha + 1.0;
  const double recip_cutoff = 2.0 * alpha * x;

  auto images = build_lattice_images(sys.lattice_bohr, real_cutoff);
  auto g_vectors = enumerate_g_vectors(sys.reciprocal_bohr(), recip_cutoff);

  // Pre-cache atom Cartesian positions.
  std::vector<Vec3> R(n);
  for (int A = 0; A < n; ++A) {
    R[A] = Vec3(sys.atoms[A].x, sys.atoms[A].y, sys.atoms[A].z);
  }

  MultipolePairTensors t;
  t.alpha = alpha;
  t.real_cutoff = real_cutoff;
  t.recip_cutoff = recip_cutoff;
  t.images = images;  // cache for gauge-correction lattice sum
  for (int a = 0; a < 3; ++a) t.sd[a] = Mat::Zero(n, n);
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) t.dd[a][b] = Mat::Zero(n, n);
  for (int p = 0; p < 6; ++p) t.sq[p] = Mat::Zero(n, n);

  const double four_pi_over_V = 4.0 * kPi / V;
  const double inv4a2 = 1.0 / (4.0 * alpha * alpha);
  const double alpha2 = alpha * alpha;
  const double alpha3 = alpha * alpha2;
  const double real_cutoff2 = real_cutoff * real_cutoff;
  const double eps2 = 1e-20;

  // Pre-cache reciprocal coefficients: (4π/V) exp(-G²/(4α²)) / G².
  std::vector<double> g_coeffs(g_vectors.size());
  for (size_t k = 0; k < g_vectors.size(); ++k) {
    const double g2 = g_vectors[k].squaredNorm();
    g_coeffs[k] = four_pi_over_V * std::exp(-g2 * inv4a2) / g2;
  }

  // G=0 background contribution (added to all (i, j) pairs).
  const double bg_dd_diag = four_pi_over_V / 6.0;
  const double bg_sq_diag = -four_pi_over_V / 9.0;

  // Self-energy per (i, i) on the diagonal (from removing the G=0 limit and
  // self-image exclusion of the real-space term). Coefficients chosen so
  // that 0.5 · μ^T · amat_dd · μ contributes -2/3 α³/√π · μ² per atom (matches
  // tblite get_multipole_matrix_3d post-loop adjustment).
  const double dd_self = 2.0 * (-2.0 / 3.0) * alpha3 / kSqrtPi;  // = -4/3 α³/√π
  const double sq_self = 4.0 / 9.0 * alpha3 / kSqrtPi;

  for (int iat = 0; iat < n; ++iat) {
    for (int jat = 0; jat < n; ++jat) {
      const Vec3 rij = R[iat] - R[jat];
      const double rco = 0.5 * (mp_radii(iat) + mp_radii(jat));

      // Real-space (direct) loop.
      double sd_acc[3] = {0.0, 0.0, 0.0};
      double dd_acc[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
      double sq_acc[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      for (const auto &im : images) {
        const Vec3 vec = rij + im.t_bohr;
        const double r2 = vec.squaredNorm();
        if (r2 < eps2 || r2 > real_cutoff2) continue;
        const double r1 = std::sqrt(r2);
        const double g1 = 1.0 / r1;
        const double g3 = g1 * g1 * g1;
        const double g5 = g3 * g1 * g1;
        const double rco_over_r = rco * g1;
        const double fdmp3 =
            1.0 / (1.0 + 6.0 * std::pow(rco_over_r, kdmp3));
        const double fdmp5 =
            1.0 / (1.0 + 6.0 * std::pow(rco_over_r, kdmp5));

        const double arg = r1 * alpha;
        const double expt = std::exp(-arg * arg) / kSqrtPi;
        const double erft = -std::erf(arg) * g1;
        const double e1 = g1 * g1 * (erft + 2.0 * expt * alpha);
        const double e2 =
            g1 * g1 * (e1 + 4.0 * expt * alpha2 * alpha / 3.0);

        const double tmp3 = fdmp3 * g3 + e1;
        const double tmp5 = fdmp5 * g5 + e2;
        const double tmp_iso = fdmp5 * g3 + e1;

        // sd[α](i, j) = +vec_α · tmp3 with vec = R_i - R_j (tblite convention,
        // matches tblite/multipole.f90 get_multipole_matrix_0d:
        //   amat_sd[α, j, i] += (R_i - R_j)_α · g3 · fdmp3).
        // Used together with `vd CT += 2·dkernel·dipm` and `H1 -= 0.5·D·v`
        // apply convention to match tblite's H1 update term-by-term.
        sd_acc[0] += vec.x() * tmp3;
        sd_acc[1] += vec.y() * tmp3;
        sd_acc[2] += vec.z() * tmp3;
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            const double iso = (a == b) ? tmp_iso : 0.0;
            dd_acc[a][b] += iso - vec(a) * vec(b) * 3.0 * tmp5;
          }
        }
        const double third = tmp_iso / 3.0;
        sq_acc[qp_xx] += vec.x() * vec.x() * tmp5 - third;
        sq_acc[qp_xy] += 2.0 * vec.x() * vec.y() * tmp5;
        sq_acc[qp_yy] += vec.y() * vec.y() * tmp5 - third;
        sq_acc[qp_xz] += 2.0 * vec.x() * vec.z() * tmp5;
        sq_acc[qp_yz] += 2.0 * vec.y() * vec.z() * tmp5;
        sq_acc[qp_zz] += vec.z() * vec.z() * tmp5 - third;
      }

      // Reciprocal loop.
      for (size_t k = 0; k < g_vectors.size(); ++k) {
        const Vec3 &G = g_vectors[k];
        const double gv = G.dot(rij);
        const double sink = std::sin(gv) * g_coeffs[k];
        const double cosk = std::cos(gv) * g_coeffs[k];
        // Tblite convention (multipole.f90 get_amat_sdq_rec_3d):
        //   amat_sd += 2·G·sin(G·rij)·g_coeff
        sd_acc[0] += 2.0 * G.x() * sink;
        sd_acc[1] += 2.0 * G.y() * sink;
        sd_acc[2] += 2.0 * G.z() * sink;
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            dd_acc[a][b] += G(a) * G(b) * cosk;
          }
        }
        sq_acc[qp_xx] += G.x() * G.x() * cosk;
        sq_acc[qp_xy] += 2.0 * G.x() * G.y() * cosk;
        sq_acc[qp_yy] += G.y() * G.y() * cosk;
        sq_acc[qp_xz] += 2.0 * G.x() * G.z() * cosk;
        sq_acc[qp_yz] += 2.0 * G.y() * G.z() * cosk;
        sq_acc[qp_zz] += G.z() * G.z() * cosk;
      }

      // G=0 background.
      for (int a = 0; a < 3; ++a) dd_acc[a][a] += bg_dd_diag;
      sq_acc[qp_xx] += bg_sq_diag;
      sq_acc[qp_yy] += bg_sq_diag;
      sq_acc[qp_zz] += bg_sq_diag;

      // Self-energy on the diagonal pair (i == j).
      if (iat == jat) {
        for (int a = 0; a < 3; ++a) dd_acc[a][a] += dd_self;
        sq_acc[qp_xx] += sq_self;
        sq_acc[qp_yy] += sq_self;
        sq_acc[qp_zz] += sq_self;
      }

      // Store.
      for (int a = 0; a < 3; ++a) t.sd[a](iat, jat) = sd_acc[a];
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) t.dd[a][b](iat, jat) = dd_acc[a][b];
      for (int p = 0; p < 6; ++p) t.sq[p](iat, jat) = sq_acc[p];
    }
  }

  return t;
}

namespace {

// kl_to_qp from anisotropic.cpp — duplicated here to avoid header churn.
constexpr int kl_to_qp_local[3][3] = {{qp_xx, qp_xy, qp_xz},
                                       {qp_xy, qp_yy, qp_yz},
                                       {qp_xz, qp_yz, qp_zz}};

// xtb's qpint storage order {xx, yy, zz, xy, xz, yz} for the AnisotropicPotentials
// vq output (matches the molecular `anisotropic_potentials` convention).
constexpr int qpint_xx = 0, qpint_yy = 1, qpint_zz = 2;
constexpr int qpint_xy = 3, qpint_xz = 4, qpint_yz = 5;

constexpr int qpint_idx(int k, int l) {
  if (k == l) return k;
  if ((k == 0 && l == 1) || (k == 1 && l == 0)) return qpint_xy;
  if ((k == 0 && l == 2) || (k == 2 && l == 0)) return qpint_xz;
  if ((k == 1 && l == 2) || (k == 2 && l == 1)) return qpint_yz;
  return -1;
}

} // namespace

AnisotropicEnergy
anisotropic_energy_ewald(const std::vector<core::Atom> &atoms, const Vec &q,
                          const CammMoments &m,
                          const MultipolePairTensors &t,
                          const Gfn2Parameters &params) {
  const int n = static_cast<int>(atoms.size());

  // 1. On-site polarization (unchanged from molecular).
  double epol = 0.0;
  for (int i = 0; i < n; ++i) {
    const auto *e = params.element(atoms[i].atomic_number);
    double dd = 0.0;
    for (int k = 0; k < 3; ++k) dd += m.dipm(k, i) * m.dipm(k, i);
    double qq = 0.0;
    for (int k = 0; k < 3; ++k) {
      for (int l = 0; l < 3; ++l) {
        const int idx = kl_to_qp_local[k][l];
        qq += m.qp(idx, i) * m.qp(idx, i);
      }
    }
    epol += e->dip_kernel * dd + e->quad_kernel * qq;
  }

  // 2. Pair contributions via tensor contractions.
  // Charge-dipole:    e01 = Σ_{i,α} dpat_α(i) · (Σ_j sd[α](i,j) · q(j))
  //                       - Σ_{j,α} dpat_α(j) · (Σ_i sd[α](i,j)^T · q(i))
  // Actually with the symmetric tensor convention from Ewald:
  // sd[α](i, j) is "field on dipole α at i due to charge j", so
  // E_qd_part = - Σ_{i, j, α} q_j · dpat_α(i) · sd[α](i, j)?
  // Easier: just use the gemv pattern from tblite directly.
  //   vd_from_q(α, i) = Σ_j sd[α](i, j) · q(j)
  //   vat_from_dp(j)  = Σ_{α, i} sd[α](i, j) · dpat(α, i)
  //   e_qd contribution = Σ_{α, i} dpat(α, i) · vd_from_q(α, i)
  //                       (charge → dipole side; the symmetric dipole → charge
  //                        contribution is captured in the same term thanks to
  //                        sd's antisymmetric construction over (i, j) pairs.)
  //
  // Following tblite's get_energy: e01 = (mur)·qat + sum(dpat · vd)
  //   where vd = sd · qat, and mur(j) = Σ_{α, i} sd^T_{αi,j} dpat(α, i).
  Mat vd = Mat::Zero(3, n);
  Vec mur = Vec::Zero(n);
  for (int i = 0; i < n; ++i) {
    for (int a = 0; a < 3; ++a) {
      double s = 0.0;
      for (int j = 0; j < n; ++j) s += t.sd[a](i, j) * q(j);
      vd(a, i) = s;
    }
  }
  for (int j = 0; j < n; ++j) {
    double s = 0.0;
    for (int i = 0; i < n; ++i)
      for (int a = 0; a < 3; ++a) s += t.sd[a](i, j) * m.dipm(a, i);
    mur(j) = s;
  }
  double e_qd = 0.0;
  for (int j = 0; j < n; ++j) e_qd += mur(j) * q(j);
  for (int i = 0; i < n; ++i)
    for (int a = 0; a < 3; ++a) e_qd += m.dipm(a, i) * vd(a, i);
  // tblite: e01 = (mur)*qat + sum(dpat * vd); then total += 0.5 * e01.
  // Sign flip vs tblite: our t.sd has the opposite sign convention (see the
  // comment on sd_acc construction), so the natural mur·q + Σ dipm·vd value
  // here is -tblite_e_qd_v1. Negate to report the same physical energy. The
  // SCC potential pipeline (anisotropic_potentials_ewald → apply...) is
  // separately consistent and is NOT changed by this — only the energy
  // reporting is corrected.
  e_qd *= -0.5;

  // Dipole-dipole: e11 = 0.5 * Σ_{i,j,α,β} dpat(α, i) · dd[α][β](i,j) · dpat(β, j)
  double e_dd = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          e_dd += m.dipm(a, i) * t.dd[a][b](i, j) * m.dipm(b, j);
        }
      }
    }
  }
  e_dd *= 0.5;

  // Charge-quadrupole: e02 = 0.5 * (Σ_{i,p} qpat(p, i) · sq[p](i,j) · q(j) +
  //                              Σ_{j,p} qpat(p, j) · sq[p](j,i)^T · q(i))
  // tblite: e02 = t1*qat + sum(qpat * vq); 0.5 prefactor.
  Mat vq_from_q = Mat::Zero(6, n);
  Vec t1 = Vec::Zero(n);
  for (int i = 0; i < n; ++i) {
    for (int p = 0; p < 6; ++p) {
      double s = 0.0;
      for (int j = 0; j < n; ++j) s += t.sq[p](i, j) * q(j);
      vq_from_q(p, i) = s;
    }
  }
  for (int j = 0; j < n; ++j) {
    double s = 0.0;
    for (int i = 0; i < n; ++i)
      for (int p = 0; p < 6; ++p) s += t.sq[p](i, j) * m.qp(p, i);
    t1(j) = s;
  }
  double e_qq = 0.0;
  for (int j = 0; j < n; ++j) e_qq += t1(j) * q(j);
  for (int i = 0; i < n; ++i)
    for (int p = 0; p < 6; ++p) e_qq += m.qp(p, i) * vq_from_q(p, i);
  e_qq *= 0.5;

  AnisotropicEnergy out;
  out.aes = e_qd + e_qq + e_dd;
  out.polariz = epol;
  return out;
}

MultipolePairTensors
build_molecular_multipole_tensors(const std::vector<core::Atom> &atoms,
                                   const Vec &mp_radii,
                                   const Gfn2Parameters &params) {
  // Same per-pair kernel as the Ewald build's direct loop (real-space damped
  // 1/r^3 and 1/r^5 with mldmp3/mldmp5), but with no lattice sum, no erfc
  // screening, no reciprocal, no self/background. Diagonals are zero.
  const auto &g = params.globals();
  const double kdmp3 = g.aesdmp3;
  const double kdmp5 = g.aesdmp5;
  const int n = static_cast<int>(atoms.size());

  MultipolePairTensors t;
  for (int a = 0; a < 3; ++a) t.sd[a] = Mat::Zero(n, n);
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) t.dd[a][b] = Mat::Zero(n, n);
  for (int p = 0; p < 6; ++p) t.sq[p] = Mat::Zero(n, n);
  t.alpha = 0.0;
  t.real_cutoff = std::numeric_limits<double>::infinity();
  t.recip_cutoff = 0.0;
  // images is left empty — molecular has no lattice translations.

  std::vector<Vec3> R(n);
  for (int i = 0; i < n; ++i) R[i] = Vec3(atoms[i].x, atoms[i].y, atoms[i].z);

  for (int iat = 0; iat < n; ++iat) {
    for (int jat = 0; jat < n; ++jat) {
      if (iat == jat) continue;
      const Vec3 vec = R[iat] - R[jat];
      const double r2 = vec.squaredNorm();
      const double r1 = std::sqrt(r2);
      const double g1 = 1.0 / r1;
      const double g3 = g1 * g1 * g1;
      const double g5 = g3 * g1 * g1;
      const double rco = 0.5 * (mp_radii(iat) + mp_radii(jat));
      const double rco_over_r = rco * g1;
      const double fdmp3 = 1.0 / (1.0 + 6.0 * std::pow(rco_over_r, kdmp3));
      const double fdmp5 = 1.0 / (1.0 + 6.0 * std::pow(rco_over_r, kdmp5));
      const double tmp3 = fdmp3 * g3;
      const double tmp5 = fdmp5 * g5;
      const double tmp_iso = fdmp5 * g3;

      // tblite convention: sd[α](i, j) = +(R_i - R_j)_α · tmp3.
      t.sd[0](iat, jat) += vec.x() * tmp3;
      t.sd[1](iat, jat) += vec.y() * tmp3;
      t.sd[2](iat, jat) += vec.z() * tmp3;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          const double iso = (a == b) ? tmp_iso : 0.0;
          t.dd[a][b](iat, jat) += iso - vec(a) * vec(b) * 3.0 * tmp5;
        }
      }
      const double third = tmp_iso / 3.0;
      t.sq[qp_xx](iat, jat) += vec.x() * vec.x() * tmp5 - third;
      t.sq[qp_xy](iat, jat) += 2.0 * vec.x() * vec.y() * tmp5;
      t.sq[qp_yy](iat, jat) += vec.y() * vec.y() * tmp5 - third;
      t.sq[qp_xz](iat, jat) += 2.0 * vec.x() * vec.z() * tmp5;
      t.sq[qp_yz](iat, jat) += 2.0 * vec.y() * vec.z() * tmp5;
      t.sq[qp_zz](iat, jat) += vec.z() * vec.z() * tmp5 - third;
    }
  }
  return t;
}

AnisotropicPotentials
anisotropic_potentials_ewald(const std::vector<core::Atom> &atoms,
                              const Vec &q, const CammMoments &m,
                              const MultipolePairTensors &t,
                              const Gfn2Parameters &params) {
  // Strict tensor-derivative path. NOTE: the molecular anisotropic_potentials
  // includes gauge-correction terms (involving absolute atom positions) that
  // make the per-atom potentials suitable for use with global-origin AO
  // dipole/quadrupole integrals via apply_anisotropic_h1. Use
  // anisotropic_potentials_ewald_gauge_corrected for that purpose; this
  // function is kept for tensor-derivative testing.
  const int n = static_cast<int>(atoms.size());
  AnisotropicPotentials out;
  out.vs = Vec::Zero(n);
  out.vd = Mat3N::Zero(3, n);
  out.vq = Mat::Zero(6, n);

  for (int i = 0; i < n; ++i) {
    double s = 0.0;
    // vs[i] = "potential at charge i due to all dipoles + quadrupoles at j",
    // matching tblite's pot.vat += amat_sd · dpat (trans=T) + amat_sq · qpat
    // (trans=T). After our +(R_i-R_j) sign convention for sd, accessing
    // t.sd[a](i, j) gives +tblite_amat_sd[a, j, i] which is the right factor
    // for the trans=T contraction. (sq is symmetric in vec, no sign issue.)
    for (int j = 0; j < n; ++j) {
      for (int a = 0; a < 3; ++a) s += t.sd[a](i, j) * m.dipm(a, j);
      for (int p = 0; p < 6; ++p) s += t.sq[p](i, j) * m.qp(p, j);
    }
    out.vs(i) = s;

    for (int a = 0; a < 3; ++a) {
      double v = 0.0;
      for (int j = 0; j < n; ++j) {
        v += t.sd[a](i, j) * q(j);
        for (int b = 0; b < 3; ++b) v += t.dd[a][b](i, j) * m.dipm(b, j);
      }
      out.vd(a, i) = v;
    }

    double tmp_xx = 0.0, tmp_xy = 0.0, tmp_yy = 0.0;
    double tmp_xz = 0.0, tmp_yz = 0.0, tmp_zz = 0.0;
    for (int j = 0; j < n; ++j) {
      const double qj = q(j);
      tmp_xx += t.sq[qp_xx](i, j) * qj;
      tmp_xy += t.sq[qp_xy](i, j) * qj;
      tmp_yy += t.sq[qp_yy](i, j) * qj;
      tmp_xz += t.sq[qp_xz](i, j) * qj;
      tmp_yz += t.sq[qp_yz](i, j) * qj;
      tmp_zz += t.sq[qp_zz](i, j) * qj;
    }
    out.vq(qpint_xx, i) = tmp_xx;
    out.vq(qpint_yy, i) = tmp_yy;
    out.vq(qpint_zz, i) = tmp_zz;
    out.vq(qpint_xy, i) = tmp_xy;
    out.vq(qpint_xz, i) = tmp_xz;
    out.vq(qpint_yz, i) = tmp_yz;
  }

  // On-site polariz (kernel) potential — matches tblite's
  // `get_kernel_potential` (multipole.f90 lines 271-291):
  //   v_dpkernel(i) = 2·dipKernel·dipm(i)
  //   v_qpkernel(i) = 2·quadKernel·qpat(i) · mpscale,  mpscale=[1,2,1,2,2,1]
  //
  // Sign: our CAMM stores dipm = -P·atom_centered_AO_dipole (electron sign);
  // for compatibility with our `apply_anisotropic_h1_periodic` (and the same
  // sign convention used for molecular benchmarks), we ADD with minus sign
  // here so the H1 contribution comes out matching the gauge-corrected
  // formulation at the molecular limit.
  // Layout: vq is stored in qpint order (xx, yy, zz, xy, xz, yz). The CAMM
  // mom.qp is in (xx, xy, yy, xz, yz, zz). Off-diag positions get scale 2.
  //   qpint:  0=xx 1=yy 2=zz 3=xy 4=xz 5=yz
  //   mom.qp: 0=xx 1=xy 2=yy 3=xz 4=yz 5=zz
  // Mapping (qpint_idx → mom.qp idx, mpscale):
  //   (0=xx → 0, 1), (1=yy → 2, 1), (2=zz → 5, 1)
  //   (3=xy → 1, 2), (4=xz → 3, 2), (5=yz → 4, 2)
  static constexpr int qp_from_qpint[6] = {0, 2, 5, 1, 3, 4};
  static constexpr double mpscale_qpint[6] = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0};
  // tblite get_kernel_potential (multipole.f90 lines 329-349):
  //   vd(:, i) += 2·dkernel·dipm(:, i)
  //   vq(:, i) += 2·qkernel·qpat(:, i) · mpscale, mpscale=[1,2,1,2,2,1]
  for (int i = 0; i < n; ++i) {
    const auto *e = params.element(atoms[i].atomic_number);
    for (int l1 = 0; l1 < 3; ++l1) {
      out.vd(l1, i) += 2.0 * e->dip_kernel * m.dipm(l1, i);
    }
    for (int p = 0; p < 6; ++p) {
      out.vq(p, i) += 2.0 * e->quad_kernel * m.qp(qp_from_qpint[p], i)
                      * mpscale_qpint[p];
    }
  }

  return out;
}

// Periodic anisotropic potentials via direct-space Ewald-corrected lattice
// sum. This is a verbatim port of the molecular `anisotropic_potentials`
// formula with `g3 → tmp3 = fdmp3·g3 + e1` and `g5 → tmp5 = fdmp5·g5 + e2`,
// and an outer loop over lattice translations T.
//
// Brute-force per SCC iteration (no pre-built tensor optimization), so cost
// is O(N² × N_translations). For typical periodic GFN2 cells this is fast.
//
// Reciprocal contribution to potentials is NOT included — for the gauge-
// corrected formula structure, the reciprocal kernel for each gauge term
// would need separate derivation. Since the Ewald-erfc screening makes the
// gauge kernel decay exponentially for r > 1/α, the missing reciprocal
// piece is small for typical molecular crystals. At the molecular limit
// (very large cell) this reduces to molecular `anisotropic_potentials`.
//
// THIS is the function to use as a drop-in replacement for the molecular
// `anisotropic_potentials` in the periodic SCC's H1 shift.
AnisotropicPotentials anisotropic_potentials_ewald_gauge_corrected(
    const std::vector<core::Atom> &atoms, const Vec &q, const Vec &mp_radii,
    const CammMoments &m, const MultipolePairTensors &t,
    const Gfn2Parameters &params) {
  const int n = static_cast<int>(atoms.size());
  const auto &g = params.globals();
  const double k3 = g.aesdmp3;
  const double k5 = g.aesdmp5;
  const double alpha = t.alpha;
  const double alpha2 = alpha * alpha;
  const double alpha3 = alpha * alpha2;
  const double real_cutoff2 = t.real_cutoff * t.real_cutoff;
  const double eps2 = 1e-20;

  AnisotropicPotentials out;
  out.vs = Vec::Zero(n);
  out.vd = Mat3N::Zero(3, n);
  out.vq = Mat::Zero(6, n);

  // Verbatim port of the molecular `anisotropic_potentials` per-(i, j) loop,
  // wrapped in an outer T-loop so each (i, j, T) image is summed. tmp3, tmp5
  // are computed inline using the Ewald kernel.
  for (int i = 0; i < n; ++i) {
    const double rai[3] = {atoms[i].x, atoms[i].y, atoms[i].z};
    double stmp = 0.0;
    double dtmp[3] = {0.0, 0.0, 0.0};
    double qtmp[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (const auto &im : t.images) {
      for (int j = 0; j < n; ++j) {
        const double rbj[3] = {atoms[j].x + im.t_bohr.x(),
                               atoms[j].y + im.t_bohr.y(),
                               atoms[j].z + im.t_bohr.z()};
        const double dra[3] = {rai[0] - rbj[0], rai[1] - rbj[1],
                               rai[2] - rbj[2]};
        const double r2 = dra[0] * dra[0] + dra[1] * dra[1] + dra[2] * dra[2];
        if (r2 < eps2 || r2 > real_cutoff2) continue;
        const double r1 = std::sqrt(r2);
        const double g1 = 1.0 / r1;
        const double g3 = g1 * g1 * g1;
        const double g5 = g3 * g1 * g1;
        const double rco = 0.5 * (mp_radii(i) + mp_radii(j));
        const double rco_over_r = rco * g1;
        const double fdmp3 =
            1.0 / (1.0 + 6.0 * std::pow(rco_over_r, k3));
        const double fdmp5 =
            1.0 / (1.0 + 6.0 * std::pow(rco_over_r, k5));
        const double arg = r1 * alpha;
        const double expt = std::exp(-arg * arg) / kSqrtPi;
        const double erft = -std::erf(arg) * g1;
        const double e1 = g1 * g1 * (erft + 2.0 * expt * alpha);
        const double e2 =
            g1 * g1 * (e1 + 4.0 * expt * alpha2 * alpha / 3.0);
        const double tmp3 = fdmp3 * g3 + e1;
        const double tmp5 = fdmp5 * g5 + e2;

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
            dum5a -= m.qp(kl_to_qp_local[l1][l2], j) * dra[l1] * dra[l2];
            dum5a -= 1.5 * q(j) * dra[l1] * dra[l2] * rai[l1] * rai[l2];
            if (l2 >= l1) continue;
            const int ki = qpint_idx(l1, l2);
            qtmp[ki] -= 3.0 * q(j) * tmp5 * dra[l2] * dra[l1];
          }
          qtmp[l1] -= 1.5 * q(j) * tmp5 * dra[l1] * dra[l1];
        }

        const double dum3a = -t1a * q(j) - t2a;
        dum5a += t3a * r2ab - 3.0 * t1a * t2a + 0.5 * q(j) * r2a * r2ab;
        stmp += dum5a * tmp5 + dum3a * tmp3;

        for (int l1 = 0; l1 < 3; ++l1) {
          const double dd3 = dra[l1] * q(j);
          const double dd5 = 3.0 * dra[l1] * t2a -
                             r2ab * m.dipm(l1, j) -
                             q(j) * r2ab * rai[l1] +
                             3.0 * q(j) * dra[l1] * t1a;
          dtmp[l1] += dd3 * tmp3 + dd5 * tmp5;
          qtmp[l1] += 0.5 * r2ab * q(j) * tmp5;
        }
      }
    }

    // CT (on-site polarization) — same as molecular, applied once per atom.
    const auto *e = params.element(atoms[i].atomic_number);
    const double qs1 = 2.0 * e->dip_kernel;
    const double qs2 = 6.0 * e->quad_kernel;
    double t3a = 0.0, t2a = 0.0;
    for (int l1 = 0; l1 < 3; ++l1) {
      t3a += rai[l1] * m.dipm(l1, i) * qs1;
      dtmp[l1] -= qs1 * m.dipm(l1, i);
      for (int l2 = 0; l2 < l1; ++l2) {
        const int ll = kl_to_qp_local[l1][l2];
        const int ki = qpint_idx(l1, l2);
        qtmp[ki] -= m.qp(ll, i) * qs2;
        t3a -= rai[l1] * rai[l2] * m.qp(ll, i) * qs2;
        dtmp[l1] += rai[l2] * m.qp(ll, i) * qs2;
        dtmp[l2] += rai[l1] * m.qp(ll, i) * qs2;
      }
      const int ll_diag = kl_to_qp_local[l1][l1];
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

} // namespace occ::xtb
