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

namespace {

// Inline gab3 / gab5 for a single (i, j, r) — same form as DampedCoulomb.
inline void gab35(double rx, double ry, double rz, double rco, double k3,
                  double k5, double &g3, double &g5) {
  const double r2 = rx * rx + ry * ry + rz * rz;
  const double r = std::sqrt(r2);
  const double rinv = 1.0 / r;
  const double rcoinvr = rco * rinv;
  const double damp3 = 1.0 / (1.0 + 6.0 * std::pow(rcoinvr, k3));
  const double damp5 = 1.0 / (1.0 + 6.0 * std::pow(rcoinvr, k5));
  const double r3inv = rinv * rinv * rinv;
  g3 = damp3 * r3inv;
  g5 = damp5 * r3inv * rinv * rinv;
}

// Pair contribution to the AES energy for a given inter-atom vector
// r = R_j_image - R_i (so signs match the molecular code's `rij`).
inline void
aes_pair_energy(int i, int j, double rx, double ry, double rz, double g3,
                double g5, const Vec &q, const CammMoments &m,
                double &e01, double &e02, double &e11) {
  const double q_i = q(i), q_j = q(j);
  const double rij[3] = {rx, ry, rz};
  const double r2 = rx * rx + ry * ry + rz * rz;
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
  e01 += ed * g3;
  e02 += eq * g5;
  e11 += edd * g5;
}

} // namespace

AnisotropicEnergy anisotropic_energy_periodic(
    const std::vector<core::Atom> &atoms,
    const std::vector<LatticeImage> &images, const Vec &q,
    const Vec &mp_radii, const CammMoments &m,
    const Gfn2Parameters &params) {
  const int nat = static_cast<int>(atoms.size());
  const auto &g = params.globals();
  const double k3 = g.aesdmp3;
  const double k5 = g.aesdmp5;

  // 1. On-site polarization (unchanged from molecular).
  double epol = 0.0;
  for (int i = 0; i < nat; ++i) {
    const auto *e = params.element(atoms[i].atomic_number);
    double dd = 0.0;
    for (int k = 0; k < 3; ++k) dd += m.dipm(k, i) * m.dipm(k, i);
    double qq = 0.0;
    for (int k = 0; k < 3; ++k) {
      for (int l = 0; l < 3; ++l) {
        const int idx = kl_to_qp[k][l];
        qq += m.qp(idx, i) * m.qp(idx, i);
      }
    }
    epol += e->dip_kernel * dd + e->quad_kernel * qq;
  }

  // 2. Pair interactions — lattice-summed.
  double e01 = 0.0, e02 = 0.0, e11 = 0.0;
  for (const auto &im : images) {
    const bool is_T0 =
        (im.hkl(0) == 0 && im.hkl(1) == 0 && im.hkl(2) == 0);
    const double w = is_T0 ? 1.0 : 0.5;
    for (int i = 0; i < nat; ++i) {
      const int j_max = is_T0 ? i : nat;
      for (int j = 0; j < j_max; ++j) {
        const double rx = atoms[j].x - atoms[i].x + im.t_bohr.x();
        const double ry = atoms[j].y - atoms[i].y + im.t_bohr.y();
        const double rz = atoms[j].z - atoms[i].z + im.t_bohr.z();
        const double rco = 0.5 * (mp_radii(i) + mp_radii(j));
        double g3, g5;
        gab35(rx, ry, rz, rco, k3, k5, g3, g5);
        double pe01 = 0.0, pe02 = 0.0, pe11 = 0.0;
        aes_pair_energy(i, j, rx, ry, rz, g3, g5, q, m, pe01, pe02, pe11);
        e01 += w * pe01;
        e02 += w * pe02;
        e11 += w * pe11;
      }
      // Self-image: i = j, T != 0. Contributes only for non-central T.
      if (!is_T0) {
        const double rx = im.t_bohr.x();
        const double ry = im.t_bohr.y();
        const double rz = im.t_bohr.z();
        const double rco = mp_radii(i);
        double g3, g5;
        gab35(rx, ry, rz, rco, k3, k5, g3, g5);
        double pe01 = 0.0, pe02 = 0.0, pe11 = 0.0;
        aes_pair_energy(i, i, rx, ry, rz, g3, g5, q, m, pe01, pe02, pe11);
        e01 += w * pe01;
        e02 += w * pe02;
        e11 += w * pe11;
      }
    }
  }

  AnisotropicEnergy out;
  out.aes = e01 + e02 + e11;
  out.polariz = epol;
  return out;
}

AnisotropicPotentials anisotropic_potentials_periodic(
    const std::vector<core::Atom> &atoms,
    const std::vector<LatticeImage> &images, const Vec &q,
    const Vec &mp_radii, const CammMoments &m,
    const Gfn2Parameters &params) {
  const int nat = static_cast<int>(atoms.size());
  const auto &g = params.globals();
  const double k3 = g.aesdmp3;
  const double k5 = g.aesdmp5;

  AnisotropicPotentials out;
  out.vs = Vec::Zero(nat);
  out.vd = Mat3N::Zero(3, nat);
  out.vq = Mat::Zero(6, nat);

  // Pair-source contributions to potentials at atom i.
  for (int i = 0; i < nat; ++i) {
    const double rai[3] = {atoms[i].x, atoms[i].y, atoms[i].z};
    double stmp = 0.0;
    double dtmp[3] = {0.0, 0.0, 0.0};
    double qtmp[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (const auto &im : images) {
      const bool is_T0 =
          (im.hkl(0) == 0 && im.hkl(1) == 0 && im.hkl(2) == 0);
      for (int j = 0; j < nat; ++j) {
        // Skip the (i, j=i, T=0) self interaction (gab → 0/0).
        if (is_T0 && j == i) continue;
        const double rbj[3] = {atoms[j].x + im.t_bohr.x(),
                               atoms[j].y + im.t_bohr.y(),
                               atoms[j].z + im.t_bohr.z()};
        const double dra[3] = {rai[0] - rbj[0], rai[1] - rbj[1],
                               rai[2] - rbj[2]};
        const double rco = 0.5 * (mp_radii(i) + mp_radii(j));
        double g3, g5;
        gab35(dra[0], dra[1], dra[2], rco, k3, k5, g3, g5);

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
            const int ki = qpint_idx(l1, l2);
            qtmp[ki] -= 3.0 * q(j) * g5 * dra[l2] * dra[l1];
          }
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
    }

    // CT (on-site polarization) terms — same as molecular.
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

void apply_anisotropic_h1(Mat &H, const Mat &S, const MatTriple &D,
                          const std::array<Mat, 6> &Q,
                          const std::vector<int> &bf_to_atom,
                          const AnisotropicPotentials &pot) {
  const int nbf = static_cast<int>(H.rows());
  for (int mu = 0; mu < nbf; ++mu) {
    const int ii = bf_to_atom[mu];
    for (int nu = 0; nu < nbf; ++nu) {
      const int jj = bf_to_atom[nu];
      // All three anisotropic contributions enter with the same sign (xtb
      // uses H += 0.5 * <op> * (v_μ + v_ν) for vs/vd/vq).
      double eh1 = 0.5 * S(mu, nu) * (pot.vs(ii) + pot.vs(jj));
      eh1 += 0.5 * (D.x(mu, nu) * (pot.vd(0, ii) + pot.vd(0, jj)) +
                    D.y(mu, nu) * (pot.vd(1, ii) + pot.vd(1, jj)) +
                    D.z(mu, nu) * (pot.vd(2, ii) + pot.vd(2, jj)));
      // vq is in xtb's qpint order {xx, yy, zz, xy, xz, yz}; my Q matrices
      // are in {xx, xy, xz, yy, yz, zz} order — remap via q_from_qpint.
      for (int l = 0; l < 6; ++l) {
        eh1 += 0.5 * Q[q_from_qpint[l]](mu, nu) *
               (pot.vq(l, ii) + pot.vq(l, jj));
      }
      H(mu, nu) += eh1;
    }
  }
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
      H(mu, nu) += eh1;
    }
  }
}

} // namespace occ::xtb
