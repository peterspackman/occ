#include <occ/xtb/camm.h>

namespace occ::xtb {

// Mapping from my quadrupole_ao_matrices index order {xx, xy, xz, yy, yz, zz}
// to xtb's qp storage order {xx, xy, yy, xz, yz, zz}.
//   xtb_qp[0]=xx ← Q[0]
//   xtb_qp[1]=xy ← Q[1]
//   xtb_qp[2]=yy ← Q[3]
//   xtb_qp[3]=xz ← Q[2]
//   xtb_qp[4]=yz ← Q[4]
//   xtb_qp[5]=zz ← Q[5]
namespace {

// xtb's qp index for the diagonal (k, k), 0-based:
//   k=0 (x→xx) → 0
//   k=1 (y→yy) → 2
//   k=2 (z→zz) → 5
constexpr int diag_qp_idx[3] = {0, 2, 5};

// xtb's qp index for off-diagonal (k, l) with k > l, 0-based:
//   (k=1, l=0) xy → 1
//   (k=2, l=0) xz → 3
//   (k=2, l=1) yz → 4
constexpr int off_qp_idx[3][3] = {
    {-1, -1, -1},
    { 1, -1, -1},
    { 3,  4, -1},
};

// Index into my Q[] for (k, l) with k > l:
constexpr int q_off_idx[3][3] = {
    {-1, -1, -1}, // k=0
    { 1, -1, -1}, // k=1, l=0 → xy → Q[1]
    { 2,  4, -1}, // k=2, l=0 → xz → Q[2];  k=2, l=1 → yz → Q[4]
};

// Index into my Q[] for diagonal (k, k):
constexpr int q_diag_idx[3] = {0, 3, 5}; // xx → Q[0], yy → Q[3], zz → Q[5]

} // namespace

CammMoments compute_camm_moments(const std::vector<core::Atom> &atoms,
                                 const std::vector<int> &bf_to_atom,
                                 const Mat &P, const Mat &S,
                                 const MatTriple &D,
                                 const std::array<Mat, 6> &Q) {
  const int nat = static_cast<int>(atoms.size());
  const int nao = static_cast<int>(P.rows());

  CammMoments out;
  out.dipm = Mat3N::Zero(3, nat);
  out.qp = Mat::Zero(6, nat);

  // Per-atom Cartesian coordinates (Bohr).
  Mat3N R(3, nat);
  for (int a = 0; a < nat; ++a) {
    R(0, a) = atoms[a].x;
    R(1, a) = atoms[a].y;
    R(2, a) = atoms[a].z;
  }

  // Per-Cartesian-component dipole AO matrix lookup.
  const Mat *D_k[3] = {&D.x, &D.y, &D.z};

  // 1. Off-diagonal (i > j) AO pair contribution. Each pair contributes to
  //    BOTH atom_i and atom_j (with their respective atom positions).
  for (int i = 0; i < nao; ++i) {
    const int ii = bf_to_atom[i];
    for (int j = 0; j < i; ++j) {
      const int jj = bf_to_atom[j];
      const double pij = P(j, i);
      const double ps = pij * S(j, i);

      double xa[3] = {R(0, ii), R(1, ii), R(2, ii)};
      double xb[3] = {R(0, jj), R(1, jj), R(2, jj)};

      double pdm[3];
      for (int k = 0; k < 3; ++k) {
        pdm[k] = pij * (*D_k[k])(j, i);
      }

      // Dipoles
      for (int k = 0; k < 3; ++k) {
        const double tii = xa[k] * ps - pdm[k];
        const double tjj = xb[k] * ps - pdm[k];
        out.dipm(k, ii) += tii;
        out.dipm(k, jj) += tjj;
      }

      // Quadrupoles
      for (int k = 0; k < 3; ++k) {
        // Off-diagonal (k > l)
        for (int l = 0; l < k; ++l) {
          const double pqm = pij * Q[q_off_idx[k][l]](j, i);
          const double tii =
              pdm[k] * xa[l] + pdm[l] * xa[k] - xa[l] * xa[k] * ps - pqm;
          const double tjj =
              pdm[k] * xb[l] + pdm[l] * xb[k] - xb[l] * xb[k] * ps - pqm;
          const int idx = off_qp_idx[k][l];
          out.qp(idx, ii) += tii;
          out.qp(idx, jj) += tjj;
        }
        // Diagonal (k == l)
        const double pqm = pij * Q[q_diag_idx[k]](j, i);
        const double tii = 2.0 * pdm[k] * xa[k] - xa[k] * xa[k] * ps - pqm;
        const double tjj = 2.0 * pdm[k] * xb[k] - xb[k] * xb[k] * ps - pqm;
        const int idx = diag_qp_idx[k];
        out.qp(idx, ii) += tii;
        out.qp(idx, jj) += tjj;
      }
    }
  }

  // 2. Diagonal (i == j) AO contribution.
  for (int i = 0; i < nao; ++i) {
    const int ii = bf_to_atom[i];
    const double pij = P(i, i);
    const double ps = pij * S(i, i);
    double xa[3] = {R(0, ii), R(1, ii), R(2, ii)};

    double pdm[3];
    for (int k = 0; k < 3; ++k) {
      pdm[k] = pij * (*D_k[k])(i, i);
    }
    for (int k = 0; k < 3; ++k) {
      out.dipm(k, ii) += xa[k] * ps - pdm[k];
    }
    for (int k = 0; k < 3; ++k) {
      for (int l = 0; l < k; ++l) {
        const double pqm = pij * Q[q_off_idx[k][l]](i, i);
        const double tii =
            pdm[k] * xa[l] + pdm[l] * xa[k] - xa[l] * xa[k] * ps - pqm;
        out.qp(off_qp_idx[k][l], ii) += tii;
      }
      const double pqm = pij * Q[q_diag_idx[k]](i, i);
      const double tii = 2.0 * pdm[k] * xa[k] - xa[k] * xa[k] * ps - pqm;
      out.qp(diag_qp_idx[k], ii) += tii;
    }
  }

  // 3. Remove trace from quadrupoles, scaling by 3/2 (xtb convention).
  //    Indices: 0=xx, 2=yy, 5=zz are the diagonals.
  for (int a = 0; a < nat; ++a) {
    const double tr = 0.5 * (out.qp(0, a) + out.qp(2, a) + out.qp(5, a));
    for (int k = 0; k < 6; ++k)
      out.qp(k, a) *= 1.5;
    out.qp(0, a) -= tr;
    out.qp(2, a) -= tr;
    out.qp(5, a) -= tr;
  }

  return out;
}

CammMoments compute_camm_moments_periodic(
    const std::vector<core::Atom> &atoms,
    const std::vector<int> &bf_to_atom,
    const Mat &P,
    const MatTriple &D_ket, const MatTriple &D_bra,
    const std::array<Mat, 6> &Q_ket, const std::array<Mat, 6> &Q_bra) {
  const int nat = static_cast<int>(atoms.size());
  const int nao = static_cast<int>(P.rows());

  CammMoments out;
  out.dipm = Mat3N::Zero(3, nat);
  out.qp = Mat::Zero(6, nat);

  // Match tblite's `get_mulliken_atomic_multipoles` (mulliken.f90 lines 76-86):
  //   for iao = 1..nao:
  //     mpat[atom_of(iao)] -= Σ_jao P(jao, iao) · mpmat(jao, iao)
  // where mpmat is centered on atom_of(iao) (the column index). In our
  // notation, that's D_bra (atom-of-col-centered, with image-T offset baked
  // in). Use ONLY Bra over the full nao × nao loop — the Ket version
  // (centered on row atom) gives the equivalent partition by index swap and
  // would double-count if added to Bra. Each ordered pair contributes once
  // to atom_of(col); the symmetric (μ ↔ ν) traversal of P × D_bra captures
  // both atom-A and atom-B contributions for off-diagonal pairs.
  // Sign convention: m.dipm = -atomic_dipole (electron sign).
  const Mat *D_bra_k[3] = {&D_bra.x, &D_bra.y, &D_bra.z};

  // Q layout: (0:xx, 1:xy, 2:xz, 3:yy, 4:yz, 5:zz) from quadrupole_ao_matrices.
  // out.qp layout: (0:xx, 1:xy, 2:yy, 3:xz, 4:yz, 5:zz) — remap.
  static constexpr int qp_from_q[6] = {0, 1, 3, 2, 4, 5};

  for (int mu = 0; mu < nao; ++mu) {
    for (int nu = 0; nu < nao; ++nu) {
      const int A_nu = bf_to_atom[nu];
      const double pmunu = P(mu, nu);
      for (int k = 0; k < 3; ++k) {
        out.dipm(k, A_nu) -= pmunu * (*D_bra_k[k])(mu, nu);
      }
      for (int kk = 0; kk < 6; ++kk) {
        out.qp(qp_from_q[kk], A_nu) -= pmunu * Q_bra[kk](mu, nu);
      }
    }
  }
  // Suppress unused-variable warning for D_ket / Q_ket — kept in the API
  // because `apply_anisotropic_h1_periodic` needs them on the row side.
  (void)D_ket;
  (void)Q_ket;

  // Quadrupole trace removal (xtb convention) — match molecular variant.
  // Diagonals in out.qp layout are at indices 0=xx, 2=yy, 5=zz.
  for (int a = 0; a < nat; ++a) {
    const double tr = 0.5 * (out.qp(0, a) + out.qp(2, a) + out.qp(5, a));
    for (int kk = 0; kk < 6; ++kk) out.qp(kk, a) *= 1.5;
    out.qp(0, a) -= tr;
    out.qp(2, a) -= tr;
    out.qp(5, a) -= tr;
  }
  return out;
}

} // namespace occ::xtb
