#include <occ/dma/binomial.h>
#include <occ/dma/shiftq.h>
#include <occ/dma/solid_harmonics.h>

namespace occ::dma {

int estimate_largest_transferred_multipole(Eigen::Ref<const Vec3> pos,
                                           const Mult &mult, int l, int m1, int m2,
                                           double eps) {
  if (eps == 0.0 || m2 < 1)
    return m2;
  double r2 = 4.0 * pos.squaredNorm();
  int n = 0;
  double a = 0.0;

  if (l == 0)
    a = mult.Q00() * mult.Q00();

  for (int k = std::max(1, l); k <= m2; k++) {
    a *= r2;
    if (k <= m1) {
      int t1 = k * k + 1; // Use 1-based indexing for temporary calculations
      int t2 = (k + 1) * (k + 1);
      for (int i = t1; i <= t2; i++) {
        a += mult.q(i - 1) * mult.q(i - 1); // Adjust index when accessing q1
      }
    }
    if (a > eps)
      n = k;
  }
  return n;
}

Mat get_cplx_sh(Eigen::Ref<const Vec3> pos, int N) {
  constexpr double RTHALF = 0.7071067811865475244;

  size_t num_components = (N + 1) * (N + 1) + 1;
  // Evaluate solid harmonics in real form
  Vec r = Vec::Zero(num_components); // One extra element
  solid_harmonics(pos, N, r);

  // Construct complex solid harmonics RC
  // RC[i][0] = real part, RC[i][1] = imaginary part
  Mat rc = Mat::Zero(num_components, 2); // One extra element/row
  rc(1, 0) = r(1);                       // Now we can use 1-based indexing
  rc(1, 1) = 0.0;

  for (int k = 1; k <= N; k++) {
    int kb = k * k + k + 1; // Use the same formula as Fortran
    int km = k * k + 1;

    rc(kb, 0) = r(km);
    rc(kb, 1) = 0.0;
    km++;

    double s = RTHALF;
    for (int m = 1; m <= k; m++) {
      s = -s;
      rc(kb - m, 0) = RTHALF * r(km);
      rc(kb - m, 1) = -RTHALF * r(km + 1);
      rc(kb + m, 0) = s * r(km);
      rc(kb + m, 1) = s * r(km + 1);
      km += 2;
    }
  }
  return rc;
}


Mat get_cplx_mults(const Mult &mult, int l1, int m1, int N) {
  constexpr double RTHALF = 0.7071067811865475244;
  size_t num_components = (N + 1) * (N + 1) + 1;
  int k1 = std::max(1, l1);

  Mat qc = Mat::Zero(num_components, 2); // One extra element/row
  if (l1 == 0) {
    qc(1, 0) = mult.Q00(); // Use index 1 for first element
    qc(1, 1) = 0.0;
  }

  if (m1 > 0) {
    for (int k = k1; k <= m1; k++) {
      int kb = k * k + k + 1;
      int km = k * k + 1;

      qc(kb, 0) = mult.q(km - 1); // Adjust index when accessing mult
      qc(kb, 1) = 0.0;
      km++;

      double s = RTHALF;
      for (int m = 1; m <= k; m++) {
        s = -s;
        qc(kb - m, 0) = RTHALF * mult.q(km - 1); // Adjust index
        qc(kb - m, 1) = -RTHALF * mult.q(km);    // Adjust index
        qc(kb + m, 0) = s * mult.q(km - 1);      // Adjust index
        qc(kb + m, 1) = s * mult.q(km);          // Adjust index
        km += 2;
      }
    }
  }

  return qc;
}

/**
 * @brief Shift multipoles from one point to another
 *
 * Implements the shiftq subroutine from the original GDMA code.
 * Shifts multipoles q1 (of ranks l1 through m1) from point (x,y,z) to
 * the origin (0,0,0) and adds them to multipole expansion q2 (keeping
 * ranks up to m2).
 */
void shiftq(const Mult &q1, int l1, int m1, Mult &q2, int m2,
            Eigen::Ref<const Vec3> pos) {
  // Constants
  constexpr double RTHALF = 0.7071067811865475244;
  constexpr double eps = 0.0; // No early termination
  const double &x = pos.x();
  const double &y = pos.y();
  const double &z = pos.z();

  // Return if parameters are invalid
  if (l1 > m1 || l1 > m2)
    return;

  // Estimate largest significant transferred multipole
  int N = estimate_largest_transferred_multipole(pos, q1, l1, m1, m2, eps);
  int k1 = std::max(1, l1);
  size_t num_components = (N + 1) * (N + 1) + 1;

  Mat rc = get_cplx_sh(pos, N);
  Mat qc = get_cplx_mults(q1, l1, m1, N);

  // Construct shifted complex multipoles QZ (only for non-negative M)
  Mat qz = Mat::Zero(num_components, 2); // Use dynamic size based on N

  if (l1 == 0) {
    qz(1, 0) = qc(1, 0); // Use index 1 for first element
    qz(1, 1) = qc(1, 1);
  }

  // Create a BinomialCoefficients object for rtbinom calculations
  BinomialCoefficients binomials(20);

  for (int l = k1; l <= N; l++) {
    int kmax = std::min(l, m1);
    int lb = l * l + l + 1;
    int lm = lb;

    for (int m = 0; m <= l; m++) {
      qz(lm, 0) = 0.0;
      qz(lm, 1) = 0.0;

      if (l1 == 0) {
        qz(lm, 0) = qc(1, 0) * rc(lm, 0) - qc(1, 1) * rc(lm, 1);
        qz(lm, 1) = qc(1, 0) * rc(lm, 1) + qc(1, 1) * rc(lm, 0);
      }

      for (int k = k1; k <= kmax; k++) {
        int qmin = std::max(-k, k - l + m);
        int qmax = std::min(k, l - k + m);
        int kb = k * k + k + 1;
        int jb = (l - k) * (l - k) + (l - k) + 1;

        for (int qq = qmin; qq <= qmax; qq++) {
          double factor = binomials.sqrt_binomial(l + m, k + qq) *
                          binomials.sqrt_binomial(l - m, k - qq);

          qz(lm, 0) += factor * (qc(kb + qq, 0) * rc(jb + m - qq, 0) -
                                 qc(kb + qq, 1) * rc(jb + m - qq, 1));
          qz(lm, 1) += factor * (qc(kb + qq, 0) * rc(jb + m - qq, 1) +
                                 qc(kb + qq, 1) * rc(jb + m - qq, 0));
        }
      }

      lm++;
    }
  }

  // Construct real multipoles and add to Q2
  if (l1 == 0)
    q2.q(0) += qz(1, 0); // Map from temp array index 1 to q2 index 0

  for (int k = k1; k <= N; k++) {
    int kb = k * k + k + 1;
    int km = k * k + 1;

    q2.q(km - 1) += qz(kb, 0); // Adjust index when writing to q2

    double s = 1.0 / RTHALF;
    km++;

    for (int m = 1; m <= k; m++) {
      s = -s;
      q2.q(km - 1) += s * qz(kb + m, 0); // Adjust index when writing to q2
      q2.q(km) += s * qz(kb + m, 1);     // Adjust index when writing to q2
      km += 2;
    }
  }
}

} // namespace occ::dma
