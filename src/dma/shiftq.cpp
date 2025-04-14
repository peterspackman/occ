#include <occ/dma/binomial.h>
#include <occ/dma/shiftq.h>
#include <occ/dma/solid_harmonics.h>

namespace occ::dma {

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
  constexpr double EPS = 0.0; // No early termination
  const double &x = pos.x();
  const double &y = pos.y();
  const double &z = pos.z();

  fmt::print(" shift\n{:2d}{:2d}{:2d}\n", l1, m1, m2);
  // Return if parameters are invalid
  if (l1 > m1 || l1 > m2)
    return;

  // Estimate largest significant transferred multipole
  int n2 = m2;
  int k1 = std::max(1, l1);

  if (EPS != 0.0 && m2 > 0) {
    double r2 = 4.0 * (x * x + y * y + z * z);
    n2 = 0;
    double a = 0.0;

    if (l1 == 0)
      a = q1.Q00() * q1.Q00();

    for (int k = k1; k <= m2; k++) {
      a *= r2;
      if (k <= m1) {
        int t1 = k * k + 1;  // Use 1-based indexing for temporary calculations
        int t2 = (k + 1) * (k + 1);
        for (int i = t1; i <= t2; i++) {
          a += q1.q(i-1) * q1.q(i-1);  // Adjust index when accessing q1
        }
      }
      if (a > EPS)
        n2 = k;
    }
  }

  // Evaluate solid harmonics in real form
  Vec r = Vec::Zero(122);  // One extra element
  solid_harmonics(pos, n2, r);
  
  // Construct complex solid harmonics RC
  // RC[i][0] = real part, RC[i][1] = imaginary part
  Mat rc = Mat::Zero(122, 2);  // One extra element/row
  rc(1, 0) = r(1);  // Now we can use 1-based indexing
  rc(1, 1) = 0.0;

  for (int k = 1; k <= n2; k++) {
    int kb = k * k + k + 1;  // Use the same formula as Fortran
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

  // Construct complex multipoles QC corresponding to original real multipoles
  Mat qc = Mat::Zero(122, 2);  // One extra element/row

  if (l1 == 0) {
    qc(1, 0) = q1.Q00();  // Use index 1 for first element
    qc(1, 1) = 0.0;
  }

  if (m1 > 0) {
    for (int k = k1; k <= m1; k++) {
      int kb = k * k + k + 1;
      int km = k * k + 1;

      qc(kb, 0) = q1.q(km-1);  // Adjust index when accessing q1
      qc(kb, 1) = 0.0;
      km++;

      double s = RTHALF;
      for (int m = 1; m <= k; m++) {
        s = -s;
        qc(kb - m, 0) = RTHALF * q1.q(km-1);  // Adjust index
        qc(kb - m, 1) = -RTHALF * q1.q(km);   // Adjust index
        qc(kb + m, 0) = s * q1.q(km-1);       // Adjust index
        qc(kb + m, 1) = s * q1.q(km);         // Adjust index
        km += 2;
      }
    }
  }

  fmt::print("qc\n");
  fmt::print("{:5d}{:5d}\n", k1, m1);
  fmt::print("{}\n", format_matrix(qc.row(1), "{:20.12f}"));
  fmt::print("{}\n", format_matrix(qc.row(2), "{:20.12f}"));
  fmt::print("{}\n", format_matrix(qc.row(3), "{:20.12f}"));
  fmt::print("{}\n", format_matrix(qc.row(4), "{:20.12f}"));
  fmt::print("{}\n", format_matrix(qc.row(5), "{:20.12f}"));
  fmt::print("{}\n", format_matrix(qc.row(6), "{:20.12f}"));

  // Construct shifted complex multipoles QZ (only for non-negative M)
  Mat qz = Mat::Zero(122, 2);  // One extra element/row

  if (l1 == 0) {
    qz(1, 0) = qc(1, 0);  // Use index 1 for first element
    qz(1, 1) = qc(1, 1);
  }

  // Create a BinomialCoefficients object for rtbinom calculations
  BinomialCoefficients binomials(20);

  for (int l = k1; l <= n2; l++) {
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
        fmt::print("{:2d}{:2d}{:2d}{:2d}{:2d}\n", l, m, lm, k1, kmax);
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

  fmt::print("qz\n");
  fmt::print("{:5d}{:5d}\n", k1, m1);
  fmt::print("{}\n", format_matrix(qz.row(1), "{:20.12f}"));
  fmt::print("{}\n", format_matrix(qz.row(2), "{:20.12f}"));
  fmt::print("{}\n", format_matrix(qz.row(3), "{:20.12f}"));
  fmt::print("{}\n", format_matrix(qz.row(4), "{:20.12f}"));
  fmt::print("{}\n", format_matrix(qz.row(5), "{:20.12f}"));
  fmt::print("{}\n", format_matrix(qz.row(6), "{:20.12f}"));


  // Construct real multipoles and add to Q2
  if (l1 == 0)
    q2.q(0) += qz(1, 0);  // Map from temp array index 1 to q2 index 0

  for (int k = k1; k <= n2; k++) {
    int kb = k * k + k + 1;
    int km = k * k + 1;

    fmt::print("{:20.12f}{:20.12f}\n", q2.q(km-1), qz(kb, 0));
    q2.q(km-1) += qz(kb, 0);  // Adjust index when writing to q2

    double s = 1.0 / RTHALF;
    km++;

    for (int m = 1; m <= k; m++) {
      s = -s;
      q2.q(km-1) += s * qz(kb + m, 0);    // Adjust index when writing to q2
      q2.q(km) += s * qz(kb + m, 1);      // Adjust index when writing to q2
      km += 2;
    }
  }
}

} // namespace occ::dma
