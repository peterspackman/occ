#include <occ/dma/solid_harmonics.h>

namespace occ::dma {

void solid_harmonics(Eigen::Ref<const Vec3> pos, int j, Eigen::Ref<Vec> r) {
  // Computes regular solid harmonics r**k Ckq(theta,phi) for ranks k up
  // to J, if J >= 0;
  // or irregular solid harmonics r**(-k-1) Ckq(theta,phi) for ranks k up
  // to |J|, if J < 0.

  // Locations in R are used as follows:
  //       1    2    3    4    5    6    7    8    9   10   11  ...
  // kq = 00   10   11c  11s  20   21c  21s  22c  22s  30   31c ...
  // R(k,0) is real and is left in location k**2 + 1.
  // R(k,mc) and r(k,ms) are sqrt(2) times the real and imaginary parts
  // respectively of the complex solid harmonic R(k,-m)* = (-1)**m R(k,m),
  // and are left in locations K**2 + 2m and K**2 + 2m + 1 respectively.

  int l = std::abs(j);
  if ((l + 1) * (l + 1) > r.rows())
    throw std::runtime_error("Insufficient space for harmonics");
  double rr = pos.squaredNorm();

  const double &x = pos.x();
  const double &y = pos.y();
  const double &z = pos.z();

  double rfx, rfy, rfz;

  if (j > 0) {
    // Regular
    r(1) = 1.0;
    r(2) = z;
    r(3) = x;
    r(4) = y;
    rfz = z;
    rfx = x;
    rfy = y;
  } else {
    // Irregular
    rr = 1.0 / rr;
    rfx = x * rr;
    rfy = y * rr;
    rfz = z * rr;
    r(1) = std::sqrt(rr);
    r(2) = rfz * r(1);
    r(3) = rfx * r(1);
    r(4) = rfy * r(1);
  }

  auto rt = [](int v) { return std::sqrt(static_cast<double>(v)); };

  //  Remaining values are found using recursion formulae, relating
  //  the new set N to the current set K and the previous set P.
  int k = 1;
  while (k < l) {
    int n = k + 1;
    int ln = n * n + 1;
    int lk = k * k + 1;
    int lp = (k - 1) * (k - 1) + 1;
    int a2kp1 = k + k + 1;
    //  Obtain R(k+1,0) from R(k,0)*R(1,0) and R(k-1,0)
    r(ln) = (a2kp1 * r(lk) * rfz - k * rr * r(lp)) / (k + 1);
    int m = 1;
    ln = ln + 1;
    lk = lk + 1;
    lp = lp + 1;
    if (k > 1) {
      while (m < k) {
        //  Obtain R(k+1,m) from R(k,m)*R(1,0) and R(k-1,m)
        r(ln) = (a2kp1 * r(lk) * rfz - rt(k + m) * rt(k - m) * rr * r(lp)) /
                (rt(n + m) * rt(n - m));
        r(ln + 1) =
            (a2kp1 * r(lk + 1) * rfz - rt(k + m) * rt(k - m) * rr * r(lp + 1)) /
            (rt(n + m) * rt(n - m));
        m = m + 1;
        ln = ln + 2;
        lk = lk + 2;
        lp = lp + 2;
      }
    }
    // Obtain R(k+1,k) from R(k,k)*R(1,0)
    r(ln) = rt(n + k) * r(lk) * rfz;
    r(ln + 1) = rt(n + k) * r(lk + 1) * rfz;
    ln = ln + 2;
    //  Obtain R(k+1,k+1) from R(k,k)*R(1,1)
    const double s = rt(n + k) / rt(n + n);
    r(ln) = s * (rfx * r(lk) - rfy * r(lk + 1));
    r(ln + 1) = s * (rfx * r(lk + 1) + rfy * r(lk));
    k = k + 1;
  }
}

} // namespace occ::dma
