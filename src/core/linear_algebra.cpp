#include <occ/core/linear_algebra.h>

namespace occ {

MatTriple MatTriple::operator+(const MatTriple &rhs) const {
  return {x + rhs.x, y + rhs.y, z + rhs.z};
}

MatTriple MatTriple::operator-(const MatTriple &rhs) const {
  return {x - rhs.x, y - rhs.y, z - rhs.z};
}

void MatTriple::scale_by(double fac) {
  x.array() *= fac;
  y.array() *= fac;
  z.array() *= fac;
}

void MatTriple::symmetrize() {
  if (x.rows() > x.cols()) {

    auto a = [](Mat &mat) {
      return mat.block(0, 0, mat.rows() / 2, mat.cols());
    };

    auto b = [](Mat &mat) {
      return mat.block(mat.rows() / 2, 0, mat.rows() / 2, mat.cols());
    };
    ;

    a(x).noalias() = (a(x) + a(x).transpose()).eval();
    a(y).noalias() = (a(y) + a(y).transpose()).eval();
    a(z).noalias() = (a(z) + a(z).transpose()).eval();

    b(x).noalias() = (b(x) + b(x).transpose()).eval();
    b(y).noalias() = (b(y) + b(y).transpose()).eval();
    b(z).noalias() = (b(z) + b(z).transpose()).eval();
  } else {
    // Restricted
    x.noalias() = 0.5 * (x + x.transpose()).eval();
    y.noalias() = 0.5 * (y + y.transpose()).eval();
    z.noalias() = 0.5 * (z + z.transpose()).eval();
  }
}

// MatSix implementations
MatSix MatSix::operator+(const MatSix &rhs) const {
  return {xx + rhs.xx, yy + rhs.yy, zz + rhs.zz, 
          xy + rhs.xy, xz + rhs.xz, yz + rhs.yz};
}

MatSix MatSix::operator-(const MatSix &rhs) const {
  return {xx - rhs.xx, yy - rhs.yy, zz - rhs.zz, 
          xy - rhs.xy, xz - rhs.xz, yz - rhs.yz};
}

void MatSix::scale_by(double fac) {
  xx.array() *= fac;
  yy.array() *= fac;
  zz.array() *= fac;
  xy.array() *= fac;
  xz.array() *= fac;
  yz.array() *= fac;
}

void MatSix::symmetrize() {
  if (xx.rows() > xx.cols()) {
    // Unrestricted case - handle alpha and beta blocks separately
    auto a = [](Mat &mat) {
      return mat.block(0, 0, mat.rows() / 2, mat.cols());
    };

    auto b = [](Mat &mat) {
      return mat.block(mat.rows() / 2, 0, mat.rows() / 2, mat.cols());
    };

    // Symmetrize alpha blocks
    a(xx).noalias() = 0.5 * (a(xx) + a(xx).transpose()).eval();
    a(yy).noalias() = 0.5 * (a(yy) + a(yy).transpose()).eval();
    a(zz).noalias() = 0.5 * (a(zz) + a(zz).transpose()).eval();
    a(xy).noalias() = 0.5 * (a(xy) + a(xy).transpose()).eval();
    a(xz).noalias() = 0.5 * (a(xz) + a(xz).transpose()).eval();
    a(yz).noalias() = 0.5 * (a(yz) + a(yz).transpose()).eval();

    // Symmetrize beta blocks
    b(xx).noalias() = 0.5 * (b(xx) + b(xx).transpose()).eval();
    b(yy).noalias() = 0.5 * (b(yy) + b(yy).transpose()).eval();
    b(zz).noalias() = 0.5 * (b(zz) + b(zz).transpose()).eval();
    b(xy).noalias() = 0.5 * (b(xy) + b(xy).transpose()).eval();
    b(xz).noalias() = 0.5 * (b(xz) + b(xz).transpose()).eval();
    b(yz).noalias() = 0.5 * (b(yz) + b(yz).transpose()).eval();
  } else {
    // Restricted case - symmetrize each matrix
    xx.noalias() = 0.5 * (xx + xx.transpose()).eval();
    yy.noalias() = 0.5 * (yy + yy.transpose()).eval();
    zz.noalias() = 0.5 * (zz + zz.transpose()).eval();
    xy.noalias() = 0.5 * (xy + xy.transpose()).eval();
    xz.noalias() = 0.5 * (xz + xz.transpose()).eval();
    yz.noalias() = 0.5 * (yz + yz.transpose()).eval();
  }
}

} // namespace occ
