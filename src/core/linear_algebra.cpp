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
    x.noalias() = 0.5 * (x + x.transpose()).eval();
    y.noalias() = 0.5 * (y + y.transpose()).eval();
    z.noalias() = 0.5 * (z + z.transpose()).eval();
}

}
