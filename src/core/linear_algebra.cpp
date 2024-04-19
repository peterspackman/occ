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
    if(x.rows() > x.cols()) {

	auto a = [](Mat &mat) {
	    return mat.block(0, 0, mat.rows() / 2, mat.cols());
	};

	auto b = [](Mat &mat) {
	    return mat.block(mat.rows() / 2, 0, mat.rows() / 2, mat.cols());
	};;

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

}
