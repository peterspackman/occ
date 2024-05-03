#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::sht {

Vec wigner3j(double l2, double l3, double m1, double m2, double m3);
double wigner3j_single(double l1, double l2, double l3, double m1, double m2, double m3);
}
