#pragma once

namespace occ::sht {

// all j, m values are 2 * the desired value
// so that 1/2 can be exactly represented etc.
double clebsch(int j1, int m1, int j2, int m2, int j, int m) noexcept;

}
