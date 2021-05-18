#include <occ/core/linear_algebra.h>

namespace occ::core::charges
{


occ::Vec eem_partial_charges(const occ::IVec &atomic_numbers, occ::Mat3N &positions, double charge = 0.0);

}
