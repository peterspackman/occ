#include <tonto/core/linear_algebra.h>

namespace tonto::core::charges
{


tonto::Vec eem_partial_charges(const tonto::IVec &atomic_numbers, tonto::Mat3N &positions, double charge = 0.0);

}
