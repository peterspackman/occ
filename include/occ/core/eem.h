#include <occ/core/linear_algebra.h>

namespace occ::core::charges {

/**
 * Determine the atomic partial charges of atoms at provided positions,
 * constraining the net charge
 *
 * \param atomic_numbers const IVec& of length N atomic numbers
 * \param positions const Mat3N& of dimensions (3, N) positions in Angstroms
 * \param charge double of the system net charge (default 0 i.e. neutral)
 *
 * \returns charges Vec representing partial charges at each of the N atomic
 * sites
 */
occ::Vec eem_partial_charges(const occ::IVec &atomic_numbers,
                             const occ::Mat3N &positions, double charge = 0.0);

} // namespace occ::core::charges
