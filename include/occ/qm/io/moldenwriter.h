#pragma once
#include <ostream>
#include <string>

namespace occ::qm {
class Wavefunction;
}

namespace occ::io {

/**
 * Write a wavefunction in Molden format.
 *
 * Coordinates are in bohr. Spherical bases are flagged with [5D] [7F] [9G];
 * Cartesian functions are written unit-normalized, as the format requires.
 * Unrestricted wavefunctions are written as alpha then beta orbitals in one
 * [MO] section. ECPs cannot be represented in Molden files.
 */
void write_molden(const occ::qm::Wavefunction &wfn, std::ostream &os);
void write_molden(const occ::qm::Wavefunction &wfn,
                  const std::string &filename);

} // namespace occ::io
