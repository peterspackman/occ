#pragma once
#include <occ/dma/mult.h>
#include <occ/core/linear_algebra.h>
#include <string>
#include <vector>

namespace occ::mults {

/**
 * @brief Parse Orient multipole format from a file or string
 *
 * Orient format specification:
 * - Components are written in order: Q00, Q10, Q11c, Q11s, Q20, Q21c, Q21s, Q22c, Q22s, ...
 * - This matches OCC's internal Mult storage order
 *
 * Example Orient site format:
 *   C  -0.10287   0.00000   0.00001      Rank 4  Type C
 *      1.142734
 *      0.000000   0.163569   0.000000
 *     -0.115661   0.000000   0.000000   0.213489   0.000000
 *      0.000000  -0.452037   0.000000   0.000000   0.000000  -1.755088   0.000000
 *      0.288890   0.000000   0.000000   0.011036   0.000000   0.000000   0.000000
 *     -0.557763   0.000000
 */

struct OrientSite {
    std::string name;
    Vec3 position;  // in bohr
    dma::Mult multipole;
};

/**
 * @brief Parse Orient multipole components from a list of value strings
 *
 * @param max_rank Maximum multipole rank (e.g., 4 for hexadecapole)
 * @param lines Vector of strings containing whitespace-separated values
 * @return Mult object with parsed multipoles
 *
 * Example:
 *   std::vector<std::string> lines = {
 *     "1.142734",
 *     "0.000000   0.163569   0.000000",
 *     "-0.115661   0.000000   0.000000   0.213489   0.000000"
 *   };
 *   auto mult = parse_orient_multipoles(2, lines);  // rank 2 = charge + dipole + quadrupole
 */
dma::Mult parse_orient_multipoles(int max_rank, const std::vector<std::string>& lines);

/**
 * @brief Parse a complete Orient input file
 *
 * @param filename Path to Orient input file
 * @return Vector of OrientSite objects
 */
std::vector<OrientSite> parse_orient_file(const std::string& filename);

} // namespace occ::mults
