#include <occ/qm/guess_density.h>
// modified routines from libint2
// include/libint2/chemistry/sto3g_atomic_density.h

namespace occ::qm::guess {

namespace impl {

/* compute orbital occupation numbers for a subshell created
 * by smearing at most num_electrons_remaining
 * (corresponds to spherical averaging)
 */

void update_occupation_subshell(std::vector<double> &destination, int size,
                                int &num_electrons_remaining) {
    const int electrons_allocated = (num_electrons_remaining > 2 * size)
                                        ? 2 * size
                                        : num_electrons_remaining;
    num_electrons_remaining -= electrons_allocated;
    const double electrons_per_orbital =
        (electrons_allocated % size == 0)
            ? static_cast<double>(electrons_allocated / size)
            : (static_cast<double>(electrons_allocated)) / size;
    for (size_t f = 0; f < size; f++) {
        destination.push_back(electrons_per_orbital);
    }
}

} // namespace impl

int minimal_basis_nao(int Z, bool spherical) {
    int nao = 1;
    if (Z == 1 || Z == 2) // H, He
        nao = 1;
    else if (Z <= 10)              // Li - Ne
        nao = 5;                   // 2p is included even for Li and Be
    else if (Z <= 18)              // Na - Ar
        nao = 9;                   // 3p is included even for Na and Mg
    else if (Z < 20)               // K, Ca
        nao = 13;                  // 4p is included
    else if (Z <= 36)              // Sc - Kr
        nao = spherical ? 18 : 19; // 1 D function = 1 extra
    else if (Z <= 38)              // Rb, Sr
        nao = spherical ? 22 : 23; // 5p is included, 1 D function = 1 extra
    else if (Z <= 53)              // Y - I
        nao = spherical ? 27 : 29; // 2 D functions = 2 extra
    else if (Z <= 86)
        nao = spherical ? 49 : 55; // 3 D functions, 1 F = 6 extra functions
    else
        throw "minimal basis not defined for elements Z > 86";
    return nao;
}

std::vector<double> minimal_basis_occupation_vector(size_t Z, bool spherical) {

    using impl::update_occupation_subshell;
    std::vector<double> occvec;
    size_t nao = minimal_basis_nao(Z, spherical);
    occvec.reserve(nao);

    int num_of_electrons = Z;
    int dsize = spherical ? 5 : 6;
    int fsize = spherical ? 9 : 10;

    update_occupation_subshell(occvec, 1, num_of_electrons);     // 1s
    if (Z > 2) {                                                 // Li+
        update_occupation_subshell(occvec, 1, num_of_electrons); // 2s
        update_occupation_subshell(occvec, 3, num_of_electrons); // 2p
    }
    if (Z > 10) {                                                // Na+
        update_occupation_subshell(occvec, 1, num_of_electrons); // 3s
        update_occupation_subshell(occvec, 3, num_of_electrons); // 3p
    }
    if (18 < Z && Z <= 36) { // K .. Kr
        // 4s is singly occupied in K, Cr, and Cu
        int num_of_4s_electrons = (Z == 19 || Z == 24 || Z == 29) ? 1 : 2;
        num_of_electrons -= num_of_4s_electrons;
        update_occupation_subshell(occvec, 1, num_of_4s_electrons); // 4s

        int num_of_4p_electrons =
            std::min(6, (Z > 30) ? static_cast<int>(Z) - 30 : 0);
        num_of_electrons -= num_of_4p_electrons;
        update_occupation_subshell(occvec, 3, num_of_4p_electrons); // 4p

        update_occupation_subshell(occvec, dsize,
                                   num_of_electrons); // 3d
    }
    if (36 < Z && Z <= 53) { // Rb .. I
        // 3d4s4p are fully occupied ...
        update_occupation_subshell(occvec, 1, num_of_electrons); // 4s
        update_occupation_subshell(occvec, 3, num_of_electrons); // 4p

        // 5s is singly occupied in Rb, Nb, Mo, Ru, Rh, and Ag
        int num_of_5s_electrons{2};
        switch (Z) {
        case 37:
        case 41:
        case 42:
        case 44:
        case 45:
        case 47:
            num_of_5s_electrons = 1;
            break;
        default:
            break;
        }
        num_of_electrons -= num_of_5s_electrons;
        update_occupation_subshell(occvec, 1, num_of_5s_electrons); // 5s

        int num_of_5p_electrons =
            std::min(static_cast<decltype(Z)>(6), (Z > 48) ? Z - 48 : 0);
        num_of_electrons -= num_of_5p_electrons;
        update_occupation_subshell(occvec, 3, num_of_5p_electrons); // 5p

        update_occupation_subshell(occvec, dsize, num_of_electrons); // 3d
        update_occupation_subshell(occvec, dsize, num_of_electrons); // 4d
    }
    if (53 < Z && Z <= 86) {
        // TODO proper setup for heavy elements!
        update_occupation_subshell(occvec, 1, num_of_electrons);     // 4s
        update_occupation_subshell(occvec, 3, num_of_electrons);     // 4p
        update_occupation_subshell(occvec, 1, num_of_electrons);     // 5s
        update_occupation_subshell(occvec, 3, num_of_electrons);     // 5p
        update_occupation_subshell(occvec, 3, num_of_electrons);     // 6p
        update_occupation_subshell(occvec, 3, num_of_electrons);     // 6p
        update_occupation_subshell(occvec, dsize, num_of_electrons); // 3d
        update_occupation_subshell(occvec, dsize, num_of_electrons); // 4d
        update_occupation_subshell(occvec, dsize, num_of_electrons); // 5d
        update_occupation_subshell(occvec, fsize, num_of_electrons); // 4f
    }

    if (occvec.size() != nao) {
        occ::log::warn(
            "Inconsistent number of atomic orbitals in minimal basis "
            "occupation vector: expected {}, have {}",
            nao, occvec.size());
    }

    return occvec;
}

} // namespace occ::qm::guess
