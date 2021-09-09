#pragma once

namespace occ::core {

struct Atom {
    int atomic_number;
    double x, y, z;
};

inline bool operator==(const Atom &atom1, const Atom &atom2) {
    return atom1.atomic_number == atom2.atomic_number && atom1.x == atom2.x &&
           atom1.y == atom2.y && atom1.z == atom2.z;
}

} // namespace occ::core
