#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::core {

struct Atom {
    int atomic_number;
    double x, y, z;

    inline void rotate(const Mat3 &rotation) {
        occ::Vec3 pos{x, y, z};
        auto pos_rot = rotation * pos;
        x = pos_rot(0);
        y = pos_rot(1);
        z = pos_rot(2);
    }

    inline void translate(const Vec3 &translation) {
        x += translation(0);
        y += translation(1);
        z += translation(2);
    }

    inline Vec3 position() const { return {x, y, z}; }

    inline void set_position(const Vec3 &v) {
        x = v.x();
        y = v.y();
        z = v.z();
    }

    inline double square_distance(double xx, double yy, double zz) const {
        double dx = xx - x, dy = yy - y, dz = zz - z;
        return dx * dx + dy * dy + dz * dz;
    }
};

inline bool operator==(const Atom &atom1, const Atom &atom2) {
    return atom1.atomic_number == atom2.atomic_number && atom1.x == atom2.x &&
           atom1.y == atom2.y && atom1.z == atom2.z;
}

template <typename AtomIterator>
inline void rotate_atoms(AtomIterator &atoms, const Mat3 &rotation) {
    for (auto &atom : atoms) {
        atom.rotate(rotation);
    }
}

template <typename AtomIterator>
inline void translate_atoms(AtomIterator &atoms, const Vec3 &translation) {
    for (auto &atom : atoms) {
        atom.translate(translation);
    }
}

} // namespace occ::core
