#pragma once
#include <occ/core/linear_algebra.h>
#include <string>

namespace occ::crystal {

/**
 * Class representing a 3D symmetry operation
 *
 * A symmetry operation describes the combined rotation and translation
 * operations, as a member of a `SpaceGroup`.
 */
class SymmetryOperation {
  public:
    /**
     * Constructor from an aribtrary 4x4 matrix
     *
     * \param seitz Mat4 matrix encompassing the rotation and
     * translation components of this symmetry operation.
     *
     * \note The matrix is not checked to ensure it's a sensible symmetry
     * operation, or even affine.
     */
    SymmetryOperation(const Mat4 &seitz);

    /**
     * Constructor from string representation.
     *
     * \param symop std::string describing the symmetry operation, e.g.
     * "x,y,z" for the identity symop
     *
     * \note The string is not checked to ensure it's a sensible symmetry
     * operation, or even affine, it is algorithmically constructed from
     * the string.
     */
    SymmetryOperation(const std::string &symop);

    /**
     * Constructor from an integer representation.
     *
     * \param symop int describing the symmetry operation, e.g. 16484
     *
     * Since there are only 3 possible entries in the rotation matrix
     * \f$(-1, 0, 1)\f$ and 8 in the translation component
     * \f$(0, \frac{1}{6}, \frac{1}{4}, \frac{1}{3},
     * \frac{1}{2}, \frac{2}{3}, \frac{3}{4}, \frac{5}{6})\f$, all of
     * which are divisible by 12, the symop can be serialized as an integer
     *
     */
    SymmetryOperation(int symop);

    /**
     * The integer representation of this symop
     *
     * \returns integer representing the symop e.g. 16484 for the identity
     */
    int to_int() const;

    /**
     * String representation of this symop
     *
     * \returns std::string representing the symop e.g. "+x,+y,+z" for the
     * identity
     */
    std::string to_string() const;

    /**
     * Returns an inverted copy of this symmetry operation e.g. (x,y,z) ->
     * (-x,-y,-z)
     *
     * \returns `SymmetryOperation` equivalent to this under inversion
     */
    SymmetryOperation inverted() const;

    /**
     * Returns an inverted copy of this symmetry operation e.g. (x,y,z) ->
     * (x+1/2,y+1/3,z-1/3) etc.
     *
     * \returns `SymmetryOperation` equivalent to this after the translation
     *
     * \note Translation is in the range [0, 1] i.e. 4/3 will be expressed as
     * 1/3
     */
    SymmetryOperation translated(const Vec3 &) const;

    /**
     * Is this the identity symop?
     *
     * \returns true if this is the identity symop, false otherwise
     */
    bool is_identity() const { return to_int() == 16484; }

    /**
     * Apply the transformation represented by this symop to a set
     * of coordinates.
     *
     * \param frac Mat3N containing fractional coordinates.
     *
     * \returns Mat3N containing the transformed coordinates.
     *
     * \note Coords are assumed to be in fractional.
     */
    Mat3N apply(const Mat3N &frac) const;

    /**
     * The 4x4 Seitz matrix representation of this symop
     *
     * \returns const reference to a Mat4 containing describing this symop.
     */
    const auto &seitz() const { return m_seitz; }

    /**
     * The 3x3 rotation component of the matrix representation of this symop
     *
     * \returns Mat3 containing describing the rotation.
     */
    Mat3 rotation() const { return m_seitz.block<3, 3>(0, 0); }

    /**
     * The translation component of the matrix representation of this symop
     *
     * \returns Vec3 containing describing the translation.
     */
    Vec3 translation() const { return m_seitz.block<3, 1>(0, 3); }

    /// Shorthand for `SymmetryOperation::apply`
    auto operator()(const Mat3N &frac) const { return apply(frac); }

    /// Check if two symops are identical
    bool operator==(const SymmetryOperation &other) const {
        return to_int() == other.to_int();
    }

    /// Ordering is based on integer representation
    bool operator<(const SymmetryOperation &other) const {
        return to_int() < other.to_int();
    }

    /// Ordering is based on integer representation
    bool operator>(const SymmetryOperation &other) const {
        return to_int() > other.to_int();
    }

    /// Ordering is based on integer representation
    bool operator<=(const SymmetryOperation &other) const {
        return to_int() <= other.to_int();
    }
    /// Ordering is based on integer representation
    bool operator>=(const SymmetryOperation &other) const {
        return to_int() >= other.to_int();
    }

    /**
     * Compose this symmetry operation with another
     *
     * \returns SymmetryOperation representing the matrix product of this
     * on the left of another symop i.e. this * other
     */
    const SymmetryOperation operator*(const SymmetryOperation &other) const {
        return SymmetryOperation(seitz() * other.seitz());
    }

  private:
    Mat4 m_seitz;
};

} // namespace occ::crystal
