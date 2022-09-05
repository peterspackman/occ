#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
#include <occ/core/rotor.h>

namespace occ::core {

using occ::core::rotor::Rotor;

/**
 * Structure/utility class for a point group symmetry operation.
 */
struct SymOp {

    /**
     * Default constructor - the identity/neutral Symop
     */
    SymOp();

    /**
     * Create a SymOp from a rotation matrix and a translation
     *
     * \param rotation a Mat3 representing a rotation matrix.
     * \param translation a Vec3 representing the translation (Angstroms)
     *
     * \warning the rotation matrix is not checked to be a valid rotation.
     */
    SymOp(const Mat3 &rotation, const Vec3 &translation);

    /**
     * Create a SymOp from a heterogeneous transformation matrix.
     *
     * \param transformation a Mat4 representing a transformation matrix
     *
     * \warning the transformation matrix is not checked to be a valid affine
     * transformation.
     */
    SymOp(const Mat4 &transformation);

    /**
     * Apply this symmetry operation to a block of coordinates
     *
     * \param coordinates an Eigen reference to a block of 3xN coordinates,
     * assumed to be in Angstroms.
     *
     * \returns a new 3xN matrix of transformed coordinates
     */
    Mat3N apply(Mat3NConstRef coordinates) const;

    /**
     * Get the rotation matrix part of this SymOp
     *
     * \returns the 3x3 block of the stored heterogeneous matrix corresponding
     * to the rotation matrix.
     */
    const auto rotation() const { return transformation.block<3, 3>(0, 0); }

    /**
     * Get the translation vector part of this SymOp
     *
     * \returns the 3x1 block of the stored heterogeneous matrix corresponding
     * to the translation vector.
     */
    const auto translation() const { return transformation.block<3, 1>(0, 3); }

    /**
     * Create a SymOp object from a rotation vector and optional origin
     *
     * \param vector a rotation vector corresponding to a unit vector which
     * corresponds to the axis of rotation, whose norm will be the magnitude of
     * the rotation in radians.
     *
     * \param origin the position of the origin, in
     * Angstroms. Default is the zero vector.
     *
     * \returns the newly constructed SymOp object
     *
     * See also SymOp::from_axis_angle.
     */
    static SymOp from_rotation_vector(const Vec3 &vector,
                                      const Vec3 &origin = Vec3::Zero());

    /**
     * Create a SymOp object from a rotation axis, angle and optional origin
     *
     * \param vector a rotation vector corresponding to a unit vector which
     * corresponds to the axis of rotation
     *
     * \param angle the magnitude of the rotation in radians.
     *
     * \param origin the position of the origin, in
     * Angstroms. Default is the zero vector.
     *
     * \returns the newly constructed SymOp object
     *
     * See also SymOp::from_rotation_vector.
     */
    static SymOp from_axis_angle(const Vec3 &vector, double angle,
                                 const Vec3 &origin = Vec3::Zero());

    /**
     * Create a SymOp object representing a reflection from a given normal
     * vector.
     *
     * \param normal a vector corresponding to the normal for the plane of
     * reflection. Will be normalized/converted to a unit vector internally.
     *
     * \param origin the position of the origin, in
     * Angstroms. Default is the zero vector.
     *
     * \returns the newly constructed SymOp object
     */
    static SymOp reflection(const Vec3 &normal,
                            const Vec3 &origin = Vec3::Zero());

    /**
     * Create a SymOp object representing a combined rotation and reflection
     * from a rotation axis, angle and optional origin.
     *
     * \param vector a vector corresponding to the normal for the plane of
     * reflection, and the axis of rotation. Will be normalized/converted to a
     * unit vector internally.
     *
     * \param angle the magnitude of the rotation in radians.
     *
     * \param origin the position of the origin, in
     * Angstroms. Default is the zero vector.
     *
     * \returns the newly constructed SymOp object
     */
    static SymOp rotoreflection(const Vec3 &vector, double angle,
                                const Vec3 &origin = Vec3::Zero());

    /**
     * The inversion symop i.e. `-I`
     *
     * \returns the newly constructed SymOp object representing an inversion.
     */
    static SymOp inversion();

    /**
     * The identity symop about the origin i.e. `I`
     *
     * \returns the newly constructed SymOp object representing an identity
     * operation.
     */
    static SymOp identity();

    /**
     * The underlying data for the transform/symmetry operation.
     */
    occ::Mat4 transformation;
};

enum class PointGroup {
    C1,   /**< No rotational symmetry */
    Ci,   /**< Ci = S2 No rotational symmetry, but with inversion */
    Cs,   /**< Cs = C1h  No rotational symmetry, but a mirror plane */
    C2,   /**< open book geometry, two-fold rotational symmetry */
    C3,   /**< propeller, three-fold rotational symmetry */
    C4,   /**< four-fold rotational symmetry */
    C5,   /**< five-fold rotational symmetry */
    C6,   /**< six-fold rotational symmetry */
    C8,   /**< eight-fold rotational symmetry */
    Coov, /**< Linear, infinite rotational symmetry*/
    Dooh, /**< Linear, with an inversion center */
    C2v,  /**< angular, see-saw, T-shape, two-fold rotational symmetry with
             mirror plane parallel to axis of rotation */
    C3v,  /**< trigonal pyramidal, three-fold rotational symmetry with
             mirror plane parallel to axis of rotation */
    C4v,  /**< square pyramidal, four-fold rotational symmetry with
              mirror plane parallel to axis of rotation */
    C5v,  /**< milking stool, five-fold rotational symmetry with
               mirror plane parallel to axis of rotation */
    C6v,  /**< six-fold rotational symmetry with
                mirror plane parallel to axis of rotation */
    C2h,  /**< two-fold rotational symmetry with
                mirror plane orthogonal to axis of rotation */
    C3h,  /**< three-fold rotational symmetry with
                  mirror plane orthogonal to axis of rotation */
    C4h,  /**< four-fold rotational symmetry with
                  mirror plane orthogonal to axis of rotation */
    C5h,  /**< five-fold rotational symmetry with
                  mirror plane orthogonal to axis of rotation */
    C6h,  /**< six-fold rotational symmetry with
                  mirror plane orthogonal to axis of rotation */
    D2,   /**< twist, two-fold rotational symmetry with an inversion */
    D3,   /**< three-fold rotational symmetry with an inversion */
    D4,   /**< four-fold rotational symmetry with an inversion */
    D5,   /**< five-fold rotational symmetry with an inversion */
    D6,   /**< six-fold rotational symmetry with an inversion */
    D7,   /**< seven-fold rotational symmetry with an inversion */
    D8,   /**< eight-fold rotational symmetry with an inversion */
    D2h,  /**< planar + inversion center, two-fold rotational symmetry with an
             and mirror plane orthogonal to axis of rotation*/
    D3h,  /**< trigonal planar, trigonal bipyramidal, three-fold rotational
             symmetry with an inversion and mirror plane orthogonal to axis of
             rotation*/
    D4h, /**< square planar, four-fold rotational symmetry with an inversion and
            mirror plane orthogonal to axis of rotation*/
    D5h, /**< pentagonal */
    D6h, /**< hexagonal */
    D7h, /**< heptagonal */
    D8h, /**< octagonal */
    D2d, /**< 90 degree twist */
    D3d, /**< 60 degree twist */
    D4d, /**< 45 degree twist */
    D5d, /**< 36 degree twist */
    D6d, /**< 30 degree twist */
    D7d,
    D8d,
    S4,
    S6,
    S8,
    T,
    Td, /**< tetrahedral */
    Th, /**< pyritohedron */
    O,
    Oh, /**< octahedral, cubic */
    I,
    Ih /**< icosahedral, dodecahedral */
};

enum class MirrorType { None, H, D, V };

PointGroup dihedral_group(int, MirrorType);
PointGroup cyclic_group(int, MirrorType);

/**
 * Storage class for a point group describing a Molecule
 */
class MolecularPointGroup {
  public:
    MolecularPointGroup(const occ::core::Molecule &);
    const char *description() const;
    const char *point_group_string() const;
    const PointGroup point_group() const { return group; }
    const auto &symops() const { return symmetry_operations; }
    const auto &rotational_symmetries() const { return rotational_symmetry; }
    int symmetry_number() const;

  private:
    void init();
    void init_linear();
    void init_asymmetric_top();
    void init_symmetric_top();
    void init_spherical_top();
    void init_dihedral();
    void init_cyclic();
    void init_no_rotational_symmetry();
    int find_rotational_symmetries(const Vec3 &);
    void find_spherical_axes();
    MirrorType find_mirror(const Vec3 &);
    bool check_perpendicular_r2_axis(const Vec3 &);
    bool check_R2_axes_asym();

    Rotor rotor_type{Rotor::Asymmetric};
    PointGroup group{PointGroup::C1};
    occ::core::Molecule centered_molecule;
    std::vector<SymOp> symmetry_operations;
    std::vector<std::pair<Vec3, int>> rotational_symmetry;
    Mat3 inertia_tensor;
    Vec eigenvalues;
    Mat3 eigenvectors;
    double tol = 0.3, etol = 1e-2, mtol = 1e-1;
};

} // namespace occ::core
