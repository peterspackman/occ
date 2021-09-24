#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
#include <occ/core/rotor.h>

namespace occ::core {

using occ::core::rotor::Rotor;

struct SymOp {
    SymOp();
    SymOp(const Mat3& rotation, const Vec3& translation);
    SymOp(const Mat4& transformation);

    Mat3N apply(const Mat3N& coordinates) const;
    const auto rotation() const { return transformation.block<3, 3>(0, 0); }
    const auto translation() const { return transformation.block<3, 1>(0, 3); }

    static SymOp from_rotation_vector(const Vec3& vector, const Vec3& origin = Vec3::Zero());
    static SymOp from_axis_angle(const Vec3& vector, double angle, const Vec3& origin = Vec3::Zero());
    static SymOp reflection(const Vec3&, const Vec3& origin = Vec3::Zero());
    static SymOp rotoreflection(const Vec3& vector, double angle, const Vec3& origin = Vec3::Zero());
    static SymOp inversion();
    static SymOp identity();

    occ::Mat4 transformation;
};

enum class PointGroup {
    C1, Ci, Cs, 
    C2, C3, C4, C5, C6, C8,
    Coov, Dooh,
    C2v, C3v, C4v, C5v, C6v,
    C2h, C3h, C4h, C5h, C6h,
    D2, D3, D4, D5, D6, D7, D8,
    D2h, D3h, D4h, D5h, D6h, D7h, D8h,
    D2d, D3d, D4d, D5d, D6d, D7d, D8d,
    S4, S6, S8,
    T, Td, Th,
    O, Oh,
    I, Ih
};

enum class MirrorType {
    None, H, D, V
};

PointGroup dihedral_group(int, MirrorType);
PointGroup cyclic_group(int, MirrorType);

class MolecularPointGroup {
public:
    MolecularPointGroup(const occ::chem::Molecule&);
    const char * description() const;
    const char * point_group_string() const;
    const PointGroup point_group() const { return group; }
    const auto& symops() const { return symmetry_operations; }
    const auto& rotational_symmetries() const { return rotational_symmetry; }

private:
    void init();
    void init_linear();
    void init_asymmetric_top();
    void init_symmetric_top();
    void init_spherical_top();
    void init_dihedral();
    void init_cyclic();
    void init_no_rotational_symmetry();
    int find_rotational_symmetries(const Vec3&);
    void find_spherical_axes();
    MirrorType find_mirror(const Vec3&);
    bool check_perpendicular_r2_axis(const Vec3&);
    bool check_R2_axes_asym();

    Rotor rotor_type{Rotor::Asymmetric};
    PointGroup group{PointGroup::C1}; 
    occ::chem::Molecule centered_molecule;
    std::vector<SymOp> symmetry_operations;
    std::vector<std::pair<Vec3, int>> rotational_symmetry;
    Mat3 inertia_tensor;
    Vec eigenvalues;
    Mat3 eigenvectors;
    double tol = 0.3, etol = 1e-2, mtol = 1e-1;
};

} // namespace occ::core
