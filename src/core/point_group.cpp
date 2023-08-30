#include <Eigen/Geometry>
#include <fmt/ostream.h>
#include <ankerl/unordered_dense.h>
#include <occ/core/log.h>
#include <occ/core/point_group.h>
#include <occ/core/units.h>

namespace occ::core {

SymOp::SymOp() : transformation(Mat4::Identity()) {}
SymOp::SymOp(const Mat4 &transform) : transformation(transform) {}
SymOp::SymOp(const Mat3 &rotation, const Vec3 &translation)
    : transformation(Mat4::Zero()) {
    transformation.block<3, 3>(0, 0) = rotation;
    transformation.block<3, 1>(0, 3) = translation;
    transformation(3, 3) = 1;
}

Mat3N SymOp::apply(Mat3NConstRef coordinates) const {
    Mat3N result = rotation() * coordinates;
    result.colwise() += translation();
    return result;
}

SymOp SymOp::from_rotation_vector(const Vec3 &rotation, const Vec3 &origin) {
    double norm = rotation.norm();
    Vec3 normalized = rotation / norm;
    return SymOp::from_axis_angle(normalized, norm, origin);
}

SymOp SymOp::from_axis_angle(const Vec3 &axis, double angle,
                             const Vec3 &origin) {
    auto aa = Eigen::AngleAxis(occ::units::radians(angle), axis.normalized());
    Mat3 rotation = aa.matrix();
    return SymOp(rotation, origin);
}

SymOp SymOp::reflection(const Vec3 &normal, const Vec3 &origin) {
    // Normalize the normal vector first.
    Vec3 n = normal.normalized();
    Vec3 n2 = n.array() * n.array();
    Mat4 translation = Mat4::Identity();
    translation.block<3, 1>(0, 3) = -origin;

    Mat4 m = Mat4::Identity();
    m(0, 0) = 1 - 2 * n2(0);
    m(1, 1) = 1 - 2 * n2(1);
    m(2, 2) = 1 - 2 * n2(2);
    m(0, 1) = -2 * n(0) * n(1);
    m(1, 0) = m(0, 1);
    m(0, 2) = -2 * n(0) * n(2);
    m(2, 0) = m(0, 2);
    m(1, 2) = -2 * n(1) * n(2);
    m(2, 1) = m(1, 2);

    if (origin.norm() > 1e-12)
        return SymOp(translation.inverse() * (m * translation));
    return SymOp(m);
}

SymOp SymOp::rotoreflection(const Vec3 &axis, double angle,
                            const Vec3 &origin) {
    SymOp rot = SymOp::from_axis_angle(axis, angle, origin);
    SymOp ref = SymOp::reflection(axis, origin);
    return SymOp(rot.transformation * ref.transformation);
}

SymOp SymOp::inversion() {
    Mat4 transformation = Mat4::Identity();
    transformation.block<3, 3>(0, 0) *= -1;
    return SymOp(transformation);
}

bool is_valid_symop(const SymOp &op, Eigen::Ref<const Mat3N> positions) {
    auto nc = op.apply(positions);
    const size_t N = positions.cols();
    for (size_t i = 0; i < N; i++) {
        const auto p1 = nc.col(i);
        bool found = false;
        for (size_t j = 0; j < N; j++) {
            const auto p2 = positions.col(j);
            if ((p2 - p1).norm() < 0.3) {
                found = true;
                break;
            }
        }
        if (!found)
            return false;
    }
    return true;
}

IVec smallest_off_axis_group(const occ::core::Molecule &mol, const Vec3 &axis) {
    double tol = 0.1;
    const auto &pos = mol.positions();
    const auto els = mol.atomic_numbers();
    const size_t N = els.rows();

    auto off_axis = [&axis, &tol](const Vec3 &site) {
        return site.cross(axis).norm() > tol;
    };

    IVec mask(N);
    ankerl::unordered_dense::map<int, int> group_size;
    for (int i = 0; i < N; i++) {
        int el = els(i);
        if (off_axis(pos.col(i))) {
            mask(i) = el;
            if (group_size.contains(el)) {
                group_size[el]++;
            } else {
                group_size[el] = 1;
            }
        } else
            mask(i) = 0;
    }

    int min_el = group_size.begin()->first;
    int min_count = group_size[min_el];
    for (const auto &el : group_size) {
        if (el.second < min_count) {
            min_el = el.first;
            min_count = el.second;
        }
    }

    for (int i = 0; i < N; i++)
        if (mask(i) != min_el)
            mask(i) = 0;
    return mask;
}

MolecularPointGroup::MolecularPointGroup(const occ::core::Molecule &mol)
    : centered_molecule(mol.translated(-mol.center_of_mass())) {
    init();
}

void MolecularPointGroup::init() {
    inertia_tensor = centered_molecule.inertia_tensor();
    double total_inertia = inertia_tensor.diagonal().array().sum() * 0.5;
    inertia_tensor.array() /= total_inertia;
    Eigen::SelfAdjointEigenSolver<Mat3> solver(inertia_tensor);
    eigenvalues = solver.eigenvalues();
    eigenvectors = solver.eigenvectors();
    rotor_type = occ::core::rotor::classify(eigenvalues(0), eigenvalues(1),
                                            eigenvalues(2), etol);
    symmetry_operations.push_back(SymOp());
    switch (rotor_type) {
    case Rotor::Linear:
        occ::log::debug("rotor is linear molecule");
        init_linear();
        break;
    case Rotor::Spherical:
        occ::log::debug("rotor is spherical top");
        init_linear();
        init_spherical_top();
        break;
    case Rotor::Asymmetric:
        occ::log::debug("rotor is asymmetric top");
        init_asymmetric_top();
        break;
    default:
        occ::log::debug("rotor is symmetric top");
        init_symmetric_top();
        break;
    }
}

void MolecularPointGroup::init_linear() {
    auto inv = SymOp::inversion();
    if (is_valid_symop(inv, centered_molecule.positions())) {
        group = PointGroup::Dooh;
        symmetry_operations.push_back(inv);
    } else
        group = PointGroup::Coov;
}

void MolecularPointGroup::find_spherical_axes() {
    ankerl::unordered_dense::map<int, int> count;
    std::array<bool, 6> rot_found{false, false, false, false, false, false};

    const auto &pos = centered_molecule.positions();
    const auto nums = centered_molecule.atomic_numbers();
    for (int i = 0; i < nums.rows(); i++) {
        int el = nums(i);
        if (!count.contains(el))
            count[el] = 1;
        else
            count[el]++;
    }
    const auto &min_el = *std::min_element(
        count.begin(), count.end(),
        [](const auto &l, const auto &r) { return l.second < r.second; });

    size_t N = min_el.second;
    Mat3N coords(3, min_el.second);
    size_t idx{0};
    for (int i = 0; i < N; i++) {
        if (nums(i) == min_el.first) {
            coords.col(idx) = pos.col(i);
            idx++;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            for (int k = j + 1; k < N; k++) {
                if (rot_found[2])
                    continue;
                Mat3 test_axes;
                test_axes.col(0) = coords.col(j) - coords.col(i);
                test_axes.col(1) = coords.col(k) - coords.col(i);
                test_axes.col(2) = coords.col(k) - coords.col(j);
                for (int c = 0; c < 3; c++) {
                    Vec3 test_axis = test_axes.col(c);
                    if (test_axis.norm() < tol)
                        continue;
                    auto op = SymOp::from_axis_angle(test_axis, 180);
                    if (is_valid_symop(op, pos)) {
                        rot_found[2] = true;
                        rotational_symmetry.push_back({test_axis, 2});
                    }
                }
                Vec3 test_axis = (coords.col(j) - coords.col(i))
                                     .cross(coords.col(k) - coords.col(i));
                if (test_axis.norm() < tol)
                    continue;
                for (int r = 3; r < 6; r++) {
                    if (rot_found[r])
                        continue;
                    auto op = SymOp::from_axis_angle(test_axis, 360 / r);
                    if (is_valid_symop(op, pos)) {
                        rot_found[r] = true;
                        rotational_symmetry.push_back({test_axis, r});
                        break;
                    }
                }

                if (rot_found[2] && rot_found[3] &&
                    (rot_found[4] || rot_found[5]))
                    break;
            }
        }
    }
}

void MolecularPointGroup::init_spherical_top() {
    // Handles T, O or I point groups.
    find_spherical_axes();
    if (rotational_symmetry.size() == 0) {
        init_symmetric_top();
    }
    const auto &m = *std::max_element(
        rotational_symmetry.begin(), rotational_symmetry.end(),
        [](const auto &l, const auto &r) { return l.second < r.second; });

    const Vec3 &main_axis = m.first;
    int rot = m.second;
    const SymOp inv = SymOp::inversion();
    const auto &pos = centered_molecule.positions();
    if (rot < 3) {
        init_symmetric_top();
    } else if (rot == 3) {
        MirrorType mirror_type = find_mirror(main_axis);
        if (mirror_type != MirrorType::None) {
            if (is_valid_symop(inv, pos)) {
                symmetry_operations.push_back(inv);
                group = PointGroup::Th;
            } else
                group = PointGroup::Td;
        } else {
            group = PointGroup::T;
        }
    } else if (rot == 4) {
        if (is_valid_symop(inv, pos)) {
            symmetry_operations.push_back(inv);
            group = PointGroup::Oh;
        } else
            group = PointGroup::O;
    } else if (rot == 5) {
        if (is_valid_symop(inv, pos)) {
            symmetry_operations.push_back(inv);
            group = PointGroup::Ih;
        } else
            group = PointGroup::I;
    }
}

void MolecularPointGroup::init_asymmetric_top() {
    check_R2_axes_asym();
    if (rotational_symmetry.size() == 0) {
        init_no_rotational_symmetry();
    } else if (rotational_symmetry.size() == 3) {
        init_dihedral();
    } else {
        init_cyclic();
    }
}

int MolecularPointGroup::find_rotational_symmetries(const Vec3 &axis) {
    IVec min_set = smallest_off_axis_group(centered_molecule, axis);
    const size_t max_symmetry = (min_set.array() > 0).count();
    const auto &positions = centered_molecule.positions();
    occ::log::debug("rotor max possible symmetry = {}", max_symmetry);

    for (size_t i = max_symmetry; i >= 0; i--) {
        if (max_symmetry % i != 0)
            continue;
        auto op = SymOp::from_axis_angle(axis, 360 / i);
        if (is_valid_symop(op, positions)) {
            symmetry_operations.push_back(op);
            rotational_symmetry.push_back(std::make_pair(axis, i));
            return i;
        }
    }
    return 1;
}

PointGroup dihedral_group(int r, MirrorType m) {
    if (m == MirrorType::H) {
        switch (r) {
        case 2:
            return PointGroup::D2h;
        case 3:
            return PointGroup::D3h;
        case 4:
            return PointGroup::D4h;
        case 5:
            return PointGroup::D5h;
        case 6:
            return PointGroup::D6h;
        case 7:
            return PointGroup::D7h;
        case 8:
            return PointGroup::D8h;
        }
    }
    if (m == MirrorType::D) {
        switch (r) {
        case 2:
            return PointGroup::D2d;
        case 3:
            return PointGroup::D3d;
        case 4:
            return PointGroup::D4d;
        case 5:
            return PointGroup::D5d;
        case 6:
            return PointGroup::D6d;
        case 7:
            return PointGroup::D7d;
        case 8:
            return PointGroup::D8d;
        }
    }
    switch (r) {
    case 2:
        return PointGroup::D2;
    case 3:
        return PointGroup::D3;
    case 4:
        return PointGroup::D4;
    case 5:
        return PointGroup::D5;
    case 6:
        return PointGroup::D6;
    case 7:
        return PointGroup::D7;
    case 8:
        return PointGroup::D8;
    }
    return PointGroup::D2;
}

PointGroup cyclic_group(int r, MirrorType m) {
    if (m == MirrorType::H) {
        switch (r) {
        case 2:
            return PointGroup::C2h;
        case 3:
            return PointGroup::C3h;
        case 4:
            return PointGroup::C4h;
        case 5:
            return PointGroup::C5h;
        case 6:
            return PointGroup::C6h;
        }
    }
    if (m == MirrorType::V) {
        switch (r) {
        case 2:
            return PointGroup::C2v;
        case 3:
            return PointGroup::C3v;
        case 4:
            return PointGroup::C4v;
        case 5:
            return PointGroup::C5v;
        case 6:
            return PointGroup::C6v;
        }
    }
    switch (r) {
    case 2:
        return PointGroup::C2;
    case 3:
        return PointGroup::C3;
    case 4:
        return PointGroup::C4;
    case 5:
        return PointGroup::C5;
    case 6:
        return PointGroup::C6;
    case 8:
        return PointGroup::C8;
    }
    return PointGroup::C2;
}

bool MolecularPointGroup::check_perpendicular_r2_axis(const Vec3 &axis) {
    IVec min_set = smallest_off_axis_group(centered_molecule, axis);
    size_t N = centered_molecule.size();
    const auto &positions = centered_molecule.positions();
    for (int i = 0; i < N; i++) {
        if (min_set(i) < 1)
            continue;
        for (int j = i + 1; j < N; j++) {
            if (min_set(j) < 1)
                continue;
            Vec3 test_axis = (positions.col(i) - positions.col(j)).cross(axis);
            if (test_axis.norm() > tol) {
                auto op = SymOp::from_axis_angle(test_axis, 180);
                if (is_valid_symop(op, positions)) {
                    symmetry_operations.push_back(op);
                    rotational_symmetry.push_back(std::make_pair(test_axis, 2));
                    return true;
                }
            }
        }
    }
    return false;
}

MirrorType MolecularPointGroup::find_mirror(const Vec3 &axis) {
    MirrorType mirror_type{MirrorType::None};
    const auto els = centered_molecule.atomic_numbers();
    const auto &positions = centered_molecule.positions();
    SymOp refl = SymOp::reflection(axis);
    if (is_valid_symop(refl, positions)) {
        symmetry_operations.push_back(refl);
        mirror_type = MirrorType::H;
        return mirror_type;
    }
    size_t N = positions.cols();
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            if (els(i) != els(j))
                continue;
            Vec3 normal = positions.col(i) - positions.col(j);
            if (normal.dot(axis) >= tol)
                continue;

            SymOp op = SymOp::reflection(normal);
            if (!is_valid_symop(op, positions))
                continue;

            if (rotational_symmetry.size() > 1) {
                mirror_type = MirrorType::D;
                for (const auto &r : rotational_symmetry) {
                    if (((r.first - axis).norm() >= tol) &&
                        ((r.first.dot(normal) < tol))) {
                        mirror_type = MirrorType::V;
                        break;
                    }
                }
            } else
                mirror_type = MirrorType::V;
            break;
        }
    }
    return mirror_type;
}

void MolecularPointGroup::init_dihedral() {
    occ::log::debug("rotor has dihedral symmetry");
    const auto &m = *std::max_element(
        rotational_symmetry.begin(), rotational_symmetry.end(),
        [](const auto &l, const auto &r) { return l.second < r.second; });

    MirrorType mirror_type = find_mirror(m.first);
    group = dihedral_group(m.second, mirror_type);
}

void MolecularPointGroup::init_no_rotational_symmetry() {
    occ::log::debug("rotor has no rotational symmetry");
    group = PointGroup::C1;
    if (is_valid_symop(SymOp::inversion(), centered_molecule.positions())) {
        symmetry_operations.push_back(SymOp::inversion());
        group = PointGroup::Ci;
        return;
    }
    for (int i = 0; i < 3; i++) {
        if (find_mirror(eigenvectors.col(i)) != MirrorType::None) {
            group = PointGroup::Cs;
            return;
        }
    }
}

bool MolecularPointGroup::check_R2_axes_asym() {
    bool found = false;
    for (int i = 0; i < 3; i++) {
        SymOp op = SymOp::from_axis_angle(eigenvectors.col(i), 180.0);
        if (is_valid_symop(op, centered_molecule.positions())) {
            symmetry_operations.push_back(op);
            rotational_symmetry.push_back(
                std::make_pair(eigenvectors.col(i), 2));
            found = true;
        }
    }
    return found;
}

void MolecularPointGroup::init_cyclic() {
    occ::log::debug("rotor has cyclic symmetry");
    const auto &m = *std::max_element(
        rotational_symmetry.begin(), rotational_symmetry.end(),
        [](const auto &l, const auto &r) { return l.second < r.second; });

    MirrorType mirror_type = find_mirror(m.first);
    if (mirror_type == MirrorType::None) {
        if (is_valid_symop(SymOp::rotoreflection(m.first, 180.0 / m.second),
                           centered_molecule.positions())) {
            switch (m.second) {
            case 2:
                group = PointGroup::S4;
                return;
            case 3:
                group = PointGroup::S6;
                return;
            case 4:
                group = PointGroup::S8;
                return;
            }
        }
    }
    group = cyclic_group(m.second, mirror_type);
}

void MolecularPointGroup::init_symmetric_top() {
    Eigen::Index ind{1};
    if (std::abs(eigenvalues(0) - eigenvalues(1)) < etol)
        ind = 2;
    else if (std::abs(eigenvalues(1) - eigenvalues(2)) < etol)
        ind = 0;

    Vec3 unique_axis = eigenvectors.col(ind);
    find_rotational_symmetries(unique_axis);
    if (rotational_symmetry.size() > 0) {
        check_perpendicular_r2_axis(unique_axis);
    }
    if (rotational_symmetry.size() >= 2) {
        init_dihedral();
    } else if (rotational_symmetry.size() == 1) {
        init_cyclic();
    } else
        init_no_rotational_symmetry();
}

const char *MolecularPointGroup::point_group_string() const {
    switch (group) {
    case PointGroup::C1:
        return "C1";
    case PointGroup::Ci:
        return "Ci";
    case PointGroup::Cs:
        return "Cs";
    case PointGroup::C2:
        return "C2";
    case PointGroup::C3:
        return "C3";
    case PointGroup::C4:
        return "C4";
    case PointGroup::C5:
        return "C5";
    case PointGroup::C6:
        return "C6";
    case PointGroup::C8:
        return "C8";
    case PointGroup::Coov:
        return "Coov";
    case PointGroup::Dooh:
        return "Dooh";
    case PointGroup::C2v:
        return "C2v";
    case PointGroup::C3v:
        return "C3v";
    case PointGroup::C4v:
        return "C4v";
    case PointGroup::C5v:
        return "C5v";
    case PointGroup::C6v:
        return "C6v";
    case PointGroup::C2h:
        return "C2h";
    case PointGroup::C3h:
        return "C3h";
    case PointGroup::C4h:
        return "C4h";
    case PointGroup::C5h:
        return "C5h";
    case PointGroup::C6h:
        return "C6h";
    case PointGroup::D2:
        return "D2";
    case PointGroup::D3:
        return "D3";
    case PointGroup::D4:
        return "D4";
    case PointGroup::D5:
        return "D5";
    case PointGroup::D6:
        return "D6";
    case PointGroup::D7:
        return "D7";
    case PointGroup::D8:
        return "D8";
    case PointGroup::D2h:
        return "D2h";
    case PointGroup::D3h:
        return "D3h";
    case PointGroup::D4h:
        return "D4h";
    case PointGroup::D5h:
        return "D5h";
    case PointGroup::D6h:
        return "D6h";
    case PointGroup::D7h:
        return "D7h";
    case PointGroup::D8h:
        return "D8h";
    case PointGroup::D2d:
        return "D2d";
    case PointGroup::D3d:
        return "D3d";
    case PointGroup::D4d:
        return "D4d";
    case PointGroup::D5d:
        return "D5d";
    case PointGroup::D6d:
        return "D6d";
    case PointGroup::D7d:
        return "D7d";
    case PointGroup::D8d:
        return "D8d";
    case PointGroup::S4:
        return "S4";
    case PointGroup::S6:
        return "S6";
    case PointGroup::S8:
        return "S8";
    case PointGroup::T:
        return "T";
    case PointGroup::Td:
        return "Td";
    case PointGroup::Th:
        return "Th";
    case PointGroup::O:
        return "O";
    case PointGroup::Oh:
        return "Oh";
    case PointGroup::I:
        return "I";
    case PointGroup::Ih:
        return "Ih";
    default:
        return "??";
    }
}

const char *MolecularPointGroup::description() const {
    switch (group) {
    case PointGroup::C1:
        return "C1";
    case PointGroup::Cs:
        return "mirror plane";
    case PointGroup::Ci:
        return "inversion center";
    case PointGroup::Coov:
        return "linear";
    case PointGroup::Dooh:
        return "linear + inversion center";
    case PointGroup::C2:
        return "open book geometry";
    case PointGroup::C3:
        return "propeller";
    case PointGroup::C2v:
        return "angular, see-saw, T-shape";
    case PointGroup::C3v:
        return "trigonal pyramidal";
    case PointGroup::C4v:
        return "square pyramidal";
    case PointGroup::C5v:
        return "milking stool";
    case PointGroup::D2:
        return "twist";
    case PointGroup::D2h:
        return "planar + inversion center";
    case PointGroup::D3h:
        return "trigonal planar, trigonal bipyramidal";
    case PointGroup::D4h:
        return "square planar";
    case PointGroup::D5h:
        return "pentagonal";
    case PointGroup::D6h:
        return "hexagonal";
    case PointGroup::D7h:
        return "heptagonal";
    case PointGroup::D8h:
        return "octagonal";
    case PointGroup::D2d:
        return "90 degree twist";
    case PointGroup::D3d:
        return "60 degree twist";
    case PointGroup::D4d:
        return "45 degree twist";
    case PointGroup::D5d:
        return "36 degree twist";
    case PointGroup::S4:
        return "S4";
    case PointGroup::Td:
        return "tetrahedral";
    case PointGroup::Th:
        return "pyritohedron";
    case PointGroup::Oh:
        return "octahedral, cubic";
    case PointGroup::Ih:
        return "icosahedral, dodecahedral";
    default:
        return "none";
    }
}

int pg_symmetry_number(PointGroup group) {
    switch (group) {
    case PointGroup::C1:
        return 1;
    case PointGroup::Cs:
        return 1;
    case PointGroup::Ci:
        return 1;
    case PointGroup::Coov:
        return 1;
    case PointGroup::Dooh:
        return 2;
    case PointGroup::C2:
        return 2;
    case PointGroup::C3:
        return 3;
    case PointGroup::C4:
        return 4;
    case PointGroup::C5:
        return 5;
    case PointGroup::C6:
        return 6;
    case PointGroup::C8:
        return 8;
    case PointGroup::C2v:
        return 2;
    case PointGroup::C3v:
        return 3;
    case PointGroup::C4v:
        return 4;
    case PointGroup::C5v:
        return 5;
    case PointGroup::C6v:
        return 6;
    case PointGroup::C2h:
        return 2;
    case PointGroup::C3h:
        return 3;
    case PointGroup::C4h:
        return 4;
    case PointGroup::C5h:
        return 5;
    case PointGroup::C6h:
        return 6;
    case PointGroup::D2:
        return 4;
    case PointGroup::D3:
        return 6;
    case PointGroup::D4:
        return 8;
    case PointGroup::D5:
        return 10;
    case PointGroup::D6:
        return 12;
    case PointGroup::D7:
        return 14;
    case PointGroup::D8:
        return 16;
    case PointGroup::D2h:
        return 4;
    case PointGroup::D3h:
        return 6;
    case PointGroup::D4h:
        return 8;
    case PointGroup::D5h:
        return 10;
    case PointGroup::D6h:
        return 12;
    case PointGroup::D7h:
        return 14;
    case PointGroup::D8h:
        return 16;
    case PointGroup::D2d:
        return 4;
    case PointGroup::D3d:
        return 6;
    case PointGroup::D4d:
        return 8;
    case PointGroup::D5d:
        return 10;
    case PointGroup::D6d:
        return 12;
    case PointGroup::D7d:
        return 14;
    case PointGroup::D8d:
        return 16;
    case PointGroup::S4:
        return 4;
    case PointGroup::S6:
        return 6;
    case PointGroup::S8:
        return 8;
    case PointGroup::T:
        return 12;
    case PointGroup::Td:
        return 12;
    case PointGroup::Th:
        return 12;
    case PointGroup::O:
        return 24;
    case PointGroup::Oh:
        return 24;
    case PointGroup::I:
        return 60;
    case PointGroup::Ih:
        return 60;
    default:
        throw std::runtime_error("Unknown point group");
    }
}

int MolecularPointGroup::symmetry_number() const {
    return pg_symmetry_number(group);
}

} // end namespace occ::core
