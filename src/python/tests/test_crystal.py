import pytest
import numpy as np
from occpy import (
    HKL,
    Crystal,
    AsymmetricUnit,
    SpaceGroup,
    UnitCell,
    Surface,
    SymmetryOperation,
    SymmetryOperationFormat,
)
import occpy
import math


@pytest.fixture
def acetic_asym():
    """Create asymmetric unit for acetic acid crystal"""
    labels = ["C1", "C2", "H1", "H2", "H3", "H4", "O1", "O2"]
    positions = np.array(
        [
            [0.16510, 0.28580, 0.17090],
            [0.08940, 0.37620, 0.34810],
            [0.18200, 0.05100, -0.11600],
            [0.12800, 0.51000, 0.49100],
            [0.03300, 0.54000, 0.27900],
            [0.05300, 0.16800, 0.42100],
            [0.12870, 0.10750, 0.00000],
            [0.25290, 0.37030, 0.17690],
        ]
    ).T
    atomic_numbers = np.array([6, 6, 1, 1, 1, 1, 8, 8])

    return AsymmetricUnit(positions, atomic_numbers, labels)


@pytest.fixture
def ice_ii_asym():
    """Create asymmetric unit for ice-II crystal"""
    labels = [
        "O1",
        "H1",
        "H2",
        "O2",
        "H3",
        "H4",
        "O3",
        "H5",
        "H6",
        "O4",
        "H7",
        "H8",
        "O5",
        "H9",
        "H10",
        "O6",
        "H11",
        "H12",
        "O7",
        "H13",
        "H14",
        "O8",
        "H15",
        "H16",
        "O9",
        "H17",
        "H18",
        "O10",
        "H19",
        "H20",
        "O11",
        "H21",
        "H22",
        "O12",
        "H23",
        "H24",
    ]

    atomic_numbers = np.array([8 if "O" in label else 1 for label in labels])

    positions = np.array(
        [
            0.273328954083,
            0.026479033257,
            0.855073668062,
            0.152000330304,
            0.043488909374,
            0.793595454907,
            0.420775085827,
            0.191165194485,
            0.996362203192,
            0.144924657237,
            0.726669877048,
            0.973520141937,
            0.206402797363,
            0.847998481439,
            0.956510183901,
            0.003636687868,
            0.579223433079,
            0.808833958746,
            0.026477491142,
            0.855072949204,
            0.273328854276,
            0.043487719387,
            0.793594459529,
            0.152000553858,
            0.191163388489,
            0.996362120061,
            0.420774953988,
            0.726670757782,
            0.973520932681,
            0.144926297633,
            0.847999275418,
            0.956510882297,
            0.206404294889,
            0.579224602173,
            0.808834869258,
            0.003637530197,
            0.855073412561,
            0.273329597478,
            0.026478702027,
            0.793594909621,
            0.152000771295,
            0.043488316376,
            0.996362312075,
            0.420775512814,
            0.191164826329,
            0.973520717390,
            0.144925628579,
            0.726671054509,
            0.956510600982,
            0.206403626100,
            0.847999547813,
            0.808834607385,
            0.003637609551,
            0.579224562315,
            0.477029330652,
            0.749805220756,
            0.331717174202,
            0.402360172390,
            0.720795433576,
            0.401054786853,
            0.368036378343,
            0.742284933413,
            0.207434128329,
            0.668282055550,
            0.522969467265,
            0.250193622013,
            0.598945169999,
            0.597639203188,
            0.279204514235,
            0.792565160978,
            0.631962548905,
            0.257714022497,
            0.749805496250,
            0.331717033025,
            0.477029827575,
            0.720795009402,
            0.401054437437,
            0.402360618546,
            0.742284706875,
            0.207433751728,
            0.368036342085,
            0.522969071341,
            0.250193392512,
            0.668282780114,
            0.597638176364,
            0.279203622225,
            0.598945231951,
            0.631962932785,
            0.257715003205,
            0.792566578018,
            0.331715381178,
            0.477028907327,
            0.749804544234,
            0.401053887354,
            0.402360576463,
            0.720795552111,
            0.207432480540,
            0.368035542438,
            0.742284142147,
            0.250193225247,
            0.668282913065,
            0.522970147212,
            0.279203658434,
            0.598945325854,
            0.597639149965,
            0.257715011998,
            0.792566781760,
            0.631964289620,
        ]
    ).reshape(3, -1)

    return AsymmetricUnit(positions, atomic_numbers, labels)


def test_asymmetric_unit_constructor(acetic_asym):
    """Test asymmetric unit construction and basic properties"""
    assert len(acetic_asym.labels) == 8
    assert len(acetic_asym.atomic_numbers) == 8
    assert acetic_asym.positions.shape == (3, 8)


def test_crystal_acetic_acid(acetic_asym):
    """Test crystal construction and basic properties for acetic acid"""
    sg = SpaceGroup(33)
    cell = occpy.orthorhombic_cell(13.31, 4.1, 5.75)

    crystal = Crystal(acetic_asym, sg, cell)

    assert len(crystal.unit_cell_molecules()) == 4
    assert len(crystal.symmetry_unique_molecules()) == 1


def test_crystal_dimers(acetic_asym):
    """Test crystal dimer analysis"""
    sg = SpaceGroup(33)
    cell = occpy.orthorhombic_cell(13.31, 4.1, 5.75)
    crystal = Crystal(acetic_asym, sg, cell)

    crystal_dimers = crystal.symmetry_unique_dimers(3.8)

    dimers = crystal_dimers.unique_dimers
    assert len(dimers) == 7

    mol_neighbors = crystal_dimers.molecule_neighbors
    assert len(mol_neighbors) > 0


def test_crystal_from_cif():
    """Test crystal creation from CIF file/string"""
    cif_string = """
    data_test
    _cell_length_a     13.31
    _cell_length_b     4.1
    _cell_length_c     5.75
    _cell_angle_alpha  90
    _cell_angle_beta   90
    _cell_angle_gamma  90
    _symmetry_space_group_name_H-M  'P n a 21'
    loop_
    _atom_site_label
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    C1 0.16510 0.28580 0.17090
    C2 0.08940 0.37620 0.34810
    """

    crystal = Crystal.from_cif_string(cif_string)
    assert crystal is not None


def test_unit_cell_transformations(acetic_asym):
    """Test coordinate transformations between fractional and cartesian"""
    sg = SpaceGroup(33)
    cell = occpy.orthorhombic_cell(13.31, 4.1, 5.75)
    crystal = Crystal(acetic_asym, sg, cell)

    frac_coords = np.random.rand(3, 100)
    cart_coords = crystal.to_cartesian(frac_coords)
    frac_coords_back = crystal.to_fractional(cart_coords)

    np.testing.assert_array_almost_equal(frac_coords, frac_coords_back)


def test_surface_construction(acetic_asym):
    """Test surface construction from Miller indices"""
    sg = SpaceGroup(33)
    cell = occpy.orthorhombic_cell(13.31, 4.1, 5.75)
    crystal = Crystal(acetic_asym, sg, cell)

    miller_indices = [
        (0, 1, 0),
        (0, 1, 1),
        (3, 2, 1),
    ]

    for h, k, l in miller_indices:
        hkl = HKL(h, k, l)
        surface = Surface(hkl, crystal)

        assert abs(np.linalg.norm(surface.depth_vector) - surface.depth()) < 1e-10
        assert surface.area() > 0


def test_unitcell_constructors():
    """Test various ways to construct a UnitCell"""
    cell = UnitCell(1.0, 2.0, 3.0, np.pi / 2, np.pi / 2, np.pi / 2)
    assert cell.a == pytest.approx(1.0)
    assert cell.b == pytest.approx(2.0)
    assert cell.c == pytest.approx(3.0)

    lengths = np.array([1.0, 2.0, 3.0])
    angles = np.array([np.pi / 2, np.pi / 2, np.pi / 2])
    cell2 = UnitCell(lengths, angles)
    assert cell2.a == pytest.approx(1.0)


def test_factory_functions():
    """Test the various factory functions for creating unit cells"""
    cubic = occpy.cubic_cell(5.0)
    assert cubic.is_cubic()
    assert cubic.a == pytest.approx(5.0)
    assert cubic.b == pytest.approx(5.0)
    assert cubic.c == pytest.approx(5.0)

    rhomb = occpy.rhombohedral_cell(5.0, np.pi / 3)
    assert rhomb.is_rhombohedral()

    ortho = occpy.orthorhombic_cell(2.0, 3.0, 4.0)
    assert ortho.is_orthorhombic()


def test_coordinate_transformations():
    """Test coordinate transformations between fractional and Cartesian"""
    cell = occpy.cubic_cell(5.0)

    frac_coords = np.array(
        [[0.5, 0.0], [0.5, 0.0], [0.5, 0.0]], dtype=np.float64, order="F"
    )

    cart_coords = cell.to_cartesian(frac_coords)
    frac_back = cell.to_fractional(cart_coords)

    np.testing.assert_array_almost_equal(frac_coords, frac_back)


def test_cell_type_checks():
    """Test methods for checking cell type"""
    cubic = occpy.cubic_cell(5.0)
    assert cubic.is_cubic()
    assert not cubic.is_triclinic()
    assert cubic.is_orthogonal()
    assert cubic.cell_type() == "cubic"

    triclinic = occpy.triclinic_cell(
        2.0, 3.0, 4.0, np.pi / 3, np.pi / 4, np.pi / 6
    )
    assert triclinic.is_triclinic()
    assert not triclinic.is_orthogonal()
    assert triclinic.cell_type() == "triclinic"


def test_adp_transformations():
    """Test ADP transformations"""
    cell = occpy.orthorhombic_cell(2.0, 3.0, 4.0)

    adp = np.array(
        [
            [1.0, 2.0],  # U11
            [1.0, 2.0],  # U22
            [1.0, 2.0],  # U33
            [0.0, 0.0],  # U12
            [0.0, 0.0],  # U13
            [0.0, 0.0],  # U23
        ],
        dtype=np.float64,
        order="F",
    )

    cart_adp = cell.to_cartesian_adp(adp)
    frac_back = cell.to_fractional_adp(cart_adp)

    np.testing.assert_array_almost_equal(adp, frac_back)


def test_hkl_limits():
    """Test HKL limits calculation"""
    cell = occpy.cubic_cell(5.0)
    limits = cell.hkl_limits(1.0)
    assert isinstance(limits.h, int)
    assert isinstance(limits.k, int)
    assert isinstance(limits.l, int)


def test_spacegroup_constructors():
    """Test various ways to construct a SpaceGroup"""
    sg1 = SpaceGroup()
    assert sg1.number() == 1  # P1

    sg2 = SpaceGroup(33)  # Pna21
    assert sg2.symbol == "P n a 21"
    assert len(sg2.symmetry_operations) == 4

    sg3 = SpaceGroup("P21/c")
    assert sg3.number() == 14
    assert sg3.short_name == "P21/c"

    symops = ["x,y,z", "-x,-y,-z"]
    sg4 = SpaceGroup(symops)
    assert len(sg4.symmetry_operations) == 2


def test_spacegroup_symmetry_operations():
    """Test applying symmetry operations"""
    sg = SpaceGroup(33)  # Pna21

    coords = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float64, order="F"
    )

    symop_indices, transformed_coords = sg.apply_all_symmetry_operations(coords)

    assert len(symop_indices) == 4 * coords.shape[1]
    assert transformed_coords.shape == (3, 4 * coords.shape[1])


def test_spacegroup_error_handling():
    """Test error handling in SpaceGroup construction"""
    with pytest.raises(ValueError):
        SpaceGroup(1000)

    with pytest.raises(ValueError):
        SpaceGroup("Invalid Symbol")


def test_spacegroup_hr_choice():
    """Test hexagonal/rhombohedral choice detection"""
    sg = SpaceGroup(146)  # R3
    assert sg.has_H_R_choice()

    sg2 = SpaceGroup(1)  # P1
    assert not sg2.has_H_R_choice()


def test_symmetry_operation_constructors():
    """Test various ways to construct a SymmetryOperation"""
    identity = SymmetryOperation("x,y,z")
    assert identity.is_identity()

    identity2 = SymmetryOperation(16484)
    assert identity2.is_identity()

    seitz = np.eye(4)
    identity3 = SymmetryOperation(seitz)
    assert identity3.is_identity()


def test_symmetry_operation_representations():
    """Test different representations of symmetry operations"""
    symop = SymmetryOperation("-x,y+1/2,-z")

    int_rep = symop.to_int()
    assert SymmetryOperation(int_rep) == symop

    str_rep = symop.to_string()
    assert SymmetryOperation(str_rep) == symop

    fmt = SymmetryOperationFormat()
    fmt.delimiter = " "
    custom_str = symop.to_string(fmt)
    assert " " in custom_str


def test_symmetry_operation_transformations():
    """Test coordinate transformations"""
    symop = SymmetryOperation("-x,y+1/2,-z")

    coords = np.array([[0.5, 0.0], [0.5, 0.0], [0.5, 0.0]], dtype=np.float64, order="F")

    transformed = symop.apply(coords)
    assert transformed.shape == coords.shape

    np.testing.assert_array_almost_equal(transformed[:, 0], np.array([-0.5, 1.0, -0.5]))


def test_symmetry_operation_composition():
    """Test composition of symmetry operations"""
    s1 = SymmetryOperation("-x,y+1/2,-z")
    s2 = SymmetryOperation("-x,-y,z")

    s3 = s1 * s2

    coords = np.array([[0.5, 0.0], [0.5, 0.0], [0.5, 0.0]], dtype=np.float64, order="F")

    result1 = s3(coords)
    result2 = s1(s2(coords))
    np.testing.assert_array_almost_equal(result1, result2)


def test_symmetry_operation_adp_rotation():
    """Test rotation of ADPs"""
    symop = SymmetryOperation("-x,y+1/2,-z")

    adp = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float64)

    rotated_adp = symop.rotate_adp(adp)
    assert rotated_adp.shape == adp.shape

    identity = SymmetryOperation("x,y,z")
    np.testing.assert_array_almost_equal(identity.rotate_adp(adp), adp)


def test_symmetry_operation_cartesian():
    """Test cartesian coordinate transformations"""
    symop = SymmetryOperation("-x,y+1/2,-z")
    cell = UnitCell(1.0, 1.0, 1.0, np.pi / 2, np.pi / 2, np.pi / 2)

    cart_rot = symop.cartesian_rotation(cell)
    assert cart_rot.shape == (3, 3)


def test_symmetry_operation_comparisons():
    """Test comparison operators"""
    s1 = SymmetryOperation("-x,y+1/2,-z")
    s2 = SymmetryOperation("-x,y+1/2,-z")
    s3 = SymmetryOperation("x,y,z")

    assert s1 == s2
    assert s1 != s3

    assert (s1 < s3) or (s1 > s3)
    assert s1 <= s2
    assert s1 >= s2
