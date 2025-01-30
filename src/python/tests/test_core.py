import pytest
import numpy as np
from occpy import (
    Dimer,
    AveragingScheme,
    eem_partial_charges,
    eeq_partial_charges,
    eeq_coordination_numbers,
    MolecularPointGroup,
    Molecule,
    Multipole,
    SymOp,
    Element,
    ElasticTensor,
    quasirandom_kgf,
)

import math
import tempfile
import os


def water_molecule():
    """Helper function to create a water molecule"""
    pos = np.array(
        [
            [-1.32695761, -1.93166418, 0.48664409],
            [-0.10593856, 1.60017351, 0.07959806],
            [0.01878821, -0.02171049, 0.00986248],
        ]
    )
    nums = np.array([8, 1, 1])
    return Molecule(nums, pos)


def oh_molecule():
    """Helper function to create an OH molecule"""
    pos = np.array(
        [
            [-1.32695761, -1.93166418],
            [-0.10593856, 1.60017351],
            [0.01878821, -0.02171049],
        ]
    )
    nums = np.array([8, 1])
    return Molecule(nums, pos)


class TestDimer:
    def test_transform(self):
        from occpy import Origin

        nums = np.array([8, 1, 1])
        pos = np.array(
            [
                [-1.32695761, -1.93166418, 0.48664409],
                [-0.10593856, 1.60017351, 0.07959806],
                [0.01878821, -0.02171049, 0.00986248],
            ]
        )
        pos2 = pos.copy()
        pos2[0, :] *= -1

        m1 = Molecule(nums, pos)
        m2 = Molecule(nums, pos2)
        dim = Dimer(m1, m2)

        transform = dim.symmetry_relation()
        m1.transform(transform, origin=Origin.CENTROID)
        assert np.allclose(m1.positions, m2.positions, rtol=1e-5, atol=1e-5)

    def test_separations(self):
        nums = np.array([8, 1, 1])
        pos = np.array(
            [
                [-1.32695761, -1.93166418, 0.48664409],
                [-0.10593856, 1.60017351, 0.07959806],
                [0.01878821, -0.02171049, 0.00986248],
            ]
        )
        pos2 = pos.copy()
        pos2[0, :] *= -1

        m1 = Molecule(nums, pos)
        m2 = Molecule(nums, pos2)
        dim = Dimer(m1, m2)

        assert dim.nearest_distance == pytest.approx(0.8605988136)
        assert dim.centroid_distance == pytest.approx(1.8479851333)
        assert dim.center_of_mass_distance == pytest.approx(2.5186418514)


class TestElement:
    def test_constructor(self):
        assert Element("H").symbol == "H"
        assert Element("He").symbol == "He"
        assert Element("He1").symbol == "He"
        assert Element(6).name == "carbon"
        assert Element("Ne") > Element("H")
        assert Element("NA").symbol == "N"
        assert Element("Na").symbol == "Na"


class TestMolecule:
    def test_constructor(self):
        pos = np.array([[-1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
        nums = np.array([1, 1])
        m = Molecule(nums, pos)
        assert len(m) == 2

    def test_atom_properties(self):
        nums = np.array([8, 1, 1])
        pos = np.array(
            [
                [-1.32695761, -0.10593856, 0.01878821],
                [-1.93166418, 1.60017351, -0.02171049],
                [0.48664409, 0.07959806, 0.00986248],
            ]
        )
        m = Molecule(nums, pos)
        masses = m.atomic_masses()
        expected_masses = np.array([15.994, 1.00794, 1.00794])
        assert np.allclose(masses, expected_masses, rtol=1e-3, atol=1e-3)

    def test_centroids(self):
        nums = np.array([8, 1, 1])
        pos = np.array(
            [
                [-1.32695761, -1.93166418, 0.48664409],
                [-0.10593856, 1.60017351, 0.07959806],
                [0.01878821, -0.02171049, 0.00986248],
            ]
        )
        m = Molecule(nums, pos)

        expected_centroid = np.array([-0.92399257, 0.524611, 0.0023134])
        calc_centroid = m.centroid()
        assert np.allclose(expected_centroid, calc_centroid, rtol=1e-5, atol=1e-5)

        expected_com = np.array([-1.25932093, -0.000102380208, 0.0160229578])
        calc_com = m.center_of_mass()
        assert np.allclose(expected_com, calc_com, rtol=1e-5, atol=1e-5)

    def test_rotation_translation(self):
        nums = np.array([8, 1, 1])
        pos = np.array(
            [
                [-1.32695761, -1.93166418, 0.48664409],
                [-0.10593856, 1.60017351, 0.07959806],
                [0.01878821, -0.02171049, 0.00986248],
            ]
        )
        m = Molecule(nums, pos)

        # 360 degree rotation around Y axis
        rotation360 = np.array(
            [
                [math.cos(2 * math.pi), 0, math.sin(2 * math.pi)],
                [0, 1, 0],
                [-math.sin(2 * math.pi), 0, math.cos(2 * math.pi)],
            ]
        )
        m.rotate(rotation360)
        assert np.allclose(pos, m.positions)

        # 180 degree rotation around X axis
        rotation180 = np.array(
            [
                [1, 0, 0],
                [0, math.cos(math.pi), -math.sin(math.pi)],
                [0, math.sin(math.pi), math.cos(math.pi)],
            ]
        )
        expected_pos = pos.copy()
        expected_pos[1:, :] *= -1
        m.rotate(rotation180)
        assert np.allclose(expected_pos, m.positions)


class TestMultipole:
    def test_constructor(self):
        c = Multipole(order=0)
        d = Multipole(order=1)
        q = Multipole(order=2)
        o = Multipole(order=3)
        assert c.charge == 0.0

    def test_addition(self):
        o = Multipole(
            order=3,
            components=[
                1.0,
                0.0,
                0.0,
                0.5,
                6.0,
                5.0,
                4.0,
                3.0,
                2.0,
                1.0,
                10.0,
                9.0,
                8.0,
                7.0,
                6.0,
                5.0,
                4.0,
                3.0,
                2.0,
                1.0,
            ],
        )
        q = Multipole(
            order=2, components=[1.0, 0.0, 0.0, 0.5, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        )
        sum_oq = o + q

        for i in range(o.num_components):
            if i < q.num_components:
                assert sum_oq.components[i] == (o.components[i] + q.components[i])
            else:
                assert sum_oq.components[i] == o.components[i]


class TestPointGroup:
    def test_water_c2v(self):
        nums = np.array([8, 1, 1])
        pos = np.array(
            [
                [-0.7021961, -0.0560603, 0.0099423],
                [-1.0221932, 0.8467758, -0.0114887],
                [0.2575211, 0.0421215, 0.0052190],
            ]
        )
        m = Molecule(nums, pos.T)
        pg = MolecularPointGroup(m)
        assert pg.point_group_string == "C2v"
        assert pg.symmetry_number == 2

    def test_o2_dooh(self):
        nums = np.array([8, 8])
        pos = np.array([[-0.616, 0.616], [0, 0], [0, 0]])
        m = Molecule(nums, pos)
        pg = MolecularPointGroup(m)
        assert pg.point_group_string == "Dooh"
        assert pg.symmetry_number == 2


class TestCharges:
    def test_eem_water(self):
        nums = np.array([8, 1, 1])
        pos = np.array(
            [
                [-0.7021961, -1.0221932, 0.2575211],
                [-0.0560603, 0.8467758, 0.0421215],
                [0.0099423, -0.0114887, 0.0052190],
            ]
        )
        q = eem_partial_charges(nums, pos, 0.0)
        expected_q = np.array([-0.637207, 0.319851, 0.317356])
        assert np.allclose(expected_q, q, rtol=1e-5, atol=1e-5)

    def test_eeq_water(self):
        nums = np.array([8, 1, 1])
        pos = np.array(
            [
                [-0.7021961, -1.0221932, 0.2575211],
                [-0.0560603, 0.8467758, 0.0421215],
                [0.0099423, -0.0114887, 0.0052190],
            ]
        )
        cn = eeq_coordination_numbers(nums, pos)
        expected_cn = np.array([2.0, 1.00401, 1.00401])
        assert np.allclose(cn, expected_cn, rtol=1e-3, atol=1e-3)

        q = eeq_partial_charges(nums, pos, 0.0)
        expected_q = np.array([-0.591878, 0.296985, 0.294893])
        assert np.allclose(expected_q, q, rtol=1e-5, atol=1e-5)


class TestElasticTensor:
    def tensor_matrix(self):
        return np.array(
            [
                [48.137, 11.411, 12.783, 0.000, -3.654, 0.000],
                [11.411, 34.968, 14.749, 0.000, -0.094, 0.000],
                [12.783, 14.749, 26.015, 0.000, -4.528, 0.000],
                [0.000, 0.000, 0.000, 14.545, 0.000, 0.006],
                [-3.654, -0.094, -4.528, 0.000, 10.771, 0.000],
                [0.000, 0.000, 0.000, 0.006, 0.000, 11.947],
            ]
        )

    @pytest.fixture
    def tensor(self):
        """Create a test elastic tensor"""
        return ElasticTensor(self.tensor_matrix())

    def test_basics(self, tensor):
        """Test basic tensor properties"""
        np.testing.assert_allclose(
            tensor.voigt_c, self.tensor_matrix(), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            tensor.voigt_s, np.linalg.inv(self.tensor_matrix()), rtol=1e-6, atol=1e-6
        )

    def test_voigt_averages(self, tensor):
        """Test Voigt averaging scheme"""
        assert tensor.average_bulk_modulus(AveragingScheme.VOIGT) == pytest.approx(
            20.778, rel=1e-3
        )
        assert tensor.average_youngs_modulus(AveragingScheme.VOIGT) == pytest.approx(
            30.465, rel=1e-3
        )
        assert tensor.average_shear_modulus(AveragingScheme.VOIGT) == pytest.approx(
            12.131, rel=1e-3
        )
        assert tensor.average_poisson_ratio(AveragingScheme.VOIGT) == pytest.approx(
            0.25564, rel=1e-4
        )

    def test_reuss_averages(self, tensor):
        """Test Reuss averaging scheme"""
        assert tensor.average_bulk_modulus(AveragingScheme.REUSS) == pytest.approx(
            19.000, rel=1e-3
        )
        assert tensor.average_youngs_modulus(AveragingScheme.REUSS) == pytest.approx(
            27.087, rel=1e-3
        )
        assert tensor.average_shear_modulus(AveragingScheme.REUSS) == pytest.approx(
            10.728, rel=1e-3
        )
        assert tensor.average_poisson_ratio(AveragingScheme.REUSS) == pytest.approx(
            0.26239, rel=1e-4
        )

    def test_hill_averages(self, tensor):
        """Test Hill averaging scheme"""
        assert tensor.average_bulk_modulus(AveragingScheme.HILL) == pytest.approx(
            19.889, rel=1e-3
        )
        assert tensor.average_youngs_modulus(AveragingScheme.HILL) == pytest.approx(
            28.777, rel=1e-3
        )
        assert tensor.average_shear_modulus(AveragingScheme.HILL) == pytest.approx(
            11.43, rel=1e-3
        )
        assert tensor.average_poisson_ratio(AveragingScheme.HILL) == pytest.approx(
            0.25886, rel=1e-4
        )

    def test_youngs_modulus(self, tensor):
        """Test Young's modulus calculations"""
        dmin = np.array([0.3540, 0.0, 0.9352])
        dmax = np.array([0.9885, 0.0000, -0.1511])

        ymin = tensor.youngs_modulus(dmin)
        assert ymin == pytest.approx(14.751, rel=1e-2)

        ymax = tensor.youngs_modulus(dmax)
        assert ymax == pytest.approx(41.961, rel=1e-2)

    def test_linear_compressibility(self, tensor):
        """Test linear compressibility calculations"""
        dmin = np.array([0.9295, -0.0000, -0.3688])
        dmax = np.array([0.3688, -0.0000, 0.9295])

        lcmin = tensor.linear_compressibility(dmin)
        assert lcmin == pytest.approx(8.2545, rel=1e-2)

        lcmax = tensor.linear_compressibility(dmax)
        assert lcmax == pytest.approx(31.357, rel=1e-2)

    def test_shear_modulus(self, tensor):
        """Test shear modulus calculations"""
        d1min = np.array([-0.2277, 0.7071, -0.6694])
        d2min = np.array([-0.2276, -0.7071, -0.6695])

        d1max = np.array([0.7352, 0.6348, 0.2378])
        d2max = np.array([-0.6612, 0.5945, 0.4575])

        smin = tensor.shear_modulus(d1min, d2min)
        assert smin == pytest.approx(6.5183, rel=1e-2)

        smax = tensor.shear_modulus(d1max, d2max)
        assert smax == pytest.approx(15.505, rel=1e-2)

    def test_poisson_ratio(self, tensor):
        """Test Poisson's ratio calculations"""
        d1min = np.array([0.5593, 0.6044, 0.5674])
        d2min = np.array([0.0525, 0.6572, -0.7519])

        d1max = np.array([0.0, 1.0, -0.0])
        d2max = np.array([-0.2611, -0.0000, -0.9653])

        vmin = tensor.poisson_ratio(d1min, d2min)
        assert vmin == pytest.approx(0.067042, rel=1e-2)

        vmax = tensor.poisson_ratio(d1max, d2max)
        assert vmax == pytest.approx(0.59507, rel=1e-2)


def test_quasirandom():
    pts = quasirandom_kgf(3, 5, 10)
    expected = np.array(
        [
            [
                0.5108976473578082,
                0.3300701607539729,
                0.1492426741501376,
                0.9684151875463005,
                0.7875877009424652,
            ],
            [
                0.8814796737416799,
                0.5525232804454685,
                0.2235668871492571,
                0.8946104938530475,
                0.5656541005568361,
            ],
            [
                0.5467052569216708,
                0.0964057348236409,
                0.646106212725611,
                0.19580669062758105,
                0.7455071685295511,
            ],
        ]
    )
    assert np.allclose(pts, expected)
