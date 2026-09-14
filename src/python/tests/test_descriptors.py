import re
from pathlib import Path

import numpy as np
import pytest

from occpy import (
    Crystal,
    Rinse,
    RinseParams,
    RinseRadialBasis,
    rinse_hash,
    rinse_hash_to_bits,
    rinse_reflections,
)

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "tests" / "data" / "rinse"
REFERENCE = REPO / "tests" / "rinse_reference_data.h"

pytestmark = pytest.mark.skipif(
    not REFERENCE.exists(), reason="needs the occ source tree's RINSE test data"
)

CIF_FIXTURES = [
    "nacl.cif",
    "si.cif",
    "ylid.cif",
    "mpox.cif",
    "QEHWEG01_P21.cif",
    "QEHWEG03_P21212.cif",
]


def reference(name):
    """Hashes and descriptor for one fixture, read from the C++ reference data."""
    text = REFERENCE.read_text()
    start = text.index('{"%s",' % name)
    end = text.find('\n    {"', start + 1)
    block = text[start : end if end != -1 else len(text)]
    hashes = re.findall(r'"([a-z]+(?:-[a-z]+)*)"', block)
    numbers = re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+[eE][-+]?\d+", block)
    return hashes[0], hashes[1], np.array([float(x) for x in numbers[-128:]])


def load(name):
    return Crystal.from_cif_file(str(DATA / name))


@pytest.mark.parametrize("name", CIF_FIXTURES)
def test_rinse_matches_reference(name):
    one_word, five_words, expected = reference(name)
    descriptor = np.asarray(Rinse().compute(load(name)))
    assert descriptor.shape == (128,)
    np.testing.assert_allclose(descriptor, expected, rtol=0, atol=1e-13)
    assert rinse_hash(descriptor) == one_word
    assert rinse_hash(descriptor, 5) == five_words


def test_rinse_hash_round_trips_through_bits():
    descriptor = Rinse().compute(load("ylid.cif"))
    bits = rinse_hash_to_bits(rinse_hash(descriptor, 3))
    assert len(bits) == 48
    assert bits[:16] == rinse_hash_to_bits(rinse_hash(descriptor, 1))


def test_rinse_fixed_uiso_is_optional():
    params = RinseParams()
    assert params.fixed_uiso is None
    params.fixed_uiso = 0.025
    assert params.fixed_uiso == 0.025
    params.fixed_uiso = None
    assert params.fixed_uiso is None

    crystal = load("QEHWEG01_P21.cif")
    zero = RinseParams()
    zero.fixed_uiso = 0.0
    assert not np.array_equal(Rinse(zero).compute(crystal), Rinse().compute(crystal))


def test_rinse_power_spectrum_from_reflections_or_crystal():
    crystal = load("QEHWEG01_P21.cif")
    rinse = Rinse(params=RinseParams())
    reflections = rinse_reflections(crystal)
    assert reflections.hkl.shape == (3, len(reflections))
    spectrum = np.asarray(rinse.power_spectrum(reflections))
    assert spectrum.shape == (8, 16)
    np.testing.assert_array_equal(spectrum, rinse.power_spectrum(crystal))
    np.testing.assert_array_equal(Rinse.flatten(spectrum), rinse.compute(crystal))


def test_rinse_parameters():
    params = RinseParams()
    assert (params.n_max, params.l_min, params.l_max, params.size) == (8, 4, 36, 128)
    assert list(params.l_values) == list(range(4, 36, 2))
    params.radial_basis = RinseRadialBasis.SmoothShellsCW
    assert params.radial_basis == RinseRadialBasis.SmoothShellsCW


def test_rinse_rejects_bad_input():
    params = RinseParams()
    params.l_min = 3
    with pytest.raises(ValueError):
        Rinse(params)
    with pytest.raises(ValueError):
        rinse_hash(np.zeros(64))
