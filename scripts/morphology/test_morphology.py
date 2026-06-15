"""Tests for the particle morphology prototype.

These need an ``occ cg ... --surface-energies N`` results JSON and its CIF.  Point the test
at them with ``OCC_MORPH_JSON`` / ``OCC_MORPH_CIF`` (or drop them at the default path below);
the tests skip cleanly if the data is absent.

    OCC_DATA_PATH=$HOME/git/occ/share \
    OCC_MORPH_JSON=tmp/morph/acetic_acid_water_cg_results.json \
    OCC_MORPH_CIF=tmp/morph/acetic_acid.cif \
    .venv/bin/python -m pytest scripts/morphology/test_morphology.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

JSON = os.environ.get("OCC_MORPH_JSON", "tmp/morph/acetic_acid_water_cg_results.json")
CIF = os.environ.get("OCC_MORPH_CIF", "tmp/morph/acetic_acid.cif")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(JSON) and os.path.exists(CIF)),
    reason=f"missing cg fixture ({JSON} / {CIF}); generate with `occ cg --surface-energies 80`",
)


@pytest.fixture(scope="module")
def model():
    from model import MorphologyModel
    return MorphologyModel.from_cg_json(JSON, CIF)


@pytest.fixture(scope="module")
def poly(model):
    from shape import wulff_shape
    return wulff_shape(model.cg)


def test_facet_energy_reproduces_json(model):
    """The stamped uc_dimers reproduce occ's reported facet energies exactly."""
    worst = 0.0
    for f in model.cg.unique_facets():
        gamma, _ = model.facet_energy(f.hkl, f.offset)
        worst = max(worst, abs(gamma - f.energy))
    assert worst < 1e-4, f"facet energy mismatch {worst}"


def test_facet_gamma_self_consistent(model):
    """Analytic facet gamma from counts x solvated matches the JSON energy."""
    for f in model.cg.facets:
        assert abs(model.cg.facet_gamma(f) - f.energy) < 1e-6


def test_wulff_is_valid_polyhedron(poly):
    """Euler characteristic V - E + F == 2 for a closed convex polyhedron."""
    assert len(poly.corners) - len(poly.edges) + len(poly.faces) == 2
    assert poly.volume > 0
    assert len(poly.faces) >= 4


def test_excess_positive_and_surface_scaling(model, poly):
    """Excess energy is positive and grows ~ N^(2/3) (surface dominated)."""
    from energy import cluster_excess, size_for_n_molecules

    data = []
    for n in (2000, 16000):
        s = size_for_n_molecules(model, poly, n)
        e, nmol = cluster_excess(model, poly, s, n_registry=2)
        assert e > 0
        data.append((nmol, e))
    (n1, e1), (n2, e2) = data
    ratio = (e2 / e1) / (n2 / n1) ** (2 / 3)
    assert 0.8 < ratio < 1.25, f"excess does not scale ~N^2/3 (ratio {ratio:.2f})"


def test_exact_vs_analytic_surface_density(model, poly):
    """Registry-minimised cluster surface density is close to the analytic optimal-cut value."""
    from cg_data import KJ_PER_MOL_TO_J_PER_M2
    from energy import cluster_excess, size_for_n_molecules

    s = size_for_n_molecules(model, poly, 16000)
    e, _ = cluster_excess(model, poly, s, n_registry=3)
    sigma_cluster = e / poly.total_area(s)
    sigma_analytic = sum((f.distance / KJ_PER_MOL_TO_J_PER_M2) * f.area for f in poly.faces) / sum(
        f.area for f in poly.faces
    )
    # the convex centroid-hull over-estimates the optimal-cut surface energy by a
    # shape-dependent amount (small for simple shapes, larger for many-facet shapes);
    # registry minimisation brings it close but not exact.
    assert 0.9 < sigma_cluster / sigma_analytic < 1.6


def test_cpp_morphology_block():
    """The C++ `occ cg --morphology` block (if present) is self-consistent."""
    import json

    with open(JSON) as fh:
        data = json.load(fh)
    if "morphology" not in data:
        pytest.skip("no morphology block (run occ cg with --morphology)")
    m = data["morphology"]
    assert m["molecular_volume"] > 0
    assert len(m["facets"]) >= 4
    assert m["samples"]
    for s in m["samples"]:
        # surface + edge + corner attribution sums to the exact excess
        assert s["e_surface"] + s["e_edge"] + s["e_corner"] == pytest.approx(
            s["e_excess"], rel=1e-6
        )
        assert s["e_excess"] > 0
        assert s["e_surface_analytic"] > 0
    # excess grows ~ N^(2/3)
    s0, s1 = m["samples"][0], m["samples"][-1]
    ratio = (s1["e_excess"] / s0["e_excess"]) / (
        s1["n_molecules"] / s0["n_molecules"]
    ) ** (2 / 3)
    assert 0.7 < ratio < 1.3


def test_crossover_runs(model, poly):
    """Polymorph machinery returns a sane crossover result comparing a model to itself."""
    from polymorph import crossover_size, g_of_size

    a = g_of_size(model, poly, [2000, 8000], n_registry=2, label="A")
    b = g_of_size(model, poly, [2000, 8000], n_registry=2, label="B")
    n_cross, small, large = crossover_size(a, b)
    # identical models => no crossover, and G(N) is finite
    assert n_cross is None
    assert np.all(np.isfinite(a.g))
