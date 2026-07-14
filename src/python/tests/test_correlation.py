import pytest
import numpy as np
from occpy import (
    Atom,
    AOBasis,
    HartreeFock,
    SpinorbitalKind,
    CorrelationOptions,
    FittingKind,
    MP2,
    ccsd,
    ccsd_t,
    exact_eris,
    df_eris,
    num_frozen_core,
    resolve_fitting_basis,
    run_correlation,
)


@pytest.fixture
def water_atoms():
    return [
        Atom(8, -1.32695761, -0.10593856, 0.01878821),
        Atom(1, -1.93166418, 1.60017351, -0.02171049),
        Atom(1, 0.48664409, 0.07959806, 0.00986248),
    ]


@pytest.fixture
def water_wfn(water_atoms):
    basis = AOBasis.load(water_atoms, "6-31G")
    hf = HartreeFock(basis)
    scf = hf.scf()
    scf.convergence_settings.energy_threshold = 1e-10
    scf.compute_scf_energy()
    return scf.wavefunction()


def test_run_correlation_mp2(water_wfn):
    res = run_correlation(water_wfn, method="mp2")
    assert res.method == "MP2"
    assert res.correlation_energy < 0
    assert pytest.approx(res.total_energy, abs=1e-8) == (
        res.scf_energy + res.correlation_energy
    )


def test_run_correlation_ri_mp2_auto_aux(water_wfn):
    res = run_correlation(water_wfn, method="ri-mp2")
    assert res.method == "RI-MP2"
    assert res.correlation_energy < 0
    # RI should be close to conventional
    conv = run_correlation(water_wfn, method="mp2")
    assert pytest.approx(res.correlation_energy, abs=2e-3) == (
        conv.correlation_energy
    )


def test_run_correlation_ccsd_t(water_wfn):
    res = run_correlation(water_wfn, method="ccsd(t)")
    assert res.method == "CCSD(T)"
    assert res.converged
    assert res.n_frozen == 1
    assert res.triples_correction < 0
    assert pytest.approx(res.ccsd_correlation, abs=1e-8) == (
        res.correlation_energy - res.triples_correction
    )


def test_run_correlation_options_object(water_wfn):
    opts = CorrelationOptions()
    opts.method = "ccsd"
    opts.backend = "df"
    res = run_correlation(water_wfn, opts)
    assert res.method == "CCSD"
    assert res.converged


def test_mp2_class_low_level(water_wfn):
    mp2 = MP2(water_wfn.basis, water_wfn.molecular_orbitals, water_wfn.total_energy)
    mp2.set_frozen_core_auto()
    e_corr = mp2.compute_correlation_energy()
    assert e_corr < 0
    r = mp2.results
    assert r.n_frozen_core == 1
    assert pytest.approx(mp2.total_energy, abs=1e-10) == (
        water_wfn.total_energy + e_corr
    )
    # matches the high-level entry point
    high = run_correlation(water_wfn, method="mp2")
    assert pytest.approx(e_corr, abs=1e-9) == high.correlation_energy


def test_ccsd_low_level_amplitudes(water_wfn):
    nfz = num_frozen_core(water_wfn.basis)
    eris = exact_eris(water_wfn.basis, water_wfn.molecular_orbitals, nfz)
    res = ccsd(eris)
    assert res.converged
    t1 = res.t1
    t2 = res.t2
    assert t1.shape == (eris.nocc, eris.nvir)
    assert t2.shape == (eris.nocc, eris.nocc, eris.nvir, eris.nvir)
    assert np.all(np.isfinite(t2))
    et = ccsd_t(res, eris)
    assert et < 0
    # matches the high-level entry point
    high = run_correlation(water_wfn, method="ccsd(t)")
    assert pytest.approx(res.e_corr, abs=1e-8) == high.ccsd_correlation
    assert pytest.approx(et, abs=1e-8) == high.triples_correction


def test_df_eris_backend(water_wfn):
    aux_name = resolve_fitting_basis("6-31G", FittingKind.Correlation)
    aux = AOBasis.load(water_wfn.basis.atoms(), aux_name)
    eris = df_eris(water_wfn.basis, aux, water_wfn.molecular_orbitals, 1)
    res = ccsd(eris)
    assert res.converged
    exact = run_correlation(water_wfn, method="ccsd")
    assert pytest.approx(res.e_corr, abs=2e-3) == exact.ccsd_correlation


def test_uhf_ccsd(water_atoms):
    from occpy import uccsd, UCCSDOptions

    basis = AOBasis.load(water_atoms, "STO-3G")
    hf = HartreeFock(basis)
    scf = hf.scf(SpinorbitalKind.Unrestricted)
    scf.set_charge_multiplicity(1, 2)
    scf.compute_scf_energy()
    wfn = scf.wavefunction()

    res = run_correlation(wfn, method="ccsd")
    assert res.method == "CCSD"
    assert res.converged
    assert res.correlation_energy < 0

    opts = UCCSDOptions()
    opts.n_frozen = 1
    opts.with_triples = False
    low = uccsd(wfn.basis, wfn.molecular_orbitals, opts)
    assert pytest.approx(low.e_corr, abs=1e-8) == res.correlation_energy
