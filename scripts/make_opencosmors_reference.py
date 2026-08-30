"""Generate openCOSMO-RS 24a reference data from the TUHH-TVT implementation.

Emits, for the .orcacosmo molecules bundled with openCOSMO-RS_py:
  * per-segment input (position, area, raw sigma, element) and the sigma and
    sigma_orth the reference derives from it, so occ can be checked on
    byte-identical input;
  * ln(gamma) split into residual and combinatorial parts for binaries, in
    both the pure-component and ideal-conductor reference states.

Run against an environment with opencosmorspy installed, from the directory
holding the .orcacosmo files:

    python scripts/make_opencosmors_reference.py tests/data/opencosmors \
        tests/data/opencosmors_reference.json
"""
import json
import os
import sys

import numpy as np

from opencosmorspy.cosmors import COSMORS
from opencosmorspy.parameterization import openCOSMORS24a

# Segments are stored for every molecule with a measured hydration free
# energy, so the solvation assembly can be checked across hydrogen-bonding
# and non-hydrogen-bonding solutes rather than on a couple of points.
SEGMENT_MOLECULES = [
    "water", "methanol", "ethanol", "acetone",
    "benzene", "cyclohexane", "acetic_acid",
]

MOLECULES = [
    "water", "methanol", "ethanol", "acetone",
    "benzene", "cyclohexane", "acetic_acid",
]

SYSTEMS = [
    ("water", "methanol"),
    ("water", "acetone"),
    ("methanol", "benzene"),
    ("benzene", "cyclohexane"),
    ("acetic_acid", "water"),
]

COMPOSITIONS = [0.05, 0.25, 0.5, 0.75, 0.95]
TEMPERATURES = [283.15, 298.15, 348.15]

# The reference clusters segments onto a (sigma, sigma_orth) grid before the
# kernel sees them, and its default 0.001 e/A^2 step costs ~0.03 in ln(gamma).
# occ does not cluster, so the reference is generated on a fine grid where
# that discretisation is small and the two agree in the same limit.
GRID_STEP = 1e-4


def parameterization():
    par = openCOSMORS24a()
    par.sigma_step = GRID_STEP
    par.sigma_grid = np.arange(par.sigma_min, par.sigma_max + GRID_STEP,
                               GRID_STEP)
    par.sigma_orth_step = GRID_STEP
    par.sigma_orth_grid = np.arange(par.sigma_orth_min,
                                    par.sigma_orth_max + GRID_STEP, GRID_STEP)
    return par


def parameter_block(par):
    return {
        "a_eff": par.a_eff,
        "r_av": par.r_av,
        "mf_alpha": par.mf_alpha,
        "mf_r_av_corr": par.mf_r_av_corr,
        "mf_f_corr": par.mf_f_corr,
        "mf_use_sigma_orth": bool(par.mf_use_sigma_orth),
        "hb_c": par.hb_c,
        "hb_c_T": par.hb_c_T,
        "hb_sigma_thresh": par.hb_sigma_thresh,
        "comb_term": par.comb_term,
        "comb_sg_z_coord": par.comb_sg_z_coord,
        "comb_sg_a_std": par.comb_sg_a_std,
        "tau": {
            str(z): getattr(par, f"tau_{z}")
            for z in (1, 6, 7, 8, 9, 14, 15, 16, 17, 35)
        },
        "eta": par.eta,
        "omega_ring": par.omega_ring,
    }


def dielectric_energy(path):
    """Gas to ideal conductor, kJ/mol, from the '# CPCM dielectric energy'
    line of the .orcacosmo file.

    Read here rather than taken from `struct.energy_dielectric`: that field
    holds the outlying-charge-corrected value where the file has one, and is
    None where it does not. occ does not apply that correction, so the
    uncorrected value is the like-for-like quantity.
    """
    hartree_to_kj = 2625.4996394798254
    for line in open(path):
        if "# CPCM dielectric energy" in line:
            return float(line.split()[0]) * hartree_to_kj
    raise ValueError(f"{path} has no CPCM dielectric energy")


def segment_block(struct, path):
    """Raw per-segment input plus the descriptors derived from it."""
    return {
        "energy_dielectric": dielectric_energy(path),
        "positions": np.asarray(struct.seg_pos).tolist(),
        "areas": np.asarray(struct.seg_area).tolist(),
        "sigma_raw": np.asarray(struct.seg_sigma_raw).tolist(),
        "atom_index": np.asarray(struct.seg_atm_nr).tolist(),
        # Element numbers after the reference remaps hydrogen to
        # 100 + Z(bonded atom).
        "element": np.asarray(struct.seg_elmnt_nr).tolist(),
        "atom_element": np.asarray(struct.atm_elmnt_nr).tolist(),
        "sigma": np.asarray(struct.seg_sigma).tolist(),
        "sigma_orth": np.asarray(struct.seg_sigma_orth).tolist(),
        "area": struct.area,
        "volume": struct.volume,
        "screening_charge": struct.screen_charge,
    }


def main(directory, destination):
    par = parameterization()
    out = {"model": "openCOSMO-RS 24a", "parameters": parameter_block(par)}

    # Per-segment descriptors. A fresh COSMORS per molecule keeps the segment
    # type collection from being shared, which does not affect the segment
    # data but keeps each block independent.
    out["segments"] = {}
    for name in SEGMENT_MOLECULES:
        path = os.path.join(directory, name + ".orcacosmo")
        crs = COSMORS(parameterization())
        crs.add_molecule([path])
        struct = crs.enth.mol_lst[0].cosmo_struct_lst[0]
        out["segments"][name] = segment_block(struct, path)
        print(f"{name}: {len(struct.seg_area)} segments")

    out["molecules"] = {}
    for name in MOLECULES:
        crs = COSMORS(parameterization())
        crs.add_molecule([os.path.join(directory, name + ".orcacosmo")])
        struct = crs.enth.mol_lst[0].cosmo_struct_lst[0]
        out["molecules"][name] = {
            "area": struct.area,
            "volume": struct.volume,
            "num_segments": len(struct.seg_area),
        }

    out["cases"] = []
    for a, b in SYSTEMS:
        for reference_state in ("pure_component", "cosmo"):
            crs = COSMORS(parameterization())
            crs.add_molecule([os.path.join(directory, a + ".orcacosmo")])
            crs.add_molecule([os.path.join(directory, b + ".orcacosmo")])
            for T in TEMPERATURES:
                for x in COMPOSITIONS:
                    crs.add_job(np.array([x, 1.0 - x]), T, reference_state)
            results = crs.calculate()
            for index in range(len(results["T"])):
                out["cases"].append({
                    "components": [a, b],
                    "reference_state": reference_state,
                    "T": float(results["T"][index]),
                    "z": results["x"][index].tolist(),
                    "lngamma": results["tot"]["lng"][index].tolist(),
                    "lngamma_resid": results["enth"]["lng"][index].tolist(),
                    "lngamma_comb": results["comb"]["lng"][index].tolist(),
                })
            print(f"{a}/{b} [{reference_state}]: {len(TEMPERATURES) * len(COMPOSITIONS)} cases")

    with open(destination, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {destination}: {len(out['cases'])} cases, "
          f"{len(out['segments'])} segment blocks")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
