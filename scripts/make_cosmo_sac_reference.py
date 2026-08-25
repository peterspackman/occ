"""Generate COSMO-SAC reference data from the NIST cCOSMO implementation.

Emits, for a set of binary systems:
  * the sigma profiles used (so occ reads byte-identical input),
  * ln(gamma) split into residual and combinatorial parts,
for both COSMO-SAC 2002 (COSMO1, 1 profile) and 2010 (COSMO3, 3 profiles).
"""
import json
import os
import sys

import cCOSMO

UD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "COSMOSAC",
                  "profiles", "UD")

KEYS = {
    "water": "XLYOFNOQVPJJNP-UHFFFAOYSA-N",
    "ethanol": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
    "methanol": "OKKJLVBELUTLKV-UHFFFAOYSA-N",
    "benzene": "UHOVQNZJYSORNB-UHFFFAOYSA-N",
    "n-hexane": "VLKZOEOYAKHREP-UHFFFAOYSA-N",
    "acetone": "CSCPPACGZOOCGX-UHFFFAOYSA-N",
    "chloroform": "HEDRZPFGACZZDS-UHFFFAOYSA-N",
}

SYSTEMS = [
    ("water", "ethanol"),
    ("water", "acetone"),
    ("methanol", "benzene"),
    ("benzene", "n-hexane"),
    ("acetone", "chloroform"),
]

COMPOSITIONS = [0.05, 0.25, 0.5, 0.75, 0.95]
TEMPERATURES = [283.15, 298.15, 348.15]


def profile_arrays(prof_set, n_profiles):
    """(sigma grid, [psigmaA per class]) for one component."""
    if n_profiles == 1:
        p = prof_set.nhb
        return list(p.sigma), [list(p.psigmaA)]
    return list(prof_set.nhb.sigma), [
        list(prof_set.nhb.psigmaA),
        list(prof_set.oh.psigmaA),
        list(prof_set.ot.psigmaA),
    ]


def build_mullins_db(dest):
    """A 1-profile COSMO-SAC 2002 database.

    UD ships Mullins-averaged single profiles in `sigma/`, but those files
    lack the dispersion metadata the Delaware parser requires. Take the
    metadata from the `sigma3/` file, swap in the Mullins averaging fields
    and the 1-profile rows, and write a loadable database.
    """
    os.makedirs(dest, exist_ok=True)
    for name, key in KEYS.items():
        with open(os.path.join(UD, "sigma3", key + ".sigma")) as f:
            meta = json.loads(f.readline().split("# meta:", 1)[1])
        with open(os.path.join(UD, "sigma", key + ".sigma")) as f:
            lines = [ln for ln in f if not ln.startswith("#")]
        mullins = json.loads(open(os.path.join(UD, "sigma", key + ".sigma"))
                             .readline().split("# meta:", 1)[1])
        for field in ("r_av [A]", "f_decay", "averaging"):
            meta[field] = mullins[field]
        with open(os.path.join(dest, key + ".sigma"), "w") as f:
            f.write("# meta: " + json.dumps(meta) + "\n")
            f.write("# Rows are given as: sigma [e/A^2] followed by a space,"
                    " then psigmaA [A^2]\n")
            f.writelines(lines)


def run(model_name, subdir, n_profiles):
    # sigma3 is used for both: the 1-profile directory lacks the dispersion
    # metadata the Delaware parser requires. For COSMO1 that means the input
    # is Hsieh-averaged rather than Mullins-averaged, so those cases are a
    # numerical cross-check of the 2002 kernel, not a physical reference.
    directory = subdir if os.path.isabs(subdir) else os.path.join(UD, subdir)
    db = cCOSMO.DelawareProfileDatabase(
        os.path.join(UD, "complist.txt"), directory + os.sep)

    out = {"model": model_name, "components": {}, "cases": []}

    for name, key in KEYS.items():
        db.add_profile(key)
        prof = db.get_profile(key)
        sigma, psigmaA = profile_arrays(prof.profiles, n_profiles)
        out["components"][name] = {
            "inchikey": key,
            "area": prof.A_COSMO_A2,
            "volume": prof.V_COSMO_A3,
            "sigma": sigma,
            "psigmaA": psigmaA,
        }

    ctor = cCOSMO.COSMO1 if n_profiles == 1 else cCOSMO.COSMO3
    for a, b in SYSTEMS:
        model = ctor([KEYS[a], KEYS[b]], db)
        for T in TEMPERATURES:
            for x in COMPOSITIONS:
                z = [x, 1.0 - x]
                out["cases"].append({
                    "components": [a, b],
                    "T": T,
                    "z": z,
                    "lngamma": list(model.get_lngamma(T, z)),
                    "lngamma_resid": list(model.get_lngamma_resid(T, z)),
                    "lngamma_comb": list(model.get_lngamma_comb(T, z)),
                })
    return out


if __name__ == "__main__":
    dest = sys.argv[1]
    mullins = os.path.join(os.path.dirname(dest), "sigma_mullins")
    build_mullins_db(mullins)
    data = {
        "cosmo_sac_2002": run("COSMO-SAC 2002", mullins, 1),
        "cosmo_sac_2010": run("COSMO-SAC 2010", "sigma3", 3),
    }
    with open(dest, "w") as f:
        json.dump(data, f, indent=1)
    for k, v in data.items():
        print(f"{k}: {len(v['components'])} components, {len(v['cases'])} cases")
