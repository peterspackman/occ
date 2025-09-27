"""
Example showing acoustic velocities in silicon from elastic tensor
"""

import numpy as np
from occpy import ElasticTensor, Crystal, AveragingScheme

SILICON_CIF = """
data_silicon
_symmetry_space_group_name_H-M   'P 1'
_symmetry_Int_Tables_number      1
_cell_length_a                   5.4310
_cell_length_b                   5.4310
_cell_length_c                   5.4310
_cell_angle_alpha                90
_cell_angle_beta                 90
_cell_angle_gamma                90
_cell_volume                     160.15
_cell_formula_units_Z            8
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_element_symbol
Si1 0.125 0.125 0.125 Si
Si2 0.875 0.875 0.875 Si
Si3 0.625 0.375 0.875 Si
Si4 0.375 0.625 0.125 Si
Si5 0.875 0.625 0.375 Si
Si6 0.125 0.375 0.625 Si
Si7 0.625 0.875 0.375 Si
Si8 0.375 0.125 0.625 Si
"""

crystal = Crystal.from_cif_string(SILICON_CIF)
print(f"Crystal: {crystal}")
print(f"Volume: {crystal.volume():.2f} Å³")
print(f"Density: {crystal.density():.3f} g/cm³")
print()

# Silicon elastic constants from McSkimin & Andreatch (1972)
C11, C12, C44 = 165.8, 63.9, 79.6  # GPa

# Cubic elastic tensor
c = np.array(
    [
        [C11, C12, C12, 0, 0, 0],
        [C12, C11, C12, 0, 0, 0],
        [C12, C12, C11, 0, 0, 0],
        [0, 0, 0, C44, 0, 0],
        [0, 0, 0, 0, C44, 0],
        [0, 0, 0, 0, 0, C44],
    ]
)

print(f"Elastic constants: C11={C11}, C12={C12}, C44={C44} GPa")
print(f"\nVoigt tensor (GPa):")
print(c)

tensor = ElasticTensor(c)
eigenvals = tensor.eigenvalues()
print(f"\nEigenvalues: {eigenvals}")

# Averaged properties
schemes = [
    ("Voigt", AveragingScheme.VOIGT),
    ("Reuss", AveragingScheme.REUSS),
    ("Hill", AveragingScheme.HILL),
]

print(
    f"\n{'Scheme':<8} {'K (GPa)':<8} {'G (GPa)':<8} {'E (GPa)':<8} {'ν':<6} {'V_s (m/s)':<10} {'V_p (m/s)':<10}"
)
print("-" * 70)
for name, scheme in schemes:
    K = tensor.average_bulk_modulus(scheme)
    G = tensor.average_shear_modulus(scheme)
    E = tensor.average_youngs_modulus(scheme)
    nu = tensor.average_poisson_ratio(scheme)
    vs, vp, _ = tensor.acoustic_velocities_with_crystal(crystal, scheme)
    print(f"{name:<8} {K:<8.1f} {G:<8.1f} {E:<8.1f} {nu:<6.3f} {vs:<10.0f} {vp:<10.0f}")

# Directional properties
directions = {"[100]": [1, 0, 0], "[110]": [1, 1, 0], "[111]": [1, 1, 1]}
print(f"\n{'Dir':<6} {'E (GPa)':<8}")
print("-" * 16)
for name, d in directions.items():
    direction = np.array(d, dtype=float)
    direction /= np.linalg.norm(direction)
    E = tensor.youngs_modulus(direction)
    print(f"{name:<6} {E:<8.1f}")
