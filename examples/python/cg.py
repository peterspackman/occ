from occpy import Crystal, CrystalGrowthConfig
import occpy
import numpy as np

np.set_printoptions(precision=2)

UREA_CIF = """
data_urea
_symmetry_space_group_name_H-M   'P -4 21 m'
_symmetry_Int_Tables_number      113
loop_
_symmetry_equiv_pos_site_id
_symmetry_equiv_pos_as_xyz
1 x,y,z
2 -y,x,-z
3 -x,-y,z
4 y,-x,-z
5 1/2-x,1/2+y,-z
6 1/2+y,1/2+x,z
7 1/2+x,1/2-y,-z
8 1/2-y,1/2-x,z
_cell_length_a                   5.582
_cell_length_b                   5.582
_cell_length_c                   4.686
_cell_angle_alpha                90
_cell_angle_beta                 90
_cell_angle_gamma                90
_cell_volume                     146.01
_cell_formula_units_Z            2
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.00000 0.50000 0.32720
H1 H 0.26900 0.76900 0.27900
H2 H 0.14200 0.64200 -0.02800
N1 N 0.14550 0.64550 0.18000
O1 O 0.00000 0.50000 0.59660
N1B N -0.14550 0.35450 0.18000
H1B H -0.26900 0.23100 0.27900
H2B H -0.14200 0.35800 -0.02800
"""

occpy.set_num_threads(6)

with open("urea_tmp.cif", "w") as f:
    f.write(UREA_CIF)

config = CrystalGrowthConfig()
config.lattice_settings.max_radius = 3.8
config.lattice_settings.crystal_filename = "urea_tmp.cif"
config.solvent = "toluene"
config.cg_radius = 3.8

# this is basically identical to running occ cg urea_tmp.cif
# it will write all the temporary files too
result = occpy.calculate_crystal_growth_energies(config)

for i, mol_res in enumerate(result.molecule_results):
    print(f"molecule {i}")
    print("Total energy = ", mol_res.total_energy())
    for d in mol_res.dimer_results:
        print(f"Dimer {d.dimer.name} energies (u={d.unique_idx})")
        for k, v in d.energy_components().items():
            print(f"{k:<24s} {v:12.5f}")
