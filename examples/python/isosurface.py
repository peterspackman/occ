from occpy import IsosurfaceCalculator as Calc, IsosurfaceGenerationParameters as Params, SurfaceKind, PropertyKind
from occpy import Molecule
import time

WATER_XYZ = """3

O   -0.7021961  -0.0560603   0.0099423
H   -1.0221932   0.8467758  -0.0114887
H    0.2575211   0.0421215   0.0052190"""

mol = Molecule.from_xyz_string(WATER_XYZ)

for sep in (1.0, 0.5, 0.25, 0.1, 0.05):
    t1 = time.time()
    params = Params()
    params.isovalue = 0.02
    params.surface_kind = SurfaceKind.PromoleculeDensity
    params.separation = sep

    calc = Calc()
    calc.set_parameters(params)
    calc.set_molecule(mol)
    calc.compute()
    surf = calc.isosurface()
    surf.save(f"example.{sep}.ply")
    t2 = time.time()
    print(f"Surface with separation = {sep:.3f} took {t2 - t1:.3f}s")
