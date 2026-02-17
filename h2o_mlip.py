from ase.build import molecule
from mace.calculators import mace_mp

# Build water molecule
atoms = molecule('H2O')

# Load universal MLIP (downloads model automatically)
calc = mace_mp(model="medium", default_dtype="float64")
atoms.calc = calc

# Get energy and forces
print(f"Energy: {atoms.get_potential_energy():.4f} eV")
print(f"Forces:\n{atoms.get_forces()}")

# Geometry optimization (find minimum energy structure)
from ase.optimize import BFGS
opt = BFGS(atoms)
opt.run(fmax=0.01)
print(f"\nOptimized energy: {atoms.get_potential_energy():.4f} eV")
print(f"O-H distance: {atoms.get_distance(0,1):.3f} A")
print(f"H-O-H angle: {atoms.get_angle(1,0,2):.1f} deg")

# Quick MD at 300K
from ase.md.langevin import Langevin
from ase import units
dyn = Langevin(atoms, timestep=0.5*units.fs, temperature_K=300, friction=0.01)
for i in range(100):
    dyn.run(1)
    if i % 10 == 0:
        print(f"Step {i}: E={atoms.get_potential_energy():.4f} eV, T={atoms.get_kinetic_energy()/(1.5*units.kB):.0f} K")

# Compare with known DFT values
print("\n=== Validation ===")
print(f"O-H bond:  {atoms.get_distance(0,1):.3f} A  (exp: 0.957 A)")
print(f"H-O-H angle: {atoms.get_angle(1,0,2):.1f} deg (exp: 104.5 deg)")

# Vibrational frequencies would be the gold standard test
# but that's a bigger exercise
