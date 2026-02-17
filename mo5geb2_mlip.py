"""
Mo5GeB2 Superconductor Simulation via MACE-MP-0

Crystal data from: doi.org/10.1016/j.jallcom.2021.160675
  Space group: I4/mcm (No. 140)
  Lattice: a = 6.0721 A, c = 11.1173 A (tetragonal)
  Structure type: Cr5B3-type

This is a Cr5B3-type structure (tetragonal, body-centered).
Mo5GeB2 has Ge substituting for one of the B/Si sites in the
parent Cr5B3 structure.

Wyckoff positions for I4/mcm Cr5B3-type with Mo5GeB2 composition:
  Mo1: 16l  (x, x+1/2, z) with x~0.166, z~0.139
  Mo2: 4c   (0, 0, 0)
  Ge:  4a   (0, 0, 1/4)
  B:   8h   (x, x+1/2, 0) with x~0.617

Links

DeepMind GNoME blog post: https://deepmind.google/discover/blog/millions-of-new-materials-discovered-with-deep-learning/
Nature paper: https://www.nature.com/articles/s41586-023-06735-9
Mo5GeB2 original paper: https://www.sciencedirect.com/science/article/abs/pii/S0925838821006381

"""

from ase import Atoms
from ase.spacegroup import crystal
from ase.optimize import BFGS
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from mace.calculators import mace_mp
import numpy as np

print("=" * 60)
print("Mo5GeB2 Superconductor - MACE-MP-0 Simulation")
print("=" * 60)

# Build crystal from space group and Wyckoff positions
# Space group I4/mcm (#140), tetragonal
# Lattice: a = b = 6.0721, c = 11.1173 A
#
# Approximate Wyckoff coordinates for Cr5B3-type:
#   Mo1 at 16l: (x, x+1/2, z), x~0.166, z~0.139
#   Mo2 at 4c:  (0, 0, 0)
#   Ge  at 4a:  (0, 0, 1/4)
#   B   at 8h:  (x, x+1/2, 0), x~0.617

atoms = crystal(
    symbols=['Mo', 'Mo', 'Ge', 'B'],
    basis=[
        (0.166, 0.666, 0.139),   # Mo1 - 16l
        (0.0, 0.0, 0.0),         # Mo2 - 4c
        (0.0, 0.0, 0.25),        # Ge  - 4a
        (0.617, 0.117, 0.0),     # B   - 8h
    ],
    spacegroup=140,
    cellpar=[6.0721, 6.0721, 11.1173, 90, 90, 90],
)

n_atoms = len(atoms)
formula = atoms.get_chemical_formula()
print(f"\nStructure built: {formula}")
print(f"Number of atoms: {n_atoms}")
print(f"Cell: a={atoms.cell.cellpar()[0]:.4f}, c={atoms.cell.cellpar()[2]:.4f} A")
print(f"Volume: {atoms.get_volume():.2f} A^3")

# Element counts
symbols = atoms.get_chemical_symbols()
from collections import Counter
counts = Counter(symbols)
print(f"Composition: {dict(counts)}")

# Attach MACE calculator
print("\nLoading MACE-MP-0 model...")
calc = mace_mp(model="medium", default_dtype="float64")
atoms.calc = calc

# Initial energy
e0 = atoms.get_potential_energy()
f0 = atoms.get_forces()
fmax0 = np.max(np.abs(f0))
print(f"\n--- Initial State ---")
print(f"Energy: {e0:.4f} eV ({e0/n_atoms:.4f} eV/atom)")
print(f"Max force: {fmax0:.4f} eV/A")

# Geometry optimization (0K)
print(f"\n--- Geometry Optimization (0K) ---")
opt = BFGS(atoms, logfile='-')
opt.run(fmax=0.05, steps=100)

e_opt = atoms.get_potential_energy()
print(f"\nOptimized energy: {e_opt:.4f} eV ({e_opt/n_atoms:.4f} eV/atom)")
print(f"Optimized cell: a={atoms.cell.cellpar()[0]:.4f}, c={atoms.cell.cellpar()[2]:.4f} A")
print(f"Volume: {atoms.get_volume():.2f} A^3")

# Stress tensor (related to pressure)
try:
    stress = atoms.get_stress(voigt=True)  # in eV/A^3
    # Convert to GPa
    stress_gpa = stress * 160.21766208  # eV/A^3 to GPa
    print(f"Stress (GPa): xx={stress_gpa[0]:.2f}, yy={stress_gpa[1]:.2f}, zz={stress_gpa[2]:.2f}")
    print(f"Pressure: {np.mean(stress_gpa[:3]):.2f} GPa")
except Exception as e:
    print(f"Stress calculation: {e}")

# Short MD at 300K
print(f"\n--- Molecular Dynamics (300K, NVT) ---")
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

dyn = Langevin(
    atoms,
    timestep=1.0 * units.fs,
    temperature_K=300,
    friction=0.01 / units.fs,
)

print(f"{'Step':>6} {'Time(fs)':>10} {'E_tot(eV)':>12} {'E/atom':>10} {'T(K)':>8}")
print("-" * 50)

for i in range(50):
    dyn.run(1)
    if i % 5 == 0:
        e = atoms.get_potential_energy()
        ek = atoms.get_kinetic_energy()
        temp = 2 * ek / (3 * n_atoms * units.kB)
        print(f"{i:6d} {i*1.0:10.1f} {e:12.4f} {e/n_atoms:10.4f} {temp:8.1f}")

# Final distances analysis
print(f"\n--- Bond Distance Analysis ---")
from ase.geometry.analysis import Analysis
ana = Analysis(atoms)

# Find Mo-Mo, Mo-B, Mo-Ge nearest neighbor distances
positions = atoms.get_positions()
cell = atoms.get_cell()

# Simple nearest-neighbor check
for pair in [('Mo', 'Mo'), ('Mo', 'B'), ('Mo', 'Ge'), ('B', 'B')]:
    dists = []
    idx_a = [i for i, s in enumerate(symbols) if s == pair[0]]
    idx_b = [i for i, s in enumerate(symbols) if s == pair[1]]
    for ia in idx_a[:4]:  # sample a few
        for ib in idx_b[:4]:
            if ia != ib:
                d = atoms.get_distance(ia, ib, mic=True)
                if d < 4.0:  # within reasonable bonding distance
                    dists.append(d)
    if dists:
        print(f"  {pair[0]}-{pair[1]}: min={min(dists):.3f} A, "
              f"avg={np.mean(dists):.3f} A (n={len(dists)} pairs < 4A)")

print(f"\n--- Summary ---")
print(f"System: Mo5GeB2 (Cr5B3-type, I4/mcm)")
print(f"Atoms: {n_atoms}")
print(f"MLIP: MACE-MP-0 (medium)")
print(f"Optimized E: {e_opt:.4f} eV ({e_opt/n_atoms:.4f} eV/atom)")
print(f"Experimental Tc: 5.8 K (type-II superconductor)")
print(f"Note: MACE cannot predict Tc directly - that requires")
print(f"      electron-phonon coupling calculations (DFT+DFPT).")
print(f"      But we CAN check structural stability & phonons.")
