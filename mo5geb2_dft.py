"""
Mo5GeB2 DFT Calculation Pipeline
=================================
Crystal: Cr5B3-type, I4/mcm (No. 140)
Lattice: a = 6.0721 A, c = 11.1173 A
Atoms: 32 per unit cell

Stages:
  1. scf   - Ground-state SCF calculation
  2. dos   - Density of States
  3. bands - Band structure along high-symmetry path

Usage:
    # Full run (all stages):
    mpirun -np 96 --use-hwthread-cpus python mo5geb2_dft.py

    # Skip SCF, reuse saved ground state:
    mpirun -np 96 --use-hwthread-cpus python mo5geb2_dft.py --skip-scf

    # Skip SCF, point to existing gpw in current directory:
    mpirun -np 96 --use-hwthread-cpus python mo5geb2_dft.py --skip-scf --gpw mo5geb2_gs.gpw

    # Only DOS (skip SCF and bands):
    mpirun -np 96 --use-hwthread-cpus python mo5geb2_dft.py --skip-scf --skip-bands

    # Only bands:
    mpirun -np 96 --use-hwthread-cpus python mo5geb2_dft.py --skip-scf --skip-dos

    # Custom parameters:
    mpirun -np 96 --use-hwthread-cpus python mo5geb2_dft.py --cutoff 500 --kpts 6 6 4 --smearing 0.05

Output:
    All files written to {prefix}/ directory (default: mo5geb2/)
"""

import argparse
import os
import sys
import time
import numpy as np
from gpaw.mpi import world


def log(msg: str) -> None:
    """Print only from MPI rank 0."""
    if world.rank == 0:
        print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Mo5GeB2 DFT calculation pipeline (SCF -> DOS -> Bands)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Stage control
    parser.add_argument('--skip-scf', action='store_true',
                        help='Skip SCF, restart from saved .gpw file')
    parser.add_argument('--skip-dos', action='store_true',
                        help='Skip DOS calculation')
    parser.add_argument('--skip-bands', action='store_true',
                        help='Skip band structure calculation')

    # SCF parameters
    parser.add_argument('--cutoff', type=float, default=400.0,
                        help='Plane-wave cutoff in eV (default: 400)')
    parser.add_argument('--kpts', type=int, nargs=3, default=[4, 4, 2],
                        metavar=('KX', 'KY', 'KZ'),
                        help='K-point grid (default: 4 4 2)')
    parser.add_argument('--smearing', type=float, default=0.1,
                        help='Fermi-Dirac smearing width in eV (default: 0.1)')
    parser.add_argument('--xc', type=str, default='PBE',
                        help='Exchange-correlation functional (default: PBE)')

    # File paths
    parser.add_argument('--gpw', type=str, default='mo5geb2_gs.gpw',
                        help='Ground state .gpw file path (default: mo5geb2_gs.gpw)')
    parser.add_argument('--prefix', type=str, default='mo5geb2',
                        help='Output file prefix (default: mo5geb2)')

    # Band structure parameters
    parser.add_argument('--band-path', type=str, default='GXMGZRAZXM',
                        help='High-symmetry k-point path (default: GXMGZRAZXM)')
    parser.add_argument('--band-npoints', type=int, default=200,
                        help='Number of k-points along band path (default: 200)')
    parser.add_argument('--band-convergence', type=str, default='occupied',
                        choices=['occupied', 'all'],
                        help='Band convergence target (default: occupied)')
    parser.add_argument('--extra-bands', type=int, default=20,
                        help='Extra empty bands beyond occupied (default: 20)')
    parser.add_argument('--band-maxiter', type=int, default=333,
                        help='Max iterations for band calculation (default: 333)')

    # DOS parameters
    parser.add_argument('--dos-width', type=float, default=0.1,
                        help='DOS Gaussian broadening width in eV (default: 0.1)')
    parser.add_argument('--dos-npts', type=int, default=3000,
                        help='Number of DOS energy points (default: 3000)')
    parser.add_argument('--dos-emin', type=float, default=-10.0,
                        help='DOS energy range min relative to Ef in eV (default: -10)')
    parser.add_argument('--dos-emax', type=float, default=5.0,
                        help='DOS energy range max relative to Ef in eV (default: 5)')

    args = parser.parse_args()

    # Output directory is the prefix name
    args.outdir = args.prefix
    # Default gpw path goes into outdir
    if args.gpw == 'mo5geb2_gs.gpw':
        args.gpw = os.path.join(args.outdir, f'{args.prefix}_gs.gpw')

    return args


def outpath(args: argparse.Namespace, filename: str) -> str:
    """Build output file path: {outdir}/{filename}"""
    return os.path.join(args.outdir, filename)


def build_crystal():
    """Build Mo5GeB2 crystal structure."""
    from ase.spacegroup import crystal as ase_crystal

    atoms = ase_crystal(
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
    return atoms


def stage_scf(atoms, args: argparse.Namespace) -> 'GPAW':
    """Run ground-state SCF calculation."""
    from gpaw import GPAW, PW, FermiDirac

    log("\n--- Stage 1: Ground-State SCF ---")
    log(f"  Functional:  {args.xc}")
    log(f"  Cutoff:      {args.cutoff} eV")
    log(f"  K-points:    {args.kpts[0]}x{args.kpts[1]}x{args.kpts[2]}")
    log(f"  Smearing:    {args.smearing} eV")
    log(f"  Output:      {outpath(args, f'{args.prefix}_scf.txt')}")

    t0 = time.time()

    calc = GPAW(
        mode=PW(args.cutoff),
        xc=args.xc,
        kpts={'size': tuple(args.kpts), 'gamma': True},
        occupations=FermiDirac(args.smearing),
        convergence={'energy': 1e-6, 'density': 1e-5},
        txt=outpath(args, f'{args.prefix}_scf.txt'),
        parallel={'band': 1},
    )

    atoms.calc = calc
    energy = atoms.get_potential_energy()
    fermi = calc.get_fermi_level()

    elapsed = time.time() - t0
    log(f"\n  Total energy: {energy:.6f} eV ({energy/len(atoms):.6f} eV/atom)")
    log(f"  Fermi level:  {fermi:.4f} eV")
    log(f"  Elapsed:      {elapsed:.1f} sec")

    # Save ground state
    calc.write(args.gpw, mode='all')
    log(f"  Saved to:     {args.gpw}")

    return calc


def load_ground_state(args: argparse.Namespace) -> 'GPAW':
    """Load ground state from saved .gpw file."""
    from gpaw import GPAW

    if not os.path.exists(args.gpw):
        log(f"ERROR: Ground state file '{args.gpw}' not found.")
        log("Run without --skip-scf first, or specify --gpw <path>")
        sys.exit(1)

    log(f"\n--- Loading ground state from {args.gpw} ---")
    t0 = time.time()
    calc = GPAW(args.gpw, txt=None)
    elapsed = time.time() - t0
    fermi = calc.get_fermi_level()
    log(f"  Fermi level:  {fermi:.4f} eV")
    log(f"  Load time:    {elapsed:.1f} sec")

    return calc


def stage_dos(calc: 'GPAW', args: argparse.Namespace) -> float:
    """Calculate density of states. Returns N(Ef)."""
    log("\n--- Stage 2: Density of States ---")
    log(f"  Broadening:  {args.dos_width} eV")
    log(f"  Points:      {args.dos_npts}")
    log(f"  Range:       [{args.dos_emin}, {args.dos_emax}] eV relative to Ef")

    t0 = time.time()
    fermi = calc.get_fermi_level()
    dos_at_fermi = 0.0

    # All ranks must call GPAW getters (they involve MPI internally)
    nbands = calc.get_number_of_bands()
    ibz_kpts = calc.get_ibz_k_points()
    nkpts = len(ibz_kpts)
    weights = calc.get_k_point_weights()

    all_eigenvalues = []
    all_weights = []
    for k in range(nkpts):
        eigenvalues_k = calc.get_eigenvalues(kpt=k, spin=0)
        for n in range(nbands):
            all_eigenvalues.append(eigenvalues_k[n])
            all_weights.append(weights[k])

    all_eigenvalues = np.array(all_eigenvalues)
    all_weights = np.array(all_weights)

    # Only rank 0 does the broadening and saves
    if world.rank == 0:
        log(f"  Eigenvalues:  {len(all_eigenvalues)} total "
            f"({nbands} bands x {nkpts} k-points)")
        log(f"  Eigenvalue range: [{all_eigenvalues.min():.2f}, "
            f"{all_eigenvalues.max():.2f}] eV")

        # Gaussian broadening DOS
        sigma = args.dos_width
        energies_abs = np.linspace(fermi + args.dos_emin,
                                   fermi + args.dos_emax,
                                   args.dos_npts)
        dos = np.zeros_like(energies_abs)

        for eig, w in zip(all_eigenvalues, all_weights):
            dos += w * np.exp(-0.5 * ((energies_abs - eig) / sigma) ** 2) / \
                   (sigma * np.sqrt(2 * np.pi))

        # Factor of 2 for spin degeneracy (non-spin-polarized calculation)
        dos *= 2.0

        energies_rel = energies_abs - fermi

        # N(Ef)
        idx_fermi = np.argmin(np.abs(energies_rel))
        dos_at_fermi = dos[idx_fermi]

        # Save
        outfile = outpath(args, f'{args.prefix}_dos.dat')
        np.savetxt(outfile,
                   np.column_stack([energies_rel, dos]),
                   header='Energy-Ef(eV)  DOS(states/eV)',
                   fmt='%.6f')

        elapsed = time.time() - t0
        log(f"\n  N(Ef):        {dos_at_fermi:.4f} states/eV")
        log(f"  Saved to:     {outfile}")
        log(f"  Elapsed:      {elapsed:.1f} sec")

    return dos_at_fermi


def stage_bands(calc: 'GPAW', atoms, args: argparse.Namespace) -> int:
    """Calculate band structure. Returns number of bands crossing Ef."""
    log("\n--- Stage 3: Band Structure ---")
    log(f"  Path:          {args.band_path}")
    log(f"  K-points:      {args.band_npoints}")
    log(f"  Convergence:   {args.band_convergence}")
    log(f"  Extra bands:   {args.extra_bands}")
    log(f"  Max iter:      {args.band_maxiter}")

    t0 = time.time()
    fermi = calc.get_fermi_level()

    path = atoms.cell.bandpath(
        args.band_path,
        npoints=args.band_npoints,
    )

    log(f"  Actual points: {len(path.kpts)}")

    # fixed_density: reuse SCF density, solve for eigenvalues only
    calc_bands = calc.fixed_density(
        kpts=path,
        symmetry='off',
        nbands=-args.extra_bands,
        convergence={'bands': args.band_convergence},
        maxiter=args.band_maxiter,
        txt=outpath(args, f'{args.prefix}_bands.txt'),
    )

    bs = calc_bands.band_structure()
    bs = bs.subtract_reference()  # shifts energies so Ef = 0

    outfile = outpath(args, f'{args.prefix}_bands.json')
    bs.write(outfile)

    # Count bands crossing Fermi level (already shifted, Ef = 0)
    energies_bands = bs.energies[0]
    n_crossings = 0
    for band_idx in range(energies_bands.shape[1]):
        band = energies_bands[:, band_idx]
        if band.min() < 0 and band.max() > 0:
            n_crossings += 1

    elapsed = time.time() - t0
    log(f"\n  Bands at Ef:   {n_crossings}")
    if n_crossings > 0:
        log("  => METALLIC (required for conventional superconductivity)")
    else:
        log("  => INSULATING or SEMICONDUCTING")
    log(f"  Saved to:      {outfile}")
    log(f"  Elapsed:       {elapsed:.1f} sec")

    return n_crossings


def main() -> None:
    args = parse_args()

    log("=" * 60)
    log("Mo5GeB2 - GPAW DFT Calculation Pipeline")
    log("=" * 60)
    log(f"  MPI ranks:     {world.size}")
    log(f"  Skip SCF:      {args.skip_scf}")
    log(f"  Skip DOS:      {args.skip_dos}")
    log(f"  Skip bands:    {args.skip_bands}")
    log(f"  Output dir:    {args.outdir}/")
    log(f"  GPW file:      {args.gpw}")
    log(f"  Prefix:        {args.prefix}")

    # Create output directory (rank 0 only, then barrier)
    if world.rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
    world.barrier()

    t_total = time.time()

    # ----- SCF or load -----
    if args.skip_scf:
        calc = load_ground_state(args)
        atoms = calc.atoms
    else:
        atoms = build_crystal()
        log(f"\nCrystal: {atoms.get_chemical_formula()}, "
            f"{len(atoms)} atoms, "
            f"a={atoms.cell.cellpar()[0]:.4f} c={atoms.cell.cellpar()[2]:.4f} A")
        calc = stage_scf(atoms, args)

    fermi = calc.get_fermi_level()

    # ----- DOS -----
    dos_at_fermi = 0.0
    if not args.skip_dos:
        dos_at_fermi = stage_dos(calc, args)

    # ----- Bands -----
    n_crossings = 0
    if not args.skip_bands:
        n_crossings = stage_bands(calc, atoms, args)

    # ----- Summary -----
    elapsed_total = time.time() - t_total
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"  System:        Mo5GeB2 (Cr5B3-type, I4/mcm)")
    log(f"  Method:        DFT-{args.xc}, PW({args.cutoff}eV), "
        f"kpts={args.kpts[0]}x{args.kpts[1]}x{args.kpts[2]}")
    log(f"  Fermi level:   {fermi:.4f} eV")
    if not args.skip_dos:
        log(f"  N(Ef):         {dos_at_fermi:.4f} states/eV")
    if not args.skip_bands:
        log(f"  Metallic:      {'YES' if n_crossings > 0 else 'NO'}")
        log(f"  Bands at Ef:   {n_crossings}")
    log(f"  Total time:    {elapsed_total:.1f} sec ({elapsed_total/60:.1f} min)")
    log(f"")
    log(f"  Experimental:  Tc = 5.8 K (type-II superconductor)")
    log(f"")
    log(f"  Output files ({args.outdir}/):")
    if not args.skip_scf:
        log(f"    {args.gpw}")
        log(f"    {outpath(args, f'{args.prefix}_scf.txt')}")
    if not args.skip_dos:
        log(f"    {outpath(args, f'{args.prefix}_dos.dat')}")
    if not args.skip_bands:
        log(f"    {outpath(args, f'{args.prefix}_bands.json')}")
        log(f"    {outpath(args, f'{args.prefix}_bands.txt')}")
    log(f"")
    log(f"  Plot: python mo5geb2_plot.py --prefix {args.prefix}")


if __name__ == '__main__':
    main()
