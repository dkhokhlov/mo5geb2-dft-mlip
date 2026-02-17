"""
Mo5GeB2 Phonon Calculation + Tc Estimate
==========================================
Uses MACE-MP-0 for fast phonon calculation via finite differences.
Then estimates Tc via McMillan formula.

This is a quick surrogate — proper electron-phonon coupling requires
DFPT (density functional perturbation theory), which is much more
expensive. MACE gives us phonon frequencies to plug into a rough estimate.

Usage:
    python mo5geb2_phonons.py
    python mo5geb2_phonons.py --supercell 2 2 1
    python mo5geb2_phonons.py --n-ef 21.6  # override N(Ef) from DFT
    python mo5geb2_phonons.py --ascii-only
"""

import argparse
import os
import time
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Mo5GeB2 phonon calculation + McMillan Tc estimate')

    parser.add_argument('--supercell', type=int, nargs=3, default=[1, 1, 1],
                        metavar=('NX', 'NY', 'NZ'),
                        help='Supercell size for finite differences (default: 1 1 1)')
    parser.add_argument('--displacement', type=float, default=0.01,
                        help='Finite difference displacement in Angstrom (default: 0.01)')
    parser.add_argument('--n-ef', type=float, default=None,
                        help='N(Ef) in states/eV from DFT (auto-read from dos.dat if not given)')
    parser.add_argument('--mu-star', type=float, default=0.13,
                        help='Coulomb pseudopotential mu* (default: 0.13, typical range 0.10-0.15)')
    parser.add_argument('--prefix', type=str, default='mo5geb2',
                        help='File prefix (default: mo5geb2)')
    parser.add_argument('--model', type=str, default='medium',
                        choices=['small', 'medium', 'large'],
                        help='MACE-MP-0 model size (default: medium)')
    parser.add_argument('--ascii-only', action='store_true',
                        help='Only print ASCII output, no PNG')
    parser.add_argument('--no-ascii', action='store_true',
                        help='Only generate PNG, no ASCII')
    parser.add_argument('--q-points', type=int, nargs=3, default=[4, 4, 2],
                        metavar=('QX', 'QY', 'QZ'),
                        help='Q-point mesh for phonon DOS (default: 4 4 2)')

    return parser.parse_args()


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


def outpath(prefix: str, filename: str) -> str:
    """Build output file path: {prefix}/{filename}"""
    return os.path.join(prefix, filename)


def read_n_ef(prefix: str) -> float:
    """Read N(Ef) from DOS data file."""
    # Check prefix/ directory first, then current directory
    dos_file = outpath(prefix, f'{prefix}_dos.dat')
    if not os.path.exists(dos_file):
        dos_file = f'{prefix}_dos.dat'
    if not os.path.exists(dos_file):
        return None

    data = np.loadtxt(dos_file)
    energy = data[:, 0]   # E - Ef
    dos = data[:, 1]      # states/eV

    idx_ef = np.argmin(np.abs(energy))
    return dos[idx_ef]


def compute_phonons(atoms, args: argparse.Namespace):
    """Compute phonon spectrum using MACE + finite differences."""
    from mace.calculators import mace_mp
    from ase.phonons import Phonons

    print("\n--- Phonon Calculation ---")
    print(f"  Calculator:    MACE-MP-0 ({args.model})")
    print(f"  Supercell:     {args.supercell[0]}x{args.supercell[1]}x{args.supercell[2]}")
    print(f"  Displacement:  {args.displacement} A")

    t0 = time.time()

    # Attach MACE calculator
    calc = mace_mp(model=args.model, default_dtype='float64')
    atoms.calc = calc

    # Geometry optimization first (MACE equilibrium, not DFT)
    from ase.optimize import BFGS
    print("  Optimizing geometry with MACE...")
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.01)
    print(f"  Optimization: {opt.nsteps} steps")

    # Phonon calculation via finite differences
    supercell = tuple(args.supercell)
    ph = Phonons(atoms, calc, supercell=supercell, delta=args.displacement,
                 name=outpath(args.prefix, f'{args.prefix}_phonon'))

    print(f"  Computing force constants...")
    ph.run()
    ph.read(acoustic=True)  # apply acoustic sum rule

    elapsed = time.time() - t0
    print(f"  Force constants computed in {elapsed:.1f} sec")

    return ph, atoms


def analyze_phonons(ph, atoms, args: argparse.Namespace):
    """Extract phonon dispersion, DOS, and key frequencies."""
    print("\n--- Phonon Analysis ---")

    # Phonon band structure along same path as electronic bands
    path = atoms.cell.bandpath(
        'GXMGZRAZXM',
        npoints=200,
    )

    # Get phonon band structure
    bs = ph.get_band_structure(path)

    # Frequencies in meV (ASE returns eV, convert)
    freqs = bs.energies[0] * 1000.0  # eV -> meV
    nkpts, nbands = freqs.shape

    print(f"  Phonon bands: {nbands}")
    print(f"  K-points:     {nkpts}")

    # Check for imaginary modes (negative frequencies = structural instability)
    min_freq = freqs.min()
    has_imaginary = min_freq < -1.0  # threshold: -1 meV (small negatives are numerical noise)

    if has_imaginary:
        n_imaginary = np.sum(freqs < -1.0)
        print(f"  WARNING: {n_imaginary} imaginary modes detected (min: {min_freq:.2f} meV)")
        print(f"  This may indicate structural instability at the MACE level")
    else:
        print(f"  No imaginary modes (structure is dynamically stable)")

    print(f"  Frequency range: [{min_freq:.2f}, {freqs.max():.2f}] meV")

    # Phonon DOS from band structure frequencies (more reliable than ph.get_dos API)
    q_mesh = tuple(args.q_points)
    print(f"  Computing phonon DOS from band frequencies...")

    # Collect all positive frequencies from the dispersion
    all_freqs_flat = freqs.flatten()
    all_freqs_positive = all_freqs_flat[all_freqs_flat > 0.1]  # skip near-zero acoustic

    dos_energies_meV = np.linspace(0, freqs.max() * 1.1, 500)
    phonon_dos = np.zeros_like(dos_energies_meV)
    sigma = 1.0  # meV broadening

    for f_val in all_freqs_positive:
        phonon_dos += np.exp(-0.5 * ((dos_energies_meV - f_val) / sigma) ** 2) / \
                      (sigma * np.sqrt(2 * np.pi))
    # Normalize
    if phonon_dos.max() > 0:
        phonon_dos /= phonon_dos.max()

    # Key phonon averages for McMillan formula
    # omega_log = exp(<ln(omega)>) - logarithmic average frequency
    # omega_2 = sqrt(<omega^2>) - RMS frequency
    # Only use positive frequencies
    all_freqs_meV = freqs[freqs > 1.0]  # skip acoustic near-zero

    if len(all_freqs_meV) > 0:
        omega_log_meV = np.exp(np.mean(np.log(all_freqs_meV)))
        omega_2_meV = np.sqrt(np.mean(all_freqs_meV ** 2))
        omega_max_meV = np.max(all_freqs_meV)

        # Convert to Kelvin: 1 meV = 11.6045 K
        meV_to_K = 11.6045
        omega_log_K = omega_log_meV * meV_to_K
        omega_2_K = omega_2_meV * meV_to_K

        print(f"\n  Phonon averages:")
        print(f"    omega_log:   {omega_log_meV:.2f} meV ({omega_log_K:.1f} K)")
        print(f"    omega_rms:   {omega_2_meV:.2f} meV ({omega_2_K:.1f} K)")
        print(f"    omega_max:   {omega_max_meV:.2f} meV")
    else:
        omega_log_meV = 0
        omega_log_K = 0
        omega_2_meV = 0
        omega_2_K = 0
        omega_max_meV = 0
        print("  WARNING: No positive phonon frequencies found")

    # Save phonon band structure data
    xcoords, label_xcoords, labels = path.get_linear_kpoint_axis()
    np.savetxt(outpath(args.prefix, f'{args.prefix}_phonon_bands.dat'),
               np.column_stack([xcoords, freqs]),
               header=f'k-distance(1/A)  freq_1..freq_{nbands}(meV)  labels:{",".join(labels)}',
               fmt='%.6f')
    print(f"  Saved {outpath(args.prefix, f'{args.prefix}_phonon_bands.dat')}")

    # Save label info separately for plotting
    np.savetxt(outpath(args.prefix, f'{args.prefix}_phonon_labels.dat'),
               np.column_stack([label_xcoords, np.zeros(len(labels))]),
               header='k-distance  0  labels: ' + ','.join(labels),
               fmt='%.6f')

    # Save phonon DOS
    np.savetxt(outpath(args.prefix, f'{args.prefix}_phonon_dos.dat'),
               np.column_stack([dos_energies_meV, phonon_dos]),
               header='Energy(meV)  PhononDOS',
               fmt='%.6f')
    print(f"  Saved {outpath(args.prefix, f'{args.prefix}_phonon_dos.dat')}")

    return {
        'freqs': freqs,
        'xcoords': xcoords,
        'label_xcoords': label_xcoords,
        'labels': labels,
        'omega_log_meV': omega_log_meV,
        'omega_log_K': omega_log_K,
        'omega_2_meV': omega_2_meV,
        'omega_max_meV': omega_max_meV,
        'has_imaginary': has_imaginary,
        'min_freq': min_freq,
        'phonon_dos_energies': dos_energies_meV,
        'phonon_dos': phonon_dos,
    }


def mcmillan_tc(omega_log_K: float, lam: float, mu_star: float) -> float:
    """McMillan formula for superconducting Tc.

    Tc = (omega_log / 1.2) * exp[-1.04(1+lambda) / (lambda - mu*(1+0.62*lambda))]

    Args:
        omega_log_K: logarithmic average phonon frequency in Kelvin
        lam: electron-phonon coupling constant lambda
        mu_star: Coulomb pseudopotential (typically 0.10-0.15)

    Returns:
        Tc in Kelvin
    """
    if lam <= mu_star * (1 + 0.62 * lam):
        return 0.0  # no superconductivity

    numerator = -1.04 * (1 + lam)
    denominator = lam - mu_star * (1 + 0.62 * lam)

    tc = (omega_log_K / 1.2) * np.exp(numerator / denominator)
    return tc


def estimate_tc(phonon_results: dict, n_ef: float, mu_star: float) -> dict:
    """Estimate Tc via McMillan formula with rough lambda estimate.

    This is a very rough estimate because we don't have the actual
    electron-phonon matrix elements. We use the simplified Hopfield
    expression:

        lambda = N(Ef) * <I^2> / (M * <omega^2>)

    where <I^2> is the average electron-phonon matrix element squared
    and M is the average atomic mass. Since we don't know <I^2>,
    we scan a range of lambda values and also try to back-calculate
    what lambda would give the experimental Tc.
    """
    print("\n--- McMillan Tc Estimate ---")
    print(f"  N(Ef):     {n_ef:.2f} states/eV")
    print(f"  omega_log: {phonon_results['omega_log_K']:.1f} K")
    print(f"  mu*:       {mu_star}")

    omega_log_K = phonon_results['omega_log_K']

    if omega_log_K <= 0:
        print("  ERROR: omega_log <= 0, cannot estimate Tc")
        return {}

    # Scan lambda values
    print(f"\n  Lambda scan:")
    print(f"  {'lambda':>8s}  {'Tc (K)':>8s}")
    print(f"  {'-'*8}  {'-'*8}")

    lambdas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    tc_values = []
    for lam in lambdas:
        tc = mcmillan_tc(omega_log_K, lam, mu_star)
        tc_values.append(tc)
        marker = " <-- exp Tc=5.8K" if abs(tc - 5.8) < 1.0 else ""
        print(f"  {lam:8.2f}  {tc:8.2f}{marker}")

    # Back-calculate lambda from experimental Tc = 5.8 K
    tc_exp = 5.8
    from scipy.optimize import brentq

    try:
        def tc_diff(lam):
            return mcmillan_tc(omega_log_K, lam, mu_star) - tc_exp

        # lambda must be > mu_star to get Tc > 0
        lam_exp = brentq(tc_diff, mu_star + 0.01, 5.0)
        tc_check = mcmillan_tc(omega_log_K, lam_exp, mu_star)
        print(f"\n  Back-calculated lambda for Tc=5.8K: {lam_exp:.3f}")
        print(f"  Verification: Tc({lam_exp:.3f}) = {tc_check:.2f} K")
    except (ValueError, RuntimeError) as e:
        lam_exp = None
        print(f"\n  Could not back-calculate lambda: {e}")

    # Physical reasonableness check
    print(f"\n  Physical context:")
    print(f"    Weak coupling:   lambda < 0.5  => Tc typically < 2K")
    print(f"    Moderate:        0.5 < lambda < 1.0 => Tc ~ 2-15K")
    print(f"    Strong:          lambda > 1.0  => Tc > 15K")
    if lam_exp:
        if lam_exp < 0.5:
            regime = "weak coupling"
        elif lam_exp < 1.0:
            regime = "moderate coupling"
        else:
            regime = "strong coupling"
        print(f"    Mo5GeB2 (Tc=5.8K): lambda~{lam_exp:.2f} => {regime}")

    return {
        'lambdas': lambdas,
        'tc_values': tc_values,
        'lam_exp': lam_exp,
        'omega_log_K': omega_log_K,
        'mu_star': mu_star,
    }


def ascii_phonon_bands(phonon_results: dict, width: int = 70, height: int = 25) -> str:
    """ASCII phonon dispersion plot."""
    freqs = phonon_results['freqs']
    xcoords = phonon_results['xcoords']
    label_xcoords = phonon_results['label_xcoords']
    labels = phonon_results['labels']

    nkpts, nbands = freqs.shape
    f_min = min(0, freqs.min())  # include 0 always
    f_max = freqs.max() * 1.05

    lines = []
    lines.append(f"  Phonon Dispersion - Mo5GeB2 (MACE-MP-0)")
    if phonon_results['has_imaginary']:
        lines.append(f"  WARNING: imaginary modes present (min: {phonon_results['min_freq']:.1f} meV)")
    else:
        lines.append(f"  Dynamically stable (no imaginary modes)")
    lines.append(f"  omega_log = {phonon_results['omega_log_meV']:.1f} meV")
    lines.append("")

    x_min, x_max = xcoords[0], xcoords[-1]

    # Build grid
    grid = [[' ' for _ in range(width)] for _ in range(height + 1)]

    for b in range(nbands):
        band = freqs[:, b]
        for col in range(width):
            k_idx = int(col * (nkpts - 1) / (width - 1))
            val = band[k_idx]
            if f_min <= val <= f_max:
                row = int((1.0 - (val - f_min) / (f_max - f_min)) * height)
                if 0 <= row <= height:
                    if grid[row][col] == ' ':
                        grid[row][col] = '.'

    # Zero line (if imaginary modes exist)
    if f_min < 0:
        zero_row = int((1.0 - (0 - f_min) / (f_max - f_min)) * height)
        if 0 <= zero_row <= height:
            for col in range(width):
                if grid[zero_row][col] == ' ':
                    grid[zero_row][col] = '-'

    # High-symmetry vertical lines
    for lx in label_xcoords:
        col = int((lx - x_min) / (x_max - x_min) * (width - 1))
        if 0 <= col < width:
            for row in range(height + 1):
                if grid[row][col] in (' ', '-'):
                    grid[row][col] = ':' if grid[row][col] == ' ' else '|'

    # Render
    for row in range(height + 1):
        if row == 0:
            label = f"{f_max:6.0f} "
        elif row == height:
            label = f"{f_min:6.0f} "
        elif row == height // 2:
            val = f_max - (f_max - f_min) * row / height
            label = f"{val:6.0f} "
        else:
            label = "       "
        lines.append(f"{label}{''.join(grid[row])}")

    lines.append("       " + "-" * width)

    if labels:
        tick_line = list(" " * (width + 7))
        for lx, lbl in zip(label_xcoords, labels):
            col = int((lx - x_min) / (x_max - x_min) * (width - 1))
            pos = col + 7
            if pos < len(tick_line):
                for c_idx, ch in enumerate(lbl):
                    if pos + c_idx < len(tick_line):
                        tick_line[pos + c_idx] = ch
        lines.append("".join(tick_line))

    lines.append(f"{'k-point path':^{width + 7}}")
    lines.append(f"{'Frequency (meV)':^{width + 7}}")

    return "\n".join(lines)


def plot_phonon_png(phonon_results: dict, prefix: str) -> None:
    """Plot phonon dispersion to PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    freqs = phonon_results['freqs']
    xcoords = phonon_results['xcoords']
    label_xcoords = phonon_results['label_xcoords']
    labels = phonon_results['labels']

    nkpts, nbands = freqs.shape

    fig, ax = plt.subplots(figsize=(8, 6))

    for b in range(nbands):
        color = '#dc2626' if freqs[:, b].min() < -1.0 else '#2563eb'
        ax.plot(xcoords, freqs[:, b], color=color, linewidth=0.8, alpha=0.7)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)

    tick_labels = [r'$\Gamma$' if l == 'G' else l for l in labels]
    for lx in label_xcoords:
        ax.axvline(lx, color='black', linewidth=0.5, alpha=0.5)
    ax.set_xticks(label_xcoords)
    ax.set_xticklabels(tick_labels)

    ax.set_xlim(xcoords[0], xcoords[-1])
    ax.set_ylabel('Frequency (meV)')
    ax.set_title('Mo$_5$GeB$_2$ - Phonon Dispersion (MACE-MP-0)')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath(prefix, f'{prefix}_phonon_bands.png'), dpi=150)
    plt.close()
    print(f"Saved {outpath(prefix, f'{prefix}_phonon_bands.png')}")


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Mo5GeB2 - Phonon Calculation + Tc Estimate")
    print("=" * 60)

    t_total = time.time()

    # Create output directory
    os.makedirs(args.prefix, exist_ok=True)

    # Build crystal
    atoms = build_crystal()
    print(f"Crystal: {atoms.get_chemical_formula()}, {len(atoms)} atoms")

    # Get N(Ef)
    if args.n_ef is not None:
        n_ef = args.n_ef
        print(f"N(Ef) from command line: {n_ef:.2f} states/eV")
    else:
        n_ef = read_n_ef(args.prefix)
        if n_ef is not None:
            print(f"N(Ef) from {args.prefix}_dos.dat: {n_ef:.2f} states/eV")
        else:
            print("WARNING: Could not read N(Ef), using default 21.6")
            n_ef = 21.6

    # Compute phonons
    ph, atoms = compute_phonons(atoms, args)

    # Analyze
    phonon_results = analyze_phonons(ph, atoms, args)

    # Tc estimate
    tc_results = estimate_tc(phonon_results, n_ef, args.mu_star)

    # Output
    if not args.no_ascii:
        print()
        print(ascii_phonon_bands(phonon_results))

    if not args.ascii_only:
        try:
            plot_phonon_png(phonon_results, args.prefix)
        except ImportError:
            print("matplotlib not installed, skipping PNG")

    # Summary
    elapsed = time.time() - t_total
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  omega_log:     {phonon_results['omega_log_meV']:.2f} meV "
          f"({phonon_results['omega_log_K']:.1f} K)")
    print(f"  omega_max:     {phonon_results['omega_max_meV']:.2f} meV")
    print(f"  Stable:        {'NO (imaginary modes)' if phonon_results['has_imaginary'] else 'YES'}")
    print(f"  N(Ef):         {n_ef:.2f} states/eV")
    print(f"  mu*:           {args.mu_star}")
    if tc_results.get('lam_exp'):
        print(f"  lambda (back): {tc_results['lam_exp']:.3f}")
    print(f"  Exp. Tc:       5.8 K")
    print(f"  Total time:    {elapsed:.1f} sec")
    print(f"\n  Output files ({args.prefix}/):")
    print(f"    {outpath(args.prefix, f'{args.prefix}_phonon_bands.dat')}")
    print(f"    {outpath(args.prefix, f'{args.prefix}_phonon_dos.dat')}")
    if not args.ascii_only:
        print(f"    {outpath(args.prefix, f'{args.prefix}_phonon_bands.png')}")


if __name__ == '__main__':
    main()
