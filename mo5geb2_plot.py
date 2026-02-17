"""
Mo5GeB2 - Plot DOS and Band Structure
======================================
Reads output from mo5_ge_b2-dft.py (or mo5geb2_dft.py).

Usage:
    python plot_mo5geb2.py
    python plot_mo5geb2.py --prefix mo5_ge_b2
    python plot_mo5geb2.py --ascii-only          # terminal output only, no PNG
    python plot_mo5geb2.py --no-ascii             # PNG only, no terminal output
"""

import argparse
import json
import os
import sys
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot Mo5GeB2 DFT results')
    parser.add_argument('--prefix', type=str, default='mo5geb2',
                        help='File prefix (default: mo5geb2)')
    parser.add_argument('--ascii-only', action='store_true',
                        help='Only print ASCII plots, no PNG')
    parser.add_argument('--no-ascii', action='store_true',
                        help='Only generate PNGs, no ASCII output')
    parser.add_argument('--emin', type=float, default=-10.0,
                        help='Energy min for plots (default: -10)')
    parser.add_argument('--emax', type=float, default=5.0,
                        help='Energy max for plots (default: 5)')
    return parser.parse_args()


# ================================================================
# ASCII plotting
# ================================================================

def ascii_dos(energy: np.ndarray, dos: np.ndarray,
              width: int = 70, height: int = 25,
              emin: float = -10.0, emax: float = 5.0) -> str:
    """Render DOS as ASCII art."""
    mask = (energy >= emin) & (energy <= emax)
    e = energy[mask]
    d = dos[mask]

    if len(e) == 0:
        return "No data in energy range"

    d_max = np.max(d)
    if d_max == 0:
        return "DOS is zero everywhere"

    lines = []
    lines.append(f"  Density of States - Mo5GeB2 (DFT-PBE)")
    lines.append(f"  N(Ef) = {d[np.argmin(np.abs(e))]:.2f} states/eV")
    lines.append("")

    # Build the plot grid
    for row in range(height, -1, -1):
        threshold = (row / height) * d_max
        line_chars = []
        for col in range(width):
            idx = int(col * len(d) / width)
            if idx >= len(d):
                idx = len(d) - 1
            if d[idx] >= threshold:
                # Check if near Fermi level
                if abs(e[idx]) < (emax - emin) / width:
                    line_chars.append('|')
                else:
                    line_chars.append('#')
            else:
                # Fermi level marker
                if abs(e[idx]) < (emax - emin) / width:
                    line_chars.append(':')
                else:
                    line_chars.append(' ')

        if row == height:
            label = f"{d_max:6.1f} "
        elif row == height // 2:
            label = f"{d_max/2:6.1f} "
        elif row == 0:
            label = f"{'0':>6s} "
        else:
            label = "       "

        lines.append(f"{label}{''.join(line_chars)}")

    # X-axis
    lines.append("       " + "-" * width)
    # Tick labels
    ticks = np.linspace(emin, emax, 7)
    tick_line = "     "
    for t in ticks:
        pos = int((t - emin) / (emax - emin) * width)
        label = f"{t:.0f}"
        tick_line = tick_line[:pos + 5] + label + tick_line[pos + 5 + len(label):]
    lines.append(tick_line)
    lines.append(f"{'Energy - Ef (eV)':^{width + 7}}")
    lines.append(f"{'DOS (states/eV)  |/:=Ef':^{width + 7}}")

    return "\n".join(lines)


def ascii_bands(bands_file: str, width: int = 70, height: int = 30,
                emin: float = -10.0, emax: float = 5.0) -> str:
    """Render band structure as ASCII art."""
    from ase.spectrum.band_structure import BandStructure

    bs = BandStructure.read(bands_file)
    e = bs.energies[0]  # spin 0, already Ef=0 from subtract_reference()
    nkpts, nbands = e.shape

    xcoords, label_xcoords, labels = bs.path.get_linear_kpoint_axis()

    lines = []
    lines.append(f"  Band Structure - Mo5GeB2 (DFT-PBE)")

    # Count crossings
    crossings = sum(1 for b in range(nbands) if e[:, b].min() < 0 and e[:, b].max() > 0)
    lines.append(f"  {crossings} bands crossing Ef => METALLIC")
    lines.append("")

    # Build grid
    grid = [[' ' for _ in range(width)] for _ in range(height + 1)]

    # Map xcoords to column positions
    x_min, x_max = xcoords[0], xcoords[-1]

    # Plot bands
    for b in range(nbands):
        band = e[:, b]
        crosses_ef = band.min() < 0 and band.max() > 0
        for col in range(width):
            k_idx = int(col * (nkpts - 1) / (width - 1))
            val = band[k_idx]
            if emin <= val <= emax:
                row = int((1.0 - (val - emin) / (emax - emin)) * height)
                if 0 <= row <= height:
                    if crosses_ef:
                        grid[row][col] = '*'
                    elif grid[row][col] == ' ':
                        grid[row][col] = '.'

    # Add Fermi level line
    ef_row = int((1.0 - (0 - emin) / (emax - emin)) * height)
    if 0 <= ef_row <= height:
        for col in range(width):
            if grid[ef_row][col] == ' ':
                grid[ef_row][col] = '-'

    # Add high-symmetry point vertical lines
    for lx in label_xcoords:
        col = int((lx - x_min) / (x_max - x_min) * (width - 1))
        if 0 <= col < width:
            for row in range(height + 1):
                if grid[row][col] in (' ', '-'):
                    grid[row][col] = '|' if grid[row][col] == '-' else ':'

    # Render
    for row in range(height + 1):
        if row == 0:
            label = f"{emax:6.1f} "
        elif row == height:
            label = f"{emin:6.1f} "
        elif row == ef_row:
            label = f"  Ef=0 "
        elif row == height // 4:
            val = emax - (emax - emin) * row / height
            label = f"{val:6.1f} "
        elif row == 3 * height // 4:
            val = emax - (emax - emin) * row / height
            label = f"{val:6.1f} "
        else:
            label = "       "
        lines.append(f"{label}{''.join(grid[row])}")

    # X-axis with labels
    lines.append("       " + "-" * width)

    if labels:
        tick_line = list(" " * (width + 7))
        for lx, lbl in zip(label_xcoords, labels):
            col = int((lx - x_min) / (x_max - x_min) * (width - 1))
            if lbl in ('G', 'Gamma', 'GAMMA'):
                lbl = 'G'
            pos = col + 7
            if pos < len(tick_line):
                for c_idx, ch in enumerate(lbl):
                    if pos + c_idx < len(tick_line):
                        tick_line[pos + c_idx] = ch
        lines.append("".join(tick_line))

    lines.append(f"{'k-point path':^{width + 7}}")
    lines.append(f"{'Energy (eV)  *=crosses Ef  .=other bands':^{width + 7}}")

    return "\n".join(lines)


# ================================================================
# PNG plotting (matplotlib)
# ================================================================

def plot_dos_png(dos_file: str, output: str,
                 emin: float = -10.0, emax: float = 5.0) -> None:
    """Plot DOS to PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = np.loadtxt(dos_file)
    energy = data[:, 0]
    dos = data[:, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(energy, dos, color='#2563eb', linewidth=1.0)
    ax.fill_between(energy, dos, alpha=0.15, color='#2563eb')
    ax.axvline(0, color='#dc2626', linestyle='--', linewidth=1.0, label='$E_F$')

    idx_ef = np.argmin(np.abs(energy))
    n_ef = dos[idx_ef]
    ax.plot(0, n_ef, 'o', color='#dc2626', markersize=6, zorder=5)
    ax.annotate(f'N($E_F$) = {n_ef:.2f} st/eV',
                xy=(0, n_ef), xytext=(1.5, n_ef * 1.1),
                fontsize=10, color='#dc2626',
                arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.0))

    ax.set_xlabel('Energy - $E_F$ (eV)')
    ax.set_ylabel('DOS (states/eV)')
    ax.set_title('Mo$_5$GeB$_2$ - Density of States (DFT-PBE)')
    ax.set_xlim(emin, emax)
    ax.set_ylim(0, None)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved {output}")


def plot_bands_png(bands_file: str, output: str,
                   emin: float = -10.0, emax: float = 5.0) -> None:
    """Plot band structure to PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from ase.spectrum.band_structure import BandStructure

    bs = BandStructure.read(bands_file)
    e = bs.energies[0]  # spin 0, already Ef=0
    xcoords, label_xcoords, labels = bs.path.get_linear_kpoint_axis()

    fig, ax = plt.subplots(figsize=(8, 6))

    for band_idx in range(e.shape[1]):
        band = e[:, band_idx]
        if band.min() < 0 and band.max() > 0:
            ax.plot(xcoords, band, color='#dc2626', linewidth=1.2, alpha=0.8)
        else:
            ax.plot(xcoords, band, color='#2563eb', linewidth=0.8, alpha=0.6)

    ax.axhline(0, color='#dc2626', linestyle='--', linewidth=1.0, alpha=0.7)

    # High-symmetry labels
    tick_labels = [r'$\Gamma$' if l in ('G', 'Gamma', 'GAMMA') else l
                   for l in labels]
    for lx in label_xcoords:
        ax.axvline(lx, color='black', linewidth=0.5, alpha=0.5)
    ax.set_xticks(label_xcoords)
    ax.set_xticklabels(tick_labels)

    ax.set_xlim(xcoords[0], xcoords[-1])
    ax.set_ylim(emin, emax)
    ax.set_ylabel('Energy - $E_F$ (eV)')
    ax.set_title('Mo$_5$GeB$_2$ - Band Structure (DFT-PBE)')
    ax.grid(True, axis='y', alpha=0.3)

    legend_elements = [
        Line2D([0], [0], color='#dc2626', lw=1.5, label='Crosses $E_F$ (metallic)'),
        Line2D([0], [0], color='#2563eb', lw=1.0, label='Other bands'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved {output}")


def plot_combined_png(dos_file: str, bands_file: str, output: str,
                      emin: float = -10.0, emax: float = 5.0) -> None:
    """Side-by-side band structure + DOS."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ase.spectrum.band_structure import BandStructure

    data = np.loadtxt(dos_file)
    energy_dos = data[:, 0]
    dos = data[:, 1]

    bs = BandStructure.read(bands_file)
    e = bs.energies[0]  # already Ef=0
    xcoords, label_xcoords, labels = bs.path.get_linear_kpoint_axis()

    fig, (ax_bands, ax_dos) = plt.subplots(
        1, 2, figsize=(12, 6), sharey=True,
        gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.05}
    )

    for band_idx in range(e.shape[1]):
        band = e[:, band_idx]
        if band.min() < 0 and band.max() > 0:
            ax_bands.plot(xcoords, band, color='#dc2626', linewidth=1.2, alpha=0.8)
        else:
            ax_bands.plot(xcoords, band, color='#2563eb', linewidth=0.8, alpha=0.6)

    ax_bands.axhline(0, color='#dc2626', linestyle='--', linewidth=1.0, alpha=0.7)

    if labels:
        tick_labels = [r'$\Gamma$' if l in ('G', 'Gamma', 'GAMMA') else l
                       for l in labels]
        for lx in label_xcoords:
            ax_bands.axvline(lx, color='black', linewidth=0.5, alpha=0.5)
        ax_bands.set_xticks(label_xcoords)
        ax_bands.set_xticklabels(tick_labels)

    ax_bands.set_xlim(xcoords[0], xcoords[-1])
    ax_bands.set_ylim(emin, emax)
    ax_bands.set_ylabel('Energy - $E_F$ (eV)')
    ax_bands.set_title('Band Structure')
    ax_bands.grid(True, axis='y', alpha=0.3)

    ax_dos.plot(dos, energy_dos, color='#2563eb', linewidth=1.0)
    ax_dos.fill_betweenx(energy_dos, dos, alpha=0.15, color='#2563eb')
    ax_dos.axhline(0, color='#dc2626', linestyle='--', linewidth=1.0, alpha=0.7)
    ax_dos.set_xlabel('DOS (st/eV)')
    ax_dos.set_title('DOS')
    ax_dos.set_xlim(0, None)
    ax_dos.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Mo$_5$GeB$_2$ - DFT-PBE Electronic Structure',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {output}")


def main() -> None:
    args = parse_args()
    prefix = args.prefix

    # Input files are in {prefix}/ directory
    outdir = prefix
    dos_file = os.path.join(outdir, f'{prefix}_dos.dat')
    bands_file = os.path.join(outdir, f'{prefix}_bands.json')

    # Fall back to current directory if not in subdir (backward compat)
    if not os.path.exists(dos_file) and os.path.exists(f'{prefix}_dos.dat'):
        dos_file = f'{prefix}_dos.dat'
        outdir = '.'
    if not os.path.exists(bands_file) and os.path.exists(f'{prefix}_bands.json'):
        bands_file = f'{prefix}_bands.json'
        outdir = '.'

    has_dos = os.path.exists(dos_file)
    has_bands = os.path.exists(bands_file)

    if not has_dos and not has_bands:
        print(f"No output files found in '{prefix}/' or current directory.")
        print(f"Run mo5geb2_dft.py first, or specify --prefix")
        sys.exit(1)

    # ASCII output
    if not args.no_ascii:
        if has_dos:
            data = np.loadtxt(dos_file)
            print(ascii_dos(data[:, 0], data[:, 1],
                            emin=args.emin, emax=args.emax))
            print()

        if has_bands:
            print(ascii_bands(bands_file,
                              emin=args.emin, emax=args.emax))
            print()

    # PNG output goes into same directory as input
    if not args.ascii_only:
        try:
            import matplotlib
        except ImportError:
            print("matplotlib not installed, skipping PNG output")
            print("Install with: pip install matplotlib")
            return

        if has_dos:
            plot_dos_png(dos_file, os.path.join(outdir, f'{prefix}_dos.png'),
                         emin=args.emin, emax=args.emax)

        if has_bands:
            plot_bands_png(bands_file, os.path.join(outdir, f'{prefix}_bands.png'),
                           emin=args.emin, emax=args.emax)

        if has_dos and has_bands:
            plot_combined_png(dos_file, bands_file,
                              os.path.join(outdir, f'{prefix}_combined.png'),
                              emin=args.emin, emax=args.emax)


if __name__ == '__main__':
    main()
