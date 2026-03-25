"""
timing_analysis.py — Pulse timing gene analysis across Experiments 3, 5, and 6.

Analyzes gene_9 (contraction_frac) and gene_10 (freq_mult in Exp3/5/6,
refractory_frac in Exp2) across evolutionary runs.

Produces three figures in output/analysis/:
  1. timing_trajectories.png  — per-generation mean ± 1σ over time
  2. timing_2d_scatter.png    — all valid individuals in timing space, coloured by generation
  3. timing_attractors.png    — 1σ/2σ ellipses for final 10 gens of each experiment

Usage:
    uv run python helpers/timing_analysis.py
"""

import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import chi2 as scipy_chi2

# ---------------------------------------------------------------------------
# Paths (relative to project root; script is called from there via uv run)
# ---------------------------------------------------------------------------

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXP_PATHS = {
    'exp3': os.path.join(BASE, 'output/cloud/3080Ti/exp3_s42/evolution_log_exp3_s42.csv'),
    'exp5': os.path.join(BASE, 'output/cloud/3080Ti/exp5_s42/evolution_log_exp5_s42.csv'),
    'exp6': os.path.join(BASE, 'output/cloud/3080Ti/exp6_s137/evolution_log_exp6_s137.csv'),
}

EXP2_SEEDS = ['s42', 's137', 's999', 's2024']
EXP2_PATHS = {
    seed: os.path.join(BASE, f'output/cloud/3080Ti/exp2_{seed}/evolution_log_exp2_{seed}.csv')
    for seed in EXP2_SEEDS
}

OUTPUT_DIR = os.path.join(BASE, 'output/analysis')

# Gene column indices (0-based after stripping header)
GENE_9_COL  = 'gene_9'
GENE_10_COL = 'gene_10'

# Experiment display config
EXP_CONFIG = {
    'exp3': {'label': 'Exp 3',  'color': 'tab:blue'},
    'exp5': {'label': 'Exp 5',  'color': 'tab:orange'},
    'exp6': {'label': 'Exp 6',  'color': 'tab:green'},
}

# Gene bounds
CONTRACTION_BOUNDS = (0.05, 0.60)
FREQMULT_BOUNDS    = (0.5,  2.0)
REFRACTORY_BOUNDS  = (0.20, 0.75)
CONTRACTION_INIT   = 0.20
FREQMULT_INIT      = 1.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path):
    """Load an evolution log CSV.  Returns list of dicts with numeric values."""
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({k: float(v) for k, v in row.items()})
            except ValueError:
                pass  # skip malformed rows
    return rows if rows else None


def load_exp_data(paths_dict):
    """Load multiple CSVs; returns {key: rows_list} for found files."""
    data = {}
    for key, path in paths_dict.items():
        rows = load_csv(path)
        if rows is None:
            print(f'  [skip] {key}: not found at {path}')
        else:
            print(f'  [ok]   {key}: {len(rows)} rows from {path}')
            data[key] = rows
    return data


def valid_rows(rows):
    """Filter to valid individuals (valid == 1)."""
    return [r for r in rows if r['valid'] == 1.0]


def best_fitness(rows, key='displacement'):
    """Return best (max) value of `key` across all valid rows."""
    vals = [r[key] for r in valid_rows(rows)]
    return max(vals) if vals else float('nan')


def per_gen_stats(rows, gene_col):
    """Compute per-generation mean and std of gene_col for valid individuals.

    Returns (generations_array, means_array, stds_array).
    """
    from collections import defaultdict
    by_gen = defaultdict(list)
    for r in valid_rows(rows):
        by_gen[int(r['generation'])].append(r[gene_col])
    gens = sorted(by_gen)
    means = [np.mean(by_gen[g]) for g in gens]
    stds  = [np.std(by_gen[g])  for g in gens]
    return np.array(gens), np.array(means), np.array(stds)


def final_n_gens_valid(rows, n=10):
    """Return valid rows from the final n generations."""
    all_gens = sorted({int(r['generation']) for r in rows})
    cutoff = all_gens[-n] if len(all_gens) >= n else all_gens[0]
    return [r for r in valid_rows(rows) if int(r['generation']) >= cutoff]


def covariance_ellipse(ax, mean, cov, n_std, color, linestyle, label=None, alpha=1.0):
    """Draw a covariance ellipse on `ax`.

    Computes eigendecomposition of `cov` and draws an ellipse at `n_std` sigma.
    The ellipse is returned so the caller can add it to a legend.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Guard against near-zero eigenvalues
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    # Scale factor: chi2 with 2 DOF to get the correct probability contour
    scale = np.sqrt(scipy_chi2.ppf(scipy_chi2.cdf(n_std**2, df=1), df=2))
    width  = 2 * scale * np.sqrt(eigenvalues[0])
    height = 2 * scale * np.sqrt(eigenvalues[1])
    angle  = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    ellipse = mpatches.Ellipse(
        xy=mean, width=width, height=height, angle=angle,
        edgecolor=color, facecolor='none', linestyle=linestyle,
        linewidth=2, alpha=alpha, label=label,
    )
    ax.add_patch(ellipse)
    return ellipse


# ---------------------------------------------------------------------------
# Figure 1: timing_trajectories.png
# ---------------------------------------------------------------------------

def plot_timing_trajectories(exp_data, exp2_data, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    fig.suptitle('Pulse Timing Gene Trajectories', fontsize=14, fontweight='bold')

    # ---- Row 1: Gene 9 (contraction_frac) ----
    ax1 = axes[0]
    for key, rows in exp_data.items():
        cfg = EXP_CONFIG[key]
        gens, means, stds = per_gen_stats(rows, GENE_9_COL)
        ax1.plot(gens, means, color=cfg['color'], label=cfg['label'], linewidth=2)
        ax1.fill_between(gens, means - stds, means + stds, color=cfg['color'], alpha=0.2)

    lo, hi = CONTRACTION_BOUNDS
    ax1.axhline(lo, color='gray', linestyle='--', linewidth=1, label=f'bound {lo}')
    ax1.axhline(hi, color='gray', linestyle='--', linewidth=1, label=f'bound {hi}')
    ax1.axhline(CONTRACTION_INIT, color='black', linestyle=':', linewidth=1,
                label=f'init {CONTRACTION_INIT}')
    ax1.set_ylabel('contraction_frac')
    ax1.set_xlabel('generation')
    ax1.legend(fontsize=8)
    ax1.set_title('Gene 9: contraction_frac')

    # ---- Row 2: Gene 10 (freq_mult vs refractory_frac) ----
    ax2 = axes[1]

    for key, rows in exp_data.items():
        cfg = EXP_CONFIG[key]
        gens, means, stds = per_gen_stats(rows, GENE_10_COL)
        ax2.plot(gens, means, color=cfg['color'], label=cfg['label'] + ' freq_mult', linewidth=2)
        ax2.fill_between(gens, means - stds, means + stds, color=cfg['color'], alpha=0.2)

    # Freq_mult bounds (primary axis)
    lo_fm, hi_fm = FREQMULT_BOUNDS
    ax2.axhline(lo_fm, color='tab:red',  linestyle=':', linewidth=1, label=f'freq_mult bound {lo_fm}')
    ax2.axhline(hi_fm, color='tab:blue', linestyle=':', linewidth=1, label=f'freq_mult bound {hi_fm}')
    ax2.set_ylabel('freq_mult (Exp3/5/6)', fontsize=9)
    ax2.set_xlabel('generation')

    # Secondary axis for Exp2 refractory_frac
    ax2b = ax2.twinx()
    if exp2_data:
        # Average the mean trajectories across available seeds
        seed_trajectories = {}
        for seed, rows in exp2_data.items():
            gens_s, means_s, _ = per_gen_stats(rows, GENE_10_COL)
            seed_trajectories[seed] = (gens_s, means_s)

        # Align on common generations
        all_gens_set = set()
        for gens_s, _ in seed_trajectories.values():
            all_gens_set.update(gens_s.tolist())
        all_gens_sorted = np.array(sorted(all_gens_set))

        stacked = []
        for seed, (gens_s, means_s) in seed_trajectories.items():
            # Interpolate to common grid (nearest, since gens are integers)
            gen_to_mean = dict(zip(gens_s.astype(int).tolist(), means_s.tolist()))
            vals = [gen_to_mean.get(g, np.nan) for g in all_gens_sorted.astype(int)]
            stacked.append(vals)
        stacked = np.array(stacked)
        avg_means = np.nanmean(stacked, axis=0)

        ax2b.plot(all_gens_sorted, avg_means, color='gray', linestyle='--',
                  linewidth=2, label='Exp2 refractory_frac')
        lo_rf, hi_rf = REFRACTORY_BOUNDS
        ax2b.axhline(lo_rf, color='silver', linestyle=':', linewidth=1,
                     label=f'refrac bound {lo_rf}')
        ax2b.axhline(hi_rf, color='silver', linestyle=':', linewidth=1,
                     label=f'refrac bound {hi_rf}')
        ax2b.set_ylabel('refractory_frac (Exp2)', fontsize=9)
        ax2b.legend(fontsize=8, loc='upper right')

    ax2.set_title('Gene 10: freq_mult vs refractory_frac')
    ax2.legend(fontsize=8, loc='upper left')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ---------------------------------------------------------------------------
# Figure 2: timing_2d_scatter.png
# ---------------------------------------------------------------------------

def plot_timing_2d_scatter(exp_data, out_path):
    n_exps = len(exp_data)
    if n_exps == 0:
        print('  [skip] timing_2d_scatter: no experiment data')
        return

    fig, axes = plt.subplots(1, n_exps, figsize=(5 * n_exps, 5))
    if n_exps == 1:
        axes = [axes]
    fig.suptitle('Timing Space: All Individuals by Generation', fontsize=14, fontweight='bold')

    for ax, (key, rows) in zip(axes, exp_data.items()):
        cfg = EXP_CONFIG[key]
        vrows = valid_rows(rows)

        xs     = np.array([r[GENE_9_COL]  for r in vrows])
        ys     = np.array([r[GENE_10_COL] for r in vrows])
        colors = np.array([r['generation'] for r in vrows])

        sc = ax.scatter(xs, ys, c=colors, cmap='viridis', alpha=0.6, s=15, edgecolors='none')
        plt.colorbar(sc, ax=ax, label='generation')

        # Mark final-generation mean with a red star
        final_gen = int(max(r['generation'] for r in vrows))
        final_rows = [r for r in vrows if int(r['generation']) == final_gen]
        final_mean_x = np.mean([r[GENE_9_COL]  for r in final_rows])
        final_mean_y = np.mean([r[GENE_10_COL] for r in final_rows])
        ax.scatter([final_mean_x], [final_mean_y], c='red', marker='*', s=200,
                   zorder=5, label='final gen mean')

        # Bound lines
        lo_c, hi_c = CONTRACTION_BOUNDS
        lo_f, hi_f = FREQMULT_BOUNDS
        for xv in (lo_c, hi_c):
            ax.axvline(xv, color='gray', linestyle='--', linewidth=1)
        for yv in (lo_f, hi_f):
            ax.axhline(yv, color='gray', linestyle='--', linewidth=1)

        best = best_fitness(rows, 'displacement')
        ax.set_title(f'{cfg["label"]}  (best disp={best:.3f})', fontsize=10)
        ax.set_xlabel('contraction_frac')
        ax.set_ylabel('freq_mult')
        ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ---------------------------------------------------------------------------
# Figure 3: timing_attractors.png
# ---------------------------------------------------------------------------

def plot_timing_attractors(exp_data, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Timing Attractors: Final 10 Generations', fontsize=13, fontweight='bold')
    ax.set_xlabel('contraction_frac')
    ax.set_ylabel('freq_mult')

    legend_handles = []
    annotations = []

    for key, rows in exp_data.items():
        cfg = EXP_CONFIG[key]
        color = cfg['color']
        label = cfg['label']

        final_rows = final_n_gens_valid(rows, n=10)
        if len(final_rows) < 3:
            print(f'  [warn] {key}: too few valid final-gen rows for ellipse ({len(final_rows)})')
            continue

        xs = np.array([r[GENE_9_COL]  for r in final_rows])
        ys = np.array([r[GENE_10_COL] for r in final_rows])
        mean = np.array([xs.mean(), ys.mean()])
        cov  = np.cov(np.stack([xs, ys]))

        # Scatter individual points
        ax.scatter(xs, ys, color=color, alpha=0.3, s=20, edgecolors='none')

        # 1σ and 2σ ellipses
        covariance_ellipse(ax, mean, cov, n_std=1, color=color, linestyle='-',  alpha=0.9)
        covariance_ellipse(ax, mean, cov, n_std=2, color=color, linestyle='--', alpha=0.6)

        # Legend proxy patches
        best = best_fitness(rows, 'displacement')
        patch = mpatches.Patch(color=color, label=f'{label} (best disp={best:.3f})')
        legend_handles.append(patch)

        # Annotation with final mean values
        annotations.append((mean, f'{label}\nc={mean[0]:.3f}\nf={mean[1]:.3f}', color))

    # Bounds
    lo_c, hi_c = CONTRACTION_BOUNDS
    lo_f, hi_f = FREQMULT_BOUNDS
    for xv in (lo_c, hi_c):
        ax.axvline(xv, color='gray', linestyle='--', linewidth=1)
    for yv in (lo_f, hi_f):
        ax.axhline(yv, color='gray', linestyle='--', linewidth=1)

    # Init point
    init_handle = ax.scatter([CONTRACTION_INIT], [FREQMULT_INIT],
                             c='black', marker='x', s=120, linewidths=2,
                             zorder=6, label=f'init ({CONTRACTION_INIT}, {FREQMULT_INIT})')
    legend_handles.insert(0, init_handle)

    # Text annotations (offset slightly to avoid overlap)
    for (mean, text, color) in annotations:
        ax.annotate(text, xy=mean, xytext=(mean[0] + 0.02, mean[1] + 0.05),
                    fontsize=7, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=1),
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor=color))

    ax.legend(handles=legend_handles, fontsize=8, loc='upper right')
    ax.set_xlim(CONTRACTION_BOUNDS[0] - 0.05, CONTRACTION_BOUNDS[1] + 0.05)
    ax.set_ylim(FREQMULT_BOUNDS[0] - 0.1, FREQMULT_BOUNDS[1] + 0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------

def print_summary(exp_data, exp2_data):
    print('\n=== Timing Analysis Summary ===')
    for key, rows in exp_data.items():
        vrows = valid_rows(rows)
        final_gen = int(max(r['generation'] for r in vrows))
        final_rows = [r for r in vrows if int(r['generation']) == final_gen]
        c_mean = np.mean([r[GENE_9_COL]  for r in final_rows])
        f_mean = np.mean([r[GENE_10_COL] for r in final_rows])
        c_std  = np.std( [r[GENE_9_COL]  for r in final_rows])
        f_std  = np.std( [r[GENE_10_COL] for r in final_rows])
        best   = best_fitness(rows, 'displacement')
        print(f'{EXP_CONFIG[key]["label"]:6s}  final contraction={c_mean:.3f}±{c_std:.3f}  '
              f'freq_mult={f_mean:.3f}±{f_std:.3f}  best_disp={best:.4f}')

    if exp2_data:
        print()
        for seed, rows in exp2_data.items():
            vrows = valid_rows(rows)
            if not vrows:
                continue
            final_gen  = int(max(r['generation'] for r in vrows))
            final_rows = [r for r in vrows if int(r['generation']) == final_gen]
            c_mean = np.mean([r[GENE_9_COL]  for r in final_rows])
            r_mean = np.mean([r[GENE_10_COL] for r in final_rows])  # refractory_frac
            best_eff = best_fitness(rows, 'fitness')
            print(f'Exp2 {seed:5s}  final contraction={c_mean:.3f}  refractory={r_mean:.3f}  '
                  f'best_eff={best_eff:.4f}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Loading experiment data...')
    exp_data  = load_exp_data(EXP_PATHS)
    exp2_data = load_exp_data(EXP2_PATHS)

    if not exp_data:
        print('ERROR: no experiment data found; cannot produce figures.')
        sys.exit(1)

    print('\nProducing Figure 1: timing_trajectories.png')
    plot_timing_trajectories(
        exp_data, exp2_data,
        os.path.join(OUTPUT_DIR, 'timing_trajectories.png'),
    )

    print('\nProducing Figure 2: timing_2d_scatter.png')
    plot_timing_2d_scatter(
        exp_data,
        os.path.join(OUTPUT_DIR, 'timing_2d_scatter.png'),
    )

    print('\nProducing Figure 3: timing_attractors.png')
    plot_timing_attractors(
        exp_data,
        os.path.join(OUTPUT_DIR, 'timing_attractors.png'),
    )

    print_summary(exp_data, exp2_data)
    print('\nDone.')


if __name__ == '__main__':
    main()
