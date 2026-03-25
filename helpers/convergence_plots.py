"""
convergence_plots.py — CMA-ES convergence diagnostics from best_genomes JSON files.

Analyzes sigma decay, covariance condition number, fitness improvement, and
per-gene variance evolution across Experiments 1, 2, 3, 5, and 6.

Produces four figures in output/analysis/:
  1. convergence_combined.png       — 2×2 convergence dashboard
  2. cov_heatmap_exp3.png           — per-gene covariance diagonal over time
  3. cov_heatmap_exp5.png
  4. cov_heatmap_exp6.png

Usage:
    uv run python helpers/convergence_plots.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAIN_EXPS = {
    'exp3': os.path.join(BASE, 'output/cloud/3080Ti/exp3_s42/best_genomes_exp3_s42.json'),
    'exp5': os.path.join(BASE, 'output/cloud/3080Ti/exp5_s42/best_genomes_exp5_s42.json'),
    'exp6': os.path.join(BASE, 'output/cloud/3080Ti/exp6_s137/best_genomes_exp6_s137.json'),
}

EXP1_PATH = os.path.join(BASE, 'output/cloud/4090/seed_42/best_genomes_seed_42.json')

EXP2_SEEDS = ['s42', 's137', 's2024']   # s999 has no JSON
EXP2_PATHS = {
    seed: os.path.join(BASE, f'output/cloud/3080Ti/exp2_{seed}/best_genomes_exp2_{seed}.json')
    for seed in EXP2_SEEDS
}

OUTPUT_DIR = os.path.join(BASE, 'output/analysis')

# Display config
EXP_CONFIG = {
    'exp1': {'label': 'Exp 1 (s42)', 'color': 'tab:purple'},
    'exp3': {'label': 'Exp 3 (s42)', 'color': 'tab:blue'},
    'exp5': {'label': 'Exp 5 (s42)', 'color': 'tab:orange'},
    'exp6': {'label': 'Exp 6 (s137)', 'color': 'tab:green'},
    'exp2_avg': {'label': 'Exp 2 (avg)', 'color': 'gray'},
}

# Gene labels for 11D genomes
GENE_LABELS_11D = [
    'cp1_x', 'cp1_y', 'cp2_x', 'cp2_y', 'end_x', 'end_y',
    't_base', 't_mid', 't_tip', 'contraction', 'freq_mult',
]

GENE_LABELS_9D = [
    'cp1_x', 'cp1_y', 'cp2_x', 'cp2_y', 'end_x', 'end_y',
    't_base', 't_mid', 't_tip',
]

SIGMA_CONVERGED = 0.1   # threshold for "90% converged" annotation


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path):
    """Load a best_genomes JSON file.  Returns list of entry dicts or None."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            return None
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f'  [error] {path}: {e}')
        return None


def extract_series(entries, field):
    """Extract (generations_array, values_array) for a given field."""
    pairs = [(e['generation'], e[field]) for e in entries if field in e]
    if not pairs:
        return np.array([]), np.array([])
    gens, vals = zip(*pairs)
    return np.array(gens), np.array(vals)


def best_displacement_series(entries):
    """Return per-generation best displacement.

    Uses the `displacement` field directly (always raw, positive = upward).
    """
    return extract_series(entries, 'displacement')


def best_fitness_series(entries):
    """Return per-generation best fitness.

    Negates if values are consistently negative (CMA-ES stores -fitness).
    """
    gens, vals = extract_series(entries, 'fitness')
    if len(vals) > 0 and vals[0] < 0 and vals[-1] < 0:
        vals = -vals
    return gens, vals


def avg_fitness_series(entries):
    """Return per-generation avg_fitness (negated if needed)."""
    gens, vals = extract_series(entries, 'avg_fitness')
    if len(vals) > 0 and vals[0] < 0 and vals[-1] < 0:
        vals = -vals
    return gens, vals


def convergence_generation(entries, threshold=SIGMA_CONVERGED):
    """Find first generation where sigma drops below `threshold`."""
    for e in entries:
        if e.get('sigma', float('inf')) < threshold:
            return e['generation']
    return None


def exp2_avg_sigma(exp2_data):
    """Compute average sigma trajectory across Exp2 seeds."""
    seed_series = {}
    for seed, entries in exp2_data.items():
        gens, sigmas = extract_series(entries, 'sigma')
        seed_series[seed] = dict(zip(gens.astype(int).tolist(), sigmas.tolist()))

    all_gens = sorted({g for series in seed_series.values() for g in series})
    avg = []
    for g in all_gens:
        vals = [series[g] for series in seed_series.values() if g in series]
        avg.append(np.mean(vals))
    return np.array(all_gens), np.array(avg)


# ---------------------------------------------------------------------------
# Figure 1: convergence_combined.png
# ---------------------------------------------------------------------------

def plot_convergence_combined(main_data, exp1_data, exp2_data, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('CMA-ES Convergence Diagnostics', fontsize=14, fontweight='bold')

    ax_sigma, ax_cond, ax_fitness, ax_gap = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    )

    # ---- Top-left: sigma on log scale ----
    ax_sigma.set_title('Step-size σ (log scale)')
    ax_sigma.set_yscale('log')
    ax_sigma.set_xlabel('generation')
    ax_sigma.set_ylabel('σ')

    for key, entries in main_data.items():
        cfg = EXP_CONFIG[key]
        gens, sigmas = extract_series(entries, 'sigma')
        ax_sigma.plot(gens, sigmas, color=cfg['color'], label=cfg['label'], linewidth=2)

    if exp1_data:
        gens, sigmas = extract_series(exp1_data, 'sigma')
        ax_sigma.plot(gens, sigmas, color=EXP_CONFIG['exp1']['color'],
                      label=EXP_CONFIG['exp1']['label'], linewidth=2, linestyle='-.')

    if exp2_data:
        gens_avg, sigma_avg = exp2_avg_sigma(exp2_data)
        ax_sigma.plot(gens_avg, sigma_avg, color=EXP_CONFIG['exp2_avg']['color'],
                      label=EXP_CONFIG['exp2_avg']['label'], linewidth=2, linestyle='--')

    ax_sigma.axhline(SIGMA_CONVERGED, color='black', linestyle='--', linewidth=1,
                     label=f'σ={SIGMA_CONVERGED} (90% converged)')
    ax_sigma.legend(fontsize=8)

    # ---- Top-right: condition number on log scale ----
    ax_cond.set_title('Covariance condition number (log scale)')
    ax_cond.set_yscale('log')
    ax_cond.set_xlabel('generation')
    ax_cond.set_ylabel('cond(C)')

    for key, entries in main_data.items():
        cfg = EXP_CONFIG[key]
        gens, conds = extract_series(entries, 'cov_cond')
        ax_cond.plot(gens, conds, color=cfg['color'], label=cfg['label'], linewidth=2)

    if exp1_data:
        gens, conds = extract_series(exp1_data, 'cov_cond')
        ax_cond.plot(gens, conds, color=EXP_CONFIG['exp1']['color'],
                     label=EXP_CONFIG['exp1']['label'], linewidth=2, linestyle='-.')

    if exp2_data:
        # Average across seeds
        seed_cond = {}
        for seed, entries in exp2_data.items():
            gens_s, conds_s = extract_series(entries, 'cov_cond')
            seed_cond[seed] = dict(zip(gens_s.astype(int).tolist(), conds_s.tolist()))
        all_gens = sorted({g for s in seed_cond.values() for g in s})
        avg_cond = [np.mean([s[g] for s in seed_cond.values() if g in s]) for g in all_gens]
        ax_cond.plot(all_gens, avg_cond, color=EXP_CONFIG['exp2_avg']['color'],
                     label=EXP_CONFIG['exp2_avg']['label'], linewidth=2, linestyle='--')

    ax_cond.legend(fontsize=8)

    # ---- Bottom-left: fitness/displacement improvement ----
    ax_fitness.set_title('Best displacement over time')
    ax_fitness.set_xlabel('generation')
    ax_fitness.set_ylabel('displacement')

    for key, entries in main_data.items():
        cfg = EXP_CONFIG[key]
        gens, best_disp = best_displacement_series(entries)
        gens_a, avg_fit = avg_fitness_series(entries)
        ax_fitness.plot(gens, best_disp, color=cfg['color'], label=f'{cfg["label"]} best',
                        linewidth=2)
        ax_fitness.plot(gens_a, avg_fit, color=cfg['color'], linestyle='--',
                        alpha=0.5, linewidth=1, label=f'{cfg["label"]} avg')

    if exp1_data:
        gens, best_disp = best_displacement_series(exp1_data)
        gens_a, avg_fit = avg_fitness_series(exp1_data)
        ax_fitness.plot(gens, best_disp, color=EXP_CONFIG['exp1']['color'],
                        label=f'{EXP_CONFIG["exp1"]["label"]} best', linewidth=2, linestyle='-.')
        ax_fitness.plot(gens_a, avg_fit, color=EXP_CONFIG['exp1']['color'],
                        linestyle=':', alpha=0.5, linewidth=1)

    ax_fitness.legend(fontsize=7, ncol=2)

    # ---- Bottom-right: best-mean gap ----
    ax_gap.set_title('Best − avg fitness gap (selection pressure)')
    ax_gap.set_xlabel('generation')
    ax_gap.set_ylabel('best − avg')

    for key, entries in main_data.items():
        cfg = EXP_CONFIG[key]
        gens_b, best_d = best_displacement_series(entries)
        gens_a, avg_f  = avg_fitness_series(entries)
        # Align on same generations
        gen_to_best = dict(zip(gens_b.astype(int).tolist(), best_d.tolist()))
        gen_to_avg  = dict(zip(gens_a.astype(int).tolist(), avg_f.tolist()))
        common_gens = sorted(set(gen_to_best) & set(gen_to_avg))
        gaps = [gen_to_best[g] - gen_to_avg[g] for g in common_gens]
        ax_gap.plot(common_gens, gaps, color=cfg['color'], label=cfg['label'], linewidth=2)

    ax_gap.axhline(0, color='black', linewidth=0.5)
    ax_gap.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ---------------------------------------------------------------------------
# Figures 2-4: covariance diagonal heatmaps
# ---------------------------------------------------------------------------

def plot_cov_heatmap(entries, exp_name, exp_num, out_path):
    """Plot per-gene, per-generation covariance diagonal as a heatmap.

    Each row (gene) is normalised by its own maximum so all rows are 0–1.
    White = low variance (locked/converged); dark = high variance.
    """
    # Determine genome dimensionality from first entry
    n_genes = len(entries[0].get('cov_diag', []))
    if n_genes == 0:
        print(f'  [skip] {exp_name}: no cov_diag data')
        return

    gene_labels = GENE_LABELS_11D[:n_genes] if n_genes <= 11 else [f'gene_{i}' for i in range(n_genes)]
    if n_genes == 9:
        gene_labels = GENE_LABELS_9D

    gens_sorted = sorted(entries, key=lambda e: e['generation'])
    gen_nums = [e['generation'] for e in gens_sorted]
    n_gens   = len(gen_nums)

    # Build matrix: shape (n_genes, n_gens)
    matrix = np.zeros((n_genes, n_gens))
    for col_idx, e in enumerate(gens_sorted):
        diag = e.get('cov_diag', [0] * n_genes)
        for row_idx in range(n_genes):
            matrix[row_idx, col_idx] = diag[row_idx] if row_idx < len(diag) else 0.0

    # Normalise each gene row by its max (avoid division by zero)
    row_maxes = matrix.max(axis=1, keepdims=True)
    row_maxes[row_maxes == 0] = 1.0
    matrix_norm = matrix / row_maxes

    # Convergence generation (first gen with sigma < threshold)
    conv_gen = convergence_generation(entries)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f'Covariance Diagonal Evolution — {exp_name}', fontsize=12, fontweight='bold')

    im = ax.imshow(
        matrix_norm, aspect='auto', cmap='hot_r', origin='upper',
        extent=[gen_nums[0] - 0.5, gen_nums[-1] + 0.5, n_genes - 0.5, -0.5],
        vmin=0, vmax=1,
    )

    plt.colorbar(im, ax=ax, label='normalised variance (per gene)')

    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(gene_labels, fontsize=9)
    ax.set_xlabel('generation')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if conv_gen is not None:
        ax.axvline(conv_gen, color='cyan', linewidth=2, linestyle='--',
                   label=f'σ<{SIGMA_CONVERGED} at gen {conv_gen}')
        ax.legend(fontsize=8, loc='lower right')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------

def print_summary(main_data, exp1_data, exp2_data):
    print('\n=== Convergence Analysis Summary ===')

    all_data = {}
    if exp1_data:
        all_data['exp1'] = exp1_data
    all_data.update(main_data)

    for key, entries in all_data.items():
        cfg = EXP_CONFIG.get(key, {'label': key})
        gens_b, best_d   = best_displacement_series(entries)
        _, best_fit      = best_fitness_series(entries)
        conv_gen         = convergence_generation(entries)
        final_sigma      = entries[-1].get('sigma', float('nan'))
        final_cond       = entries[-1].get('cov_cond', float('nan'))
        best_disp_val    = best_d[-1] if len(best_d) else float('nan')
        best_fit_val     = best_fit[-1] if len(best_fit) else float('nan')

        print(f'{cfg["label"]:20s}  '
              f'conv_gen={str(conv_gen):>4s}  '
              f'final_σ={final_sigma:.4f}  '
              f'cond={final_cond:.1f}  '
              f'best_disp={best_disp_val:.4f}  '
              f'best_fit={best_fit_val:.4f}')

    if exp2_data:
        print()
        for seed, entries in exp2_data.items():
            conv_gen     = convergence_generation(entries)
            final_sigma  = entries[-1].get('sigma', float('nan'))
            _, best_fit  = best_fitness_series(entries)
            best_fit_val = best_fit[-1] if len(best_fit) else float('nan')
            print(f'Exp2 {seed:5s}               '
                  f'conv_gen={str(conv_gen):>4s}  '
                  f'final_σ={final_sigma:.4f}  '
                  f'best_fit={best_fit_val:.4f}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Loading JSON data...')

    main_data = {}
    for key, path in MAIN_EXPS.items():
        entries = load_json(path)
        if entries is None:
            print(f'  [skip] {key}: not found at {path}')
        else:
            print(f'  [ok]   {key}: {len(entries)} entries from {path}')
            main_data[key] = entries

    exp1_data = load_json(EXP1_PATH)
    if exp1_data is None:
        print(f'  [skip] exp1: not found at {EXP1_PATH}')
    else:
        print(f'  [ok]   exp1: {len(exp1_data)} entries')

    exp2_data = {}
    for seed, path in EXP2_PATHS.items():
        entries = load_json(path)
        if entries is None:
            print(f'  [skip] exp2_{seed}: not found at {path}')
        else:
            print(f'  [ok]   exp2_{seed}: {len(entries)} entries')
            exp2_data[seed] = entries

    if not main_data:
        print('ERROR: no main experiment data found; cannot produce figures.')
        sys.exit(1)

    print('\nProducing Figure 1: convergence_combined.png')
    plot_convergence_combined(
        main_data, exp1_data, exp2_data,
        os.path.join(OUTPUT_DIR, 'convergence_combined.png'),
    )

    print('\nProducing covariance heatmaps...')
    heatmap_specs = [
        ('exp3', 'Exp 3', 3),
        ('exp5', 'Exp 5', 5),
        ('exp6', 'Exp 6', 6),
    ]
    for key, exp_name, exp_num in heatmap_specs:
        if key not in main_data:
            print(f'  [skip] heatmap for {key}: data not loaded')
            continue
        plot_cov_heatmap(
            main_data[key], exp_name, exp_num,
            os.path.join(OUTPUT_DIR, f'cov_heatmap_{key}.png'),
        )

    print_summary(main_data, exp1_data, exp2_data)
    print('\nDone.')


if __name__ == '__main__':
    main()
