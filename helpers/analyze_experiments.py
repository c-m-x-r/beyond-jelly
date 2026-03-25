"""
analyze_experiments.py — Comparative analysis across Jellyfih evolutionary experiments.

Loads CSV data from all available experiments and produces four multi-panel figures
saved to output/analysis/.

Experiments:
  Exp1 (9D, efficiency):  output/cloud/4090/seed_42/evolution_log_seed_42.csv
  Exp2 (11D, efficiency): output/cloud/3080Ti/exp2_s{42,137,999,2024}/...
  Exp3 (11D, displacement): output/cloud/3080Ti/exp3_s42/...
  Exp5 (11D, axisym displacement): output/cloud/3080Ti/exp5_s42/...
  Exp6 (11D, efficiency): output/cloud/3080Ti/exp6_s137/...

Usage:
    uv run python helpers/analyze_experiments.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_HERE)
_OUT    = os.path.join(_ROOT, "output")
_ADIR   = os.path.join(_OUT, "analysis")

EXP_PATHS = {
    "exp1": os.path.join(_OUT, "cloud", "4090",   "seed_42",    "evolution_log_seed_42.csv"),
    "exp2_s42":   os.path.join(_OUT, "cloud", "3080Ti", "exp2_s42",   "evolution_log_exp2_s42.csv"),
    "exp2_s137":  os.path.join(_OUT, "cloud", "3080Ti", "exp2_s137",  "evolution_log_exp2_s137.csv"),
    "exp2_s999":  os.path.join(_OUT, "cloud", "3080Ti", "exp2_s999",  "evolution_log_exp2_s999.csv"),
    "exp2_s2024": os.path.join(_OUT, "cloud", "3080Ti", "exp2_s2024", "evolution_log_exp2_s2024.csv"),
    "exp3": os.path.join(_OUT, "cloud", "3080Ti", "exp3_s42",   "evolution_log_exp3_s42.csv"),
    "exp5": os.path.join(_OUT, "cloud", "3080Ti", "exp5_s42",   "evolution_log_exp5_s42.csv"),
    "exp6": os.path.join(_OUT, "cloud", "3080Ti", "exp6_s137",  "evolution_log_exp6_s137.csv"),
}

# ---------------------------------------------------------------------------
# Gene metadata
# ---------------------------------------------------------------------------

GENE_NAMES_11 = ["cp1_x", "cp1_y", "cp2_x", "cp2_y", "end_x", "end_y",
                  "t_base", "t_mid", "t_tip", "contraction", "freq_mult"]

GENE_BOUNDS = {
    "cp1_x":       (0.00,  0.25),
    "cp1_y":       (-0.15, 0.15),
    "cp2_x":       (0.00,  0.30),
    "cp2_y":       (-0.20, 0.15),
    "end_x":       (0.05,  0.35),
    "end_y":       (-0.30, 0.10),
    "t_base":      (0.025, 0.08),
    "t_mid":       (0.025, 0.10),
    "t_tip":       (0.01,  0.04),
    "contraction": (0.05,  0.60),
    "freq_mult":   (0.50,  2.00),
}

# ---------------------------------------------------------------------------
# CSV loader — uses numpy only (pandas not in deps)
# ---------------------------------------------------------------------------

def load_csv(path):
    """Load an experiment CSV into a dict of column-name -> 1-D numpy array.

    Returns None if the file is missing or contains no numeric data rows
    (e.g. an incomplete / stub run with only a header line).
    """
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    # genfromtxt returns a 1-D array when there is exactly one data row, and
    # a 0-D / scalar when there are zero rows — detect both bad cases.
    if data.ndim != 2:
        if data.ndim == 1 and data.shape[0] == len(header):
            # Exactly one row — promote to 2-D so the rest of the code works.
            data = data[np.newaxis, :]
        else:
            # Zero rows or unparseable — treat as missing.
            return None
    if data.shape[0] == 0:
        return None
    # Discard any rows where 'generation' (column 0) is NaN — this catches
    # stub files where the second line is a duplicate header.
    gen_col = data[:, 0]
    valid_rows = ~np.isnan(gen_col)
    if not valid_rows.any():
        return None
    data = data[valid_rows]
    return {col: data[:, i] for i, col in enumerate(header)}


def load_all():
    """Load every experiment CSV; return dict of key -> column dict (or None if missing)."""
    dfs = {}
    for key, path in EXP_PATHS.items():
        d = load_csv(path)
        if d is None:
            print(f"  WARNING: no usable data at {path}")
        else:
            n = len(d["generation"])
            print(f"  Loaded {key}: {n} rows, gens 0-{int(d['generation'].max())}")
        dfs[key] = d
    return dfs


# ---------------------------------------------------------------------------
# Per-generation aggregation helpers
# ---------------------------------------------------------------------------

def best_per_gen(d, col, valid_only=True):
    """Return (generations, best_values) for the given column."""
    if d is None:
        return np.array([]), np.array([])
    gens = np.unique(d["generation"]).astype(int)
    values = []
    for g in gens:
        mask = d["generation"] == g
        if valid_only:
            mask = mask & (d["valid"] == 1)
        if not mask.any():
            values.append(np.nan)
        else:
            values.append(np.nanmax(d[col][mask]))
    return gens, np.array(values)


def mean_sigma_per_gen(d, col, valid_only=True):
    """Return (generations, mean, std) arrays for the given column."""
    if d is None:
        return np.array([]), np.array([]), np.array([])
    gens = np.unique(d["generation"]).astype(int)
    means, stds = [], []
    for g in gens:
        mask = d["generation"] == g
        if valid_only:
            mask = mask & (d["valid"] == 1)
        vals = d[col][mask] if mask.any() else np.array([np.nan])
        means.append(np.nanmean(vals))
        stds.append(np.nanstd(vals))
    return gens, np.array(means), np.array(stds)


def validity_per_gen(d):
    """Return (generations, fraction_valid) arrays."""
    if d is None:
        return np.array([]), np.array([])
    gens = np.unique(d["generation"]).astype(int)
    fracs = []
    for g in gens:
        mask = d["generation"] == g
        fracs.append(d["valid"][mask].mean())
    return gens, np.array(fracs)


def final_gen_valid(d, gens_range=(40, 49)):
    """Return a row-mask for individuals in the final generation window that are valid."""
    if d is None:
        return None
    mask = (d["generation"] >= gens_range[0]) & (d["generation"] <= gens_range[1]) & (d["valid"] == 1)
    return mask


def compute_efficiency_exp1(d):
    """
    Compute efficiency for Exp1 (9D genome, no refractory gene).
    active_frac = 1.0 (no refractory gene; refractory_frac defaults to 0.40 historically,
    but the CSV already has an 'efficiency' column — use it directly).
    """
    if d is None:
        return None
    # Exp1 CSV has an efficiency column; use it directly.
    return d["efficiency"]


# ---------------------------------------------------------------------------
# Figure 1: Fitness trajectories
# ---------------------------------------------------------------------------

def fig_fitness_comparison(dfs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Fitness Trajectories by Experiment", fontsize=14, fontweight="bold")

    # --- Left panel: Efficiency experiments (Exp1, Exp2, Exp6) ---
    ax = axes[0]
    ax.set_title("Efficiency Experiments (Exp 1, 2, 6)")
    ax.set_xlabel("generation")
    ax.set_ylabel("best efficiency per generation")

    # Exp1 — use the efficiency column it already has
    g1, v1 = best_per_gen(dfs["exp1"], "efficiency", valid_only=True)
    if len(g1):
        ax.plot(g1, v1, color="purple", linewidth=2, label="Exp1 (9D, 4090)")

    # Exp2 — four seeds, thin gray lines + thick mean
    exp2_keys = ["exp2_s42", "exp2_s137", "exp2_s999", "exp2_s2024"]
    exp2_curves = []
    for k in exp2_keys:
        g, v = best_per_gen(dfs[k], "efficiency", valid_only=True)
        if len(g):
            ax.plot(g, v, color="gray", linewidth=0.8, alpha=0.6)
            exp2_curves.append((g, v))
    if exp2_curves:
        # Interpolate all to common generation grid then average
        max_gen = max(c[0].max() for c in exp2_curves)
        common_g = np.arange(0, max_gen + 1)
        stacked = []
        for g, v in exp2_curves:
            interp = np.interp(common_g, g, v)
            stacked.append(interp)
        mean_v = np.nanmean(stacked, axis=0)
        ax.plot(common_g, mean_v, color="gray", linewidth=2.5, label="Exp2 (11D, 3080Ti) — 4 seeds")

    # Exp6
    g6, v6 = best_per_gen(dfs["exp6"], "efficiency", valid_only=True)
    if len(g6):
        ax.plot(g6, v6, color="green", linewidth=2, label="Exp6 (11D, efficiency control)")

    ax.legend(fontsize=8)

    # --- Right panel: Displacement experiments (Exp3, Exp5) ---
    ax = axes[1]
    ax.set_title("Displacement Experiments (Exp 3, 5)")
    ax.set_xlabel("generation")
    ax.set_ylabel("best displacement per generation")

    g3, v3 = best_per_gen(dfs["exp3"], "displacement", valid_only=True)
    if len(g3):
        ax.plot(g3, v3, color="blue", linewidth=2, label="Exp3 (11D, displacement)")

    g5, v5 = best_per_gen(dfs["exp5"], "displacement", valid_only=True)
    if len(g5):
        ax.plot(g5, v5, color="orange", linewidth=2, label="Exp5 (axisym, not comparable)")

    ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(_ADIR, exist_ok=True)
    out = os.path.join(_ADIR, "fitness_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Gene trajectories
# ---------------------------------------------------------------------------

def fig_gene_trajectories(dfs):
    n_genes = 11
    fig, axes = plt.subplots(3, 4, figsize=(18, 14))
    fig.suptitle("Gene Trajectories: Mean ± 1σ (valid individuals only)", fontsize=14, fontweight="bold")
    axes_flat = axes.flatten()

    exp_configs = [
        ("exp3", "Exp3", "blue"),
        ("exp5", "Exp5", "orange"),
        ("exp6", "Exp6", "green"),
    ]

    for i, gene_name in enumerate(GENE_NAMES_11):
        ax = axes_flat[i]
        col = f"gene_{i}"
        lo, hi = GENE_BOUNDS[gene_name]

        for key, label, color in exp_configs:
            d = dfs[key]
            g, mean, std = mean_sigma_per_gen(d, col, valid_only=True)
            if len(g) == 0:
                continue
            ax.plot(g, mean, color=color, linewidth=1.5, label=label)
            ax.fill_between(g, mean - std, mean + std, color=color, alpha=0.2)

        # Gene bounds as dashed horizontal lines
        ax.axhline(hi, color="red",  linestyle="--", linewidth=0.8, alpha=0.7, label="_upper bound")
        ax.axhline(lo, color="blue", linestyle="--", linewidth=0.8, alpha=0.7, label="_lower bound")

        ax.set_title(gene_name, fontsize=10)
        ax.set_xlabel("generation", fontsize=7)
        ax.set_ylabel("value", fontsize=7)
        ax.tick_params(labelsize=7)

    # Panel 11: legend box
    ax = axes_flat[11]
    ax.axis("off")
    legend_elements = [
        mpatches.Patch(color="blue",   label="Exp3 (displacement)"),
        mpatches.Patch(color="orange", label="Exp5 (axisym disp)"),
        mpatches.Patch(color="green",  label="Exp6 (efficiency)"),
        plt.Line2D([0], [0], color="red",  linestyle="--", linewidth=1, label="upper bound"),
        plt.Line2D([0], [0], color="blue", linestyle="--", linewidth=1, label="lower bound"),
    ]
    ax.legend(handles=legend_elements, loc="center", fontsize=10, frameon=True, title="Legend")

    plt.tight_layout()
    out = os.path.join(_ADIR, "gene_trajectories.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: Attractor fingerprint (box plots, genes normalised to [0,1])
# ---------------------------------------------------------------------------

def normalise_gene(values, gene_name):
    lo, hi = GENE_BOUNDS[gene_name]
    return (values - lo) / (hi - lo)


def fig_attractor_fingerprint(dfs):
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        "Attractor Fingerprint: Final 10 Generations (Genes Normalized to [0,1])",
        fontsize=13, fontweight="bold"
    )

    exp_configs = [
        ("exp3", "Exp3", "blue"),
        ("exp5", "Exp5", "orange"),
        ("exp6", "Exp6", "green"),
    ]

    n_genes = 11
    n_exps  = len(exp_configs)
    group_w = 0.8
    box_w   = group_w / n_exps

    positions_all = []
    data_all      = []
    colors_all    = []

    for gi, gene_name in enumerate(GENE_NAMES_11):
        col = f"gene_{gi}"
        group_centre = gi + 1.0
        offsets = np.linspace(-group_w / 2 + box_w / 2, group_w / 2 - box_w / 2, n_exps)

        for ei, (key, label, color) in enumerate(exp_configs):
            d = dfs[key]
            if d is None:
                continue
            mask = final_gen_valid(d)
            if mask is None or not mask.any():
                continue
            raw = d[col][mask]
            norm = normalise_gene(raw, gene_name)
            pos = group_centre + offsets[ei]
            positions_all.append(pos)
            data_all.append(norm)
            colors_all.append(color)

    bp = ax.boxplot(
        data_all,
        positions=positions_all,
        widths=box_w * 0.85,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        flierprops=dict(marker=".", markersize=2, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Reference lines for normalised bounds
    ax.axhline(1.0, color="red",  linestyle="--", linewidth=0.8, alpha=0.6, label="upper bound")
    ax.axhline(0.0, color="blue", linestyle="--", linewidth=0.8, alpha=0.6, label="lower bound")

    ax.set_xticks(np.arange(1, n_genes + 1))
    ax.set_xticklabels(GENE_NAMES_11, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("normalised gene value (0 = lower bound, 1 = upper bound)")
    ax.set_ylim(-0.15, 1.15)

    legend_patches = [
        mpatches.Patch(color="blue",   alpha=0.6, label="Exp3 (displacement)"),
        mpatches.Patch(color="orange", alpha=0.6, label="Exp5 (axisym disp)"),
        mpatches.Patch(color="green",  alpha=0.6, label="Exp6 (efficiency)"),
        plt.Line2D([0], [0], color="red",  linestyle="--", linewidth=1, label="upper bound"),
        plt.Line2D([0], [0], color="blue", linestyle="--", linewidth=1, label="lower bound"),
    ]
    ax.legend(handles=legend_patches, fontsize=9)

    plt.tight_layout()
    out = os.path.join(_ADIR, "attractor_fingerprint.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4: Validity rate
# ---------------------------------------------------------------------------

def fig_validity_rate(dfs):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Population Validity Rate", fontsize=13, fontweight="bold")
    ax.set_xlabel("generation")
    ax.set_ylabel("validity rate")
    ax.set_ylim(0, 1.1)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    g3, v3 = validity_per_gen(dfs["exp3"])
    if len(g3):
        ax.plot(g3, v3, color="blue",   linewidth=2, label="Exp3 (displacement)")

    g5, v5 = validity_per_gen(dfs["exp5"])
    if len(g5):
        ax.plot(g5, v5, color="orange", linewidth=2, label="Exp5 (axisym)")

    g6, v6 = validity_per_gen(dfs["exp6"])
    if len(g6):
        ax.plot(g6, v6, color="green",  linewidth=2, label="Exp6 (efficiency)")

    # Exp2 mean across seeds
    exp2_keys = ["exp2_s42", "exp2_s137", "exp2_s999", "exp2_s2024"]
    exp2_val_curves = []
    for k in exp2_keys:
        g, v = validity_per_gen(dfs[k])
        if len(g):
            exp2_val_curves.append((g, v))
    if exp2_val_curves:
        max_gen = max(c[0].max() for c in exp2_val_curves)
        common_g = np.arange(0, max_gen + 1)
        stacked = [np.interp(common_g, g, v) for g, v in exp2_val_curves]
        ax.plot(common_g, np.nanmean(stacked, axis=0),
                color="gray", linewidth=2, linestyle="-.", label="Exp2 mean (4 seeds)")

    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(_ADIR, "validity_rate.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Summary table (stdout)
# ---------------------------------------------------------------------------

def print_summary(dfs):
    print("\nSummary Table")
    print("=" * 100)
    header = f"{'Experiment':<15} {'Best-fitness':>13} {'Best-disp':>10} {'Final-sigma':>12} {'Converged-gen':>14}"
    for gi, gn in enumerate(GENE_NAMES_11):
        header += f" {gn:>10}"
    print(header)
    print("-" * len(header))

    def converged_gen(d, col, threshold=0.95):
        """Generation at which fitness reaches 95% of its final best."""
        if d is None:
            return -1
        gens = np.unique(d["generation"]).astype(int)
        bests = []
        for g in gens:
            mask = (d["generation"] == g) & (d["valid"] == 1)
            bests.append(np.nanmax(d[col][mask]) if mask.any() else np.nan)
        bests = np.array(bests)
        final_best = np.nanmax(bests)
        thresh = threshold * final_best
        for i, b in enumerate(bests):
            if not np.isnan(b) and b >= thresh:
                return int(gens[i])
        return int(gens[-1])

    def final_gene_means(d):
        if d is None:
            return [np.nan] * 11
        mask = final_gen_valid(d)
        if mask is None or not mask.any():
            return [np.nan] * 11
        means = []
        for gi in range(11):
            col = f"gene_{gi}"
            if col in d:
                means.append(np.nanmean(d[col][mask]))
            else:
                means.append(np.nan)
        return means

    def final_sigma(d):
        if d is None:
            return np.nan
        mask = d["generation"] == d["generation"].max()
        return np.nanmean(d["sigma"][mask]) if mask.any() else np.nan

    experiments = {
        "Exp1":       ("exp1",    "efficiency"),
        "Exp2-s42":   ("exp2_s42",  "efficiency"),
        "Exp2-s137":  ("exp2_s137", "efficiency"),
        "Exp2-s999":  ("exp2_s999", "efficiency"),
        "Exp2-s2024": ("exp2_s2024","efficiency"),
        "Exp3":       ("exp3",    "displacement"),
        "Exp5":       ("exp5",    "displacement"),
        "Exp6":       ("exp6",    "efficiency"),
    }

    for exp_label, (key, fit_col) in experiments.items():
        d = dfs.get(key)
        if d is None:
            print(f"{exp_label:<15}  (data missing)")
            continue

        mask_valid = d["valid"] == 1
        best_fit  = np.nanmax(d[fit_col][mask_valid]) if mask_valid.any() else np.nan
        best_disp = np.nanmax(d["displacement"][mask_valid]) if mask_valid.any() else np.nan
        fsig      = final_sigma(d)
        cgen      = converged_gen(d, fit_col)
        gene_means = final_gene_means(d)

        row = f"{exp_label:<15} {best_fit:>13.4f} {best_disp:>10.4f} {fsig:>12.5f} {cgen:>14}"
        for gm in gene_means:
            row += f" {gm:>10.4f}" if not np.isnan(gm) else f" {'N/A':>10}"
        print(row)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(_ADIR, exist_ok=True)
    print("Loading experiment CSVs...")
    dfs = load_all()

    print("\nGenerating figures...")
    fig_fitness_comparison(dfs)
    fig_gene_trajectories(dfs)
    fig_attractor_fingerprint(dfs)
    fig_validity_rate(dfs)

    print_summary(dfs)
    print(f"All figures saved to {_ADIR}")


if __name__ == "__main__":
    main()
