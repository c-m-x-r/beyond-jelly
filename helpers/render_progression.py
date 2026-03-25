"""
render_progression.py — Render representative generations from Experiments 3, 5, and 6
in the tall tank (128×256, 2-unit domain) by calling tall_tank.py as a subprocess.

For each experiment, five target generations are selected (default: 1, 12, 25, 37, 49).
The nearest available generation is used when an exact match is not found.

Output: output/progression/<exp_name>_gen<NN>.mp4

Usage:
    uv run python helpers/render_progression.py --dry-run
    uv run python helpers/render_progression.py --exp exp3 --cycles 5
    uv run python helpers/render_progression.py --gens 1 25 49 --palette web
"""

import argparse
import json
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {
        "name": "exp3",
        "json": "output/cloud/3080Ti/exp3_s42/best_genomes_exp3_s42.json",
        "axisym": False,
    },
    {
        "name": "exp5",
        "json": "output/cloud/3080Ti/exp5_s42/best_genomes_exp5_s42.json",
        "axisym": True,
    },
    {
        "name": "exp6",
        "json": "output/cloud/3080Ti/exp6_s137/best_genomes_exp6_s137.json",
        "axisym": False,
    },
]

DEFAULT_TARGET_GENS = [1, 12, 25, 37, 49]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TALL_TANK_PATH = os.path.join(REPO_ROOT, "helpers", "tall_tank.py")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output", "progression")


def load_json(path):
    """Load best_genomes JSON from an absolute or repo-relative path."""
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    if not os.path.exists(path):
        return None, path
    with open(path) as f:
        return json.load(f), path


def find_nearest_entry(data, target_gen):
    """Return the entry whose generation is closest to target_gen."""
    return min(data, key=lambda e: abs(e["generation"] - target_gen))


def build_summary_table(plan):
    """Return a formatted summary table string."""
    col_widths = [8, 5, 8, 10]
    header = (
        f"{'Exp':<{col_widths[0]}}  {'Gen':>{col_widths[1]}}  "
        f"{'Fitness':>{col_widths[2]}}  {'Disp':>{col_widths[3]}}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for row in plan:
        fitness_str = f"{row['fitness']:.4f}" if row["fitness"] is not None else "n/a"
        disp_str = f"{row['displacement']:.4f}" if row["displacement"] is not None else "n/a"
        lines.append(
            f"{row['exp']:<{col_widths[0]}}  {row['gen']:>{col_widths[1]}}  "
            f"{fitness_str:>{col_widths[2]}}  {disp_str:>{col_widths[3]}}"
        )
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render evolutionary progression for Experiments 3, 5, and 6 in the tall tank"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        metavar="NAME",
        help="Only process one experiment (e.g. exp3, exp5, exp6)",
    )
    parser.add_argument(
        "--gens",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Override target generation list (e.g. --gens 1 25 49)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="Actuation cycles to simulate per render (default: 3)",
    )
    parser.add_argument(
        "--palette",
        choices=["abyss", "web"],
        default="abyss",
        help="Colour palette for rendering (default: abyss)",
    )
    args = parser.parse_args()

    target_gens = args.gens if args.gens is not None else DEFAULT_TARGET_GENS

    # Filter experiments if --exp was supplied
    experiments = EXPERIMENTS
    if args.exp is not None:
        experiments = [e for e in EXPERIMENTS if e["name"] == args.exp]
        if not experiments:
            print(f"ERROR: unknown experiment '{args.exp}'. "
                  f"Available: {', '.join(e['name'] for e in EXPERIMENTS)}")
            sys.exit(1)

    # Build the render plan: load JSON for each experiment, resolve nearest gens
    plan = []   # list of dicts describing each render job
    skipped_experiments = []

    for exp in experiments:
        data, resolved_path = load_json(exp["json"])
        if data is None:
            print(f"WARNING: JSON not found for {exp['name']}: {resolved_path} — skipping")
            skipped_experiments.append(exp["name"])
            continue

        for tgen in target_gens:
            entry = find_nearest_entry(data, tgen)
            actual_gen = entry["generation"]
            genome = entry.get("genome") or entry.get("best_genome")
            if genome is None:
                print(f"WARNING: no genome field in entry gen={actual_gen} for {exp['name']} — skipping")
                continue

            plan.append({
                "exp": exp["name"],
                "axisym": exp["axisym"],
                "gen": actual_gen,
                "requested_gen": tgen,
                "genome": genome,
                "fitness": entry.get("fitness"),
                "displacement": entry.get("displacement"),
                "out_path": os.path.join(OUTPUT_DIR, f"{exp['name']}_gen{actual_gen:02d}.mp4"),
            })

    if not plan:
        print("No render jobs to execute.")
        sys.exit(0)

    # Print summary table
    print(f"\nRender plan — {len(plan)} job(s):")
    print(build_summary_table(plan))
    if skipped_experiments:
        print(f"Skipped experiments (missing JSON): {', '.join(skipped_experiments)}")
    print()

    # Create output directory (unless dry-run where it is harmless to create)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Execute render jobs
    for i, job in enumerate(plan, start=1):
        out_path = job["out_path"]
        genome_str = json.dumps(job["genome"])

        # Subprocess command
        cmd = [
            sys.executable,
            TALL_TANK_PATH,
            "--genome", genome_str,
            "--cycles", str(args.cycles),
            "--output", out_path,
            "--palette", args.palette,
        ]

        # Environment — tall-tank env vars + optional axisym flag
        env = os.environ.copy()
        env["JELLY_INSTANCES"] = "1"
        env["JELLY_GRID_Y"] = "256"
        env["JELLY_PARTICLES"] = "160000"
        env["JELLY_DOMAIN_H"] = "2.0"
        if job["axisym"]:
            env["JELLY_AXISYM"] = "1"
        else:
            # Explicitly unset so a prior export does not bleed through
            env.pop("JELLY_AXISYM", None)

        gen_note = (
            f"(requested gen {job['requested_gen']})"
            if job["requested_gen"] != job["gen"]
            else ""
        )
        print(
            f"[{i}/{len(plan)}] {job['exp']}  gen {job['gen']:02d} {gen_note}"
            f"  axisym={job['axisym']}  → {os.path.relpath(out_path, REPO_ROOT)}"
        )
        print(f"  cmd: {' '.join(cmd)}")

        if args.dry_run:
            print("  [dry-run — skipping]\n")
            continue

        print()
        try:
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"ERROR: tall_tank.py exited with code {exc.returncode} for job {i}. "
                  f"Continuing with remaining jobs.\n")
            continue
        print()

    if args.dry_run:
        print("Dry-run complete. No files written.")
    else:
        print(f"Done. Output in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
