"""
view_axisym.py — Render a jellyfish in axisymmetric MPM mode (JELLY_AXISYM=1).

The axisymmetric formulation treats x as the radial axis and y as the axis of
symmetry, applying a geometric correction factor r·dr·dθ to each MPM cell.
Uses a tall tank (128×256 grid, 160K particles, 2-unit domain) matching the
domain used during Experiment 5. Output video is 512×1024 (1:2 aspect ratio).

Usage:
    uv run python helpers/view_axisym.py --aurelia
    uv run python helpers/view_axisym.py --gen 49 --log output/cloud/3080Ti/exp5_s42/best_genomes_exp5_s42.json
    uv run python helpers/view_axisym.py --genome "[0.15,...]" --cycles 3
    uv run python helpers/view_axisym.py --aurelia --palette web
"""

import os as _os

# CRITICAL: must be set before Taichi / mpm_sim initialise (kernel compilation at import time).
_os.environ["JELLY_AXISYM"] = "1"

import sys as _sys

_os.environ.setdefault("JELLY_INSTANCES", "1")
_os.environ.setdefault("JELLY_GRID_Y",    "256")
_os.environ.setdefault("JELLY_PARTICLES", "160000")
_os.environ.setdefault("JELLY_DOMAIN_H",  "2.0")

import argparse
import json

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import imageio.v3 as iio

import mpm_sim as sim
from mpm_sim import WEB_PALETTE
from make_jelly import fill_tank, AURELIA_GENOME, DEFAULT_SPAWN

OUTPUT_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "output")

DOMAIN_HEIGHT = 2.0
RENDER_EVERY  = 500   # substeps between frames
FPS           = 30


def load_genome(args):
    if args.aurelia:
        print("Genome: Aurelia aurita reference  [AXISYM mode]")
        return AURELIA_GENOME, "aurelia"

    if args.log is not None:
        # Load from an explicitly specified JSON file.
        log_path = args.log
        if not _os.path.isabs(log_path):
            log_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), log_path)
        with open(log_path) as f:
            data = json.load(f)
        if args.gen is not None:
            entry = next((e for e in data if e["generation"] == args.gen), None)
            if entry is None:
                entry = max(data, key=lambda e: e["generation"])
                print(f"Generation {args.gen} not found in {log_path}; using gen {entry['generation']}")
        else:
            entry = max(data, key=lambda e: e["generation"])
        genome = np.array(entry["genome"])
        fit_str = f"{entry.get('fitness', '?'):.4f}" if isinstance(entry.get('fitness'), float) else str(entry.get('fitness', '?'))
        print(f"Genome: gen {entry['generation']}, fitness {fit_str}  [AXISYM mode]")
        return genome, f"gen{entry['generation']}"

    if args.gen is not None:
        # Fall back to the default output directory search (no --log specified).
        path = _os.path.join(OUTPUT_DIR, "best_genomes.json")
        if not _os.path.exists(path):
            for d in sorted(_os.listdir(OUTPUT_DIR)):
                candidate = _os.path.join(OUTPUT_DIR, d, f"best_genomes_{d}.json")
                if _os.path.exists(candidate):
                    path = candidate
                    break
        with open(path) as f:
            data = json.load(f)
        entry = next((e for e in data if e["generation"] == args.gen), None)
        if entry is None:
            entry = max(data, key=lambda e: e["generation"])
            print(f"Generation {args.gen} not found; using gen {entry['generation']}")
        genome = np.array(entry["genome"])
        print(f"Genome: gen {entry['generation']}, fitness {entry.get('fitness', '?'):.4f}  [AXISYM mode]")
        return genome, f"gen{entry['generation']}"

    if args.genome:
        genome = np.array(json.loads(args.genome))
        print("Genome: custom  [AXISYM mode]")
        return genome, "custom"

    raise ValueError("Specify --aurelia, --gen N, --log PATH, or --genome [...]")


def render_frame(palette):
    if palette == "web":
        sim.clear_frame_buffer_white()
        for mat_id, r, g, b in WEB_PALETTE:
            sim.render_flat_pass(sim.video_res, 1, 3.0, mat_id, r, g, b)
    else:
        sim.clear_frame_buffer()
        sim.render_frame_abyss(sim.video_res, 1, 3.0)
        sim.tone_map_and_encode()
    raw = sim.frame_buffer.to_numpy()   # (1024, 1024, 3)
    raw = np.clip(raw, 0.0, 1.0)
    # Tall-tank crop: left 512 columns → (1024, 512, 3) — 1:2 aspect for 2-unit domain
    cropped = raw[:, :512, :]           # (1024, 512, 3)
    return (cropped * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Axisymmetric jellyfish viewer (JELLY_AXISYM=1, 128×256 tall grid, 160K particles)"
    )
    parser.add_argument("--aurelia", action="store_true", help="Use Aurelia aurita reference genome")
    parser.add_argument("--gen",  type=int,  default=None, help="Use best genome from generation N")
    parser.add_argument("--log",  type=str,  default=None, help="Path to best_genomes JSON file")
    parser.add_argument("--genome", type=str, default=None, help="JSON genome array")
    parser.add_argument("--cycles", type=int, default=3, help="Number of actuation cycles to simulate (default 3)")
    parser.add_argument("--palette", choices=["abyss", "web"], default="abyss")
    args = parser.parse_args()

    genome, label = load_genome(args)

    print(f"Building particle set (axisym, domain 1×{DOMAIN_HEIGHT})...")
    pos, mat, fiber, stats = fill_tank(
        genome,
        max_particles=sim.n_particles,
        grid_res=128,
        spawn_offset=DEFAULT_SPAWN,
        domain_height=DOMAIN_HEIGHT,
    )
    if stats['self_intersecting']:
        print("WARNING: morphology is self-intersecting")
    print(f"  {stats['n_robot']} robot  |  {stats['n_water']} water  |  {stats['muscle_count']} muscle  |  {sim.n_particles} total slots")

    sim.sim_time[None] = 0.0
    sim.load_particles(0, pos, mat, fiber)

    # Apply timing genes if present (genes 9 and 10)
    if len(genome) > 9:
        sim.instance_act_contraction[0] = float(np.clip(genome[9],  0.05, 0.60))
    if len(genome) > 10:
        sim.instance_freq[0] = float(np.clip(genome[10], 0.5, 2.0))

    steps_per_cycle = int(round(1.0 / (sim.actuation_freq * sim.dt)))
    total_steps     = steps_per_cycle * args.cycles
    print(f"Simulating {args.cycles} cycles × {steps_per_cycle} steps = {total_steps} total steps  [AXISYM mode]...")

    frames = []
    for step in range(total_steps):
        sim.substep()
        if step % RENDER_EVERY == 0:
            frames.append(render_frame(args.palette))
            if step % (RENDER_EVERY * 10) == 0:
                pct = 100 * step / total_steps
                print(f"  {pct:.0f}%  (step {step}/{total_steps})")

    # Final frame
    frames.append(render_frame(args.palette))

    # Write video
    out_path = _os.path.join(OUTPUT_DIR, f"view_axisym_{label}.mp4")
    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    iio.imwrite(out_path, frames, fps=FPS, codec="libx264", quality=8)
    print(f"\nSaved → {out_path}  ({len(frames)} frames, {sim.video_res // 2}×{sim.video_res} px)")

    # Final CoM stats
    payload_stats = sim.get_payload_stats()
    print(f"Final payload CoM y = {payload_stats[0,0]:.4f}  (started at ~{DEFAULT_SPAWN[1]:.2f})  [AXISYM mode]")


if __name__ == "__main__":
    main()
