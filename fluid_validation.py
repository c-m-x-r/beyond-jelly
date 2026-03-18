"""
fluid_validation.py - Water physics quality validation with vector field visualization.

Tests:
  1. Vorticity video      : Aurelia swimming — left=particles, right=vorticity+velocity arrows
  2. Grid KE comparison   : Energy dissipation over 3 actuation cycles at visc=0 vs visc=0.05
  3. Velocity profile     : Horizontal slice through the wake at 3 time snapshots

Usage:
    uv run python fluid_validation.py           # all three tests
    uv run python fluid_validation.py --video   # vorticity video only
    uv run python fluid_validation.py --ke      # KE comparison plot only
    uv run python fluid_validation.py --profile # velocity profile only

Outputs → output/
    fluid_validation.mp4        vorticity + particle video (Aurelia, visc=0)
    wake_ke_comparison.png      grid KE over time: visc=0, 0.01, 0.05
    velocity_profile.png        horizontal velocity slice at 3 actuation snapshots
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio.v3 as iio
import taichi as ti

import mpm_sim as sim
from make_jelly import fill_tank, AURELIA_GENOME

OUTPUT_DIR = "output"
N_GRID = sim.n_grid
DX = sim.dx
DT = sim.dt
ACTUATION_STEPS = int(1.0 / sim.actuation_freq / DT)  # steps per actuation cycle


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_aurelia_all_instances():
    """Load Aurelia genome into every instance so the GPU stays busy (same morphology)."""
    pos, mat, fiber, _ = fill_tank(AURELIA_GENOME, sim.n_particles)
    for i in range(sim.n_instances):
        sim.load_particles(i, pos, mat, fiber)
    sim.sim_time[None] = 0.0
    ti.sync()


def substep_with_visc(visc: float):
    """Single physics step, optionally applying Laplacian diffusion at the given visc."""
    sim._substep_p2g()
    sim._substep_grid_ops()
    if visc > 0.0:
        sim._substep_viscosity(visc)
        sim._substep_copy_lap()
    sim._substep_g2p()


def grid_ke_instance0() -> float:
    """Grid kinetic energy for instance 0 (numpy side, forces GPU sync)."""
    gv = sim.grid_v.to_numpy()[0]   # (N, N, 2)
    gm = sim.grid_m.to_numpy()[0]   # (N, N)
    mask = gm > 0
    ke = 0.5 * np.sum(gm[mask] * np.sum(gv[mask] ** 2, axis=1))
    return float(ke)


def compute_vorticity(gv: np.ndarray) -> np.ndarray:
    """
    Compute ω_z = ∂vy/∂x − ∂vx/∂y from grid velocity array (N,N,2).
    Uses central differences; boundary wrapped (artifacts are masked out by grid_m).
    """
    vy = gv[:, :, 1]
    vx = gv[:, :, 0]
    dvydx = (np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)) / (2.0 * DX)
    dvxdy = (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)) / (2.0 * DX)
    return dvydx - dvxdy


def render_particles_np(x_np: np.ndarray, mat_np: np.ndarray, size: int = 512) -> np.ndarray:
    """
    Fast numpy particle renderer for a single instance.
    Water → blue density histogram.  Structure (jelly/muscle/payload) → solid splat.
    Returns float32 RGB (size, size, 3) in [0,1].
    """
    img = np.zeros((size, size, 3), dtype=np.float32)

    # Water: 2D density histogram mapped to a dark blue glow
    wm = mat_np == 0
    if np.any(wm):
        wx = x_np[wm, 0]
        wy = x_np[wm, 1]
        hist, _, _ = np.histogram2d(wx, 1.0 - wy, bins=size, range=[[0, 1], [0, 1]])
        density = np.clip(hist.T / 10.0, 0, 1)
        img[:, :, 0] += density * 0.04
        img[:, :, 1] += density * 0.12
        img[:, :, 2] += density * 0.55

    # Structural particles: small disc splat (radius 2 px)
    MAT_COLORS = {
        1: np.array([0.30, 0.65, 1.00]),   # jelly:   cyan-blue
        3: np.array([0.50, 1.00, 0.45]),   # muscle:  green
        2: np.array([1.00, 0.35, 0.10]),   # payload: orange-red
    }
    for mat_id, color in MAT_COLORS.items():
        mm = mat_np == mat_id
        if not np.any(mm):
            continue
        px = np.clip((x_np[mm, 0] * size).astype(np.int32), 0, size - 1)
        py = np.clip(((1.0 - x_np[mm, 1]) * size).astype(np.int32), 0, size - 1)
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr * dr + dc * dc <= 5:
                    r = np.clip(py + dr, 0, size - 1)
                    c = np.clip(px + dc, 0, size - 1)
                    img[r, c] = color

    return np.clip(img, 0.0, 1.0)


def render_vorticity_np(gv: np.ndarray, gm: np.ndarray, size: int = 512,
                        vort_clim: float = 40.0, arrow_skip: int = 10) -> np.ndarray:
    """
    Render vorticity heatmap (RdBu_r) + downsampled velocity arrows.
    Returns float32 RGB (size, size, 3) in [0,1].
    """
    omega = compute_vorticity(gv)
    omega[gm < 1e-8] = 0.0  # blank empty cells

    # Vorticity → colormap → RGB image
    norm = mcolors.Normalize(vmin=-vort_clim, vmax=vort_clim)
    cmap = plt.get_cmap('RdBu_r')
    omega_img = cmap(norm(omega.T))[:, :, :3].astype(np.float32)  # (N,N,3), y-axis flipped below

    # Resize to output size using nearest-neighbour upscale
    scale = size // N_GRID
    if scale > 1:
        omega_img = np.repeat(np.repeat(omega_img, scale, axis=0), scale, axis=1)
    omega_img = omega_img[:size, :size]  # crop if needed

    # Velocity arrow overlay drawn on top via matplotlib (returned as numpy splice)
    fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
    ax.imshow(omega_img, origin='lower', extent=[0, 1, 0, 1], aspect='equal')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

    # Downsampled quiver — only in cells with mass
    si = np.arange(arrow_skip // 2, N_GRID, arrow_skip)
    xs = (si + 0.5) / N_GRID
    XX, YY = np.meshgrid(xs, xs)
    gv_T = gv.transpose(1, 0, 2)  # (j, i, 2) so indexing matches (y, x)
    VX = gv_T[np.ix_(si, si, [0])][..., 0]
    VY = gv_T[np.ix_(si, si, [1])][..., 0]
    GM = gm.T[np.ix_(si, si)]
    mask_q = GM > 1e-8
    speed_max = max(np.sqrt(VX**2 + VY**2).max(), 1e-6)
    ax.quiver(XX[mask_q], YY[mask_q],
              VX[mask_q], VY[mask_q],
              color='white', alpha=0.75,
              scale=speed_max * 14, width=0.003, headwidth=3, headlength=4)

    ax.text(0.02, 0.97, f'ω_z  (±{vort_clim:.0f})', transform=ax.transAxes,
            color='white', fontsize=7, va='top')

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return buf[:size, :size, :3].astype(np.float32) / 255.0


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — Vorticity video
# ──────────────────────────────────────────────────────────────────────────────

def run_vorticity_video(n_cycles: int = 3, substeps_per_frame: int = 200,
                        panel_size: int = 512, visc: float = 0.0):
    """
    Render Aurelia swimming.
    Left panel  : particles (water density + structure splat)
    Right panel : vorticity heatmap (RdBu) + velocity field arrows
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = f"visc{visc:.3f}".replace('.', 'p')
    out_path = os.path.join(OUTPUT_DIR, f"fluid_validation_{tag}.mp4")

    n_frames = int(n_cycles * ACTUATION_STEPS / substeps_per_frame)
    total_steps = n_frames * substeps_per_frame

    print(f"\n[Vorticity Video] visc={visc}  frames={n_frames}  steps={total_steps}")
    print(f"  Output → {out_path}")

    load_aurelia_all_instances()

    # Warmup — let the jellyfish settle for half a cycle
    for _ in range(ACTUATION_STEPS // 2):
        substep_with_visc(visc)
    sim.sim_time[None] = 0.0

    frames_rgb = []
    for f in range(n_frames):
        for _ in range(substeps_per_frame):
            substep_with_visc(visc)
        ti.sync()

        t = sim.sim_time[None]

        # Export instance 0
        x_np   = sim.x.to_numpy()[0]
        mat_np = sim.material.to_numpy()[0]
        gv_np  = sim.grid_v.to_numpy()[0]
        gm_np  = sim.grid_m.to_numpy()[0]

        left  = render_particles_np(x_np, mat_np, size=panel_size)
        right = render_vorticity_np(gv_np, gm_np, size=panel_size)

        # Composite: side by side with a thin separator
        sep = np.ones((panel_size, 2, 3), dtype=np.float32) * 0.3
        frame = np.concatenate([left, sep, right], axis=1)

        # Timestamp overlay (using matplotlib on composite is slow; burn text via numpy skip — just print)
        frames_rgb.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))

        if f % 30 == 0:
            ke = grid_ke_instance0()
            phase = (t % (1.0 / sim.actuation_freq)) / (1.0 / sim.actuation_freq)
            print(f"  frame {f:4d}/{n_frames}  t={t:.3f}s  phase={phase:.2f}  gridKE={ke:.4f}")

    iio.imwrite(out_path, frames_rgb, fps=30)
    print(f"  Saved {len(frames_rgb)} frames → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — Grid KE comparison across viscosity values
# ──────────────────────────────────────────────────────────────────────────────

def run_ke_comparison(visc_values=(0.0, 0.01, 0.05), n_cycles: int = 3,
                      record_every: int = 100):
    """
    For each viscosity value: run n_cycles of Aurelia, record grid KE every record_every steps.
    Plots all curves on the same axis with the actuation waveform as background shading.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "wake_ke_comparison.png")
    total_steps = n_cycles * ACTUATION_STEPS

    print(f"\n[KE Comparison] cycles={n_cycles}  steps={total_steps}  record_every={record_every}")

    results = {}
    for visc in visc_values:
        print(f"  Running visc={visc} ...")
        load_aurelia_all_instances()

        times, ke_vals = [], []
        for step in range(total_steps):
            substep_with_visc(visc)
            if step % record_every == 0:
                ti.sync()
                times.append(float(sim.sim_time[None]))
                ke_vals.append(grid_ke_instance0())

        results[visc] = (np.array(times), np.array(ke_vals))
        peak = np.max(ke_vals)
        mean = np.mean(ke_vals)
        print(f"    peak KE={peak:.5f}  mean KE={mean:.5f}")

    # Plot
    fig, (ax_ke, ax_wave) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                          gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Grid Kinetic Energy — Aurelia aurita — viscosity comparison', fontsize=12)

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    for idx, (visc, (t, ke)) in enumerate(results.items()):
        label = f'visc = {visc:.3f}' if visc > 0 else 'visc = 0  (baseline)'
        ax_ke.plot(t, ke, color=colors[idx % len(colors)], lw=1.5, label=label)

    ax_ke.set_ylabel('Grid kinetic energy (sim units)')
    ax_ke.legend(fontsize=9)
    ax_ke.grid(True, alpha=0.3)
    ax_ke.set_yscale('log')

    # Actuation waveform background (raised cosine)
    t_wave = np.linspace(0, n_cycles / sim.actuation_freq, 500)
    period = 1.0 / sim.actuation_freq
    phase = (t_wave % period) / period
    wave = np.where(phase < 0.2,
                    0.5 * (1 - np.cos(phase / 0.2 * np.pi)),
                    0.5 * (1 + np.cos((phase - 0.2) / 0.8 * np.pi)))
    ax_wave.fill_between(t_wave, 0, wave, alpha=0.4, color='gray')
    ax_wave.plot(t_wave, wave, color='gray', lw=1)
    ax_wave.set_ylabel('Actuation')
    ax_wave.set_xlabel('Simulation time (s)')
    ax_wave.set_ylim(0, 1.1)
    ax_wave.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — Velocity profile: horizontal slice through wake
# ──────────────────────────────────────────────────────────────────────────────

def run_velocity_profile(visc_values=(0.0, 0.05), snapshot_phases=(0.1, 0.5, 0.9)):
    """
    At 3 actuation phases (early contraction, mid-relaxation, late), extract the
    horizontal velocity component along a vertical strip at x=0.5 (bell centreline).
    Compares two viscosity values to show how diffusion smooths the velocity gradient.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "velocity_profile.png")
    period = 1.0 / sim.actuation_freq

    print(f"\n[Velocity Profile] phases={snapshot_phases}  visc_values={visc_values}")

    # Target snapshot steps within the second actuation cycle (past transient)
    snap_steps = [int((1 + ph) * ACTUATION_STEPS) for ph in snapshot_phases]
    max_step = max(snap_steps) + 1

    fig, axes = plt.subplots(1, len(snapshot_phases), figsize=(13, 4), sharey=True)
    fig.suptitle('Vertical velocity (vy) profile at x ≈ 0.5 — actuation phase snapshots', fontsize=11)

    colors_visc = {0.0: '#2196F3', 0.05: '#FF9800'}

    for visc in visc_values:
        print(f"  Running visc={visc} ...")
        load_aurelia_all_instances()
        snapshots = {}

        for step in range(max_step):
            substep_with_visc(visc)
            if step in snap_steps:
                ti.sync()
                gv_np = sim.grid_v.to_numpy()[0]   # (N, N, 2)
                gm_np = sim.grid_m.to_numpy()[0]   # (N, N)
                # Column at x=N//2 (centreline), all y
                col_x = N_GRID // 2
                vy_col = gv_np[col_x, :, 1]        # vy along y axis at x=0.5
                mass_col = gm_np[col_x, :]
                y_coords = (np.arange(N_GRID) + 0.5) / N_GRID
                snapshots[step] = (y_coords, vy_col, mass_col)

        for ax, step, ph in zip(axes, snap_steps, snapshot_phases):
            y_coords, vy_col, mass_col = snapshots[step]
            # Only plot cells that have particles
            valid = mass_col > 1e-8
            label = f'visc={visc:.3f}' if visc > 0 else 'visc=0 (baseline)'
            color = colors_visc.get(visc, 'gray')
            ax.plot(vy_col[valid], y_coords[valid], color=color, lw=1.5, label=label)
            ax.axvline(0, color='gray', lw=0.5, ls='--')
            ax.set_title(f'phase = {ph:.1f}', fontsize=9)
            ax.set_xlabel('vy (sim units/s)')

    axes[0].set_ylabel('y position')
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fluid physics validation")
    parser.add_argument('--video',   action='store_true', help='Run vorticity video only')
    parser.add_argument('--ke',      action='store_true', help='Run KE comparison only')
    parser.add_argument('--profile', action='store_true', help='Run velocity profile only')
    args = parser.parse_args()
    run_all = not (args.video or args.ke or args.profile)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outputs = []

    if run_all or args.video:
        p = run_vorticity_video(n_cycles=3, substeps_per_frame=200, panel_size=512, visc=0.0)
        outputs.append(p)

    if run_all or args.ke:
        p = run_ke_comparison(visc_values=(0.0, 0.01, 0.05), n_cycles=3, record_every=100)
        outputs.append(p)

    if run_all or args.profile:
        p = run_velocity_profile(visc_values=(0.0, 0.05), snapshot_phases=(0.1, 0.5, 0.9))
        outputs.append(p)

    print(f"\n=== Fluid validation complete ===")
    for p in outputs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
