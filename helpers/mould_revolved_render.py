"""
mould_revolved_render.py — Side-by-side renders of two-piece revolved mould options

The revolved mould has two pieces. Both options share the same outer mould body.
The difference is how the second piece (inner core) self-locates at the bell tip.

Coordinate system (r-z half-plane, revolved around z-axis):
  r = 0            bell axis (left edge of plot)
  z = 0            base plate bottom (sits on table)
  z = base_mm      base of payload boss
  z = z_collar     bell base (where arms begin)
  z = z_tip        bell tips (top of outer mould, open end)

──────────────────────────────────────────────────────────────────────────────
Option 1 — Inner core with normal-direction flange  (self-locating)

  The inner core fills the subumbrellar void.  At the bell tip, a short
  flange extends from inner_1_tip to outer_1_tip along the local spine
  normal.  This flange rests exactly on the outer mould's rim — geometry-
  derived, no measurement needed, works for any valid genome.

  Contact geometry:
    flange outer edge  == outer_bell_tip  == outer mould rim inner edge
    flange surface ⊥ spine tangent at t=1  (conical seat when revolved)

──────────────────────────────────────────────────────────────────────────────
Option 2 — Flat cap with wide outward collar  (simpler)

  The outer mould is extended with a flat collar at the tip level.
  Collar width = outer_mould_tip_r + collar_extra_mm.  A simple flat cap
  (ring disc) rests on this collar under gravity.  No self-registration —
  just rests on the ledge.  Easy to print and use, less precise.

  Contact geometry:
    cap underside (flat)  on  collar top face (flat)
    inner radius of cap hole  = inner_1_tip_r  (spans bell opening)

──────────────────────────────────────────────────────────────────────────────
Usage:
  uv run python helpers/mould_revolved_render.py --aurelia
  uv run python helpers/mould_revolved_render.py --diameter 120 --scale 2.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

from make_jelly import (AURELIA_GENOME, PAYLOAD_WIDTH, PAYLOAD_HEIGHT,
                        cubic_bezier, get_normals_2d, random_genome)
from make_cad import _scale_factor


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _bell_curves(genome, n_pts=80, t_scale=1.0):
    start_p = np.array([PAYLOAD_WIDTH / 2.0, 0.0])
    cp1 = start_p + np.array([abs(genome[0]), genome[1]])
    cp2 = start_p + np.array([abs(genome[2]), genome[3]])
    end_p = start_p + np.array([abs(genome[4]), genome[5]])
    t_base = abs(genome[6]) * t_scale
    t_mid  = abs(genome[7]) * t_scale
    t_tip  = abs(genome[8]) * t_scale
    ts     = np.linspace(0, 1, n_pts)
    spine  = np.array([cubic_bezier(start_p, cp1, cp2, end_p, t) for t in ts])
    normals = get_normals_2d(spine)
    half_t = np.interp(ts, [0, 0.5, 1], [t_base, t_mid, t_tip])[:, None] / 2.0
    return spine + normals * half_t, spine - normals * half_t


def _rz(curve, s, z_collar):
    """
    Convert sim-local (x, y) curve to mould (r, z) half-plane.
      r = x_sim * s          (radial, away from axis)
      z = z_collar − y_sim*s  (axial, tips go UP after flip)
    """
    r = np.clip(curve[:, 0] * s, 0.0, None)
    z = z_collar - curve[:, 1] * s
    return r, z


# ── Shared geometry computation ────────────────────────────────────────────────

def _geom(genome, diameter_mm, mould_scale, base_mm, collar_extra_mm, n_pts=80):
    """
    Return all r-z arrays needed for both option renders.
    """
    s         = _scale_factor(genome, diameter_mm)
    payload_r = PAYLOAD_WIDTH  / 2.0 * s
    payload_h = PAYLOAD_HEIGHT * s
    z_collar  = base_mm + payload_h          # z where bell wall starts
    z_base    = 0.0                          # bottom of base plate

    outer_1, inner_1 = _bell_curves(genome, n_pts=n_pts, t_scale=1.0)
    outer_m, _       = _bell_curves(genome, n_pts=n_pts, t_scale=mould_scale)

    o1_r, o1_z = _rz(outer_1, s, z_collar)
    i1_r, i1_z = _rz(inner_1, s, z_collar)
    om_r, om_z = _rz(outer_m, s, z_collar)

    # Bell tip points
    tip_o1 = (o1_r[-1], o1_z[-1])           # outer_1 tip: inner edge of outer mould rim
    tip_i1 = (i1_r[-1], i1_z[-1])           # inner_1 tip: outer edge of inner core
    tip_om = (om_r[-1], om_z[-1])           # outer_m tip: outer edge of outer mould

    # Option 2 collar outer radius
    collar_outer_r = om_r[-1] + collar_extra_mm

    return dict(
        s=s, payload_r=payload_r, payload_h=payload_h,
        z_collar=z_collar, z_base=z_base, base_mm=base_mm,
        o1_r=o1_r, o1_z=o1_z,
        i1_r=i1_r, i1_z=i1_z,
        om_r=om_r, om_z=om_z,
        tip_o1=tip_o1, tip_i1=tip_i1, tip_om=tip_om,
        collar_outer_r=collar_outer_r,
    )


# ── Shared: draw outer mould body ──────────────────────────────────────────────

def _draw_outer_mould(ax, g, cap_z=None, mould_color='#7a8fa6'):
    """
    Draw the outer mould piece in r-z half-plane:
      - base plate + payload boss  (solid rectangles)
      - outer bell wall            (region between outer_m and outer_1)

    cap_z: if given, draw the collar rim outline up to cap_z (Option 2).
    """
    o1_r, o1_z = g['o1_r'], g['o1_z']
    om_r, om_z = g['om_r'], g['om_z']
    pr, ph     = g['payload_r'], g['payload_h']
    z_collar   = g['z_collar']
    z_base     = g['z_base']
    base_mm    = g['base_mm']

    # Base plate: r=[0, om_r.max()+margin], z=[z_base, base_mm]
    r_max = om_r.max() + 3
    ax.fill([0, r_max, r_max, 0],
            [z_base, z_base, base_mm, base_mm],
            color=mould_color, alpha=0.75, zorder=2)

    # Payload boss: r=[0, payload_r], z=[base_mm, z_collar]
    ax.fill([0, pr, pr, 0],
            [base_mm, base_mm, z_collar, z_collar],
            color=mould_color, alpha=0.75, zorder=2)

    # Outer mould wall: polygon between outer_m and outer_1 curves
    wall_r = np.concatenate([om_r, o1_r[::-1]])
    wall_z = np.concatenate([om_z, o1_z[::-1]])
    ax.fill(wall_r, wall_z, color=mould_color, alpha=0.75, zorder=2,
            label='Outer mould (rigid)')

    # Rim highlight at bell tip (top edge of outer mould wall)
    ax.plot([g['tip_o1'][0], g['tip_om'][0]],
            [g['tip_o1'][1], g['tip_om'][1]],
            color='#ffffff', lw=2.0, zorder=5, alpha=0.7)


def _draw_silicone_cavity(ax, g, cavity_color='#44cc88'):
    """Bell wall cavity (between outer_1 and inner_1) — where silicone goes."""
    o1_r, o1_z = g['o1_r'], g['o1_z']
    i1_r, i1_z = g['i1_r'], g['i1_z']

    cav_r = np.concatenate([o1_r, i1_r[::-1]])
    cav_z = np.concatenate([o1_z, i1_z[::-1]])
    ax.fill(cav_r, cav_z, color=cavity_color, alpha=0.75, zorder=3,
            label='Silicone cavity')


def _draw_payload_boss(ax, g, boss_color='#ff8844'):
    """Payload slot boundary (dashed outline on the boss)."""
    pr, ph   = g['payload_r'], g['payload_h']
    base_mm  = g['base_mm']
    z_collar = g['z_collar']
    ax.plot([0, pr, pr, 0, 0],
            [base_mm, base_mm, z_collar, z_collar, base_mm],
            color=boss_color, lw=1.2, ls='--', zorder=5, alpha=0.7,
            label='Payload slot (insert after curing)')


def _style_ax(ax, title, g):
    bg = '#12121f'
    ax.set_facecolor(bg)
    ax.set_aspect('equal')
    ax.axvline(0, color='#555', lw=0.8, ls=':', alpha=0.6, label='Axis of revolution')
    ax.set_xlabel('r  (mm)', color='#aaa', fontsize=8)
    ax.set_ylabel('z  (mm)', color='#aaa', fontsize=8)
    ax.tick_params(colors='#666')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')
    ax.grid(True, alpha=0.15, color='#555')
    ax.set_title(title, color='#ddd', fontsize=10, pad=6)
    # Axis label
    ax.text(-1.5, g['z_collar'] + (g['tip_o1'][1] - g['z_collar']) / 2,
            'axis', color='#555', fontsize=7, ha='right', va='center',
            rotation=90)


def _set_limits(ax, g, margin=5):
    r_max = g['collar_outer_r'] + margin
    z_max = max(g['tip_om'][1], g['tip_o1'][1]) + margin + 10
    ax.set_xlim(-3, r_max)
    ax.set_ylim(g['z_base'] - margin, z_max)


# ── Option 1: inner core with normal-direction flange ─────────────────────────

def draw_option1(ax, g):
    """
    Option 1 — Inner core with normal-direction flange.

    The flange is the segment from inner_1_tip → outer_1_tip.
    When revolved, this becomes a conical ring seat.
    The core body fills the subumbrellar void from the axis to inner_1.
    """
    _draw_outer_mould(ax, g)
    _draw_silicone_cavity(ax, g)
    _draw_payload_boss(ax, g)

    i1_r, i1_z = g['i1_r'], g['i1_z']
    z_collar    = g['z_collar']
    tip_i1      = g['tip_i1']
    tip_o1      = g['tip_o1']

    # ── Inner core body ──
    # Polygon: axis at collar → inner_1 base → inner_1 (base→tip) → axis at tip → close
    core_r = np.concatenate([[0], i1_r, [0]])
    core_z = np.concatenate([[z_collar], i1_z, [i1_z[-1]]])
    ax.fill(core_r, core_z, color='#5588cc', alpha=0.75, zorder=4,
            label='Inner core (rigid)')

    # ── Flange: inner_1_tip → outer_1_tip (along spine normal at t=1) ──
    flange_r = [tip_i1[0], tip_o1[0]]
    flange_z = [tip_i1[1], tip_o1[1]]
    ax.fill(
        [tip_i1[0], tip_o1[0], tip_o1[0], tip_i1[0]],
        [tip_i1[1], tip_o1[1], tip_o1[1] + 1.5, tip_i1[1] + 1.5],
        color='#5588cc', alpha=0.85, zorder=5,
    )
    ax.plot(flange_r, flange_z, color='#ff4444', lw=2.5, zorder=6,
            label='Flange contact (self-locating)')

    # Contact annotation
    mid_r = (tip_i1[0] + tip_o1[0]) / 2
    mid_z = (tip_i1[1] + tip_o1[1]) / 2
    ax.annotate(
        'flange contact\n(along tip normal)',
        xy=(mid_r, mid_z), xytext=(mid_r + 15, mid_z + 8),
        color='#ff8888', fontsize=7,
        arrowprops=dict(arrowstyle='->', color='#ff8888', lw=0.8),
        ha='left',
    )
    ax.annotate(
        'inner core\n(fills void)',
        xy=(i1_r[len(i1_r)//2] / 2, i1_z[len(i1_z)//2]),
        xytext=(i1_r[len(i1_r)//2] / 2 + 12, i1_z[len(i1_z)//2] - 5),
        color='#88aaee', fontsize=7,
        arrowprops=dict(arrowstyle='->', color='#88aaee', lw=0.8),
        ha='left',
    )

    _style_ax(ax, 'Option 1 — Inner core with normal-direction flange', g)
    _set_limits(ax, g)
    ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#444',
              labelcolor='#ccc', loc='upper right')


# ── Option 2: flat cap with wide collar ───────────────────────────────────────

def draw_option2(ax, g, cap_thick_mm=6.0):
    """
    Option 2 — Flat cap resting on wide outer mould collar.

    The outer mould has a flat collar extending outward at the bell tip level.
    A simple flat ring cap rests on the collar under gravity.
    The cap inner hole spans the bell opening (inner_1_tip_r).
    """
    # Outer mould with collar extension
    _draw_outer_mould(ax, g)

    om_r, om_z   = g['om_r'], g['om_z']
    tip_o1       = g['tip_o1']
    tip_om       = g['tip_om']
    tip_i1       = g['tip_i1']
    collar_r     = g['collar_outer_r']
    z_tip        = tip_o1[1]                # Z level of inner mould rim at tip
    z_tip_om     = tip_om[1]               # Z level of outer mould outer wall at tip

    # ── Collar extension on outer mould ──────────────────────────────────────
    # Flat horizontal collar from outer_m tip outward to collar_outer_r
    # Sits at z = z_tip (level of the outer_1 tip = mould rim inner edge)
    collar_z = z_tip  # collar sits at the outer_1 tip level (inside the mould rim)
    ax.fill(
        [tip_om[0], collar_r, collar_r, tip_om[0]],
        [collar_z - cap_thick_mm * 0.4, collar_z - cap_thick_mm * 0.4,
         collar_z, collar_z],
        color='#7a8fa6', alpha=0.85, zorder=4,
        label='_nolegend_',
    )
    # Collar top face (contact surface)
    ax.plot([tip_om[0], collar_r], [collar_z, collar_z],
            color='#ff4444', lw=2.5, zorder=6,
            label='Collar contact (flat)')

    _draw_silicone_cavity(ax, g)
    _draw_payload_boss(ax, g)

    # ── Flat cap ──────────────────────────────────────────────────────────────
    # A ring: inner_r = tip_i1[0], outer_r = collar_r + overhang
    # Sits with its underside at collar_z
    cap_inner_r = tip_i1[0]
    cap_outer_r = collar_r + 8.0           # slight overhang past collar for grip

    cap_pts_r = [cap_inner_r, cap_outer_r, cap_outer_r, cap_inner_r]
    cap_pts_z = [collar_z, collar_z, collar_z + cap_thick_mm, collar_z + cap_thick_mm]
    ax.fill(cap_pts_r, cap_pts_z, color='#5588cc', alpha=0.75, zorder=5,
            label='Cap (rigid, rests by gravity)')

    # Cap annotation
    ax.annotate(
        'flat cap\n(gravity + optional clip)',
        xy=(cap_inner_r + (cap_outer_r - cap_inner_r) / 2, collar_z + cap_thick_mm),
        xytext=(cap_outer_r + 5, collar_z + cap_thick_mm + 8),
        color='#88aaee', fontsize=7,
        arrowprops=dict(arrowstyle='->', color='#88aaee', lw=0.8),
        ha='left',
    )
    ax.annotate(
        'wide collar\n(contact surface)',
        xy=(tip_om[0] + (collar_r - tip_om[0]) / 2, collar_z),
        xytext=(tip_om[0] + (collar_r - tip_om[0]) / 2 + 10, collar_z - 12),
        color='#ff8888', fontsize=7,
        arrowprops=dict(arrowstyle='->', color='#ff8888', lw=0.8),
        ha='left',
    )

    _style_ax(ax, 'Option 2 — Flat cap on wide collar', g)
    _set_limits(ax, g)
    ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#444',
              labelcolor='#ccc', loc='upper right')


# ── Main render ────────────────────────────────────────────────────────────────

def render(genome, diameter_mm=120.0, mould_scale=2.0, base_mm=8.0,
           collar_extra_mm=12.0, output='output/mould_revolved_options.png'):
    """
    Side-by-side render of Option 1 and Option 2 in r-z half-plane.
    """
    g = _geom(genome, diameter_mm, mould_scale, base_mm, collar_extra_mm)

    print(f"  Scale      : {g['s']:.1f} mm/sim-unit  (⌀{diameter_mm:.0f} mm)")
    print(f"  z_collar   : {g['z_collar']:.1f} mm")
    print(f"  tip outer_1: r={g['tip_o1'][0]:.1f}  z={g['tip_o1'][1]:.1f} mm")
    print(f"  tip inner_1: r={g['tip_i1'][0]:.1f}  z={g['tip_i1'][1]:.1f} mm")
    print(f"  tip outer_m: r={g['tip_om'][0]:.1f}  z={g['tip_om'][1]:.1f} mm")
    print(f"  flange len : {np.hypot(g['tip_o1'][0]-g['tip_i1'][0], g['tip_o1'][1]-g['tip_i1'][1]):.1f} mm  (= t_tip at scale 1×)")

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    fig.patch.set_facecolor('#1a1a2e')

    draw_option1(axes[0], g)
    draw_option2(axes[1], g)

    fig.suptitle(
        f'Two-piece revolved mould — r-z half-plane cross-section\n'
        f'⌀{diameter_mm:.0f} mm bell  |  mould_scale = {mould_scale:.1f}×  |  base = {base_mm:.0f} mm',
        color='#ddd', fontsize=11, y=1.01,
    )
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  Saved: {output}")


def main():
    p = argparse.ArgumentParser(
        description='Render two-piece revolved mould options (r-z half-plane)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--aurelia',  action='store_true', help='Aurelia aurita reference genome')
    p.add_argument('--diameter', type=float, default=120.0, help='Bell diameter (mm)')
    p.add_argument('--scale',    type=float, default=2.0,   help='Mould thickness scale')
    p.add_argument('--base',     type=float, default=8.0,   help='Base plate thickness (mm)')
    p.add_argument('--collar',   type=float, default=12.0,  help='Option 2 collar extra width (mm)')
    p.add_argument('--output',   default='output/mould_revolved_options.png')
    args = p.parse_args()

    genome = AURELIA_GENOME if args.aurelia else random_genome()
    render(genome, args.diameter, args.scale, args.base, args.collar, args.output)


if __name__ == '__main__':
    main()
