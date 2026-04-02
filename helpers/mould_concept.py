"""
helpers/mould_concept.py — 2D cross-section render of single-piece extruded mould

Concept:
  The mould is derived directly from the genome's Bezier spine.
  Instead of the original thickness profile (t_base, t_mid, t_tip),
  we scale it up by `mould_scale` to produce a thicker envelope.
  Subtracting the original jelly+muscle region leaves the mould cavity.
  The payload slot is excluded — the payload is inserted separately after curing.

Usage:
  uv run python helpers/mould_concept.py --aurelia
  uv run python helpers/mould_concept.py --diameter 120 --wall-scale 2.5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Polygon
from shapely.ops import unary_union

from make_jelly import (
    AURELIA_GENOME, PAYLOAD_WIDTH, PAYLOAD_HEIGHT,
    cubic_bezier, get_normals_2d, random_genome,
)
from make_cad import _scale_factor


# ── Geometry helpers ───────────────────────────────────────────────────────────

def bell_curves_scaled(genome, n_pts=80, thickness_scale=1.0):
    """
    Returns (outer, inner) bell curves in sim-local coords.
    thickness_scale > 1.0 expands both sides symmetrically around the spine.
    """
    start_p = np.array([PAYLOAD_WIDTH / 2.0, 0.0])
    cp1 = start_p + np.array([abs(genome[0]), genome[1]])
    cp2 = start_p + np.array([abs(genome[2]), genome[3]])
    end_p = start_p + np.array([abs(genome[4]), genome[5]])

    t_base = abs(genome[6]) * thickness_scale
    t_mid  = abs(genome[7]) * thickness_scale
    t_tip  = abs(genome[8]) * thickness_scale

    t_steps = np.linspace(0, 1, n_pts)
    spine   = np.array([cubic_bezier(start_p, cp1, cp2, end_p, t) for t in t_steps])
    normals = get_normals_2d(spine)
    half_t  = np.interp(t_steps, [0, 0.5, 1], [t_base, t_mid, t_tip])[:, None] / 2.0

    return spine + normals * half_t, spine - normals * half_t


def full_cross_section(outer, inner, scale=1.0):
    """
    Build full symmetric bell cross-section polygon from right-side curves.
    Same winding as make_cad.export_extruded. Returns a Shapely Polygon.
    """
    o = outer * scale
    i = inner * scale
    pts = np.vstack([
        o,                                        # right outer  base → tip
        i[::-1],                                  # right inner  tip → base
        np.c_[-i[:, 0],    i[:, 1]],              # left  inner  base → tip
        np.c_[-o[::-1, 0], o[::-1, 1]],           # left  outer  tip → base
    ])
    return Polygon(pts).buffer(0)


# ── Render ─────────────────────────────────────────────────────────────────────

def render(genome, diameter_mm=120.0, mould_scale=2.5, output="output/mould_concept.png"):
    """
    mould_scale: factor applied to t_base/t_mid/t_tip for the mould envelope.
      e.g. 2.5 means mould wall thickness = 2.5× the jellyfish wall thickness.
    """
    s = _scale_factor(genome, diameter_mm)

    # Original bell curves (the part — silicone jellyfish shape)
    outer_part, inner_part = bell_curves_scaled(genome, thickness_scale=1.0)

    # Expanded bell curves (the mould envelope)
    outer_mould, inner_mould = bell_curves_scaled(genome, thickness_scale=mould_scale)

    # 2D cross-section polygons
    part_poly  = full_cross_section(outer_part,  inner_part,  scale=s)
    mould_poly = full_cross_section(outer_mould, inner_mould, scale=s)

    # Payload slot rectangle (excluded from mould — inserted separately)
    payload_r = PAYLOAD_WIDTH  / 2.0 * s
    payload_h = PAYLOAD_HEIGHT * s
    payload_slot = Polygon([
        [-payload_r, 0], [payload_r, 0],
        [payload_r, payload_h], [-payload_r, payload_h],
    ])

    # Mould body = expanded envelope − part cavity − payload slot
    mould_body = mould_poly.difference(part_poly).difference(payload_slot)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#1a1a2e')

    for ax, show_mould in zip(axes, [False, True]):
        ax.set_facecolor('#12121f')
        ax.set_aspect('equal')

        def fill(poly, color, alpha, zorder, label=None):
            if poly is None or poly.is_empty:
                return
            geoms = list(poly.geoms) if poly.geom_type == 'MultiPolygon' else [poly]
            for k, g in enumerate(geoms):
                x, y = g.exterior.xy
                ax.fill(x, y, color=color, alpha=alpha, zorder=zorder,
                        label=label if k == 0 else None)
                for hole in g.interiors:
                    hx, hy = hole.xy
                    ax.fill(hx, hy, color='#12121f', alpha=1.0, zorder=zorder + 0.05)

        if show_mould:
            fill(mould_body,  '#7a8fa6', 0.85, 1, f'Mould body (×{mould_scale:.1f} thickness)')
        fill(part_poly,   '#44cc88', 0.80, 2, 'Silicone cavity (jelly + muscle)')
        fill(payload_slot, '#ff8844', 0.55, 3, 'Payload slot (open)')

        # Bell curve overlays
        op = outer_part * s; ip = inner_part * s
        om = outer_mould * s; im = inner_mould * s
        for sign in (1, -1):
            ax.plot(sign * op[:, 0], op[:, 1], color='#44cc88', lw=1.0, alpha=0.6)
            ax.plot(sign * ip[:, 0], ip[:, 1], color='#44cc88', lw=0.6, alpha=0.4, ls='--')
            if show_mould:
                ax.plot(sign * om[:, 0], om[:, 1], color='#7a8fa6', lw=1.0, alpha=0.5)
                ax.plot(sign * im[:, 0], im[:, 1], color='#7a8fa6', lw=0.6, alpha=0.3, ls='--')

        # Payload label
        ax.text(0, payload_h * 0.5, 'PAYLOAD\n(insert after\ncuring)',
                ha='center', va='center', fontsize=7, color='#ffaa66',
                bbox=dict(facecolor='#12121f', edgecolor='none', alpha=0.6, pad=1))

        bounds = mould_poly.bounds
        pad = diameter_mm * 0.08
        ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
        ax.set_ylim(bounds[1] - pad, bounds[3] + pad)
        ax.set_xlabel('x  (mm)', color='#aaa', fontsize=8)
        ax.set_ylabel('y  (mm)', color='#aaa', fontsize=8)
        ax.tick_params(colors='#666')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        ax.grid(True, alpha=0.15, color='#555')

        title = 'Part cross-section' if not show_mould else 'Mould cross-section'
        ax.set_title(title, color='#ccc', fontsize=10)
        ax.legend(loc='lower right', fontsize=7,
                  facecolor='#1a1a2e', edgecolor='#444', labelcolor='#ccc')

    fig.suptitle(
        f'Single-piece extruded mould  |  ⌀{diameter_mm:.0f} mm bell  '
        f'|  mould thickness = {mould_scale:.1f}× bell thickness',
        color='#ddd', fontsize=11, y=1.01,
    )
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {output}")


def main():
    p = argparse.ArgumentParser(description='Render extruded mould concept')
    p.add_argument('--aurelia', action='store_true', help='Aurelia aurita reference genome')
    p.add_argument('--diameter', type=float, default=120.0, help='Bell diameter (mm)')
    p.add_argument('--wall-scale', type=float, default=2.5,
                   help='Mould thickness as multiple of bell thickness (default 2.5)')
    p.add_argument('--output', default='output/mould_concept.png')
    args = p.parse_args()

    genome = AURELIA_GENOME if args.aurelia else random_genome()
    render(genome, diameter_mm=args.diameter, mould_scale=args.wall_scale, output=args.output)


if __name__ == '__main__':
    main()
