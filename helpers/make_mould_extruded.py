"""
make_mould_extruded.py — Single-piece frame mould for extruded flat jellyfish

Architecture
────────────
The mould is a flat frame (2D cross-section extruded to the robot's Z-depth).
The frame material fills both the outer mould walls AND the subumbrellar void,
so silicone only enters the two crescent-shaped bell wall cavities.
The payload slot area stays solid — it acts as a boss that creates the slot
in the cured silicone. After curing the payload is inserted separately.

Cross-section anatomy (X-Y plane, Y-up):
  ┌─────────────────────────────┐
  │  outer mould wall  (solid)  │  ← follows outer_bell × mould_scale
  │  ┌───────────────────────┐  │
  │  │  subumbrellar void    │  │  ← solid frame material (blocks silicone)
  │  │    ┌────┐             │  │
  │  │    │boss│ payload     │  │  ← solid boss → creates slot in silicone
  │  │    └────┘             │  │
  │  │  [cavity] [cavity]    │  │  ← crescents removed → silicone fills here
  │  └───────────────────────┘  │
  └─────────────────────────────┘

Assembly:
  1. Place frame on any flat base plate (glass, aluminium)
  2. Pour silicone into the two crescent holes from the top
  3. Place flat top plate; clamp lightly
  4. Cure, slide frame off in Z (demould direction — zero draft needed)
  5. Insert payload into the formed slot

Orientation: payload boss at Y=0 (sits on base plate), bell tips at Y_max.

Usage:
  uv run python helpers/make_mould_extruded.py --aurelia
  uv run python helpers/make_mould_extruded.py --aurelia --preview
  uv run python helpers/make_mould_extruded.py --gen 5 --diameter 120 --depth 15 --scale 2.0
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.affinity import scale as shapely_scale, affine_transform

from make_jelly import PAYLOAD_WIDTH, PAYLOAD_HEIGHT, cubic_bezier, get_normals_2d
from make_cad import _scale_factor, load_genome


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _bell_curves(genome, n_pts=80, t_scale=1.0):
    """
    Bell outer and inner curves in sim-local coordinates.
    t_scale multiplies the thickness genes (t_base, t_mid, t_tip).
    """
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


# ── 2D frame cross-section ─────────────────────────────────────────────────────

def build_frame_2d(genome, s, mould_scale, n_pts=80):
    """
    Shapely Polygon describing the mould frame cross-section.

    Exterior: full mould envelope at mould_scale (horseshoe covers walls + void +
              payload boss area — everything enclosed).
    Interior holes: two crescent-shaped bell wall cavities at 1× thickness
                    (these are the spaces where silicone is poured).

    The payload slot region is NOT subtracted → stays solid as the boss.

    Returns (frame, right_cresc, left_cresc) for optional preview.
    """
    outer_1, inner_1 = _bell_curves(genome, n_pts=n_pts, t_scale=1.0)
    outer_m, _       = _bell_curves(genome, n_pts=n_pts, t_scale=mould_scale)

    o1 = outer_1 * s
    i1 = inner_1 * s
    om = outer_m * s

    # ── Right crescent (silicone cavity, right side) ──────────────────────────
    right_cresc = Polygon(np.vstack([o1, i1[::-1]])).buffer(0)
    if right_cresc.geom_type == 'MultiPolygon':
        right_cresc = max(right_cresc.geoms, key=lambda g: g.area)

    # ── Left crescent (mirror of right) ──────────────────────────────────────
    left_cresc = shapely_scale(right_cresc, xfact=-1.0, origin=(0, 0, 0))

    # ── Mould envelope: rectangular bounding box ──────────────────────────────
    # The outer_m spine-normal expansion cannot reliably enclose outer_1 near the
    # bell base because the normal at t=0 can point inward (the attachment tangent
    # is near-vertical), so a larger t_scale moves outer_m toward the axis rather
    # than outward.  A rectangle sized from the outer_m extent is simple, robustly
    # contains both crescents, and gives clean flat edges for clamping.
    #
    # Wall thickness (each side) is the peak genome thickness × (mould_scale - 1),
    # so the sizing is still genome-derived.
    payload_h_mm = PAYLOAD_HEIGHT * s
    wall_mm  = max(abs(genome[6]), abs(genome[7]), abs(genome[8])) * s * (mould_scale - 1)
    wall_mm  = max(wall_mm, 4.0)          # floor: at least 4 mm regardless of genome

    x_max = float(om[:, 0].max()) + wall_mm
    y_min = float(om[:, 1].min()) - wall_mm   # below bell tips (y negative in sim coords)
    y_max = payload_h_mm + wall_mm * 0.5      # above payload

    mould_env = Polygon([
        (-x_max, y_min), (x_max, y_min),
        (x_max, y_max), (-x_max, y_max),
    ])

    # ── Frame = envelope minus both crescent cavities ─────────────────────────
    frame = mould_env.difference(right_cresc).difference(left_cresc)
    return frame, right_cresc, left_cresc


def _orient(poly):
    """
    Flip y → −y (so bell tips point up, payload boss at bottom)
    then translate so y_min = 0 (base sits on the table).
    """
    flipped = affine_transform(poly, [1, 0, 0, -1, 0, 0])
    ymin = flipped.bounds[1]
    return affine_transform(flipped, [1, 0, 0, 1, 0, -ymin])


# ── STL generation ─────────────────────────────────────────────────────────────

def generate(genome, diameter_mm=120.0, depth_mm=15.0, mould_scale=2.0,
             output="output/mould_extruded.stl"):
    """
    Build and export the extruded frame mould as a watertight STL.

    Parameters
    ----------
    depth_mm   : extrusion depth = physical robot thickness in Z
    mould_scale: thickness multiplier (2.0 → mould wall = 2× bell wall thickness)
    """
    s = _scale_factor(genome, diameter_mm)
    print(f"  Scale       : {s:.1f} mm/sim-unit  (⌀{diameter_mm:.0f} mm)")
    print(f"  Mould scale : {mould_scale:.1f}×  (wall = {mould_scale:.1f}× bell thickness)")
    print(f"  Depth (Z)   : {depth_mm:.1f} mm")

    frame_2d, right_c, left_c = build_frame_2d(genome, s, mould_scale)
    frame_oriented = _orient(frame_2d)

    bounds = frame_oriented.bounds
    print(f"  Frame X     : {bounds[0]:.1f} → {bounds[2]:.1f} mm  "
          f"(width {bounds[2]-bounds[0]:.1f} mm)")
    print(f"  Frame Y     : {bounds[1]:.1f} → {bounds[3]:.1f} mm  "
          f"(height {bounds[3]-bounds[1]:.1f} mm)")
    if hasattr(frame_oriented, 'interiors'):
        n_holes = len(list(frame_oriented.interiors))
        print(f"  Cavities    : {n_holes} crescent hole(s)")

    print("  Extruding …")
    mesh = trimesh.creation.extrude_polygon(frame_oriented, depth_mm)

    wt = "watertight" if mesh.is_watertight else "NOT watertight"
    vol = abs(mesh.volume) / 1e3
    print(f"  Volume      : {vol:.1f} cm³   triangles: {len(mesh.faces)}   [{wt}]")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    mesh.export(output)
    print(f"  Saved       : {output}")
    return mesh


# ── 2D preview ─────────────────────────────────────────────────────────────────

def preview(genome, diameter_mm=120.0, mould_scale=2.0,
            output="output/mould_extruded_preview.png"):
    """
    Matplotlib render of the oriented frame cross-section.
    Left panel: before orienting (sim coords).
    Right panel: after orienting (mould coords, Y-up).
    """
    import matplotlib.pyplot as plt

    s = _scale_factor(genome, diameter_mm)
    frame_2d, right_c, left_c = build_frame_2d(genome, s, mould_scale)
    frame_oriented = _orient(frame_2d)

    payload_r = PAYLOAD_WIDTH / 2.0 * s
    payload_h = PAYLOAD_HEIGHT * s
    payload_slot_orig = Polygon([
        [-payload_r, 0], [payload_r, 0],
        [payload_r, payload_h], [-payload_r, payload_h],
    ])
    payload_slot = _orient(payload_slot_orig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')

    def fill_poly(ax, poly, color, alpha=0.75, label=None, bg='#12121f'):
        if poly.is_empty:
            return
        geoms = list(poly.geoms) if poly.geom_type == 'MultiPolygon' else [poly]
        for k, g in enumerate(geoms):
            x, y = g.exterior.xy
            ax.fill(x, y, color=color, alpha=alpha,
                    label=label if k == 0 else None, zorder=2)
            for hole in g.interiors:
                hx, hy = hole.xy
                ax.fill(hx, hy, color=bg, alpha=1.0, zorder=3)

    def setup_ax(ax, title):
        ax.set_facecolor('#12121f')
        ax.set_aspect('equal')
        ax.set_xlabel('x  (mm)', color='#aaa', fontsize=8)
        ax.set_ylabel('y  (mm)', color='#aaa', fontsize=8)
        ax.tick_params(colors='#666')
        for sp in ax.spines.values():
            sp.set_edgecolor('#333')
        ax.grid(True, alpha=0.15, color='#555')
        ax.set_title(title, color='#ccc', fontsize=9)

    # Original orientation
    ax = axes[0]
    setup_ax(ax, 'Frame cross-section  (sim coords, tips down)')
    fill_poly(ax, frame_2d,  '#7a8fa6', label='Frame (rigid)')
    fill_poly(ax, right_c,   '#44cc88', label='Silicone cavity (right)')
    fill_poly(ax, left_c,    '#44cc88')
    fill_poly(ax, payload_slot_orig, '#ff8844', alpha=0.55, label='Payload boss (solid)')
    ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#444', labelcolor='#ccc',
              loc='lower right')

    # Mould orientation
    ax = axes[1]
    setup_ax(ax, 'Frame cross-section  (mould coords, tips up — pour from top)')
    right_oriented = _orient(right_c)
    left_oriented  = _orient(left_c)
    fill_poly(ax, frame_oriented, '#7a8fa6', label='Frame (rigid)')
    fill_poly(ax, right_oriented, '#44cc88', label='Silicone cavity')
    fill_poly(ax, left_oriented,  '#44cc88')
    fill_poly(ax, payload_slot,   '#ff8844', alpha=0.55, label='Payload boss')
    ax.annotate('pour from top →', xy=(0, frame_oriented.bounds[3]),
                xytext=(0, frame_oriented.bounds[3] + 5),
                ha='center', color='#aaa', fontsize=7, arrowprops=dict(arrowstyle='->',
                color='#aaa', lw=0.8))
    ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#444', labelcolor='#ccc',
              loc='lower right')

    fig.suptitle(
        f'Single-piece extruded mould  |  ⌀{diameter_mm:.0f} mm  |  '
        f'mould_scale={mould_scale:.1f}×',
        color='#ddd', fontsize=11,
    )
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  Preview saved: {output}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate single-piece extruded frame mould STL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--aurelia', action='store_true', help='Aurelia aurita reference genome')
    p.add_argument('--gen', type=int, default=None, help='Best genome from this generation')
    p.add_argument('--diameter', type=float, default=120.0, help='Bell diameter (mm)')
    p.add_argument('--depth',    type=float, default=15.0,  help='Robot Z-depth / frame thickness (mm)')
    p.add_argument('--scale',    type=float, default=2.0,   help='Mould thickness scale (2.0 = 2× bell wall)')
    p.add_argument('--preview',  action='store_true',       help='Show 2D cross-section preview only')
    p.add_argument('--output',   default=None,              help='Output STL path (default: output/mould_extruded_<tag>.stl)')
    args = p.parse_args()

    genome, tag = load_genome(args)
    print(f"Genome : {tag}")

    if args.preview:
        preview(genome, args.diameter, args.scale,
                output=f'output/mould_extruded_preview_{tag}.png')
    else:
        out = args.output or f'output/mould_extruded_{tag}.stl'
        generate(genome, args.diameter, args.depth, args.scale, output=out)


if __name__ == '__main__':
    main()
