#!/usr/bin/env python3
"""
jelly_dxf.py — Genomically-defined jellyfish bell profile as DXF.

Outputs a 2D half-section in the r-z plane (revolution profile):
  - JELLY_OUTER: outer bell wall
  - JELLY_INNER: inner bell wall (subumbrellar surface)
  - MOULD_OUTER: expanded mould envelope (thickness × mould_scale)
  - PAYLOAD:     payload cavity reference rectangle (not silicon — insert after curing)
  - SPINE:       bell spine construction curve
  - AXIS:        axis of symmetry

Manufacturing orientation: tips up (positive z), collar attachment at z=0.

Import into Fusion360/FreeCAD:
  - Select JELLY_OUTER + JELLY_INNER closed profile → Revolve for axisymmetric bell
  - Select MOULD_OUTER + JELLY_OUTER closed profile → Revolve for mould block
  - Use full symmetric extrusion for planar 2D swimmer version

Usage:
    uv run python helpers/jelly_dxf.py                         # Aurelia reference
    uv run python helpers/jelly_dxf.py --genome "[0.05,...]"   # custom genome
    uv run python helpers/jelly_dxf.py --preview               # matplotlib only
    uv run python helpers/jelly_dxf.py --diameter 80 --mould-scale 2.0
"""

import argparse
import json
import os
import sys

import numpy as np
import ezdxf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from make_jelly import AURELIA_GENOME, PAYLOAD_HEIGHT, PAYLOAD_WIDTH, cubic_bezier, get_normals_2d


def get_bell_profiles(genome, n_pts=80, scale_mm=192.0, mould_scale=0.5):
    """
    Compute bell profile curves from genome in manufacturing coordinates (r-z half-plane).

    r = x_sim * scale_mm   (radial distance from axis, ≥ 0)
    z = -y_sim * scale_mm  (height; sim tips are negative y, so they become positive z = up)

    All geometry is computed in sim coords first, then transformed together.

    Collar geometry matches make_jelly.generate_phenotype exactly: a Bezier socket
    that bridges the payload attachment edge to the bell wall with C1 continuity.

    Returns dict: spine, outer, inner, mould_outer (each Nx2 array in mm),
                  payload_rect (4x2 array of corners in mm).
    """
    # --- Bell spine and wall curves (sim coords) ---
    start_p = np.array([PAYLOAD_WIDTH / 2.0, 0.0])
    cp1 = start_p + np.array([abs(genome[0]), genome[1]])
    cp2 = start_p + np.array([abs(genome[2]), genome[3]])
    end_p = start_p + np.array([abs(genome[4]), genome[5]])

    t_steps = np.linspace(0, 1, n_pts)
    spine = np.array([cubic_bezier(start_p, cp1, cp2, end_p, t) for t in t_steps])

    t_base, t_mid, t_tip = abs(genome[6]), abs(genome[7]), abs(genome[8])
    semi = np.interp(t_steps, [0.0, 0.5, 1.0], [t_base, t_mid, t_tip])[:, None] / 2.0

    normals = get_normals_2d(spine)

    outer = spine + normals * semi
    inner = spine - normals * semi

    # Mould outer: wall thickness halved vs. naive mould_scale × semi,
    # so expansion beyond jelly outer = semi * (mould_scale - 1) / 2.
    mould_outer = outer + normals * semi * (mould_scale - 1) / 2.0

    # --- Collar geometry (matches make_jelly.generate_phenotype) ---
    # Smooth socket that connects payload attachment edge to the bell wall.
    collar_top_y = PAYLOAD_HEIGHT * 0.50
    collar_thickness_top = t_base * 0.35
    n_collar = 20
    collar_t = np.linspace(0, 1, n_collar)

    # Inner collar edge: Bezier from payload side wall down to inner[0]
    ci_P0 = np.array([PAYLOAD_WIDTH / 2.0, collar_top_y])
    ci_P3 = inner[0]
    ci_P1 = np.array([PAYLOAD_WIDTH / 2.0, collar_top_y - collar_top_y * 0.4])
    ci_P2 = ci_P3 + np.array([0.0, min(collar_top_y * 0.3, abs(ci_P3[1] - ci_P0[1]) * 0.4)])
    collar_inner = np.array([cubic_bezier(ci_P0, ci_P1, ci_P2, ci_P3, t) for t in collar_t])

    bell_outer_start = outer[0]
    if bell_outer_start[0] > PAYLOAD_WIDTH / 2.0:
        # Outer collar edge: Bezier from collar top to outer[0] with C1 continuity
        P0 = np.array([PAYLOAD_WIDTH / 2.0 + collar_thickness_top, collar_top_y])
        P3 = bell_outer_start
        P1 = np.array([P0[0], P0[1] - collar_top_y * 0.5])
        bell_tangent = outer[min(2, len(outer) - 1)] - outer[0]
        bt_len = np.linalg.norm(bell_tangent)
        P2 = P3 - (bell_tangent / bt_len) * (collar_top_y * 0.6) if bt_len > 1e-10 else P3 + np.array([0.0, 0.01])
        collar_outer = np.array([cubic_bezier(P0, P1, P2, P3, t) for t in collar_t])

        # Mould collar outer: same shape but ending at mould_outer[0]
        mould_wall_r = semi[0, 0] * (mould_scale - 1) / 2.0  # radial expansion at base
        P0_m = np.array([P0[0] + mould_wall_r, collar_top_y])
        P3_m = mould_outer[0]
        P1_m = np.array([P0_m[0], P0_m[1] - collar_top_y * 0.5])
        mould_tangent = mould_outer[min(2, len(mould_outer) - 1)] - mould_outer[0]
        mt_len = np.linalg.norm(mould_tangent)
        P2_m = P3_m - (mould_tangent / mt_len) * (collar_top_y * 0.6) if mt_len > 1e-10 else P3_m + np.array([0.0, 0.01])
        collar_mould = np.array([cubic_bezier(P0_m, P1_m, P2_m, P3_m, t) for t in collar_t])
    else:
        # Degenerate: bell folds inside payload width — no collar
        collar_outer = np.array([bell_outer_start] * n_collar)
        collar_mould = np.array([mould_outer[0]] * n_collar)

    # Prepend collar to bell wall curves to make full profiles from collar top to bell tip
    full_outer      = np.vstack([collar_outer, outer])
    full_inner      = np.vstack([collar_inner, inner])
    full_mould      = np.vstack([collar_mould, mould_outer])

    # --- Coordinate transform: sim (x, y) → manufacturing r-z half-plane ---
    def to_rz(pts):
        r = pts[:, 0] * scale_mm
        z = -pts[:, 1] * scale_mm   # flip: sim tips (negative y) → positive z (up)
        return np.column_stack([r, z])

    # Payload cavity rectangle: below z=0 (payload hangs below collar in mfg coords)
    pw = PAYLOAD_WIDTH / 2.0 * scale_mm
    ph = PAYLOAD_HEIGHT * scale_mm
    payload_rect = np.array([[0.0, 0.0], [pw, 0.0], [pw, -ph], [0.0, -ph]])

    return {
        "spine":        to_rz(spine),
        "outer":        to_rz(full_outer),
        "inner":        to_rz(full_inner),
        "mould_outer":  to_rz(full_mould),
        "payload_rect": payload_rect,
    }


def write_dxf(profiles, path):
    """Write layered bell profiles to a DXF file."""
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    layer_defs = [
        ("JELLY_OUTER", ezdxf.colors.WHITE),
        ("JELLY_INNER", ezdxf.colors.CYAN),
        ("MOULD_OUTER", ezdxf.colors.YELLOW),
        ("PAYLOAD",     ezdxf.colors.RED),
        ("SPINE",       8),   # dark grey
        ("AXIS",        8),
    ]
    for name, color in layer_defs:
        doc.layers.new(name=name, dxfattribs={"color": color})

    def add_poly(pts, layer, closed=False):
        msp.add_lwpolyline(
            [(float(x), float(y)) for x, y in pts],
            close=closed,
            dxfattribs={"layer": layer},
        )

    add_poly(profiles["outer"],       "JELLY_OUTER")
    add_poly(profiles["inner"],       "JELLY_INNER")
    add_poly(profiles["mould_outer"], "MOULD_OUTER")
    add_poly(profiles["spine"],       "SPINE")
    add_poly(profiles["payload_rect"], "PAYLOAD", closed=True)

    # Axis: vertical line at r=0, spanning full z range
    all_z = np.concatenate([profiles["outer"][:, 1], profiles["payload_rect"][:, 1]])
    msp.add_line(
        (0.0, float(all_z.min()) - 5.0),
        (0.0, float(all_z.max()) + 5.0),
        dxfattribs={"layer": "AXIS"},
    )

    doc.saveas(path)
    print(f"DXF written: {path}")


def preview(profiles):
    """Matplotlib preview of the r-z half-profile."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 9), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    ax.plot(*profiles["mould_outer"].T, color="#f0c040", lw=1.5, label="Mould outer")
    ax.plot(*profiles["outer"].T,       color="#ffffff", lw=1.5, label="Jelly outer")
    ax.plot(*profiles["inner"].T,       color="#40e0d0", lw=1.5, label="Jelly inner")
    ax.plot(*profiles["spine"].T,       color="#555566", lw=0.8, ls="--", label="Spine")
    ax.add_patch(plt.Polygon(profiles["payload_rect"], closed=True, fill=False,
                             edgecolor="#ff4444", lw=1.2, label="Payload (void)"))
    ax.axvline(0, color="#444455", lw=0.6, ls=":")

    ax.set_aspect("equal")
    ax.set_xlabel("r (mm)", color="white")
    ax.set_ylabel("z (mm) — tips up", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#555566")
    ax.legend(facecolor="#222233", labelcolor="white", loc="upper right", fontsize=9)
    ax.set_title("Jellyfish bell — revolution half-profile", color="white", pad=10)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate jellyfish bell DXF profile from genome")
    parser.add_argument("--genome", type=str, default=None,
                        help="Genome as JSON list (default: Aurelia reference)")
    parser.add_argument("--diameter", type=float, default=100.0,
                        help="Target bell diameter in mm (default: 100)")
    parser.add_argument("--mould-scale", type=float, default=2.5,
                        help="Mould wall thickness multiplier relative to jelly (default: 2.5)")
    parser.add_argument("--n-pts", type=int, default=80,
                        help="Spine sample points (default: 80)")
    parser.add_argument("--preview", action="store_true",
                        help="Show matplotlib preview; skip DXF output")
    parser.add_argument("--out", type=str, default=None,
                        help="Output DXF path (default: output/jelly_profile.dxf)")
    args = parser.parse_args()

    genome = np.array(json.loads(args.genome)) if args.genome else AURELIA_GENOME

    # Scale so the bell outer tip radius ≈ diameter/2
    # Bell tip r ≈ (PAYLOAD_WIDTH/2 + end_x) in sim units
    bell_r_sim = PAYLOAD_WIDTH / 2.0 + abs(genome[4])
    scale_mm = (args.diameter / 2.0) / bell_r_sim

    profiles = get_bell_profiles(genome, n_pts=args.n_pts,
                                 scale_mm=scale_mm, mould_scale=args.mould_scale)

    if args.preview:
        preview(profiles)
        return

    out_path = args.out or "output/jelly_profile.dxf"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_dxf(profiles, out_path)


if __name__ == "__main__":
    main()
