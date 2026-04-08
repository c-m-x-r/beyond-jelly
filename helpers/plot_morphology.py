"""
plot_morphology.py — Jellyfish morphology visualization

Three views driven purely from the genome curve math (no particle rasterization).
Colors: jelly/collar/bridge = blue, muscle = red, payload = gold.

Coordinate system: normalized sim units.  Payload sits at y=[0, 0.05].
Bell curves hang below y=0.  All curve geometry is right-half (x≥0); plots
are mirrored about x=0 for 2D and fully revolved for the axisymmetric view.

Usage:
    uv run python helpers/plot_morphology.py --aurelia
    uv run python helpers/plot_morphology.py --aurelia --mode 2d
    uv run python helpers/plot_morphology.py --aurelia --mode extrude
    uv run python helpers/plot_morphology.py --aurelia --mode revolve
    uv run python helpers/plot_morphology.py --genome "[0.05,0.04,0.18,-0.03,0.22,-0.12,0.04,0.05,0.015]"
    uv run python helpers/plot_morphology.py --gen 10 --mode all
    uv run python helpers/plot_morphology.py --aurelia --mode all --out output/morphology.png
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from make_jelly import (
    AURELIA_GENOME,
    PAYLOAD_HEIGHT,
    PAYLOAD_WIDTH,
    cubic_bezier,
    get_normals_2d,
    random_genome,
)

# ── colours ────────────────────────────────────────────────────────────────────

JELLY_BLUE   = '#3A7DC9'
MUSCLE_RED   = '#C93535'
PAYLOAD_GOLD = '#FFD700'

# ── geometry ───────────────────────────────────────────────────────────────────

def _build_geometry(genome, n_pts=60):
    """
    Pure-math geometry from a genome vector — no rasterization.

    Returns a dict of right-half curve arrays in sim-local coords (origin = bell
    attachment point; no spawn offset applied).

    Keys
    ----
    outer, inner   : (n_pts, 2) bell-wall outer / inner surfaces
    muscle         : (n_pts, 2) muscle–jelly interface (inner 25% of wall, exaggerated
                     slightly from the sim's ~10% for visual clarity)
    collar_outer,
    collar_inner,
    collar_muscle  : (20, 2) collar curves, or None if bell folds inside payload base
    bridge_half_w  : float  half-width of the transverse bridge rectangle
    """
    g        = np.asarray(genome, dtype=float)
    start_p  = np.array([PAYLOAD_WIDTH / 2.0, 0.0])
    cp1      = start_p + np.array([abs(g[0]),  g[1]])
    cp2      = start_p + np.array([abs(g[2]),  g[3]])
    end_p    = start_p + np.array([abs(g[4]),  g[5]])
    t_base, t_mid, t_tip = abs(g[6]), abs(g[7]), abs(g[8])

    t       = np.linspace(0, 1, n_pts)
    spine   = np.array([cubic_bezier(start_p, cp1, cp2, end_p, ti) for ti in t])
    normals = get_normals_2d(spine)
    half_t  = np.interp(t, [0, 0.5, 1], [t_base, t_mid, t_tip])[:, None] / 2.0

    outer  = spine + normals * half_t
    inner  = spine - normals * half_t
    # Muscle: inner 25% of wall from the inner (subumbrellar) surface.
    # The simulation uses ~10% but 25% is clearer at plot resolution.
    muscle = inner + 0.25 * (outer - inner)

    # ── collar ────────────────────────────────────────────────────────────────
    collar_top_y  = PAYLOAD_HEIGHT * 0.50          # mid-height of payload
    col_thick_top = t_base * 0.35
    ct            = np.linspace(0, 1, 20)

    # Inner collar edge: Bezier from payload side down to inner bell base
    ci_P0 = np.array([PAYLOAD_WIDTH / 2.0, collar_top_y])
    ci_P3 = inner[0].copy()
    ci_P1 = np.array([PAYLOAD_WIDTH / 2.0, collar_top_y * 0.6])
    ci_P2 = ci_P3 + np.array([0.0, min(collar_top_y * 0.3,
                                        abs(ci_P3[1] - ci_P0[1]) * 0.4)])
    collar_inner = np.array([cubic_bezier(ci_P0, ci_P1, ci_P2, ci_P3, s) for s in ct])

    collar_outer  = None
    collar_muscle = None
    bridge_half_w = PAYLOAD_WIDTH / 2.0 + 0.015

    if outer[0, 0] > PAYLOAD_WIDTH / 2.0:
        # Outer collar edge: Bezier from outer collar top to bell outer base
        P0    = np.array([PAYLOAD_WIDTH / 2.0 + col_thick_top, collar_top_y])
        P3    = outer[0].copy()
        P1    = np.array([P0[0], P0[1] - collar_top_y * 0.5])
        bt    = outer[min(2, n_pts - 1)] - outer[0]
        bt_n  = np.linalg.norm(bt)
        P2    = (P3 - (bt / bt_n) * (collar_top_y * 0.6)
                 if bt_n > 1e-10 else P3 + np.array([0.0, 0.01]))
        collar_outer  = np.array([cubic_bezier(P0, P1, P2, P3, s) for s in ct])
        collar_muscle = collar_inner + 0.25 * (collar_outer - collar_inner)
        bridge_half_w = max(outer[0, 0], PAYLOAD_WIDTH / 2.0 + col_thick_top)

    return dict(
        outer=outer, inner=inner, muscle=muscle,
        collar_inner=collar_inner,
        collar_outer=collar_outer,
        collar_muscle=collar_muscle,
        bridge_half_w=bridge_half_w,
    )


# ── shared utilities ───────────────────────────────────────────────────────────

def _mirror(c):
    """Reflect a curve about x=0."""
    m = c.copy()
    m[:, 0] *= -1
    return m


def _wall_poly(a, b):
    """Closed 2D polygon: curve a (base→tip) then curve b reversed (tip→base)."""
    return np.vstack([a, b[::-1]])


# ── 2D cross-section ───────────────────────────────────────────────────────────

def _fill_sym(ax, rout, rin, **kw):
    """Fill a wall region on both sides of the axis."""
    for a, b in [(rout, rin), (_mirror(rout), _mirror(rin))]:
        ax.fill(*_wall_poly(a, b).T, **kw)
        kw.pop('label', None)


def plot_2d(ax, geom, title):
    g  = geom
    bw = g['bridge_half_w']

    # Drawing order: bridge → collar jelly → bell jelly → muscle layers → payload
    bridge_xy = np.array([[-bw, -0.03], [bw, -0.03], [bw, 0.0], [-bw, 0.0]])
    ax.fill(*bridge_xy.T, color=JELLY_BLUE, alpha=0.9)

    if g['collar_outer'] is not None:
        _fill_sym(ax, g['collar_outer'], g['collar_inner'], color=JELLY_BLUE, alpha=0.9)

    _fill_sym(ax, g['outer'], g['inner'], color=JELLY_BLUE, alpha=0.9)

    if g['collar_muscle'] is not None:
        _fill_sym(ax, g['collar_muscle'], g['collar_inner'], color=MUSCLE_RED, alpha=0.9)

    _fill_sym(ax, g['muscle'], g['inner'], color=MUSCLE_RED, alpha=0.9)

    pw, ph  = PAYLOAD_WIDTH, PAYLOAD_HEIGHT
    pay_xy  = np.array([[-pw/2, 0], [pw/2, 0], [pw/2, ph], [-pw/2, ph]])
    ax.fill(*pay_xy.T, color=PAYLOAD_GOLD, alpha=1.0)

    # Thin outer outline
    for c in (g['outer'], _mirror(g['outer'])):
        ax.plot(*c.T, color='k', lw=0.7, alpha=0.4)

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x  (sim units)')
    ax.set_ylabel('y  (sim units)')
    ax.axhline(0, color='gray', lw=0.4, ls='--', alpha=0.4)
    ax.axvline(0, color='gray', lw=0.4, ls='--', alpha=0.4)
    ax.grid(True, alpha=0.18, linestyle='--')
    ax.legend(handles=[
        mpatches.Patch(color=JELLY_BLUE,   label='Jelly / Collar / Bridge'),
        mpatches.Patch(color=MUSCLE_RED,   label='Muscle'),
        mpatches.Patch(color=PAYLOAD_GOLD, label='Payload'),
    ], loc='lower right', fontsize=8)


# ── 3D extrusion ───────────────────────────────────────────────────────────────

def _extrude_faces(poly2d, depth):
    """
    Extrude a 2D polygon (N×2: x_sim, y_sim) by ±depth/2 along the depth axis.

    Returned face vertex coords: (x_sim, depth_coord, y_sim).
    This maps to matplotlib 3D as (X, Y, Z) with Z = y_sim = "up".
    """
    d0, d1 = -depth / 2, depth / 2
    v = poly2d
    n = len(v)
    faces = []
    faces.append([(v[i, 0], d0, v[i, 1]) for i in range(n)])                    # front
    faces.append([(v[i, 0], d1, v[i, 1]) for i in range(n - 1, -1, -1)])        # back
    for i in range(n):
        j = (i + 1) % n
        faces.append([
            (v[i, 0], d0, v[i, 1]), (v[j, 0], d0, v[j, 1]),
            (v[j, 0], d1, v[j, 1]), (v[i, 0], d1, v[i, 1]),
        ])
    return faces


def _add_extrude_region(ax, rout, rin, depth, color, alpha=0.9):
    for a, b in [(rout, rin), (_mirror(rout), _mirror(rin))]:
        ax.add_collection3d(Poly3DCollection(
            _extrude_faces(_wall_poly(a, b), depth),
            facecolor=color, edgecolor='none', alpha=alpha,
        ))


def plot_3d_extrude(ax, geom, title='', depth=0.12):
    g  = geom
    bw = g['bridge_half_w']

    bridge_xy = np.array([[-bw, -0.03], [bw, -0.03], [bw, 0.0], [-bw, 0.0]])
    ax.add_collection3d(Poly3DCollection(
        _extrude_faces(bridge_xy, depth),
        facecolor=JELLY_BLUE, edgecolor='none', alpha=0.9,
    ))

    if g['collar_outer'] is not None:
        _add_extrude_region(ax, g['collar_outer'], g['collar_inner'], depth, JELLY_BLUE)

    _add_extrude_region(ax, g['outer'], g['inner'], depth, JELLY_BLUE)

    if g['collar_muscle'] is not None:
        _add_extrude_region(ax, g['collar_muscle'], g['collar_inner'], depth, MUSCLE_RED)

    _add_extrude_region(ax, g['muscle'], g['inner'], depth, MUSCLE_RED)

    pw, ph  = PAYLOAD_WIDTH, PAYLOAD_HEIGHT
    pay_xy  = np.array([[-pw/2, 0], [pw/2, 0], [pw/2, ph], [-pw/2, ph]])
    ax.add_collection3d(Poly3DCollection(
        _extrude_faces(pay_xy, depth),
        facecolor=PAYLOAD_GOLD, edgecolor='none', alpha=1.0,
    ))

    xr = g['outer'][:, 0].max() + 0.05
    yr = abs(g['outer'][:, 1].min()) + 0.05
    zr = depth / 2 + 0.02
    ax.set_xlim(-xr, xr)
    ax.set_ylim(-zr, zr)
    ax.set_zlim(-yr, PAYLOAD_HEIGHT + 0.02)
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('depth', fontsize=8)
    ax.set_zlabel('y', fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.view_init(elev=25, azim=-55)
    _axes_equal_3d(ax)


# ── 3D revolution ──────────────────────────────────────────────────────────────

def _rev_arrays(curve, n_theta):
    """
    Revolve a right-half curve (N×2: x_sim, y_sim) 360° around the bell axis.

    Convention: r = x_sim (radial), z = y_sim (axial = up).
    Returns (X, Y, Z) grid arrays ready for ax.plot_surface.
    """
    r     = np.clip(curve[:, 0], 0.0, None)
    z     = curve[:, 1]
    theta = np.linspace(0, 2.0 * np.pi, n_theta)
    R  = r[np.newaxis, :]             # (1, N)
    TH = theta[:, np.newaxis]         # (n_theta, 1)
    X  = R * np.cos(TH)               # (n_theta, N)
    Y  = R * np.sin(TH)               # (n_theta, N)
    Z  = np.tile(z[np.newaxis, :], (n_theta, 1))
    return X, Y, Z


def plot_3d_revolve(ax, geom, title='', n_theta=64):
    g = geom

    # Build continuous outer/inner profiles: collar (top→bell base) + bell (base→tip).
    # collar_outer[-1] == outer[0] and collar_inner[-1] == inner[0] by construction,
    # so skip the duplicated junction point.
    if g['collar_outer'] is not None:
        outer_full  = np.vstack([g['collar_outer'],  g['outer'][1:]])
        inner_full  = np.vstack([g['collar_inner'],  g['inner'][1:]])
        muscle_full = np.vstack([g['collar_muscle'], g['muscle'][1:]])
    else:
        outer_full  = g['outer'].copy()
        inner_full  = g['inner'].copy()
        muscle_full = g['muscle'].copy()

    # For some bell shapes (upward-curving collar) the "outer" curve (in normal-direction
    # sense) may have smaller r than "inner" near the base.  Swap pointwise so that the
    # radially-outward surface is always plotted in blue and the subumbrellar in red.
    swap = outer_full[:, 0] < inner_full[:, 0]
    outer_r  = np.where(swap, inner_full[:, 0], outer_full[:, 0])
    outer_z  = np.where(swap, inner_full[:, 1], outer_full[:, 1])
    inner_r  = np.where(swap, outer_full[:, 0], inner_full[:, 0])
    inner_z  = np.where(swap, outer_full[:, 1], inner_full[:, 1])

    outer_rz  = np.column_stack([outer_r, outer_z])
    inner_rz  = np.column_stack([inner_r, inner_z])
    muscle_rz = inner_rz + 0.25 * (outer_rz - inner_rz)

    surf_kw = dict(rstride=1, cstride=1, linewidth=0, shade=True)

    # Jelly outer surface
    ax.plot_surface(*_rev_arrays(outer_rz, n_theta),
                    color=JELLY_BLUE, alpha=0.82, **surf_kw)

    # Muscle inner surface (subumbrellar face)
    ax.plot_surface(*_rev_arrays(inner_rz, n_theta),
                    color=MUSCLE_RED, alpha=0.90, **surf_kw)

    # Top annular cap — closes the bell opening at the collar/payload junction
    r0 = float(inner_rz[0, 0])
    r1 = float(outer_rz[0, 0])
    z_cap = float((inner_rz[0, 1] + outer_rz[0, 1]) / 2.0)
    if r1 > r0 + 1e-4:
        r_vals = np.linspace(r0, r1, 8)
        theta  = np.linspace(0, 2.0 * np.pi, n_theta)
        R_cap, TH_cap = np.meshgrid(r_vals, theta)
        ax.plot_surface(R_cap * np.cos(TH_cap), R_cap * np.sin(TH_cap),
                        np.full_like(R_cap, z_cap),
                        color=JELLY_BLUE, alpha=0.82, shade=False,
                        rstride=1, cstride=1, linewidth=0)

    # Tip cap — connects outer tip to inner tip
    tip_pts = np.array([
        (1 - s) * outer_rz[-1] + s * inner_rz[-1]
        for s in np.linspace(0, 1, 6)
    ])
    ax.plot_surface(*_rev_arrays(tip_pts, n_theta),
                    color=JELLY_BLUE, alpha=0.82, **surf_kw)

    # Payload cylinder + top disk
    theta    = np.linspace(0, 2.0 * np.pi, n_theta)
    z_pay    = np.linspace(0, PAYLOAD_HEIGHT, 6)
    TH_c, Z_c = np.meshgrid(theta, z_pay)
    r_cyl    = PAYLOAD_WIDTH / 2.0
    ax.plot_surface(r_cyl * np.cos(TH_c), r_cyl * np.sin(TH_c), Z_c,
                    color=PAYLOAD_GOLD, alpha=1.0, shade=True,
                    rstride=1, cstride=1, linewidth=0)
    r_disk   = np.linspace(0, r_cyl, 5)
    R_d, TH_d = np.meshgrid(r_disk, theta)
    ax.plot_surface(R_d * np.cos(TH_d), R_d * np.sin(TH_d),
                    np.full_like(R_d, PAYLOAD_HEIGHT),
                    color=PAYLOAD_GOLD, alpha=1.0, shade=False,
                    rstride=1, cstride=1, linewidth=0)

    r_max = outer_rz[:, 0].max() + 0.05
    z_min = outer_rz[:, 1].min() - 0.05
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_zlim(z_min, PAYLOAD_HEIGHT + 0.05)
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('y', fontsize=8)
    ax.set_zlabel('z  (up)', fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.view_init(elev=20, azim=45)
    _axes_equal_3d(ax)


# ── 3D equal-aspect helper ─────────────────────────────────────────────────────

def _axes_equal_3d(ax):
    """Force equal axis scaling on a 3D axes."""
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    ctr  = lims.mean(axis=1)
    rad  = 0.5 * (lims[:, 1] - lims[:, 0]).max()
    ax.set_xlim3d(ctr[0] - rad, ctr[0] + rad)
    ax.set_ylim3d(ctr[1] - rad, ctr[1] + rad)
    ax.set_zlim3d(ctr[2] - rad, ctr[2] + rad)


# ── genome loading ─────────────────────────────────────────────────────────────

def _load_genome(args):
    if args.aurelia:
        return AURELIA_GENOME, 'Aurelia aurita'
    if args.genome:
        return np.array(json.loads(args.genome)), 'Custom Genome'
    if args.gen is not None:
        path = 'output/best_genomes.json'
        if not os.path.exists(path):
            sys.exit(f'ERROR: {path} not found — run evolve.py first.')
        with open(path) as f:
            records = json.load(f)
        matches = [r for r in records if r['generation'] == args.gen]
        if not matches:
            sys.exit(f'ERROR: generation {args.gen} not in {path}.')
        rec = matches[0]
        print(f'Generation {args.gen}: fitness={rec["fitness"]:.4f}')
        return np.array(rec['genome']), f'Gen {args.gen} Best'
    g = random_genome()
    print(f'Random genome: {np.round(g, 4)}')
    return g, 'Random Genome'


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Jellyfish morphology visualizer')
    p.add_argument('--aurelia', action='store_true',
                   help='Aurelia aurita reference genome')
    p.add_argument('--genome',  type=str,   default=None,
                   help='JSON genome array, e.g. "[0.05,0.04,...]"')
    p.add_argument('--gen',     type=int,   default=None,
                   help='Best genome from generation N (reads output/best_genomes.json)')
    p.add_argument('--mode',    type=str,   default='all',
                   choices=['2d', 'extrude', 'revolve', 'all'],
                   help='Plot mode (default: all)')
    p.add_argument('--out',     type=str,   default=None,
                   help='Save to file instead of showing interactively')
    p.add_argument('--depth',   type=float, default=0.12,
                   help='Extrusion depth in sim units (default: 0.12)')
    p.add_argument('--slices',  type=int,   default=64,
                   help='Angular slices for revolution (default: 64)')
    args = p.parse_args()

    genome, label = _load_genome(args)
    geom          = _build_geometry(genome)

    if args.mode == 'all':
        fig = plt.figure(figsize=(18, 7))
        fig.suptitle(label, fontsize=13)
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        plot_2d(ax1, geom, '2D cross-section')
        plot_3d_extrude(ax2, geom, '3D extrusion', depth=args.depth)
        plot_3d_revolve(ax3, geom, '3D revolution', n_theta=args.slices)

    elif args.mode == '2d':
        fig, ax = plt.subplots(figsize=(7, 8))
        fig.suptitle(label)
        plot_2d(ax, geom, '2D cross-section')

    elif args.mode == 'extrude':
        fig = plt.figure(figsize=(9, 8))
        fig.suptitle(label)
        ax = fig.add_subplot(111, projection='3d')
        plot_3d_extrude(ax, geom, '3D extrusion', depth=args.depth)

    else:  # revolve
        fig = plt.figure(figsize=(9, 8))
        fig.suptitle(label)
        ax = fig.add_subplot(111, projection='3d')
        plot_3d_revolve(ax, geom, '3D revolution', n_theta=args.slices)

    plt.tight_layout()

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(args.out, dpi=150, bbox_inches='tight')
        print(f'Saved → {args.out}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
