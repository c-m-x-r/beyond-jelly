"""
make_cad.py — CAD export for evolved jellyfish morphologies

Given a genome vector, produces two STL files:
  bell_extruded_<tag>.stl  — L1: symmetric 2D cross-section extruded along Z
  bell_revolved_<tag>.stl  — L2: half-profile revolved 360° around the bell axis (watertight)

Coordinate conventions (sim-local, spawn offset not applied):
  x = left/right  (bell symmetric around x=0)
  y = up/down     (bell hangs below y=0; payload at y=[0, PAYLOAD_HEIGHT])
  For L2: (x,y) → (r,z) in cylindrical coords, revolve around z-axis.

Usage:
    uv run python make_cad.py                            # random genome
    uv run python make_cad.py --aurelia                  # Aurelia aurita reference
    uv run python make_cad.py --gen 5                    # best genome from generation 5
    uv run python make_cad.py --diameter 120 --depth 15  # physical scale (mm)
    uv run python make_cad.py --remesh 5                 # isotropic remesh, 5mm target edge
    uv run python make_cad.py --slices 120               # finer revolution (default 72)
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh
from shapely.geometry import Polygon

from make_jelly import (
    AURELIA_GENOME,
    PAYLOAD_HEIGHT,
    PAYLOAD_WIDTH,
    cubic_bezier,
    get_normals_2d,
    random_genome,
)


# ── Geometry ──────────────────────────────────────────────────────────────────

def get_bell_curves(genome, n_pts=80):
    """
    Extract outer and inner bell-wall curves from a genome vector.
    Returns (outer, inner): arrays of shape (n_pts, 2), sim-local coordinates.
    Reuses the same Bezier math as generate_phenotype(), without rasterizing.
    """
    start_p = np.array([PAYLOAD_WIDTH / 2.0, 0.0])
    cp1     = start_p + np.array([abs(genome[0]), genome[1]])
    cp2     = start_p + np.array([abs(genome[2]), genome[3]])
    end_p   = start_p + np.array([abs(genome[4]), genome[5]])

    t_base, t_mid, t_tip = abs(genome[6]), abs(genome[7]), abs(genome[8])

    t_steps = np.linspace(0, 1, n_pts)
    spine   = np.array([cubic_bezier(start_p, cp1, cp2, end_p, t) for t in t_steps])
    normals = get_normals_2d(spine)
    half_t  = np.interp(t_steps, [0, 0.5, 1], [t_base, t_mid, t_tip])[:, None] / 2.0

    outer = spine + normals * half_t
    inner = spine - normals * half_t
    return outer, inner


def _scale_factor(genome, diameter_mm):
    """mm per sim-unit: maps bell-tip radius to diameter_mm/2."""
    sim_radius = PAYLOAD_WIDTH / 2.0 + abs(genome[4])
    return (diameter_mm / 2.0) / max(sim_radius, 1e-6)


# ── Remeshing ─────────────────────────────────────────────────────────────────

def remesh_isotropic(mesh, target_edge_mm, iterations=5):
    """
    Isotropic explicit remeshing via pymeshlab.
    Produces uniform triangles regardless of how the source mesh was built.
    Skips gracefully if pymeshlab is not installed.
    """
    try:
        import pymeshlab
    except ImportError:
        print("  (remesh skipped: run 'uv add pymeshlab' to enable)")
        return mesh

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=mesh.vertices.astype(np.float64),
        face_matrix=mesh.faces.astype(np.int32),
    ))
    ms.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.PureValue(target_edge_mm),
        iterations=iterations,
    )
    rm = ms.current_mesh()
    return trimesh.Trimesh(
        vertices=rm.vertex_matrix(),
        faces=rm.face_matrix(),
        process=True,
    )


# ── L1: Flat extrusion ────────────────────────────────────────────────────────

def export_extruded(genome, diameter_mm=120.0, depth_mm=10.0, remesh_mm=None,
                    output="output/bell_extruded.stl"):
    """
    L1: Symmetric 2D cross-section extruded by depth_mm along Z.

    Cross-section polygon (single closed boundary, no inner subtraction):
      right outer (base→tip) → [tip wall] → right inner (tip→base)
      → [inner base bridge] → left inner (base→tip) → [tip wall]
      → left outer (tip→base) → [outer base bridge, implicit close]

    This gives short wall-thickness edges at each tip instead of the
    center-spanning line that the naive outer/inner-ring approach produces.
    The subumbrellar void is filled solid in this 2D slice — acceptable for
    a cross-section preview. The hollow shell matters in L2 (revolution).

    buffer(0) repairs any self-intersections from non-monotone Bezier spines.
    """
    outer, inner = get_bell_curves(genome)
    s = _scale_factor(genome, diameter_mm)

    ro = outer * s   # right outer  (base → tip)
    ri = inner * s   # right inner  (base → tip)

    # Single polygon: traces outer surface right, crosses tip wall, traces inner
    # surface right, crosses inner base to left side, mirrors back.
    single = np.vstack([
        ro,                                              # right outer  base→tip
        ri[::-1],                                        # right inner  tip→base
        np.column_stack([-ri[:, 0],    ri[:, 1]]),       # left  inner  base→tip
        np.column_stack([-ro[::-1, 0], ro[::-1, 1]]),   # left  outer  tip→base
    ])

    cross_section = Polygon(single).buffer(0)   # repair self-intersections
    if cross_section.geom_type == 'MultiPolygon':
        cross_section = max(cross_section.geoms, key=lambda g: g.area)

    if cross_section.is_empty:
        print("ERROR: cross-section is empty (degenerate genome?)")
        return None

    if cross_section.geom_type == 'MultiPolygon':
        meshes = [trimesh.creation.extrude_polygon(g, depth_mm)
                  for g in cross_section.geoms]
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = trimesh.creation.extrude_polygon(cross_section, depth_mm)

    if remesh_mm is not None:
        print(f"  remeshing L1 (target {remesh_mm} mm)...")
        mesh = remesh_isotropic(mesh, remesh_mm)

    os.makedirs("output", exist_ok=True)
    mesh.export(output)
    _print_stats("L1 extrude", output, mesh)
    return mesh


# ── L2: Revolution solid ──────────────────────────────────────────────────────

def export_revolved(genome, diameter_mm=120.0, n_slices=72, remesh_mm=None,
                    output="output/bell_revolved.stl"):
    """
    L2: Half-profile revolved 360° around the Z-axis. Watertight.

    Half-profile in (r, z) = (x_sim, y_sim) * scale:
      outer_curve (base→tip) + inner_curve (tip→base), forming a closed loop.

    The lateral surface is generated by sweeping all n_p-1 profile edges around
    the axis. The base seam — the profile's closing edge (inner_base → outer_base)
    — is also swept, closing the face that faces toward the payload attachment and
    making the mesh fully watertight.

    Payload bore not cut in this POC. The bell sits below z=0 and the bore (a
    cylinder at r < PAYLOAD_WIDTH/2, z ∈ [0, PAYLOAD_HEIGHT]) barely intersects.
    """
    outer, inner = get_bell_curves(genome)
    s = _scale_factor(genome, diameter_mm)

    outer_rz = outer * s   # (n_pts, 2): col0=r, col1=z
    inner_rz = inner * s

    # Closed half-profile: outer (base→tip) then inner (tip→base).
    # profile[0]      = outer_rz[0]  = outer base
    # profile[n_pts-1] = outer_rz[-1] = outer tip
    # profile[n_pts]   = inner_rz[-1] = inner tip
    # profile[-1]      = inner_rz[0]  = inner base
    # Closing edge (profile[-1] → profile[0]) = inner_base → outer_base = the base seam.
    n_pts = len(outer_rz)
    profile_rz = np.vstack([outer_rz, inner_rz[::-1]])
    profile_rz[:, 0] = np.clip(profile_rz[:, 0], 0.0, None)  # r ≥ 0

    n_p    = len(profile_rz)   # = 2 * n_pts
    angles = np.linspace(0, 2 * np.pi, n_slices, endpoint=False)

    # Vertices: rotate each profile point around z at each angle.
    verts = np.empty((n_slices * n_p, 3), dtype=np.float64)
    for i, theta in enumerate(angles):
        c, ss = np.cos(theta), np.sin(theta)
        base  = i * n_p
        verts[base : base + n_p, 0] = profile_rz[:, 0] * c
        verts[base : base + n_p, 1] = profile_rz[:, 0] * ss
        verts[base : base + n_p, 2] = profile_rz[:, 1]

    # Faces — two parts:
    # (a) Lateral: sweep each of the n_p-1 adjacent profile edges around the axis.
    # (b) Base seam: sweep the profile closing edge (profile[-1]→profile[0]).
    #     This is the face "facing toward the payload" — without it the mesh has
    #     an open annular gap at the top of the bell.

    n_lateral  = n_slices * (n_p - 1) * 2
    n_base_cap = n_slices * 2
    faces = np.empty((n_lateral + n_base_cap, 3), dtype=np.int64)
    fi = 0

    for a in range(n_slices):
        b      = (a + 1) % n_slices
        base_a = a * n_p
        base_b = b * n_p

        # (a) Lateral strips
        for p in range(n_p - 1):
            v00, v01 = base_a + p,     base_a + p + 1
            v10, v11 = base_b + p,     base_b + p + 1
            faces[fi]     = [v00, v10, v11]
            faces[fi + 1] = [v00, v11, v01]
            fi += 2

        # (b) Base seam: sweep the closing edge profile[-1] → profile[0]
        #     (inner_base → outer_base), forming the annular face at the top.
        v_ib_a = base_a + (n_p - 1)   # inner base, angle a
        v_ob_a = base_a + 0            # outer base, angle a
        v_ib_b = base_b + (n_p - 1)   # inner base, angle b
        v_ob_b = base_b + 0            # outer base, angle b
        faces[fi]     = [v_ib_a, v_ob_b, v_ib_b]
        faces[fi + 1] = [v_ib_a, v_ob_a, v_ob_b]
        fi += 2

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    if remesh_mm is not None:
        print(f"  remeshing L2 (target {remesh_mm} mm)...")
        mesh = remesh_isotropic(mesh, remesh_mm)

    os.makedirs("output", exist_ok=True)
    mesh.export(output)
    _print_stats("L2 revolve", output, mesh)
    return mesh


def _print_stats(label, path, mesh):
    vol  = abs(mesh.volume) / 1000.0
    wt   = "watertight" if mesh.is_watertight else "NOT watertight"
    print(f"{label} → {path}")
    print(f"  bounds     : {mesh.bounds[0].round(1)} → {mesh.bounds[1].round(1)} mm")
    print(f"  volume     : {vol:.2f} cm³   triangles: {len(mesh.triangles)}   [{wt}]")


# ── Genome loading ────────────────────────────────────────────────────────────

def load_genome(args):
    if args.aurelia:
        print("Genome : Aurelia aurita reference")
        return AURELIA_GENOME, "aurelia"

    if args.gen is not None:
        path = "output/best_genomes.json"
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run evolve.py first.")
            sys.exit(1)
        with open(path) as f:
            records = json.load(f)
        matches = [r for r in records if r["generation"] == args.gen]
        if not matches:
            print(f"ERROR: No record for generation {args.gen} in {path}.")
            sys.exit(1)
        rec = matches[0]
        print(f"Genome : generation {args.gen} best  (fitness={rec['fitness']:.4f})")
        return np.array(rec["genome"]), f"gen{args.gen}"

    genome = random_genome()
    print(f"Genome : random  {genome.round(4)}")
    return genome, "random"


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export jellyfish bell morphology to STL (extruded + revolved)"
    )
    parser.add_argument("--aurelia", action="store_true",
                        help="Use Aurelia aurita reference genome")
    parser.add_argument("--gen", type=int, default=None,
                        help="Use best genome from this generation")
    parser.add_argument("--diameter", type=float, default=120.0,
                        help="Target bell diameter in mm (default: 120)")
    parser.add_argument("--depth", type=float, default=10.0,
                        help="Extrusion depth for L1 in mm (default: 10)")
    parser.add_argument("--slices", type=int, default=72,
                        help="Angular slices for L2 revolution (default: 72)")
    parser.add_argument("--remesh", type=float, default=None, metavar="MM",
                        help="Isotropic remesh with this target edge length in mm")
    parser.add_argument("--extrude-only", action="store_true")
    parser.add_argument("--revolve-only", action="store_true")
    args = parser.parse_args()

    genome, tag = load_genome(args)
    print(f"Scale  : {_scale_factor(genome, args.diameter):.1f} mm/sim-unit"
          f"  (⌀ {args.diameter} mm)\n")

    if not args.revolve_only:
        export_extruded(
            genome,
            diameter_mm=args.diameter,
            depth_mm=args.depth,
            remesh_mm=args.remesh,
            output=f"output/bell_extruded_{tag}.stl",
        )
        print()

    if not args.extrude_only:
        export_revolved(
            genome,
            diameter_mm=args.diameter,
            n_slices=args.slices,
            remesh_mm=args.remesh,
            output=f"output/bell_revolved_{tag}.stl",
        )


if __name__ == "__main__":
    main()
