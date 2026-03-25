"""
make_mould.py — Silicone mould generator for jellyfish bell morphologies

Produces two STL files:
  mould_outer_<tag>.stl  — outer mould: square-base body, inverted bell cavity,
                            screw-hole pins in the collar zone
  mould_core_<tag>.stl   — inner core: fills subumbrellar void; wide base rests
                            on screw pins; ring flange balances on mould rim

──────────────────────────────────────────────────────────────────────────────
Orientation (inverted vs. simulation — bell tips face +Z / upward)

  Z = 0                    table surface
  Z = 0 → base_mm          solid base plate
  Z = base_mm              floor of payload bore
  Z = base_mm → z_collar   payload bore  (r < payload_r)
  Z = z_collar             collar / mesoglea — bell attachment level
  Z = above z_collar       bell cavity (tips at top)

Coordinate transform:  mould_z = z_collar − z_sim_cad
  (sim y=0 at bell base → z_collar; sim y<0 at tips → above z_collar)

──────────────────────────────────────────────────────────────────────────────
Outer mould cavity

  The outer bell profile is non-monotone in z (it dips below z_collar near the
  payload attachment before sweeping out toward the tips).  The void is therefore
  constructed from up to two solids and a payload bore, all unioned together:

    payload_bore  — cylinder r=payload_r, z = base_mm → z_collar
    bell_upper    — outer bell profile from z_min (deepest point) up to tip
                    (profile: (0,z_min)→outer_curve[idx_min:]→(0,z_top), closed)
    bell_pocket   — lower overhang from z_min back up to outer_curve[0]
                    (only present when idx_min > 0)

  Combined void = union(payload_bore, bell_upper [, bell_pocket]).
  Mould body    = box − combined_void.

  Two solid cylindrical SCREW-HOLE PINS are then unioned back at the collar
  mid-radius, 0° and 180°:
    • Create M-screw through-holes in the cast silicone collar.
    • Tops sit flush with z_collar — support the inner core's base plate.

──────────────────────────────────────────────────────────────────────────────
Inner core

  Body   Revolved solid of inner-bell profile, plus a wide base disc
         (r = 0 → base_r) at z = z_collar so it rests on the screw-pin tops.
  Ring   Annular flange at the top: inner_r_tip → outer_r_tip + overhang.
         Rests on the mould's top face; protrudes above for a grip.

  Core rests at two points:
    ① lower: wide base plate on screw-pin tops (z = z_collar)
    ② upper: ring flange on mould rim     (z ≈ outer_z_m[-1])

──────────────────────────────────────────────────────────────────────────────
Requires manifold3d (already added by `uv add manifold3d`):

Usage:
  uv run python helpers/make_mould.py                   # random genome
  uv run python helpers/make_mould.py --aurelia         # Aurelia aurita
  uv run python helpers/make_mould.py --gen 5           # best of gen 5
  uv run python helpers/make_mould.py --diameter 120    # 120 mm bell (default)
  uv run python helpers/make_mould.py --wall 8          # 8 mm wall (default)
  uv run python helpers/make_mould.py --base 10         # 10 mm base plate
  uv run python helpers/make_mould.py --screw-dia 3.3   # M3 clearance (default)
  uv run python helpers/make_mould.py --remesh 2        # isotropic remesh 2 mm
  uv run python helpers/make_mould.py --outer-only      # skip inner core
  uv run python helpers/make_mould.py --core-only       # skip outer mould
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh

from make_jelly import PAYLOAD_HEIGHT, PAYLOAD_WIDTH
from make_cad import _scale_factor, get_bell_curves, load_genome, remesh_isotropic


# ── Boolean helper ─────────────────────────────────────────────────────────────

def _bool(op, meshes, engines=("manifold", "openscad", "blender")):
    """
    Boolean op ('difference' | 'union') trying each engine in order.
    Raises RuntimeError with install hint if all fail.
    """
    fn = getattr(trimesh.boolean, op)
    errors = {}
    for eng in engines:
        try:
            result = fn(meshes, engine=eng)
            if isinstance(result, trimesh.Trimesh) and len(result.faces) > 0:
                return result
            errors[eng] = "returned empty mesh"
        except Exception as exc:
            errors[eng] = str(exc)
    details = "  " + "\n  ".join(f"{e}: {m}" for e, m in errors.items())
    raise RuntimeError(
        f"Boolean '{op}' failed with all engines.\n{details}\n"
        "Install manifold3d:  uv add manifold3d\n"
        "Or OpenSCAD:         https://openscad.org/downloads.html"
    )


# ── Revolution helper ──────────────────────────────────────────────────────────

def _revolve(profile_rz, n_slices=120):
    """
    Revolve a CLOSED 2D polygon (r ≥ 0, z) around the z-axis.
    Uses trimesh.creation.revolve which produces watertight solids.

    The profile must be closed: profile[-1] == profile[0] (approximately).
    If not already closed, the closing edge is appended automatically.

    Returns a watertight trimesh.Trimesh suitable for boolean operations.
    """
    rz = np.asarray(profile_rz, dtype=np.float64).copy()
    rz[:, 0] = np.clip(rz[:, 0], 0.0, None)

    # Ensure closed: append first point if the last differs
    if not np.allclose(rz[0], rz[-1]):
        rz = np.vstack([rz, rz[[0]]])

    m = trimesh.creation.revolve(rz, sections=n_slices)
    trimesh.repair.fix_normals(m)
    return m


def _cylinder(radius, z_bot, z_top, sections=64):
    """Upright solid cylinder from z_bot to z_top, centred on Z-axis."""
    h = z_top - z_bot
    cyl = trimesh.creation.cylinder(radius=radius, height=h, sections=sections)
    cyl.apply_translation([0.0, 0.0, z_bot + h / 2.0])
    return cyl


# ── Bell cavity builder ────────────────────────────────────────────────────────

_MIN_BELL_DEPTH_MM = 1.0   # below this the bell is considered flat / degenerate


def _build_bell_cavity(outer_r, outer_z_m, n_slices):
    """
    Build a watertight solid of revolution for the outer bell surface.

    The outer bell profile may be non-monotone in z (it dips below the
    attachment level before rising to the tip).  We handle this by splitting
    at the z-minimum and building up to two solids:

      upper solid  — from z_min upward along the ascending outer curve to tip
      lower pocket — from z_min back up to the first profile point (only when
                     a meaningful descent exists before z_min)

    Returns the union of these solids (or just the upper solid if monotone or
    if the pocket has negligible depth).
    """
    idx_min = int(np.argmin(outer_z_m))
    z_min   = outer_z_m[idx_min]

    # Guard: if the bell has essentially zero depth, just use a thin disc
    bell_depth = outer_z_m[-1] - z_min
    if bell_depth < _MIN_BELL_DEPTH_MM:
        # Degenerate bell — create a minimal cylinder to avoid zero-volume solid
        r_max  = float(outer_r.max()) if outer_r.max() > 0 else 1.0
        z_top  = max(outer_z_m[-1], z_min + _MIN_BELL_DEPTH_MM)
        return _revolve(np.array([
            [0.0,   z_min], [r_max, z_min],
            [r_max, z_top], [0.0,   z_top],
        ]), n_slices)

    # ── Upper solid (z_min → tip) ──────────────────────────────────────────
    # Closed profile (r, z):
    #   (0, z_min) → (outer_r[idx_min], z_min)  bottom cap
    #   → outer_curve[idx_min+1 … -1]            ascending wall
    #   → (outer_r[-1], outer_z_m[-1])           tip outer edge
    #   → (0, outer_z_m[-1])                     top cap to axis
    #   → close back to (0, z_min)
    up_r = np.concatenate([[0.0, outer_r[idx_min]], outer_r[idx_min + 1:], [0.0]])
    up_z = np.concatenate([[z_min, z_min          ], outer_z_m[idx_min + 1:], [outer_z_m[-1]]])
    upper = _revolve(np.column_stack([up_r, up_z]), n_slices)

    # If no descent before z_min, the profile is already monotone
    if idx_min == 0:
        return upper

    # Guard: skip the pocket if the height gain is negligible
    pocket_height = outer_z_m[0] - z_min
    if pocket_height < _MIN_BELL_DEPTH_MM:
        return upper

    # ── Lower pocket (z_min → outer_z_m[0]) ───────────────────────────────
    # Closed profile:
    #   (0, z_min)  → (outer_r[idx_min], z_min)   bottom cap at z_min
    #   → descending curve reversed: indices idx_min-1 down to 0 (z rising)
    #   → (outer_r[0], outer_z_m[0])
    #   → (0, outer_z_m[0])                        top cap to axis
    #   → close back to (0, z_min)
    lo_r = np.concatenate([[0.0, outer_r[idx_min]], outer_r[idx_min - 1::-1], [0.0]])
    lo_z = np.concatenate([[z_min, z_min          ], outer_z_m[idx_min - 1::-1], [outer_z_m[0]]])
    pocket = _revolve(np.column_stack([lo_r, lo_z]), n_slices)

    print("  Boolean: bell_upper ∪ bell_pocket …")
    return _bool("union", [upper, pocket])


# ── Inner core profile builder ─────────────────────────────────────────────────

def _build_core_body(inner_r, inner_z_m, outer_r_base, z_collar, n_slices):
    """
    Build the inner-core body solid.

    The core fills the subumbrellar void.  Its base is widened to outer_r_base
    at z = z_collar so it rests on the screw-pin tops.

    Profile (r, z_mould), closed polygon:
      A(0, z_collar)           axis at base
      B(outer_r_base, z_collar) wide base — rests on pin tops
      [step inward to inner_r[0] if inner_r[0] < outer_r_base]
      vertical rise from z_collar to inner_z_m[0] at inner_r[0]
        (some genomes have inner_z_m[0] > z_collar)
      inner bell curve inner_r[0..n-1] / inner_z_m[0..n-1]
      F(0, inner_z_m[-1])      axis at tip
      close back to A
    """
    pts_r = [0.0, outer_r_base]
    pts_z = [z_collar, z_collar]

    # Step inward to inner bell base if needed
    if outer_r_base > inner_r[0] + 0.05:
        pts_r.append(inner_r[0])
        pts_z.append(z_collar)

    # Vertical rise from z_collar to inner_z_m[0] at inner_r[0]
    if inner_z_m[0] > z_collar + 0.05:
        pts_r.append(inner_r[0])
        pts_z.append(inner_z_m[0])
        curve_start = 1        # index 0 already added above
    else:
        curve_start = 0

    core_r = np.concatenate([pts_r, inner_r[curve_start:], [0.0]])
    core_z = np.concatenate([pts_z, inner_z_m[curve_start:], [inner_z_m[-1]]])
    return _revolve(np.column_stack([core_r, core_z]), n_slices)


# ── Stats printer ──────────────────────────────────────────────────────────────

def _stats(label, path, mesh):
    vol = abs(mesh.volume) / 1e3
    wt  = "watertight" if mesh.is_watertight else "NOT watertight"
    print(f"  {label} → {path}")
    print(f"    bounds  : {mesh.bounds[0].round(1)} → {mesh.bounds[1].round(1)} mm")
    print(f"    volume  : {vol:.2f} cm³    triangles: {len(mesh.faces)}    [{wt}]")


# ── Outer mould ────────────────────────────────────────────────────────────────

def build_outer_mould(
    genome,
    diameter_mm  = 120.0,
    wall_mm      = 8.0,
    base_mm      = 10.0,
    tip_gap_mm   = 3.0,
    n_slices     = 120,
    screw_dia_mm = 3.3,
    remesh_mm    = None,
    output       = "output/mould_outer.stl",
):
    """
    Outer mould body: square-base block with bell cavity and screw-hole pins.

    Geometry
    --------
    Square side = outer_bell_tip_diameter + 2*wall_mm + 40 mm base flange.
    Total height = bell_tip_z + tip_gap_mm.
    Minimum wall at bell tip: wall_mm.  Base plate: base_mm.

    Void = union(payload_bore, bell_cavity_solid).
    Body = box − void.
    Then: body ∪ pin₁ ∪ pin₂  (screw-hole pillars).
    """
    outer, _ = get_bell_curves(genome, n_pts=80)
    s = _scale_factor(genome, diameter_mm)

    outer_rz  = outer * s
    payload_r = PAYLOAD_WIDTH  / 2.0 * s
    payload_h = PAYLOAD_HEIGHT * s
    z_collar  = base_mm + payload_h

    outer_r   = np.clip(outer_rz[:, 0], 0.0, None)
    outer_z_m = z_collar - outer_rz[:, 1]      # flipped; index 0 ≈ base, -1 = tip

    outer_cyl_r  = outer_r[-1] + wall_mm       # body radius at tip level
    sq_half      = outer_cyl_r + 20.0          # square half-side (+20 mm flange)
    total_height = outer_z_m[-1] + tip_gap_mm

    print(f"  payload bore   : ⌀{payload_r*2:.1f} mm  h={payload_h:.1f} mm")
    print(f"  outer r base→tip : {outer_r[0]:.1f} → {outer_r[-1]:.1f} mm")
    print(f"  z_collar       : {z_collar:.1f} mm")
    print(f"  bell z range   : {outer_z_m.min():.1f} → {outer_z_m[-1]:.1f} mm")
    print(f"  total height   : {total_height:.1f} mm")
    print(f"  square side    : {sq_half*2:.1f} mm")

    # ── Bell cavity solid ──────────────────────────────────────────────────────
    print("  Building bell cavity solid …")
    bell_cavity = _build_bell_cavity(outer_r, outer_z_m, n_slices)

    # ── Payload bore solid ─────────────────────────────────────────────────────
    # Extend 1 mm past z_collar so the union seam is clean.
    payload_bore = _cylinder(payload_r, base_mm, z_collar + 1.0, sections=n_slices)

    # ── Combined void ──────────────────────────────────────────────────────────
    print("  Boolean: payload_bore ∪ bell_cavity …")
    void = _bool("union", [payload_bore, bell_cavity])

    # ── Outer square block ─────────────────────────────────────────────────────
    box = trimesh.creation.box(extents=[sq_half * 2, sq_half * 2, total_height])
    box.apply_translation([0.0, 0.0, total_height / 2.0])

    # ── Subtract void from box ─────────────────────────────────────────────────
    print("  Boolean: box − void …")
    mould = _bool("difference", [box, void])

    # ── Screw-hole pins ────────────────────────────────────────────────────────
    # Solid cylinders in the collar void: z = base_mm → z_collar.
    # r = midpoint of collar annulus (between payload bore and bell outer base).
    # The pin tops (at z_collar) physically touch and support the inner core's
    # wide base plate.  In the cast silicone, each pin creates a screw through-hole.
    pin_r     = (payload_r + max(outer_r[0], payload_r)) / 2.0
    pin_h     = payload_h                     # spans base_mm → z_collar
    pin_ctr_z = base_mm + pin_h / 2.0

    print(f"  Screw pins     : ⌀{screw_dia_mm:.1f} mm  r={pin_r:.1f} mm  "
          f"z {base_mm:.0f}→{z_collar:.0f} mm")

    for angle in (0.0, np.pi):
        px = pin_r * np.cos(angle)
        py = pin_r * np.sin(angle)
        pin = trimesh.creation.cylinder(
            radius   = screw_dia_mm / 2.0,
            height   = pin_h,
            sections = 24,
        )
        pin.apply_translation([px, py, pin_ctr_z])
        print(f"  Boolean: mould ∪ pin @ ({px:+.1f}, {py:+.1f}) …")
        mould = _bool("union", [mould, pin])

    if remesh_mm is not None:
        print(f"  Remesh → {remesh_mm} mm target edge …")
        mould = remesh_isotropic(mould, remesh_mm)

    os.makedirs("output", exist_ok=True)
    mould.export(output)
    _stats("Outer mould", output, mould)
    return mould


# ── Inner core ─────────────────────────────────────────────────────────────────

def build_inner_core(
    genome,
    diameter_mm      = 120.0,
    base_mm          = 10.0,
    ring_overhang_mm = 15.0,
    ring_thick_mm    = 5.0,
    n_slices         = 120,
    remesh_mm        = None,
    output           = "output/mould_core.stl",
):
    """
    Inner core: fills the subumbrellar void during casting.

    Body
    ~~~~
    Revolved solid from the inner-bell profile, with a wide base disc at
    z = z_collar extending to base_r = max(inner_r[0], outer_r[0], pin_r + gap).
    This disc covers the screw-pin tops — the lower support seat.

    Ring flange
    ~~~~~~~~~~~
    Annular prism added at the top of the core body:
      inner radius = inner_r[-1]  (matches the core tip)
      outer radius = outer_r[-1] + ring_overhang_mm  (extends past mould rim)
      z range      = mould_rim_z − 0.5 mm (overlap) to mould_rim_z + ring_thick_mm

    The ring's lower face rests on the mould's top face (z ≈ outer_z_m[-1]),
    centring and supporting the core from above.

    Assembly:  inner_core = union(core_body, ring_solid)
    """
    outer, inner = get_bell_curves(genome, n_pts=80)
    s = _scale_factor(genome, diameter_mm)

    outer_rz  = outer * s
    inner_rz  = inner * s
    payload_h = PAYLOAD_HEIGHT * s
    z_collar  = base_mm + payload_h

    inner_r   = np.clip(inner_rz[:, 0], 0.0, None)
    inner_z_m = z_collar - inner_rz[:, 1]

    outer_r   = np.clip(outer_rz[:, 0], 0.0, None)
    outer_z_m = z_collar - outer_rz[:, 1]

    # Screw pin position (matches build_outer_mould)
    payload_r = PAYLOAD_WIDTH / 2.0 * s
    pin_r     = (payload_r + max(outer_r[0], payload_r)) / 2.0

    # Wide base radius: must reach pin centres so the core rests on them
    base_r = max(inner_r[0], outer_r[0], pin_r + 3.3)

    mould_rim_z  = outer_z_m[-1]              # Z of outer-bell tip = mould top face
    ring_inner_r = inner_r[-1]
    ring_outer_r = outer_r[-1] + ring_overhang_mm
    ring_bot_z   = mould_rim_z                # ring bottom flush with mould rim
    ring_top_z   = mould_rim_z + ring_thick_mm

    print(f"  z_collar       : {z_collar:.1f} mm")
    print(f"  base_r         : {base_r:.1f} mm  (pin_r={pin_r:.1f}, "
          f"inner_r[0]={inner_r[0]:.1f})")
    print(f"  inner z range  : {inner_z_m[0]:.1f} → {inner_z_m[-1]:.1f} mm")
    print(f"  mould rim Z    : {mould_rim_z:.1f} mm")
    print(f"  ring           : r {ring_inner_r:.1f}→{ring_outer_r:.1f} mm  "
          f"z {ring_bot_z:.1f}→{ring_top_z:.1f} mm")

    # ── Core body ─────────────────────────────────────────────────────────────
    core_body = _build_core_body(inner_r, inner_z_m, base_r, z_collar, n_slices)

    # ── Ring flange ────────────────────────────────────────────────────────────
    # Rectangular cross-section annular prism.
    # 0.5 mm z-overlap below ring_bot_z ensures a clean boolean union
    # (avoids a coincident-face artefact at the junction with the core tip).
    overlap = 0.5
    ring_profile = np.array([
        [ring_inner_r, ring_bot_z - overlap],
        [ring_outer_r, ring_bot_z - overlap],
        [ring_outer_r, ring_top_z          ],
        [ring_inner_r, ring_top_z          ],
    ])
    ring_solid = _revolve(ring_profile, n_slices)

    print("  Boolean: core ∪ ring …")
    inner_core = _bool("union", [core_body, ring_solid])

    if remesh_mm is not None:
        print(f"  Remesh → {remesh_mm} mm target edge …")
        inner_core = remesh_isotropic(inner_core, remesh_mm)

    os.makedirs("output", exist_ok=True)
    inner_core.export(output)
    _stats("Inner core", output, inner_core)
    return inner_core


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate silicone mould STLs for an evolved jellyfish bell",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Genome selection (mirrors make_cad.py)
    p.add_argument("--aurelia", action="store_true",
                   help="Use Aurelia aurita reference genome")
    p.add_argument("--gen", type=int, default=None,
                   help="Best genome from this generation (output/best_genomes.json)")

    # Physical scale
    p.add_argument("--diameter", type=float, default=120.0,
                   help="Target bell outer diameter (mm)")

    # Outer mould geometry
    p.add_argument("--wall", type=float, default=8.0,
                   help="Mould wall thickness around bell cavity (mm)")
    p.add_argument("--base", type=float, default=10.0,
                   help="Solid base-plate thickness below payload bore (mm)")
    p.add_argument("--tip-gap", type=float, default=3.0,
                   help="Clearance above bell-tip cavity (mm)")
    p.add_argument("--screw-dia", type=float, default=3.3,
                   help="Screw-hole pin diameter (mm): M3=3.3, M4=4.5")

    # Inner core geometry
    p.add_argument("--ring-overhang", type=float, default=15.0,
                   help="Ring extends this far past outer-bell-tip radius (mm)")
    p.add_argument("--ring-thick", type=float, default=5.0,
                   help="Ring flange thickness (mm)")

    # Mesh quality
    p.add_argument("--slices", type=int, default=120,
                   help="Angular slices for revolution meshes")
    p.add_argument("--remesh", type=float, default=None, metavar="MM",
                   help="Isotropic remesh target edge length (mm); skip if omitted")

    # Partial outputs
    p.add_argument("--outer-only", action="store_true",
                   help="Build outer mould only")
    p.add_argument("--core-only", action="store_true",
                   help="Build inner core only")

    args = p.parse_args()

    genome, tag = load_genome(args)
    s = _scale_factor(genome, args.diameter)
    print(f"Genome  : {tag}")
    print(f"Scale   : {s:.1f} mm/sim-unit  (⌀ {args.diameter} mm)\n")

    if not args.core_only:
        print("── Outer mould ──────────────────────────────────────────────")
        build_outer_mould(
            genome,
            diameter_mm  = args.diameter,
            wall_mm      = args.wall,
            base_mm      = args.base,
            tip_gap_mm   = args.tip_gap,
            n_slices     = args.slices,
            screw_dia_mm = args.screw_dia,
            remesh_mm    = args.remesh,
            output       = f"output/mould_outer_{tag}.stl",
        )
        print()

    if not args.outer_only:
        print("── Inner core ───────────────────────────────────────────────")
        build_inner_core(
            genome,
            diameter_mm      = args.diameter,
            base_mm          = args.base,
            ring_overhang_mm = args.ring_overhang,
            ring_thick_mm    = args.ring_thick,
            n_slices         = args.slices,
            remesh_mm        = args.remesh,
            output           = f"output/mould_core_{tag}.stl",
        )


if __name__ == "__main__":
    main()
