# Experiment 1 — Baseline Morphology Search

**Status:** Complete
**Date:** 2026-03-24
**Code commit:** `3c5ad12` — Rendering overhaul, morphology fixes, physics corrections
**Hardware:** 1× RTX 4090 (seed 42, sequential) + 4× RTX 3080 (seeds R0–R3, parallel)
**Instance:** vast.ai — 4090 @ $0.30/hr; 4× 3080 @ $0.275/hr
**Output:** `output/cloud/4090/seed_42/`, `output/cloud/3080x4/`

---

## Hypothesis

Evolutionary search over bell morphology alone (shape + thickness) will discover non-biomimetic geometries optimised for upward payload transport in 2D MPM water.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Genome | 9D — Bezier shape (6) + thickness (3) |
| end_y bounds | [−0.30, −0.03] — tips constrained downward |
| Population (λ) | 16 |
| Generations | 50 |
| Steps/eval | 150,000 (7.5 cycles @ 1 Hz) |
| Fitness | `displacement / sqrt(muscle_count / 500)` |
| Spawn | [0.5, 0.40] |
| Actuation | Fixed 20/40/40 contraction/relaxation/refractory |
| Gravity | 10.0, payload 2.5× water density, no buoyancy |
| Tank | 128×128 (square), ceiling cap y=0.93 |

---

## Results

| Run | Seed | Best fitness | Gen of best | Notes |
|-----|------|-------------|-------------|-------|
| 4090-s42 | 42 | **0.536** | 42 | Full 50 gens |
| 3080-R0 | — | ~0.52 | ~10 | |
| 3080-R1 | — | ~0.53 | ~10 | |
| 3080-R2 | — | **0.535** | 41 | Only run to cross 0.53 |
| 3080-R3 | — | ~0.52 | ~10 | cp1_x outlier, cond# 51 |

**All 4 runs converge to the same attractor within 10 generations.** Cross-run genome correlation ~0.96.

### Morphology attractor

- Wide, nearly-flat bell with tips pressing against upper end_y bound (−0.03)
- **Locked genes** (CV < 0.15): `end_x` ≈ 0.277, `t_base` ≈ 0.054, `t_mid` ≈ 0.045
- `end_y` at 88% of bounds range → tips want to be flatter/curl upward beyond −0.03
- `cp1_y` pulled downward → outward flare at bell margin
- **Dominant coupling:** `cp2_x ↔ end_y` (−0.26 to −0.36): wider bell → shallower tip

### Ceiling exploit

Payload reaches y ≈ 0.88 within 3 cycles from spawn y = 0.40. Fitness plateaus after ~gen 10 because ceiling cap (y=0.93) limits raw displacement. Later generations optimise efficiency (muscle count), not displacement.

### Fluid dynamics

From `helpers/fluid_analysis.py` on best genome:
- 35% of peak momentum remains at refractory start — wake not dissipated before next stroke
- Vorticity rises during first 150ms of refractory (vortex ring still rolling up)
- Ceiling impact (top_flux) real but small (~1.3% of domain momentum at peak)
- Floor damping working correctly

---

## Discussion

The attractor is a **wide outward-flaring bell** — non-biomimetic relative to *Aurelia aurita*. The shape optimises the downstroke for maximum thrust while the fixed timing fires into its own undissipated wake (35% residual momentum). The morphology may be locally optimal *for fixed 20/40/40 timing* but globally suboptimal.

**R3** (cond# 51, pressing against cp1_x and end_y bounds): found a wider-swept bell on a different ridge. High condition number and low σ at gen 49 suggests it was still exploring and likely would converge to the same attractor with more generations.

**Bound-pressing on end_y** is the dominant signal: evolution wants upward-curling tips beyond −0.03 but cannot reach them. This directly motivates relaxing the end_y upper bound in Experiment 2.

---

## Open questions carried to Exp 2

1. Does relaxing end_y unlock cup-shaped bells with qualitatively different propulsion?
2. Would evolved timing find the same morphology or unlock coupling between shape and timing?
3. Is 35% residual wake momentum a fundamental property of the morphology or the fixed timing?
