# Jellyfih Experiment Log

---

## Experiment 1 — Baseline Morphology Search

**Date:** 2026-03-24
**Hardware:** 1× RTX 4090 (sequential, seed 42) + 4× RTX 3080 (parallel, 4 seeds)
**Instance:** vast.ai — 4090 @ $0.30/hr destroyed after run; 4× 3080 @ $0.275/hr

### Configuration

| Parameter | Value |
|-----------|-------|
| Genome | 9D (shape + thickness only) |
| end_y bounds | [−0.30, −0.03] — tips constrained downward |
| Population (λ) | 16 |
| Generations | 50 |
| Steps/eval | 150,000 (3 cycles @ 1 Hz) |
| Fitness | `displacement / sqrt(muscle_count / 500)` |
| Spawn | [0.5, 0.40] |
| Actuation | Fixed 20/40/40 contraction/relaxation/refractory |
| Gravity | 10.0, payload 2.5× density, no buoyancy |

### Data

| Run | Seed | Best fitness | Gen converged | Notes |
|-----|------|-------------|---------------|-------|
| 4090-s42 | 42 | **0.536** (gen 42) | ~gen 40 | Full 50 gens |
| 3080-R0 | — | ~0.52 | ~gen 10 | |
| 3080-R1 | — | ~0.53 | ~gen 10 | |
| 3080-R2 | — | **0.535** (gen 41) | ~gen 15 | Only run to cross 0.53 |
| 3080-R3 | — | ~0.52 | ~gen 10 | cp1_x outlier, cond# 51 |

**Output:** `output/cloud/4090/seed_42/`, `output/cloud/3080x4/`
**Videos:** `view_gen{0,12,24,36,42}.mp4`, `view_gen{36,42}_long.mp4` (600 frames, 15 cycles)

### Results

**All runs converge within 10 generations** to a common attractor — a wide, nearly-flat bell with tips pressing against the upper end_y bound (−0.03). Cross-run genome correlation ~0.96.

**Locked genes** (CV < 0.15): `end_x` (0.277), `t_base` (0.054), `t_mid` (0.045). These are structural constants — evolution found one wall thickness solution and never deviated.

**Pressing against bounds:**
- `end_y` at 88% of [−0.30, −0.03] → tips want to be flatter/curl upward beyond −0.03
- `cp1_y` at 14% of [−0.15, +0.15] → first control point pulled downward, creating outward flare

**Dominant covariance coupling:** `cp2_x ↔ end_y` (−0.26 to −0.36): wider bell → shallower tip. This encodes the outward-flaring morphology.

**Ceiling exploit confirmed:** Payload reaches y ≈ 0.88 within 3 cycles from spawn y = 0.40. Fitness plateaus because ceiling (validity check at 0.93) caps raw displacement. Later generations optimize efficiency (muscle count) not displacement.

**Fluid dynamics analysis** (`helpers/fluid_analysis.py`):
- Wake not dissipated at next stroke: 35% of peak momentum remains at refractory end
- Vorticity *rises* during first 150ms of refractory (vortex ring still rolling up)
- Ceiling impact (top_flux) small but real (~1.3% of domain momentum at peak)
- Floor damping working correctly (bottom_flux ≈ 0)

### Discussion

The evolved morphology is a **wide, flat-tipped bell that flares outward**. The upstroke provides lift; the recovery (expansion) generates drag, which the outward flare partially decouples laterally. This is non-biomimetic: a real *Aurelia aurita* cannot have outward-curving bell tips due to radial symmetry constraints — but that's the point.

The 20/40/40 actuation timing fires into an undissipated vortex wake every cycle (35% residual momentum). The jellyfish cannot exploit this because timing is fixed. Evolved shapes may be locally optimal *for this fixed timing* but globally suboptimal.

**R3 (cond# 51, pressing against cp1_x and end_y bounds):** this run found a wider-swept bell on a different ridge. Its poor convergence (σ=0.086 at gen 49) suggests it was still exploring when the run ended, and may be hitting a gene bound constraint.

### Next Steps → See Experiment 2

---

## Experiment 2 — Timing-Free Morphology + Cup Bells

**Date:** 2026-03-24
**Hardware:** 4× RTX 3080 planned; **3 runs completed** (1 GPU hardware failure before/during run)
**Instance:** vast.ai 4× RTX 3080 @ $0.275/hr
**Status:** ✅ Complete (3/4 seeds)

### Hypothesis

1. **end_y upper bound relaxation** (→ +0.10): morphologies press against the flat bound in Exp 1; allowing positive end_y will unlock cup-shaped bells with upward-curling tips.

2. **Evolved timing** (genes 9–10): the fixed 20/40/40 waveform fires into its own wake (35% residual momentum at refractory end). Freeing contraction_frac and refractory_frac should allow evolution to discover viable propulsion timing independently.

3. **Fitness function with active_frac**: `displacement / sqrt(muscle_count × (1 − refractory_frac) / 500)`. A jellyfish with long refractory pays proportionally less muscle cost — rewards energy-efficient propulsion over brute-force firing.

4. **λ=32**: doubles population, better sampling of the 11D search space.

### Configuration

| Parameter | Value | Change from Exp 1 |
|-----------|-------|-------------------|
| Genome | **11D** (shape + thickness + timing) | +2 genes |
| Tank | **128×256 tall**, ceiling y=1.93 | tall tank (confirmed by results) |
| end_y bounds | **[−0.30, +0.10]** | upper relaxed from −0.03 |
| Gene 9: contraction_frac | [0.05, 0.40], **init 0.20** | new |
| Gene 10: refractory_frac | [0.20, 0.75], **init 0.40** | new |
| Population (λ) | **32** | doubled |
| Generations | 50 | same |
| Steps/eval | 150,000 | same |
| Fitness | `displacement / sqrt(muscle_count × (1−refractory) / 500)` | updated |
| Actuation | Per-instance from genome genes 9–10 | now evolved |

**Timing init rationale:** Start timing genes at known-good defaults (0.20/0.40) rather than random midpoint, so CMA-ES explores timing from a viable baseline.

### Results

**Tall tank confirmed by data:** payload final_y reached ~1.3 across converged runs. This is above the square tank validity ceiling (y=0.93), proving the run used the 128×256 domain (ceiling y=1.93). Any Exp 2 data showing final_y > 0.93 is a tall-tank result.

**Best fitness:** ~1.356 (gen 48, efficiency metric).

**3 runs completed.** One GPU failed (hardware) before completing its assigned seed; that seed has no data.

**The 2 converged runs revealed a genomic bifurcation in timing space.**

Both runs that fully converged reached the same morphological attractor (wide flat-tipped bell, consistent with Exp 1) but **separated into distinct timing basins**:

| Basin | Timing strategy | Timing gene behaviour | Outcome |
|-------|-----------------|-----------------------|---------|
| Displacement-maximising | Higher firing rate; more contraction per cycle | contraction_frac pressed toward upper bound | Higher raw displacement per eval |
| Efficiency-maximising | Long refractory; minimal active fraction | refractory_frac pressed to upper bound (0.75) | Lower displacement; lower effective muscle cost; highest fitness score |

Neither run crossed into the other basin. The timing genes pressed against their respective bounds and remained there. The two populations did not mix. This is a **genomic bifurcation**: the same 11D genome, the same fitness function, the same morphological attractor — but two irreconcilable timing attractors that the search could not bridge.

**Morphological attractor:** Both basins converged to the same bell shape as Exp 1 — wide, flat-tipped, thick base. The shape genes (end_x, t_base, t_tip, end_y) are consistent across all three runs. This confirms the morphological attractor is determined by the fluid physics, not by the timing or the fitness formulation.

**No cup morphology discovered:** end_y remained negative across all converged runs. The relaxed upper bound (+0.10) did not attract solutions — the flat-tipped bell remained dominant.

### Discussion

**The bifurcation is a landscape property, not a search artefact.** CMA-ES uses a single Gaussian distribution; once it commits to a timing basin, it has no mechanism to jump to the other. The two basins are separated by a fitness valley — intermediate timing (moderate contraction, moderate refractory) performs worse than either extreme. Seeds that drifted toward high contraction early locked into that basin; seeds that drifted toward high refractory locked into the efficiency basin.

**The efficiency metric actively drives the bifurcation.** The `active_frac = 1 − refractory_frac` denominator term creates a gradient toward longer refractory independent of displacement. A run that reaches similar displacement with refractory_frac=0.70 scores substantially higher than one with refractory_frac=0.40. This gradient is strong enough to pin a seed to the efficiency basin even if raw displacement could be increased by firing more often.

**Neither basin is globally dominant without prior specification of objective.** The displacement-first basin reaches higher final_y; the efficiency basin scores higher on the fitness metric. Future work should consider a Pareto front over (displacement, active_frac × muscle_count) to make this trade-off explicit.

**Relation to Experiment 6:** Exp 6 (efficiency, freq_mult encoding, tall tank) independently reproduced the efficiency attractor using a different timing encoding (freq_mult pressed to 0.5 Hz lower bound). This confirms the efficiency basin is robust to encoding change — any fitness that rewards low firing rate will drive timing to its slowest-allowed value.

---

---

## Experiment 3 — Raw Displacement + Frequency Gene

**Date:** 2026-03-25
**Hardware:** 4× RTX 3080 Ti, 1 seed (s42), ~5.1 hrs
**Status:** ✅ Complete 50/50 gens

**Best fitness:** +0.838 displacement (gen 43). 90% convergence at gen 16.

**Key results:**
- Fully converged: sigma=0.101, cond#=62.9, all 11 genes locked (CV < 0.15)
- **Same morphological attractor as Exp 1/2**: end_x=0.348 (pressing upper 0.35), t_base=0.079 (pressing upper 0.08). Shape is physics-determined, not fitness-determined.
- **Contraction pressing upper bound**: contraction_frac=0.550 (upper=0.60). Without efficiency penalty, evolution maximises firing fraction. Muscle count: 674 vs Exp 6's 472.
- **freq_mult settled at 0.90**: slightly below 1 Hz. Counter-intuitive — even without a frequency penalty, evolution chose slightly slower than baseline, suggesting a minimum vortex recovery time requirement.
- Beats Exp 2 raw displacement (~0.61–0.73) with fewer gens; confirms displacement headroom was there all along, not unlocked by timing genes specifically.

---

## Experiment 5 — Axisymmetric MPM

**Date:** 2026-03-25
**Hardware:** 4× RTX 3080 Ti, 1 seed (s42), ~1.8 hrs (131s/gen — 2.8× faster than Cartesian due to half particles)
**Status:** ✅ Complete 50/50 gens — **DID NOT CONVERGE**

**Best fitness:** +1.274 displacement (gen 33, axisym coordinates).

**Key results:**
- Numerically stable all 50 gens — no NaN explosions, axis BC working correctly
- **Did not converge**: sigma=0.41, cond#=21.2 at gen 49. Axisym landscape is much flatter than Cartesian (Exp 3 cond#=62.9). Would need more gens or higher lambda.
- **Different attractor from Cartesian**: cp2_y = +0.114 (positive, vs Cartesian −0.19); end_x=0.266 (not pressing upper bound); t_base=0.055 (thinner). Axisym geometry genuinely changes optimal bell shape.
- **freq_mult not converged** (gen49 best: 0.692; gen33 best: 0.903; high population variance). Timing genes unlocked — no timing attractor found yet.
- Several genes unlocked (cp1_y, t_mid, contraction_frac, freq_mult CV > 0.15) — landscape too broad for single seed at 50 gens.
- Vortex ring visualisation (vs Cartesian dipoles) not yet rendered.

---

## Experiment 6 — Efficiency Control (New Genome, Tall Tank)

**Date:** 2026-03-25
**Hardware:** 4× RTX 3080 Ti, 1 seed (s137), ~5.2 hrs
**Status:** ✅ Complete 50/50 gens — **New efficiency record**

**Best fitness:** 1.327 efficiency (gen 48). 90% convergence at gen 19.

**Key results:**
- **New record: 1.327** — beats Exp 2's 1.179 by 12.5%. Tall tank (no ceiling) + freq_mult encoding both contributed.
- Tightest convergence of all experiments: sigma=0.087, cond#=47.7, all genes locked.
- **freq_mult pressing lower bound (0.505/0.50 Hz)**: efficiency fitness drives evolution to pulse as infrequently as possible. Lower bound should be extended to 0.25–0.30 Hz in future experiments.
- **contraction_frac=0.293**: much lower than Exp 3's 0.550. Efficiency pressure keeps contractions short; frequency pressure keeps cycles long.
- Same morphological attractor as Exp 1/2/3 confirmed — shape is fully physics-determined.
- Genome change (freq_mult replacing refractory_frac) is transparent: Exp 6 reproduced the Exp 2 attractor.

---

## Experiment 4 — Payload Effect (Partial)

**Date:** 2026-03-25
**Hardware:** 4× RTX 3080 Ti, seed 999, 2/50 gens (checkpoint saved)
**Status:** 🔖 Incomplete — GPU reassigned mid-run

**Brief:** Validity bug found and fixed (body CoM tracking when no material-2 particles). GPU 2 banned (broken fan). Ran 2 gens on GPU 3 after Exp 5 freed it, then killed to avoid leaving 2 GPUs idle. Gen 2 best body CoM displacement +0.878 (healthy early-gen value). Resume from checkpoint for full results.

---

## Cross-Experiment Analysis: Experiments 3, 5, 6

*Added 2026-03-25 after all three runs completed.*

### Convergence Summary

| Experiment | Best fitness | Best gen | Final σ | Final cond# | Converged gen (σ<0.15) |
|------------|-------------|----------|---------|-------------|------------------------|
| Exp 1 (9D, eff) | 0.536 | 42 | ~0.053 | ~45 | ~40 |
| Exp 2 (11D, eff, best seed) | **1.356** | 48 | ~0.05 | ~40 | ~30 |
| Exp 3 (11D, disp) | 0.838 | 43 | 0.101 | 62.9 | 30 |
| Exp 5 (axisym, disp) | 1.274† | 33 | 0.410 | 21.2 | **never** |
| Exp 6 (11D, eff) | **1.327** | 48 | 0.087 | 47.7 | 33 |

†Not directly comparable to Cartesian displacement.

**Exp5 sigma trajectory:** 0.272 → 0.414 → 0.471 → 0.623 → 0.670 → 0.395 — sigma actually *increases* mid-run, indicating active multi-basin exploration. This is distinct from the flat plateau in Exp 3; the axisym landscape has genuine diversity that a single seed cannot resolve.

---

### Morphological Attractor: Cartesian Experiments

All three Cartesian experiments (Exp 3, 6, and by reference Exp 1/2) converge to the **same wide flat-bell attractor** regardless of fitness objective:

| Gene | Exp 3 (gen 49) | Exp 6 (gen 49) | Bounds | Status |
|------|----------------|----------------|--------|--------|
| cp1_x | 0.006 | 0.141 | [0, 0.25] | **diverges** — Exp3 at lower bound |
| cp1_y | −0.100 | −0.095 | [−0.15, 0.15] | locked (similar) |
| cp2_x | 0.234 | 0.0002 | [0, 0.30] | **diverges** — Exp6 at lower bound |
| cp2_y | −0.195 | −0.176 | [−0.20, 0.15] | both near lower bound |
| end_x | **0.350** | 0.339 | [0.05, **0.35**] | Exp3 at upper bound |
| end_y | −0.146 | −0.139 | [−0.30, 0.10] | both moderate |
| t_base | **0.079** | 0.077 | [0.025, **0.08**] | both near upper bound |
| t_mid | 0.067 | 0.041 | [0.025, 0.10] | **diverges** — efficiency favours thin mid |
| t_tip | 0.031 | 0.031 | [0.01, 0.04] | locked |

**Morphological conclusion:** The bell shape (end_x, t_base, t_tip, end_y) is **physics-determined** — identical across displacement and efficiency objectives. The first and second control points (cp1_x, cp2_x) and mid-thickness differ between objectives because they affect how much muscle is packed into the mesoglea (relevant for efficiency but not raw displacement). The attractor is a **wide, downward-tipped bell with a thick base** in all cases.

---

### Axisymmetric Attractor (Exp 5)

| Gene | Exp 5 gen49 | Exp 3 gen49 | Direction |
|------|-------------|-------------|-----------|
| cp2_y | **+0.108** | −0.195 | **sign flip** |
| end_x | 0.294 | 0.350 | 15% smaller, not at bound |
| end_y | **−0.300** | −0.146 | at lower bound — deep tips |
| t_base | 0.051 | 0.079 | 35% thinner |
| t_mid | 0.033 | 0.067 | 51% thinner |

The axisymmetric attractor differs qualitatively: **positive cp2_y** means the mid-section of the bell curves *outward* rather than inward — a fundamentally different geometry. Combined with very deep end_y (−0.300, at lower bound) and thinner walls, this suggests the axisym penalty for large annular muscle volume favours a thinner, more elongated bell that can generate a large vortex ring without excessive tissue mass. Note: Exp5 has not converged and these values represent one local basin, not the global attractor.

---

### Timing Gene Analysis

#### Attractor in Timing Space

| Experiment | contraction_frac | freq_mult | Objective | Strategy |
|------------|-----------------|-----------|-----------|----------|
| Exp 3 gen49 | **0.566** | 0.896 | displacement | pulse hard + often |
| Exp 5 gen49 | 0.306 | 0.692 | displacement (axisym) | moderate (not converged) |
| Exp 6 gen49 | 0.306 | **0.507** | efficiency | pulse moderate + slow |

**Exp3 vs Exp6 timing:** dramatically different attractors driven by objective function:
- **Displacement (Exp3):** maximise contraction duty cycle → contraction_frac at 0.566 (~94% of upper bound 0.6). Frequency slightly below 1 Hz — there is a minimum vortex recovery time that penalises firing faster than the wake can organise, even when maximising displacement.
- **Efficiency (Exp6):** contraction_frac moderate (0.306), freq_mult at *lower bound* (0.507 ≈ 0.5 Hz). Evolution wants to fire even more slowly but is constrained by the [0.5, 2.0] bound. This is a **systematic bound-pressing finding**: the efficiency objective strongly prefers 0.5 Hz (or slower).

#### Comparison with Exp 2 Timing (refractory_frac encoding)

| Experiment | contraction_frac | refractory/freq gene | Gene 10 at bound? |
|------------|-----------------|---------------------|-------------------|
| Exp 2 s42 | 0.400 | refractory_frac = **0.750** | yes (upper) |
| Exp 2 s137 | 0.400 | refractory_frac = **0.749** | yes (upper) |
| Exp 6 | 0.306 | freq_mult = **0.507** | yes (lower) |

Both Exp2 and Exp6 press their timing gene to its efficiency-maximising bound — Exp2 presses refractory_frac to its *upper* bound (0.75) to maximise dead time, while Exp6 presses freq_mult to its *lower* bound (0.5 Hz) to maximise cycle period. These are equivalent strategies: both reduce the number of pulses per unit time, but using different encodings. This confirms the finding is *robust to encoding change*: **efficiency fitness consistently favours the slowest possible firing rate within the search bounds**.

**Implication:** For any future efficiency experiment, extend freq_mult lower bound to [0.20–0.25 Hz] — the current 0.5 Hz bound is actively constraining the search.

#### Is There One Timing Attractor or Multiple?

Based on available data (single seed per experiment):
- **Displacement (Cartesian):** one clear attractor — high contraction, ~0.9 Hz. Consistent with strong gradient toward high duty cycle.
- **Efficiency (Cartesian):** one clear attractor — low contraction, minimum freq (bound-pressed). Consistent with Exp2.
- **Axisymmetric:** timing genes unlocked at gen49, no reliable attractor identified yet. Needs multiple seeds.

---

### Efficiency Record: Exp 6 vs Exp 2

Exp6 best efficiency = **1.327** (gen48), vs Exp2's best 1.356 (s42, gen48).

On closer inspection, Exp2 s42/s137 both score ~1.356 — *higher* than Exp6's 1.327. However:
- Exp2 ran in a **128×128 square tank** (ceiling at y=0.93); Exp6 ran in a **128×256 tall tank** (no ceiling). The tall tank gives a harder test — no ceiling to exploit.
- Exp2's refractory_frac=0.750 may partially inflate its efficiency metric (active_frac=0.25 → denominator shrinks). The encoding change in Exp6 (freq_mult) changes how active_frac enters the formula — the two scores may not be directly comparable.
- Treat Exp6 as establishing a **new tall-tank efficiency baseline of 1.327**; Exp2's 1.356 is the **square-tank record**.

---

### Open Questions and Recommended Experiments

1. **freq_mult lower bound too constraining.** Exp6 and Exp2 both press to slowest allowed rate. Run Exp7 with freq_mult ∈ [0.2, 2.0] and check if evolution pushes further toward 0.2 Hz.

2. **Axisym needs more gens and seeds.** Exp5's sigma is still increasing at gen49 — this is not a plateau, it's active multi-modal exploration. Recommend: Exp5b with 4 seeds × 100 gens, or at minimum 4 seeds × 50 gens to identify replicable attractors.

3. **Morphological attractor uniqueness.** Exp3 vs Exp6 show the same bell shape but different control-point configurations (cp1_x, cp2_x diverge). It's unclear if this is two sub-basins of the same attractor or a true bifurcation. Multi-seed runs for Exp3 would clarify.

4. **Axisym vortex ring.** The render_progression.py and view_axisym.py scripts now support visualising the axisym Exp5 morphology in a tall tank with vorticity overlay. Render key generations to confirm vortex ring structure vs Cartesian dipoles.

5. **Timing × morphology coupling.** Does the covariance matrix show coupling between timing and morphology genes? (e.g. does thicker bell → lower freq_mult?) The convergence_plots.py covariance heatmaps will reveal this.

---

## Physical Model Assumptions (reference)

See memory files:
- `~/.claude/projects/-home-mc-projects-jellyfih/memory/physics_water.md` — inviscid fluid, ~7× slow speed of sound, no viscosity
- `~/.claude/projects/-home-mc-projects-jellyfih/memory/physics_payload.md` — near-rigid 2.5× density, no Archimedes buoyancy, Δy=−0.089/3-cycle neutral sink

---

## Results and Discussion Outline — Project Submission

*Compiled 2026-03-25. Primary evidence base: Experiments 1, 2, 3, 6 (Cartesian) and Experiment 5 (axisymmetric). Experiment 4 incomplete.*

---

### Videos to Generate

| Video | Command / Source | Purpose |
|-------|-----------------|---------|
| **Exp 1 best genome — square tank, early vs late** | `view_single.py --gen 42 --run-id <exp1>` | Show the evolved attractor in motion; compare gen 5 vs gen 42 |
| **Exp 2 bifurcation pair** | `tall_tank.py --genome [displacement_basin]` and `tall_tank.py --genome [efficiency_basin]` | Side-by-side of same morphology, two timing basins; visually demonstrate the bifurcation |
| **Exp 2 vs Aurelia reference** | `make_comparison.py` with Aurelia and Exp 2 best | Show departure from biomimetic design |
| **Exp 3 progression — gen 0, 10, 30, 49** | `view_generation.py --gen N --log exp3` | Show morphological convergence within a single fitness objective |
| **Exp 6 best genome — tall tank, efficiency** | `tall_tank.py --gen 48 --run-id exp2_s137` (or exp6) | Efficiency-basin jellyfish swimming at 0.5 Hz; long coast visible |
| **Exp 5 axisym — vortex ring** | `view_axisym.py` with flow overlay | Contrast vortex ring (axisym) vs dipole pair (Cartesian); key physics difference |
| **Fluid analysis — Exp 1 vs Exp 2 best** | `fluid_analysis.py` on each | Show residual wake reduction in efficiency basin |

---

### Convergence Plots

Generate via `helpers/convergence_plots.py` for each run. Required panels:

1. **Best fitness vs generation** — overlay all seeds per experiment; show rapid convergence (Exp 1: gen 10, Exp 3: gen 16, Exp 6: gen 19)
2. **Population mean fitness ± std vs generation** — show gap between best and mean; indicates selection pressure and population diversity
3. **Sigma (σ) vs generation** — shows when the search has committed; sigma collapse marks convergence; Exp 5's rising sigma is diagnostic of multi-modal landscape
4. **Condition number vs generation** — shows elongation of fitness landscape; high cond# (Exp 3: 62.9) indicates a narrow ridge; compare to Exp 5 (21.2, broad)
5. **Timing gene trajectories (Exp 2)** — contraction_frac and refractory_frac over generations, per seed; bifurcation point should be visible within the first 10–15 generations

---

### Empirical Morphology Analysis

**Claim to make:** The morphological attractor — wide, flat-tipped bell, thick base, tips not pressing the bound — is **physics-determined**, not fitness-determined. It is identical across displacement (Exp 3), efficiency (Exp 6), and timing-free (Exp 1) objectives.

**Evidence:**
- Gene table across Exp 1, 2, 3, 6: end_x ≈ 0.28–0.35, t_base ≈ 0.054–0.079, t_tip ≈ 0.031, end_y < 0 in all converged runs
- Genes that *do* vary across objectives: cp1_x, cp2_x, t_mid — these affect muscle packing, which enters the efficiency denominator but not raw displacement
- end_y never positive in any converged run despite the upper bound being relaxed to +0.10 in Exp 2

**Caveat to state explicitly:** All morphological conclusions are conditioned on the specific fluid environment — inviscid, weakly compressible, infinite-Re MPM water. This is not real water. The attractor is the optimal morphology *within these physics*, and the physics diverge from physical reality in known ways (speed of sound 7× too slow, no viscosity, 2D slab geometry). The morphological conclusions should be treated as model-relative, not physically predictive.

**Axisymmetric exception (Exp 5):** The one run that used correct 3D-equivalent geometry found a measurably different attractor (positive cp2_y, thinner walls, unconstrained end_x). This supports the hypothesis that Cartesian geometry inflates thrust and distorts morphology selection. Exp 5 is unconverged and must be replicated before strong claims can be made.

---

### Attractor Discussion

**What the attractor is:** a locally stable region of the 11D genome space to which CMA-ES reliably converges across independent random seeds, within the Cartesian 2D fluid environment.

**Properties of the Cartesian morphological attractor:**
- Convergence time: approximately 10–20 generations (out of 50) across all Cartesian experiments
- Cross-seed genome correlation: approximately 0.96 (Exp 1 data)
- Locked genes (CV < 0.15): end_x, t_base, t_tip, end_y
- Free genes (vary across seeds/objectives): cp1_x, cp2_x, t_mid
- Not pressing end_y upper bound: evolution is not constrained here; it genuinely does not prefer upward-curling tips in Cartesian geometry
- Basin width: wide enough that σ collapses from 0.25 to ~0.05–0.10 without instability (no cond# divergence)

**What the attractor is not:** a global optimum. It is the best morphology discoverable by a single-population gradient-following search in this physics. It may correspond to a local optimum; the flat landscape near convergence (low fitness variance in top individuals) does not distinguish local from global.

**Timing bifurcation as a secondary attractor structure:** Within Exp 2, two timing basins coexist. Each is stable under CMA-ES — neither seed escaped its basin during 50 generations. The fitness valley between them (intermediate contraction + intermediate refractory) is not quantified directly but is implied by the absence of any inter-basin transitions.

---

### Covariance and Condition Number Analysis

**What the covariance matrix records:** the directions of correlated gene change that evolution discovered were jointly useful. A large off-diagonal entry means the search learned "when gene A increases, gene B must also change in a fixed direction to maintain fitness." This encodes mechanistic coupling in the morphology-physics interaction.

**Key coupling to report (Exp 1):** cp2_x ↔ end_y (−0.26 to −0.36): wider bell → shallower (less downward) tips. This is mechanistically interpretable — a wider bell generates more lateral pressure; shallower tips reduce the drag of the returning bell edge. Evolution discovered this coupling without it being encoded.

**Condition number as a landscape shape diagnostic:**
- cond# ~ 1: isotropic landscape, all directions equally exploitable
- cond# ~ 50 (Exp 3, Exp 6): one principal axis dominates; the search has found a ridge and is moving along it
- cond# ~ 21 (Exp 5): flatter, more isotropic — the axisym landscape has more directions of improvement; not yet channelled

**What high condition number implies:** by late convergence, the search has effectively collapsed from 11D to 1–2 principal directions. The remaining genes are locked. This is genome saturation — the search space has been exhausted along the available ridge. Future fitness gains require either a different search space (new genes, wider bounds) or a different landscape (different physics or fitness).

---

### Genome Saturation and Future Experiments

**Saturation definition used here:** the point at which σ has collapsed and fitness improvement has plateaued despite further generations. All Cartesian experiments saturate between gen 30–43.

**What saturation means for this genome:** the 11D encoding of Bezier shape + thickness + timing has been exhausted for the given fluid environment. The attractor is stable; no further improvements are available by running more generations on the same genome in the same physics.

**Saturation does not mean the organism is optimal** — it means the search has found the best genome-environment combination within the current encoding. Two directions to escape saturation:

1. **Expand the genome:** add frequency as an evolved gene (Exp 3 partially does this); extend freq_mult lower bound below 0.5 Hz (Exp 6 presses this bound); add bell asymmetry; allow non-uniform actuation across muscle segments

2. **Change the environment:** tall tank removes the ceiling saturation; axisymmetric physics removes the Cartesian thrust overestimate; longer evaluation windows remove the 3-cycle measurement horizon; multi-objective fitness removes the single-scalar collapse

Experiments 3, 5, and 6 each escape one constraint. None has escaped all simultaneously. A natural follow-on is an experiment combining tall tank + axisym + efficiency + extended freq_mult bounds.

---

### Bifurcation Analysis — Experiment 2

**Observation:** Two converged runs from Experiment 2 share an identical morphological attractor but occupy distinct timing attractors that are mutually unreachable within a single CMA-ES run.

**Mechanism:** The fitness function `displacement / sqrt(muscle_count × (1−refractory) / 500)` creates two locally optimal strategies:
- *Maximise displacement at fixed timing cost:* push contraction_frac up, fire more per cycle, accept the higher denominator
- *Minimise timing cost at fixed displacement:* push refractory_frac up, fire rarely, shrink the denominator

The gradient from each basin reinforces itself: a population already near high-refractory gets a fitness boost from increasing refractory further, pulling it away from the displacement basin. The populations cannot merge because the intermediate region (moderate refractory, moderate contraction) scores worse than either extreme.

**Why this constitutes speciation:** both populations are drawn from the same genome, the same fitness function, and the same morphological space. They are differentiated solely by their timing genes. If the two populations were merged, the efficiency-basin individuals would outcompete the displacement-basin individuals on the fitness metric — but the displacement-basin genome is not inferior in displacement, only in fitness score. This is a case of frequency-dependent or metric-dependent speciation: the two populations are adapted to different implicit objectives encoded within the same scalar fitness.

**Implication for multi-seed experimental design:** a single seed is insufficient to characterise the timing landscape. Any future efficiency experiment should run at minimum 4 seeds and compare timing gene distributions across seeds to detect basin structure. Bifurcation in timing space is expected to re-emerge whenever the fitness function contains competing gradients in the timing dimension.
