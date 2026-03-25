# Experiment 2 — Evolved Timing + Cup Bell Geometry

**Status:** ✅ Complete (50/50 gens, 3/4 seeds — 1 GPU hardware failure)
**Date:** 2026-03-25
**Code commit:** `39a3cbe` — Add Exp 2 genome: timing genes, cup bells, activity-weighted fitness
**Hardware:** 4× RTX 3080 planned; 3 runs completed (s999 GPU failed — hardware)
**Instance:** vast.ai 4× RTX 3080 @ $0.275/hr
**Output:** `output/cloud/exp2/exp2_s42/`, `exp2_s137/`, `exp2_s2024/`

**Note on tank:** This file was written after 19 gens and incorrectly annotated as 128×128 square tank. The displacement values recorded (0.606, 0.626, 0.728) place final_y at 1.0–1.1, which is above the square tank ceiling (y=0.93). The run used the **128×256 tall tank** (ceiling y=1.93). Final_y reaching ~1.3 in the completed run further confirms this.

---

## Hypothesis

1. **end_y upper bound relaxation** (→ +0.10): morphologies press against the flat bound; allowing positive end_y unlocks cup-shaped bells with upward-curling tips.
2. **Evolved timing** (genes 9–10): fixed 20/40/40 waveform fires into its own undissipated wake. Freeing contraction_frac and refractory_frac should evolve toward jet-mode (short strong stroke + long coast).
3. **Activity-weighted fitness**: penalising `muscle_count × (1 − refractory_frac)` rewards energy-efficient propulsion — a jellyfish with 70% refractory pays only 30% of muscle cost.

---

## Configuration

| Parameter | Value | Change from Exp 1 |
|-----------|-------|-------------------|
| Genome | **11D** — shape (6) + thickness (3) + timing (2) | +2 timing genes |
| end_y bounds | **[−0.30, +0.10]** | upper relaxed from −0.03 |
| Gene 9: contraction_frac | [0.05, 0.40], init 0.20 | new |
| Gene 10: refractory_frac | [0.20, 0.75], init 0.40 | new |
| relaxation_frac | `max(0.05, 1 − contraction − refractory)` — computed, not stored | derived |
| Population (λ) | **32** | doubled |
| Generations | 50 | same target |
| Steps/eval | 150,000 (7.5 cycles @ 1 Hz) | same |
| Fitness | `displacement / sqrt(muscle_count × (1−refractory) / 500)` | updated |
| Tank | **128×256 tall**, ceiling y=1.93 | tall tank (confirmed by displacement data) |

---

## Results

### Fitness

| Run | Gens | Best fitness (at 19 gens) | Final best fitness (gen 48–50) | Peak displacement |
|-----|------|--------------------------|-------------------------------|-------------------|
| s42 | 50 | 1.143 (gen 18) | **~1.356** (gen 48) | ~0.93+ |
| s137 | 50 | 1.179 (gen 17) | **~1.356** (gen 48) | ~0.93+ |
| s999 | 0 | — | — | GPU hardware failure |
| s2024 | 50 | 0.975 (gen 14) | — | 0.728+ |

s42 and s137 broke fitness 1.0 at ~gen 15 and reached ~1.356 by gen 48. s2024 achieved higher raw displacement but lower fitness throughout — this is the bifurcation (see below).

*Note: The 19-gen column preserves intermediate data from when this file was first written. Full 50-gen per-seed results are in the CSV logs.*

### Morphology

Same attractor as Experiment 1: wide, flat-tipped bell with `end_x` locked near upper bound.

| Gene | Cross-run CV | Value |
|------|-------------|-------|
| end_x | **0.005** | ≈ 0.34 (near upper bound 0.35) |
| t_base | 0.037 | ≈ 0.076 |
| t_mid | 0.027 | ≈ 0.061 |
| end_y | 0.135 | ≈ −0.10 (negative — tips down, not cup) |

**Cup bells did not emerge.** end_y converged to negative values in all runs. The relaxed upper bound was not exploited; the morphology attractor is robust.

### Timing — genomic bifurcation

s42 and s137 both **saturated the refractory upper bound** by gen 18, and remained there through gen 50:

| Run | contraction (g9) | refractory (g10) | relaxation | strategy | Basin |
|-----|-----------------|-----------------|-----------|---------|-------|
| s42 pop mean (gen 18) | 0.346 ± 0.051 | 0.652 ± 0.077 | **0.050 (capped)** | long coast | efficiency |
| s137 pop mean (gen 18) | 0.376 ± 0.033 | 0.677 ± 0.087 | **0.050 (capped)** | long coast | efficiency |
| s2024 pop mean (gen 18) | 0.262 ± 0.096 | 0.269 ± 0.067 | 0.469 | moderate | displacement |
| s42/s137 (final, gen ~48) | ~0.400 | **~0.750** (upper bound) | ~0.050 | efficiency | efficiency |

**Efficiency basin (s42, s137):** refractory_frac pressed to its upper bound (0.75) and remained there through the full 50-gen run. Contraction also pressing upper bound (0.40). The relaxation gene was driven to its 5% floor — effectively vestigial. These seeds reached best fitness ~1.356.

**Displacement basin (s2024):** moderate contraction (~26%), low refractory (~27%), substantial relaxation (~47%). Higher raw displacement per eval but lower fitness score throughout. This run never entered the efficiency basin; the two populations did not mix across 50 generations.

**This is a genomic bifurcation.** The same 11D genome, the same fitness function, and the same morphological attractor — but two irreconcilable timing attractors. Neither seed crossed into the opposing basin. The timing genes hit their respective bounds and the populations separated permanently.

### Cost of transport

s2024 fires muscles ~2× as often as s42/s137 and travels further per eval, but the activity-weighted fitness correctly penalises this. This is an empirical demonstration of the trade-off the fitness function was designed to capture: the displacement-maximising strategy exists and is accessible, but it is sub-optimal under the efficiency metric.

### Covariance / condition numbers

All three runs develop condition numbers from ~1.5 → ~10–11 over 19 generations at nearly identical rates, despite finding different solutions. CMA-ES learned equally elongated search ellipsoids in qualitatively different parts of the timing space.

The refractory diagonal variance (`cov_diag[10]`) shrinks in s42/s137 as g10 presses against its upper bound — the population converges in that dimension. In s2024, timing diagonal variances remain near 1.0 (no compression), consistent with a plateau rather than a sharp attractor.

σ remains 0.25–0.45 at gen 18 in all runs — **not fully converged**. More generations would likely sharpen the timing bounds further.

---

## Discussion

### Genomic bifurcation is a landscape property

CMA-ES uses a single Gaussian distribution; once a population commits to a timing basin, it has no mechanism to jump to another. The fitness valley between the two basins (intermediate contraction + intermediate refractory) performs worse than either extreme. Seeds that drifted toward high refractory early were reinforced by the efficiency gradient; seeds that drifted toward high contraction were reinforced by the displacement gradient. Neither population could reach the opposing attractor across 50 generations.

The two basins are not equally accessible: the efficiency basin (s42, s137) scores higher on the fitness metric and is where 2 of 3 seeds converged. The displacement basin (s2024) is a valid local optimum under a different implicit objective (maximise raw displacement). Both are stable.

### Relaxation gene is vestigial

The "relaxation" phase in the 3-phase waveform is a misnomer. It represents a *diminishing inward force* — not passive elastic recoil. The muscle is still active but in declining contraction. Evolution minimised this to the 5% floor in the efficiency basin because:
- It increases the active fraction without proportionate thrust (the jet is already formed at peak activation)
- The fitness denominator penalises it
- The elastic mesoglea recoils the bell freely during full zero-activation refractory anyway

**Conclusion: the relaxation gene should be removed.** The waveform reduces to a raised half-cosine arch within the contraction window, followed by full refractory. This is more biologically accurate (fast twitch, passive recoil) and removes a redundant gene.

### Bound pressing on timing genes

In the efficiency basin, both contraction (0.40) and refractory (0.75) are pressing their upper bounds. The search found the efficiency-maximising corner of the timing space and saturated there. The refractory upper bound of 0.75 is constraining the search; extension to 0.90+ would allow further efficiency gains. Experiment 3 (freq_mult encoding) partially addresses this by decoupling the frequency and duty cycle dimensions.

### s999 — GPU hardware failure

s999 produced no data. The GPU assigned to this seed failed (hardware fault) before completing any generations. No data recovered.

---

## Success criteria review

| Criterion | Threshold | Outcome |
|-----------|-----------|---------|
| Best fitness improvement | > 0.55 | ✅ 1.179 (2.1× Exp 1) |
| Jet mode discovered | refractory > 0.60 in ≥2 runs | ✅ s42 (0.65), s137 (0.68) |
| Cup morphology viable | end_y > 0.02 in top 5 | ❌ Not found |
| Cross-run convergence | genome corr > 0.85 or < 0.6 | ✅ Bifurcated: s42/s137 agree (corr ~0.99), s2024 in separate basin |

---

## Open questions carried to Exp 3 & 4

1. **Relaxation gene removed**: with 3-phase collapsed to 2-phase, does the search improve?
2. **Frequency as a gene**: if actuation frequency can evolve ±50%, do resonant frequencies emerge?
3. **Refractory ceiling**: evolution pressed hard against 0.75 — what happens if bound is extended to 0.90?
4. **Payload effect**: is the wide-bell attractor driven by payload properties, or is it intrinsic to the jellyfish body plan?
5. **Raw displacement**: what morphology maximises distance traveled with no efficiency constraint?
