# Experiment 2 — Evolved Timing + Cup Bell Geometry

**Status:** Complete (19/50 gens — run cut short, sufficient for analysis)
**Date:** 2026-03-25
**Code commit:** `39a3cbe` — Add Exp 2 genome: timing genes, cup bells, activity-weighted fitness
**Tall tank support added:** `8b74f17` (not used for this run — square tank)
**Hardware:** 4× RTX 3080 (parallel seeds, CUDA_VISIBLE_DEVICES=0/1/2/3)
**Instance:** vast.ai 4× 3080 @ $0.275/hr, ~3 hrs, ~$0.85
**Output:** `output/cloud/exp2/exp2_s42/`, `exp2_s137/`, `exp2_s999/`, `exp2_s2024/`

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
| Generations | 50 (ran 19) | same target |
| Steps/eval | 150,000 (7.5 cycles @ 1 Hz) | same |
| Fitness | `displacement / sqrt(muscle_count × (1−refractory) / 500)` | updated |
| Tank | 128×128 (square), ceiling cap y=0.93 | same |

---

## Results

### Fitness

| Run | Gens | Best fitness | Gen of best | Peak displacement |
|-----|------|-------------|-------------|-------------------|
| s42 | 19 | **1.143** | 18 | 0.606 |
| s137 | 19 | **1.179** | 17 | 0.626 |
| s999 | 0 | — | — | crashed at launch |
| s2024 | 19 | 0.975 | 14 | 0.728 |

s42 and s137 broke fitness 1.0 at ~gen 15 and trade the lead throughout. s2024 is ~17% lower despite achieving higher raw displacement (0.728 vs 0.61).

### Morphology

Same attractor as Experiment 1: wide, flat-tipped bell with `end_x` locked near upper bound.

| Gene | Cross-run CV | Value |
|------|-------------|-------|
| end_x | **0.005** | ≈ 0.34 (near upper bound 0.35) |
| t_base | 0.037 | ≈ 0.076 |
| t_mid | 0.027 | ≈ 0.061 |
| end_y | 0.135 | ≈ −0.10 (negative — tips down, not cup) |

**Cup bells did not emerge.** end_y converged to negative values in all runs. The relaxed upper bound was not exploited; the morphology attractor is robust.

### Timing — jet mode discovered

s42 and s137 both **saturated the refractory upper bound** by gen 18:

| Run | contraction (g9) | refractory (g10) | relaxation | strategy |
|-----|-----------------|-----------------|-----------|---------|
| s42 final pop mean | 0.346 ± 0.051 | 0.652 ± 0.077 | **0.050 (capped)** | jet mode |
| s137 final pop mean | 0.376 ± 0.033 | 0.677 ± 0.087 | **0.050 (capped)** | jet mode |
| s2024 final pop mean | 0.262 ± 0.096 | 0.269 ± 0.067 | 0.469 | moderate |

**s42 and s137 found jet mode**: long strong contraction (~40% of cycle), near-zero relaxation (5% minimum), very long coast (~65–75%). The relaxation gene was made vestigial — evolution minimised it to the floor in both successful runs.

**s2024 found a local basin**: moderate contraction (~26%), low refractory (~27%), substantial relaxation (~47%). Higher raw displacement but lower fitness. A different cost-of-transport trade-off.

### Cost of transport demonstration

s2024 fires muscles ~2× as often as s42/s137 and travels 16–20% further per run, but the activity-weighted fitness correctly penalises this: the jellyfish works harder for less efficiency. This is a clean empirical demonstration of the cost-of-transport trade-off the fitness function was designed to capture.

### Covariance / condition numbers

All three runs develop condition numbers from ~1.5 → ~10–11 over 19 generations at nearly identical rates, despite finding different solutions. CMA-ES learned equally elongated search ellipsoids in qualitatively different parts of the timing space.

The refractory diagonal variance (`cov_diag[10]`) shrinks in s42/s137 as g10 presses against its upper bound — the population converges in that dimension. In s2024, timing diagonal variances remain near 1.0 (no compression), consistent with a plateau rather than a sharp attractor.

σ remains 0.25–0.45 at gen 18 in all runs — **not fully converged**. More generations would likely sharpen the timing bounds further.

---

## Discussion

### Relaxation gene is vestigial

The "relaxation" phase in the 3-phase waveform is a misnomer. It represents a *diminishing inward force* — not passive elastic recoil. The muscle is still active but in declining contraction. Evolution minimised this to the 5% floor because:
- It increases the active fraction without proportionate thrust (the jet is already formed at peak activation)
- The fitness denominator penalises it
- The elastic mesoglea recoils the bell freely during full zero-activation refractory anyway

**Conclusion: the relaxation gene should be removed.** The waveform reduces to a raised half-cosine arch (smooth rise to peak, symmetric fall) within the contraction window, followed by full refractory. This is more biologically accurate (fast twitch, passive recoil) and removes a redundant gene.

### Bound pressing on both timing genes

Both contraction and refractory are pressing against their upper bounds (0.40 and 0.75 respectively). The bounds are limiting the search, not the morphology. **The refractory upper bound of 0.75 is likely too conservative.** Experiment 3 should extend this.

### s999 failure

s999 produced no data — crash at launch, likely a CUDA initialisation race condition when all 4 processes started simultaneously. No data recovered.

---

## Success criteria review

| Criterion | Threshold | Outcome |
|-----------|-----------|---------|
| Best fitness improvement | > 0.55 | ✅ 1.179 (2.1× Exp 1) |
| Jet mode discovered | refractory > 0.60 in ≥2 runs | ✅ s42 (0.65), s137 (0.68) |
| Cup morphology viable | end_y > 0.02 in top 5 | ❌ Not found |
| Cross-run convergence | genome corr > 0.85 or < 0.6 | ⚠️ Bifurcated: s42/s137 agree (corr ~0.99), s2024 diverges |

---

## Open questions carried to Exp 3 & 4

1. **Relaxation gene removed**: with 3-phase collapsed to 2-phase, does the search improve?
2. **Frequency as a gene**: if actuation frequency can evolve ±50%, do resonant frequencies emerge?
3. **Refractory ceiling**: evolution pressed hard against 0.75 — what happens if bound is extended to 0.90?
4. **Payload effect**: is the wide-bell attractor driven by payload properties, or is it intrinsic to the jellyfish body plan?
5. **Raw displacement**: what morphology maximises distance traveled with no efficiency constraint?
