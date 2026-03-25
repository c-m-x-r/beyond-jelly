# Jellyfih Experiment Index

Evolutionary optimisation of soft-robotic jellyfish morphologies for upward payload transport.
MPM simulation (Taichi), CMA-ES, 2D.

---

## Experiments

| # | File | Status | Key question | Best fitness |
|---|------|--------|-------------|-------------|
| 1 | [EXPERIMENT_1.md](EXPERIMENT_1.md) | ✅ Complete | What morphology emerges from shape-only search? | 0.536 |
| 2 | [EXPERIMENT_2.md](EXPERIMENT_2.md) | ✅ Complete (19/50 gens) | Does evolved timing unlock better performance? | 1.179 |
| 3 | [EXPERIMENT_3.md](EXPERIMENT_3.md) | 🔲 Planned | Raw displacement + frequency as gene | — |
| 4 | [EXPERIMENT_4.md](EXPERIMENT_4.md) | 🔲 Planned | Does payload shape the morphology attractor? | — |

---

## Genome evolution

| Exp | Genes | Key change |
|-----|-------|------------|
| 1 | 9D | Shape (6) + thickness (3), fixed timing |
| 2 | 11D | + contraction_frac + refractory_frac |
| 3 | 11D | relaxation removed → **freq_mult** replaces refractory gene |
| 4 | 11D | Same as Exp 3 (or Exp 2), `with_payload=False` |

---

## Default configuration (as of Exp 2+)

| Parameter | Value |
|-----------|-------|
| Population λ | 32 |
| Tank | 128×256 tall (JELLY_GRID_Y=256, JELLY_DOMAIN_H=2.0) |
| Steps/eval | 150,000 (7.5s at dt=2e-5) |
| Seeds (standard) | 42, 137, 999, 2024 |
| Hardware target | 4× RTX 3080, one seed per GPU |
| Time/gen (tall tank, λ=32) | ~480s |
| Time total (50 gens) | ~6.7 hrs |

---

## Key findings to date

- **Morphology attractor**: wide flat-tipped bell, `end_x` ≈ 0.34, `t_base` ≈ 0.076, robust across all seeds and experiments
- **Jet mode**: evolution converges to high refractory (~65–75%), minimal relaxation (5% floor), long coast
- **Relaxation gene vestigial**: should be removed from Exp 3 onward
- **Cost of transport trade-off**: demonstrated explicitly in Exp 2 (s2024 higher displacement, lower fitness)
- **Ceiling exploit**: still present in square tank; tall tank required for unconstrained displacement experiments
- **Cup bells not found**: relaxing end_y upper bound to +0.10 did not unlock cup morphologies

---

## Code commits by experiment

| Exp | Commit | Description |
|-----|--------|-------------|
| 1 | `3c5ad12` | Rendering overhaul, morphology fixes, physics corrections |
| 2 | `39a3cbe` | Add Exp 2 genome: timing genes, cup bells, activity-weighted fitness |
| 2 (tall tank support) | `8b74f17` | add evolve compatibility for tall tank |
| current HEAD | `0d62bd8` | bugfix |

---

## Other documentation

| File | Location | Contents |
|------|----------|---------|
| EXPERIMENT_LOG.md | storage/ | Original combined log (Exp 1 + 2 planning notes) |
| DESIGN_CRITIQUE.md | storage/ | Physics model limitations and design decisions |
| CLOUD_TEST_PLAN.md | storage/ | vast.ai deployment runbook |
| PROBLEMS.md | storage/ | Known issues and workarounds |
| SESSION_2026-03-24.md | storage/ | Session notes from initial cloud runs |
