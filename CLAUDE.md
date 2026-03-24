# Jellyfih Project Instructions

## Project Overview

Evolutionary optimization of soft robotic jellyfish morphologies using CMA-ES and 2D MPM simulation in Taichi. The goal is to discover bell shapes optimized for carrying instrumented payloads, diverging from biomimetic designs where necessary.

**Current experiment:** Experiment 2 — 11D genome with evolved actuation timing and relaxed bell geometry. See `storage/EXPERIMENT_LOG.md` for full history.

## Current Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Genome | 11D | Bezier shape (6) + thickness (3) + actuation timing (2) |
| Fitness | Activity-weighted efficiency | `displacement / sqrt(muscle_count × (1−refractory_frac) / 500)` |
| Resolution | 128×128 grid, 80K particles | quality=1 |
| Payload | 0.08 × 0.05 normalized units | Material 2, density=2.5× water, full gravity |
| Boundaries | **All four sides damped** | n_grid/20 damping layer; floor and ceiling both absorbed |
| CMA-ES | lambda=JELLY_INSTANCES, sigma=0.25, 50 gens | Set JELLY_INSTANCES=32 for Exp 2 |
| Sim Duration | 150,000 steps (7.5 cycles @ 1 Hz) | dt=2e-5 |
| Frequency | 1.0 Hz | Fixed |
| Spawn | [0.5, 0.40] | Payload CoM starts below midline; 0.53 units headroom to ceiling |
| Ceiling cap | y = 0.93 | Displacement capped, not invalidated — ceiling-riders score same as y=0.93 |
| Actuation strength | 500 | Per-instance field; timing now also per-instance from genome |

## Architecture

Three main components, fully integrated:

1. **mpm_sim.py** - GPU-accelerated MPM simulation engine
   - N simulation instances batched on one GPU (N = `JELLY_INSTANCES` env var, default 16)
   - Materials: Water(0), Jelly(1), Payload(2), Muscle(3)
   - Fixed tensor allocation (N × 80,000 particles)
   - Raised cosine actuation waveform — **per-instance timing** via `instance_act_contraction` and `instance_act_refractory` fields (genes 9–10)
   - Per-instance actuation strength (`instance_actuation`), hue, muscle hue, payload density
   - GPU-side fitness evaluation via `fitness_buffer` field
   - Headless batch runner (`run_batch_headless`)
   - HDR particle splatting renderer (abyss/web/random palettes) with per-instance color tinting
   - Vorticity overlay (`render_vorticity_overlay`) — additive curl visualization
   - Enhanced water rendering: velocity-direction HSV colouring above threshold 0.02

2. **make_jelly.py** - Genotype-phenotype mapping
   - 11-gene encoding: Bezier curve bell shape + thickness + actuation timing
   - Structural features: bell, muscle layer, mesoglea collar, transverse bridge
   - `fill_tank()` / `generate_phenotype()`: accept `with_payload` param
   - Water generation, KDTree boolean subtraction, dead-particle padding
   - Returns muscle particle count for CPU-side stats
   - Includes `AURELIA_GENOME`: hand-designed moon jelly reference (9D, compatible)
   - `random_genome()`: 11D, timing genes init at known-good defaults (0.20/0.40)

3. **evolve.py** - Evolutionary optimizer (CMA-ES)
   - `--gens N`: generation count
   - `--seed N`: CMA-ES random seed (use different seeds for independent replicate runs)
   - `--run-id LABEL`: output subdirectory label (auto-creates `output/<LABEL>/`)
   - `--view` / `--view --gen N`: render best genomes as grid video
   - `--aurelia`: evaluate moon jelly reference genome
   - Zero-actuation + payload-sink baselines at startup (fresh runs only)
   - Checkpoint/resume via pickle (every 5 gens)
   - Full CSV logging (11 genes) + JSON best-genome history + covariance diagnostics

## Fitness Evaluation

**Activity-weighted efficiency** (Experiment 2):
```
active_frac = 1.0 - refractory_frac          # genome gene 10
effective_muscle = muscle_count × active_frac
fitness = displacement / sqrt(effective_muscle / 500)
```

Rationale: a jellyfish with 70% refractory fires muscles only 30% of the time — it should pay only 30% of the muscle-count cost. This rewards jet-mode propulsion (short strong stroke, long coast) alongside morphological efficiency.

**Displacement cap:**
```
displacement = min(final_y, 0.93) - init_y
```
Ceiling-riders are not invalidated — their displacement is capped. Once the population saturates against the ceiling, evolution pivots to reducing effective muscle cost.

**Validity checks** (get `worst_valid_fitness + 1.0` penalty):
- Payload CoM `y < 0.01` (floor contact) — GPU side
- `muscle_count < 200` (degenerate morphology) — CPU side
- Self-intersecting morphology — CPU side

Drift penalty (`- 1.0 * drift`) is implemented but commented out.

## Material System

| ID | Material | Properties | Role |
|----|----------|------------|------|
| 0 | Water | Fluid, mu=0, lambda=100000, **inviscid** | Background medium |
| 1 | Jelly | Hyperelastic, E=0.7e3, nu=0.3 | Passive bell structure |
| 2 | Payload | Near-rigid, E=2e5, nu=0.2, density=2.5×, **full gravity** | Instrumented cargo |
| 3 | Muscle | Same elasticity as jelly + per-instance active stress | Actuation tissue |
| -1 | Dead | Skipped in all kernels | Padding to fixed count |

**Physics notes:** No buoyancy model — water pressure provides partial support but is not calibrated to Archimedes' principle. Payload sinks ~0.09 units/3-cycle run without jellyfish. Water speed of sound ~200 m/s (7× slower than real water — deliberate trade-off for CFL stability). Effectively infinite Reynolds number (mu=0).

## Genome Encoding

11-dimensional vector. Genes 0–8 are morphology; genes 9–10 are actuation timing.

| Index | Parameter | Bounds | Init | Description |
|-------|-----------|--------|------|-------------|
| 0 | cp1_x | [0.0, 0.25] | mid | Control Point 1 x-offset |
| 1 | cp1_y | [-0.15, 0.15] | mid | Control Point 1 y-offset |
| 2 | cp2_x | [0.0, 0.3] | mid | Control Point 2 x-offset |
| 3 | cp2_y | [-0.2, 0.15] | mid | Control Point 2 y-offset |
| 4 | end_x | [0.05, 0.35] | mid | Bell tip x-extent |
| 5 | end_y | [-0.30, **+0.10**] | mid | Bell tip y-extent (**positive = tips curl upward**) |
| 6 | t_base | [0.025, 0.08] | mid | Thickness at payload connection |
| 7 | t_mid | [0.025, 0.1] | mid | Thickness at bell middle |
| 8 | t_tip | [0.01, 0.04] | mid | Thickness at bell tip |
| 9 | act_contraction_frac | [0.05, 0.40] | **0.20** | Fraction of cycle spent contracting |
| 10 | act_refractory_frac | [0.20, 0.75] | **0.40** | Fraction of cycle in refractory rest |

`relaxation_frac = max(0.05, 1.0 − contraction_frac − refractory_frac)` — computed in kernel, not stored.

Timing genes start at historical defaults (not midpoint) so CMA-ES explores from a viable baseline.

CMA-ES uses built-in bounds handling (not hard clipping) to preserve covariance estimation. Gene ranges passed as `CMA_stds` for proportional mutation scaling.

## Key Experimental Findings (Experiment 1)

- All 4+ independent runs converge to **same attractor within 10 generations** — wide flat-tipped bell, tips pressing against former upper bound (−0.03)
- **Locked genes** (CV < 0.15): end_x ≈ 0.277, t_base ≈ 0.054, t_mid ≈ 0.045
- **Dominant covariance coupling**: cp2_x ↔ end_y (−0.26 to −0.36): wider bell → shallower tip
- **Ceiling exploit**: payload reaches y ≈ 0.88 within 3 cycles from spawn y=0.40
- **Undissipated wake**: 35% of peak momentum remains at refractory end; vortex ring still organizing for 150ms into refractory before decaying
- Best fitness plateau: ~0.53 (Exp 1), ceiling-limited not morphology-limited

## Cloud Deployment

Standard workflow via `deploy.sh` / `sync_results.sh`:

```bash
# Provision on vast.ai, then:
bash deploy.sh <ssh_addr> <port>        # rsync + setup + launch experiments

# Sync results back anytime:
bash sync_results.sh <ssh_addr> <port> <label>

# Destroy when done:
vastai destroy instance <ID>
```

**Multi-GPU parallel runs** (4× 3080, one seed per GPU):
```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 JELLY_INSTANCES=32 \
  uv run python evolve.py --gens 50 --seed 42 --run-id exp2_s42 > logs/s42.log 2>&1 &
# repeat for CUDA_VISIBLE_DEVICES=1,2,3 with different seeds
```

`CUDA_VISIBLE_DEVICES` pins each process to one GPU; Taichi picks it up automatically.

## Usage

```bash
# Quick morphology test (matplotlib plot)
uv run python make_jelly.py
uv run python make_jelly.py --aurelia

# Evolve (lambda set by JELLY_INSTANCES, default 16)
JELLY_INSTANCES=32 uv run python evolve.py --gens 50 --seed 42 --run-id myrun

# Render single genome at full res (1024×1024), 15 cycles
JELLY_INSTANCES=1 uv run python helpers/view_single.py --aurelia --steps 300000
JELLY_INSTANCES=1 uv run python helpers/view_single.py --genome "[0.15,...]" --flow

# Render a generation as 4×4 grid
JELLY_INSTANCES=16 uv run python helpers/view_generation.py --gen 5 \
    --log output/myrun/evolution_log_myrun.csv --palette random

# 16 independently random jellyfish
JELLY_INSTANCES=16 uv run python helpers/view_random.py --flow --steps 100000

# Fluid dynamics analysis (captures grid momentum/vorticity over time → CSV)
JELLY_INSTANCES=1 uv run python helpers/fluid_analysis.py --genome "[...]" --steps 60000

# Payload sink baseline
uv run python helpers/payload_sink.py

# Web viewer
cd web && python app.py   # http://localhost:5000
```

## Files

| File | Purpose |
|------|---------|
| mpm_sim.py | MPM physics engine + renderer + fitness kernels |
| make_jelly.py | Morphology generator + tank filler + Aurelia reference |
| evolve.py | CMA-ES evolutionary loop + visualization |
| helpers/view_single.py | Full-res single-genome render; `--flow` vorticity overlay |
| helpers/view_generation.py | All individuals from one generation as N×N grid |
| helpers/view_random.py | 16 independently random genomes; `--flow`, `--steps` |
| helpers/fluid_analysis.py | Grid momentum/vorticity time-series for one genome |
| helpers/payload_sink.py | Payload-only baseline (no jellyfish) |
| helpers/fluid_test.py | Oscillating paddle fluid dynamics test |
| helpers/make_cad.py | Genome → STL (extruded + revolved) |
| helpers/make_comparison.py | Side-by-side comparison video |
| helpers/tune_actuation.py | Actuation strength sweep across GPU batch |
| web/ | Flask web viewer: genome sliders, history, Bezier editor, /custom designer |
| setup_cloud.sh | Bootstrap cloud instance (uv, deps, X11 libs, GPU smoke test) |
| deploy.sh | Full pipeline: rsync + setup + launch experiments |
| run_experiments.sh | Sequential/parallel seed runner with rsync-back |
| sync_results.sh | Pull output from remote instance to output/cloud/<label>/ |
| CLOUD_TEST_PLAN.md | Cloud run runbook |
| storage/EXPERIMENT_LOG.md | Full experiment history, methodology, results, analysis |
| pyproject.toml | Dependencies; use `uv sync --extra simulation` for GPU deps |

## Output Files

All outputs in `output/<run-id>/` (or `output/` for default runs):

| File | Description |
|------|-------------|
| `evolution_log_<id>.csv` | Every individual: generation, gene_0..gene_10, fitness, final_y, displacement, drift, muscle_count, valid, sigma, efficiency |
| `best_genomes_<id>.json` | Best genome per generation + covariance diagnostics |
| `checkpoint_<id>.pkl` | CMA-ES state for crash recovery (every 5 gens) |
| `view_gen*.mp4` | Rendered generation grids |
| `fluid_analysis_*.csv` | Grid momentum/vorticity time-series |

## Performance

| Hardware | λ | Steps/eval | Time/gen | 50-gen total |
|----------|---|------------|----------|--------------|
| RTX 4090 | 16 | 60K | ~74s | ~62 min |
| RTX 4090 | 16 | 150K | ~112s | ~93 min |
| RTX 3080 | 32 | 150K | ~212s | ~177 min |
| 4× RTX 3080 | 32 each | 150K | ~212s | ~177 min (all 4 parallel) |

GPU is SM-compute bound (~100% SM, ~6% VRAM at λ=16). Two processes on same GPU time-slice — use `CUDA_VISIBLE_DEVICES` to assign one process per GPU.

## Known Issues / TODO

1. **No buoyancy model**: payload sinks ~0.09 units/3-cycle run; jellyfish must overcome this plus generate net upward thrust
2. **Mirror overlap**: particles at x=0 duplicated by symmetry mirroring (density spike at midline)
3. **Ceiling exploit**: population saturates at y≈0.88 within ~10 gens; activity-weighted fitness partially addresses this by pivoting to efficiency
4. **Undissipated wake**: 35% residual momentum at refractory end (Exp 1); evolved timing (Exp 2) should reduce this
5. **No adaptive resolution**: 128→256 grid phase transition not implemented
6. **Drift penalty disabled**: `- 1.0 * drift` commented out in `compute_fitness()`
7. **Water speed of sound ~7× too slow**: deliberate for CFL stability; affects acoustic thrust realism
8. **2D only**: no out-of-plane dynamics; overestimates thrust vs real 3D jellyfish
