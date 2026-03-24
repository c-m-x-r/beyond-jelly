# Beyond Jelly: Evolutionary Soft Robotic Jellyfish

GPU-accelerated evolutionary optimization of soft robotic jellyfish morphologies using CMA-ES and 2D Material Point Method (MPM) simulation.

<img width="1432" height="1235" alt="Screenshot 2026-02-24 220055" src="https://github.com/user-attachments/assets/81f5cb3b-f274-4c53-977a-f27e51b2a7ad" />

## Overview

This project explores whether strict biomimetic copying of natural jellyfish shapes is suboptimal for soft robots carrying instrumented payloads. By leveraging evolutionary computation (CMA-ES) within a GPU-accelerated Taichi simulation, we discover novel bell morphologies specifically optimized for payload-carrying applications — unconstrained by biological symmetry or manufacturing conventions.

<img width="1139" height="1281" alt="Screenshot 2026-02-24 220202" src="https://github.com/user-attachments/assets/24f17ad6-8eb4-4262-beb1-c4bec5769942" />

### Hypothesis

Evolutionary strategies applied within GPU-accelerated simulation will converge upon novel bell morphologies that differ significantly from biological baselines, exhibiting higher propulsive efficiency when carrying centralized payloads.

**Experiment 1 finding:** All independent runs converged within 10 generations to a wide, flat-tipped bell with outward-curling tips — pressing against the gene bound that prevented positive end_y. This morphology is not achievable in real *Aurelia aurita* (radial symmetry prevents outward bell tips). The point is to escape biomimicry.

**Experiment 2 (current):** Relaxed end_y to allow cup-shaped bells; added evolved actuation timing (11D genome, λ=32 population).

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      CMA-ES Optimizer                        │
│           (evolve.py, popsize=JELLY_INSTANCES, 11 genes)     │
└──────────────────────┬───────────────────────────────────────┘
                       │ Genome Vector (11 floats)
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                 Genotype-Phenotype Mapping                    │
│                      (make_jelly.py)                         │
│  • Bezier curve bell shape (6 params)                        │
│  • Variable thickness profile (3 params)                     │
│  • Actuation timing: contraction_frac, refractory_frac (2)  │
│  • Muscle layer, mesoglea collar, transverse bridge          │
└──────────────────────┬───────────────────────────────────────┘
                       │ Particle positions + materials + timing
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   MPM Simulation Engine                       │
│                      (mpm_sim.py)                            │
│  • N instances batched on one GPU (JELLY_INSTANCES)         │
│  • 128×128 grid, 80K particles per instance                  │
│  • Per-instance raised-cosine actuation waveform             │
│  • All four boundaries damped (n_grid/20 layer)             │
│  • GPU-side payload CoM tracking + fitness buffer            │
└──────────────────────┬───────────────────────────────────────┘
                       │ Payload displacement (N floats)
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                    Fitness Evaluation                         │
│  • displacement / sqrt(muscle_count × (1−refractory) / 500) │
│  • Activity-weighted: resting more = lower effective cost    │
│  • Ceiling cap at y=0.93 (not invalidated, just capped)     │
│  • Invalid penalty: worst_valid + 1.0 (not hard -100)       │
└──────────────────────────────────────────────────────────────┘
```

## Project Structure

```
jellyfih/
├── mpm_sim.py              # GPU MPM engine + renderer + fitness kernels
├── make_jelly.py           # Genotype-phenotype mapping + tank filler
├── evolve.py               # CMA-ES evolutionary loop + visualization
├── helpers/
│   ├── view_single.py      # Full-res single-genome render + vorticity overlay
│   ├── view_generation.py  # All individuals from one generation as grid
│   ├── view_random.py      # N independently random jellyfish
│   ├── fluid_analysis.py   # Grid momentum/vorticity time-series analysis
│   ├── payload_sink.py     # Payload-only baseline (no jellyfish)
│   ├── fluid_test.py       # Oscillating paddle fluid test
│   ├── make_cad.py         # Genome → STL (extruded + revolved)
│   ├── make_comparison.py  # Side-by-side comparison video
│   └── tune_actuation.py   # Actuation strength sweep
├── web/                    # Flask web viewer + interactive designer
├── setup_cloud.sh          # Cloud instance bootstrap
├── deploy.sh               # Full rsync + setup + launch pipeline
├── sync_results.sh         # Pull results from remote instance
├── run_experiments.sh      # Multi-seed experiment runner
├── storage/
│   └── EXPERIMENT_LOG.md   # Full experiment history + methodology
├── pyproject.toml
└── output/                 # Evolution results (generated, gitignored)
```

## Installation

```bash
git clone <repo-url>
cd jellyfih

# Web viewer, CAD, videos (no GPU required)
uv sync

# Full simulation + evolution (requires CUDA GPU)
uv sync --extra simulation
```

Requirements: Python 3.10+, CUDA GPU (for simulation), `libx11-6 libxext6 libxi6` (headless Linux).

## Usage

```bash
# Evolve — lambda set by JELLY_INSTANCES (default 16, use 32 for Exp 2)
JELLY_INSTANCES=32 uv run python evolve.py --gens 50 --seed 42 --run-id myrun

# Resume from checkpoint (automatic)
JELLY_INSTANCES=32 uv run python evolve.py --gens 50 --run-id myrun

# Render single genome, full-res, many cycles
JELLY_INSTANCES=1 uv run python helpers/view_single.py --aurelia --steps 300000
JELLY_INSTANCES=1 uv run python helpers/view_single.py --genome "[0.15,-0.06,...]" --flow

# View all individuals from a generation
JELLY_INSTANCES=16 uv run python helpers/view_generation.py \
    --gen 5 --log output/myrun/evolution_log_myrun.csv --palette random

# 16 random jellyfish with vorticity overlay
JELLY_INSTANCES=16 uv run python helpers/view_random.py --flow --steps 100000

# Fluid dynamics analysis (grid momentum + vorticity → CSV)
JELLY_INSTANCES=1 uv run python helpers/fluid_analysis.py \
    --genome "[0.15,-0.06,0.167,-0.12,0.289,-0.041,0.056,0.046,0.014,0.2,0.4]"

# Aurelia reference baseline
uv run python evolve.py --aurelia

# Web viewer
cd web && python app.py   # http://localhost:5000
```

## Genome Encoding

11-dimensional vector. Genes 0–8 are morphology; genes 9–10 are actuation timing.

| Index | Parameter | Bounds | Description |
|-------|-----------|--------|-------------|
| 0 | cp1_x | [0.0, 0.25] | Bezier Control Point 1 x |
| 1 | cp1_y | [-0.15, 0.15] | Bezier Control Point 1 y |
| 2 | cp2_x | [0.0, 0.3] | Bezier Control Point 2 x |
| 3 | cp2_y | [-0.2, 0.15] | Bezier Control Point 2 y |
| 4 | end_x | [0.05, 0.35] | Bell tip x-extent |
| 5 | end_y | [-0.30, **+0.10**] | Bell tip y-extent (positive = tips curl upward) |
| 6 | t_base | [0.025, 0.08] | Thickness at payload connection |
| 7 | t_mid | [0.025, 0.1] | Thickness at bell middle |
| 8 | t_tip | [0.01, 0.04] | Thickness at bell tip |
| 9 | act_contraction_frac | [0.05, 0.40] | Fraction of cycle contracting (init: 0.20) |
| 10 | act_refractory_frac | [0.20, 0.75] | Fraction of cycle resting (init: 0.40) |

`relaxation_frac = max(0.05, 1 − contraction − refractory)` — implicit, clamped in kernel.

## Physics Model

| Aspect | Model | Notes |
|--------|-------|-------|
| Fluid | Weakly-compressible MPM, mu=0 | Inviscid; infinite Reynolds number |
| Speed of sound | ~200 m/s (sim) vs 1480 m/s (real) | 7× slower — deliberate CFL trade-off |
| Buoyancy | None | Payload gets full gravity; water pressure provides partial support only |
| Payload | E=2e5, 2.5× density, plasticity ±0.5% | Near-rigid; sinks ~0.09 units/3 cycles without jellyfish |
| Boundaries | All four walls: 5% damping per cell in n_grid/20 band | Absorbs wall reflections; no free surface |
| Actuation | Raised cosine pulse, per-instance timing | Tangent-aligned stress on muscle fiber direction |

## Performance

| Hardware | λ | Steps/eval | Time/gen | Cost for 4 seeds × 50 gens |
|----------|---|------------|----------|-----------------------------|
| RTX 4090 | 16 | 150K | ~112s | ~3.1 hrs sequential, ~$0.93 |
| 4× RTX 3080 | 32 each | 150K | ~212s | ~3 hrs parallel, ~$0.85 |

GPU is SM-compute bound at λ=16–32 (~6% VRAM). Use `CUDA_VISIBLE_DEVICES=N` to pin one process per GPU on multi-GPU machines.

## Cloud Deployment

```bash
# 1. Provision instance on vast.ai (RTX 4090 or 4× RTX 3080)
vastai create instance <offer_id> --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
    --disk 30 --ssh --direct --label jellyfih

# 2. Deploy (rsync code + install deps + launch experiments)
bash deploy.sh <ssh_addr> <ssh_port>

# 3. Monitor
ssh -p <port> root@<addr> tail -f /root/jellyfih/logs/run_experiments.log

# 4. Sync results
bash sync_results.sh <ssh_addr> <port> <label>

# 5. Destroy
vastai destroy instance <ID>
```

Multi-GPU parallel (4× 3080, one seed per GPU):
```bash
for i in 0 1 2 3; do
    SEED=$((42 + i * 95))
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$i JELLY_INSTANCES=32 \
        uv run python evolve.py --gens 50 --seed $SEED --run-id exp2_s$SEED \
        > logs/s$SEED.log 2>&1 &
done
```

## Current Status

### Implemented
- [x] MPM simulation engine, N-instance GPU batch, 11D genome
- [x] Per-instance actuation timing (genes 9–10, evolved)
- [x] Activity-weighted fitness: `displacement / sqrt(muscle × (1−refractory) / 500)`
- [x] Bezier-curve morphology + cup-bell support (end_y up to +0.10)
- [x] All four boundaries damped
- [x] CMA-ES with checkpoint/resume, full CSV/JSON/diagnostics logging
- [x] HDR abyss renderer + web palette + vorticity overlay
- [x] view_single / view_generation / view_random / fluid_analysis helpers
- [x] Cloud deployment scripts (deploy.sh, sync_results.sh, setup_cloud.sh)
- [x] Multi-GPU parallel runs via CUDA_VISIBLE_DEVICES
- [x] Fluid dynamics analysis tool (grid momentum, vorticity, wall flux)
- [x] Web viewer: genome sliders, evolution history, Bezier editor, /custom designer

### Known Limitations
- No buoyancy model (intentional simplification)
- 2D only — overestimates thrust vs 3D jellyfish
- Speed of sound 7× slower than real water
- Drift penalty disabled (lateral stability not penalized)
- Ceiling exploit: population saturates at y≈0.88 within ~10 gens

### Next (Experiment 3 candidates)
- Frequency as an evolved gene [0.3, 3.0 Hz]
- Multi-objective: Pareto front over (displacement, effective_muscle)
- Longer evaluation (6+ cycles) to measure sustained propulsion
- Physical validation: 120mm diameter bell, ~30g payload

## References

1. Gemmell et al. "Passive energy recapture in jellyfish" PNAS 2013
2. Hansen, N. "The CMA Evolution Strategy: A Tutorial" 2016
3. Hu et al. "Taichi: High-performance computation" ACM TOG 2019

## License

MIT
