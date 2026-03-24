# Beyond Jelly: Evolutionary Soft Robotic Jellyfish

GPU-accelerated evolutionary optimization of soft robotic jellyfish morphologies using CMA-ES and 2D Material Point Method (MPM) simulation.

<img width="1432" height="1235" alt="Screenshot 2026-02-24 220055" src="https://github.com/user-attachments/assets/81f5cb3b-f274-4c53-977a-f27e51b2a7ad" />

## Overview

This project explores whether strict biomimetic copying of natural jellyfish shapes is suboptimal for soft robots carrying instrumented payloads. By leveraging evolutionary computation (CMA-ES) within a GPU-accelerated Taichi simulation, we aim to discover novel bell morphologies specifically optimized for payload-carrying applications.

<img width="1139" height="1281" alt="Screenshot 2026-02-24 220202" src="https://github.com/user-attachments/assets/24f17ad6-8eb4-4262-beb1-c4bec5769942" />

### Hypothesis

Evolutionary strategies applied within GPU-accelerated simulation will converge upon novel bell morphologies that:
- Differ significantly from biological baselines when subjected to centralized payloads
- Exhibit higher propulsive efficiency compared to standard biomimetic designs
- Demonstrate improved station-keeping stability

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CMA-ES Optimizer                       │
│               (evolve.py, popsize=16, 9 genes)              │
└─────────────────────┬───────────────────────────────────────┘
                      │ Genome Vector (9 floats)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Genotype-Phenotype Mapping                  │
│                     (make_jelly.py)                          │
│  • Bezier curve bell shape (6 params)                       │
│  • Variable thickness profile (3 params)                    │
│  • Muscle layer, mesoglea collar, transverse bridge         │
│  • Radial symmetry mirroring                                │
└─────────────────────┬───────────────────────────────────────┘
                      │ Particle positions + materials
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tank Assembly Stage                       │
│                  (fill_tank in make_jelly.py)                │
│  • Background water generation (lattice grid)               │
│  • KDTree boolean subtraction (carve robot from fluid)      │
│  • Padding to fixed 80K particle count                      │
└─────────────────────┬───────────────────────────────────────┘
                      │ Fixed-size particle arrays (CPU → GPU)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  MPM Simulation Engine                       │
│                     (mpm_sim.py)                             │
│  • 16 simulation instances batched on one GPU               │
│  • 128×128 grid, 80K particles per instance                 │
│  • Materials: Water(0), Jelly(1), Payload(2), Muscle(3)     │
│  • Pulsed active stress actuation on muscle tissue          │
│  • GPU-side payload CoM tracking (fitness_buffer)           │
└─────────────────────┬───────────────────────────────────────┘
                      │ Payload displacement + stability (16 floats)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Fitness Evaluation                        │
│  • Efficiency: displacement / sqrt(muscle_count / 500)      │
│  • Muscle floor (≥200 particles) rejects degenerate shapes  │
│  • Dynamic invalid penalty: worst_valid_fitness + 1         │
│  • Boundary-stuck detection (y > 0.93 or y < 0.01)         │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
jellyfih/
├── mpm_sim.py          # GPU MPM engine + renderer + fitness kernels
├── make_jelly.py       # Genotype-phenotype mapping + tank filler
├── evolve.py           # CMA-ES evolutionary loop + visualization
├── run_population.py   # Batch population runner with CV2 rendering
├── tune_actuation.py   # Per-instance actuation strength sweep tool
├── helpers/            # Utility scripts (not required for core evolution)
│   ├── fluid_test.py       # Fluid dynamics test visualization
│   ├── make_cad.py         # CAD export: genome → STL (extruded + revolved)
│   ├── make_comparison.py  # Side-by-side comparison video generator
│   └── payload_sink.py     # Baseline: payload sinking without jellyfish
├── web/                # Flask web viewer + interactive designer
│   ├── app.py
│   ├── templates/
│   └── static/
├── pyproject.toml      # Project dependencies
├── CLAUDE.md           # AI assistant project instructions
├── README.md           # This file
└── output/             # Evolution results (generated)
    ├── evolution_log.csv
    ├── best_genomes.json
    ├── checkpoint.pkl
    └── view_*.mp4
```

## Installation

### Install uv

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### Set up the project

Dependencies are split by use case. Choose the sync command that matches your needs:

```bash
git clone <repo-url>
cd jellyfih

# Web viewer, CAD export, comparison videos, morphology plots
# (no GPU required)
uv sync

# Full simulation + evolution (requires CUDA GPU)
uv sync --extra simulation
```

| Extra | Adds | Use when |
|-------|------|----------|
| *(none)* | numpy, scipy, flask, moviepy, trimesh, shapely, matplotlib, imageio | Web viewer, CAD, videos |
| `simulation` | taichi, cma, opencv-python | Running evolution or simulation |

### Requirements
- Python 3.10+
- CUDA-capable GPU (simulation only)

## Usage

```bash
# Quick 5-generation test run (~6 min)
uv run python evolve.py --gens 5

# Full 50-generation evolution (~60 min)
uv run python evolve.py

# Resume from checkpoint (automatic)
uv run python evolve.py --gens 50

# Render best genomes as 4x4 grid video
# Rows = generations, columns = lime|green|turquoise|cyan
uv run python evolve.py --view

# Render a specific generation
uv run python evolve.py --view --gen 3

# Test morphology generator standalone
uv run python make_jelly.py

# Payload sink baseline (no jellyfish)
uv run python helpers/payload_sink.py

# Fluid dynamics test visualization
uv run python helpers/fluid_test.py

# CAD export to STL
uv run python helpers/make_cad.py --aurelia
uv run python helpers/make_cad.py --gen 5 --diameter 120
uv run python helpers/make_cad.py --gen 5 --diameter 120 --remesh 5  # isotropic remesh

# Side-by-side comparison video
uv run python helpers/make_comparison.py

# Web viewer (genome explorer + evolutionary history + interactive designer)
cd web && python app.py   # http://localhost:5000
# /          — genome sliders, morphology preview, evolution history, convergence plots
# /custom    — interactive jellyfish designer (symposium mode, shared aquarium)

# Actuation strength sweep (renders N strengths simultaneously across GPU instances)
uv run python tune_actuation.py
```

## Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Actuation | Raised cosine pulse, strength=500 | 20% contraction / 40% relaxation / 40% refractory; tangent-aligned stress on muscle fiber |
| Fitness | Efficiency metric | `displacement / sqrt(muscle_count / 500)` — penalises large muscle mass relative to thrust |
| Resolution | 128×128 grid | 80K particles, quality=1 |
| Payload | 0.08 × 0.05 | Material 2, 2.5× density, 0.44× gravity (slightly neg. buoyant) |
| Boundaries | Damped sides, clamped walls | Damping layer = grid/20 |
| CMA-ES | lambda=16, sigma=0.1 | Population matches GPU batch size (λ=48 used on cloud runs) |
| Sim Duration | 3 cycles (60K steps) | dt=2e-5, freq=1Hz |
| Spawn | [0.5, 0.4] | Centered, 40% up — gives 0.53 headroom before ceiling threshold |

## Genome Encoding

9-dimensional vector controlling bell morphology via cubic Bezier curves:

| Index | Parameter | Description |
|-------|-----------|-------------|
| 0-1 | cp1_x, cp1_y | Control Point 1 (curve shaping) |
| 2-3 | cp2_x, cp2_y | Control Point 2 (curve shaping) |
| 4-5 | end_x, end_y | Tip position (bell extent) |
| 6 | t_base | Thickness at payload connection |
| 7 | t_mid | Thickness at bell middle |
| 8 | t_tip | Thickness at bell tip |

## Materials

| ID | Material | Properties |
|----|----------|------------|
| 0 | Water | Fluid, zero shear modulus |
| 1 | Jelly | Hyperelastic soft body (mesoglea) |
| 2 | Payload | Near-rigid, high density (2.5x) |
| 3 | Muscle | Soft body + pulsed active stress |
| -1 | Dead | Padding particles (skipped in kernels) |

## Outputs

| File | Description |
|------|-------------|
| `evolution_log.csv` | All genomes with fitness, efficiency, displacement, drift, muscle count, validity per generation |
| `best_genomes.json` | Best genome per generation for replay |
| `checkpoint.pkl` | CMA-ES state for crash recovery |
| `view_*.mp4` | Rendered 4x4 grid videos (column-color-coded) |
| `custom_submissions/` | Genomes + thumbnail PNGs submitted via the `/custom` web designer |

## Performance

Benchmarked on a single CUDA GPU (RTX 4090, 16 simulation instances batched, 80K particles each):

| Metric | Value |
|--------|-------|
| Substep throughput | ~1.2 ms/step |
| Per-generation time (λ=16) | ~74s (72s sim + 2s CPU) |
| 50-generation run (λ=16) | ~62 minutes |
| Per-generation time (λ=48, cloud) | ~244s |
| GPU-CPU transfer | λ×5 floats/generation |

The GPU is SM-compute bound at λ=16 (100% SM utilisation, ~6% VRAM). Running two `evolve.py` processes time-slices rather than parallelises — scale via larger λ in a single process instead.

## Current Status

### Implemented
- [x] MPM simulation engine with 16 simulation instances batched on one GPU
- [x] Bezier-curve morphology generator with tangent-aligned muscle fibers
- [x] Tank filler with KDTree boolean subtraction
- [x] Pulsed active stress actuation — 20/40/40 waveform with refractory period
- [x] Efficiency-based fitness with muscle floor and dynamic invalid penalty
- [x] GPU-side payload CoM fitness evaluation
- [x] CMA-ES evolutionary loop with bounds handling
- [x] Checkpoint/resume for crash recovery
- [x] Full CSV + JSON logging (fitness, efficiency, displacement, covariance diagnostics)
- [x] HDR particle splatting renderer
- [x] 4x4 grid visualization with per-column color coding
- [x] Zero-actuation baseline validation
- [x] Boundary-stuck payload detection
- [x] Per-instance actuation field (enables parameter sweeps across GPU batch)
- [x] Web viewer: genome sliders, evolution history, convergence plots, click-drag Bezier points
- [x] `/custom` interactive designer: 3-step wizard, colour picker, shared aquarium, fitness prediction
- [x] Vast.ai cloud deployment (`setup_vastai.sh`, λ=48 cloud runs validated)

### TODO
- [ ] Full Cost of Transport fitness (GPU energy tracking)
- [ ] Re-enable drift penalty in fitness function
- [ ] Adaptive resolution (128 → 256 grid transition)
- [ ] Genome heatmap visualization
- [ ] Automatic per-generation video export

## Cloud Deployment (Vast.ai)

The project supports running long evolution jobs on rented GPU instances via [Vast.ai](https://vast.ai).

```bash
# On a fresh Vast.ai CUDA instance (RTX 4090 recommended):
bash setup_vastai.sh   # installs uv, deps, verifies Taichi CUDA, starts tmux session

# Run with larger population for better GPU utilisation
uv run python evolve.py --gens 50   # λ=48 recommended for cloud runs

# Sync results back to local machine
rsync -az -e "ssh -p <port>" root@<host>:~/jellys/output/ ./output/
```

Key finding from cloud runs: the GPU is SM-compute bound, not memory bound (~6% VRAM at λ=16). Running two `evolve.py` processes does not parallelise — it time-slices. Scale by increasing λ in a single process.

## References

1. Gemmell et al. "Passive energy recapture in jellyfish" PNAS 2013
2. Hansen, N. "The CMA Evolution Strategy: A Tutorial" 2016
3. Hu et al. "Taichi: High-performance computation" ACM TOG 2019

## License

MIT
