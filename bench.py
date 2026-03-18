"""Quick substep throughput benchmark.
Loads Aurelia genome into all 16 instances, warms up 100 steps,
then times N_BENCH steps and reports ms/step.
Usage: uv run python bench.py
"""
import time
import numpy as np

N_BENCH = 2000

# Import sim (triggers ti.init)
from mpm_sim import n_instances, n_particles, substep, load_particles, sim_time
import taichi as ti
from make_jelly import fill_tank, AURELIA_GENOME

print("Building phenotype...")
pos_np, mat_np, fiber_np, stats = fill_tank(AURELIA_GENOME, n_particles)

print(f"Loading {n_instances} instances...")
for i in range(n_instances):
    load_particles(i, pos_np, mat_np, fiber_np)

sim_time[None] = 0.0

print("Warming up (100 steps)...")
for _ in range(100):
    substep()
ti.sync()

print(f"Benchmarking {N_BENCH} steps...")
t0 = time.perf_counter()
for _ in range(N_BENCH):
    substep()
ti.sync()
elapsed = time.perf_counter() - t0

ms_per_step = elapsed / N_BENCH * 1000
steps_per_sec = N_BENCH / elapsed

print(f"\n{'='*40}")
print(f"  {N_BENCH} steps in {elapsed:.3f}s")
print(f"  {ms_per_step:.3f} ms/step")
print(f"  {steps_per_sec:.1f} steps/sec")
print(f"{'='*40}")

# Sanity check: verify sim hasn't exploded
from mpm_sim import get_payload_stats
stats = get_payload_stats()
print(f"\nPayload CoM Y (all instances): {stats[:, 0]}")
valid = np.sum((stats[:, 0] > 0.01) & (stats[:, 0] < 0.99))
print(f"Valid instances: {valid}/{n_instances}")
