"""
Plot GPU temperature timeline from gpu_temp_timeline.csv.
Run anytime during or after evolution: uv run python plot_thermal.py
Saves output/gpu_thermal_profile.png
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = "output"

def load_timeline(path):
    times, temps, clocks, throttles, utils = [], [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['wall_time']))
            temps.append(int(row['temp_c']))
            clocks.append(float(row['clock_ratio']))
            throttles.append(row['throttle'])
            utils.append(int(row['util_pct']))
    return (np.array(times), np.array(temps),
            np.array(clocks), throttles, np.array(utils))

def load_gen_events(thermal_path):
    """Return list of (wall_elapsed_approx, generation, phase) from thermal_log.
    Since thermal_log doesn't store wall time, we infer from elapsed_s offsets."""
    events = []
    with open(thermal_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append({
                'phase': row['phase'],
                'gen': int(row['generation']),
                'elapsed_s': float(row['elapsed_s']),
                'temp': int(row['temp_c']),
            })
    return events

def main():
    timeline_path = os.path.join(OUTPUT_DIR, 'gpu_temp_timeline.csv')
    thermal_path  = os.path.join(OUTPUT_DIR, 'thermal_log.csv')

    if not os.path.exists(timeline_path):
        print(f"No timeline data at {timeline_path}. Run gpu_monitor.py first.")
        return

    times, temps, clocks, throttles, utils = load_timeline(timeline_path)
    t0 = times[0]
    elapsed_min = (times - t0) / 60.0

    # Throttle reason color map
    colors = {
        'none':         '#4CAF50',   # green — uncapped
        'idle':         '#2196F3',   # blue
        'power_cap':    '#FF9800',   # orange — normal laptop sustained
        'sw_thermal':   '#F44336',   # red — thermal throttle
        'hw_slowdown':  '#9C27B0',   # purple
        'hw_power_brake': '#E91E63',
        'other':        '#9E9E9E',
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#1a1a2e')
    for ax in (ax1, ax2):
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#444')

    # ── Temperature plot ──────────────────────────────────────────
    # Color segments by throttle reason
    prev_throttle = throttles[0]
    seg_start = 0
    for i in range(1, len(throttles) + 1):
        curr = throttles[i] if i < len(throttles) else None
        if curr != prev_throttle or i == len(throttles):
            col = colors.get(prev_throttle, colors['other'])
            ax1.plot(elapsed_min[seg_start:i], temps[seg_start:i],
                     color=col, linewidth=1.5, solid_capstyle='round')
            seg_start = i
            prev_throttle = curr

    # Threshold lines
    ax1.axhline(80, color='#FF5722', linestyle='--', linewidth=1, alpha=0.7, label='Pause (80°C)')
    ax1.axhline(60, color='#00BCD4', linestyle='--', linewidth=1, alpha=0.7, label='Resume (60°C)')
    ax1.axhline(83, color='#F44336', linestyle=':', linewidth=1, alpha=0.5, label='Thermal throttle (~83°C)')

    # Annotate generation completions from thermal_log
    if os.path.exists(thermal_path):
        try:
            events = load_gen_events(thermal_path)
            # Find cooling phase start events (first cooling entry per gen)
            seen_gens = set()
            for ev in events:
                if ev['phase'] == 'cooling' and ev['gen'] not in seen_gens:
                    seen_gens.add(ev['gen'])
                    # Find closest timestamp where temp matches
                    target_temp = ev['temp']
                    diffs = np.abs(temps - target_temp)
                    # Among recent points, find the cooling-start moment
                    # (rough: just annotate at the gen number along x)
                    best = np.argmin(diffs[-len(diffs)//2:]) + len(diffs)//2
                    ax1.axvline(elapsed_min[best], color='white', alpha=0.15,
                                linewidth=0.8, linestyle='-')
                    ax1.text(elapsed_min[best], ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 90,
                             f'G{ev["gen"]}', color='white', fontsize=7, alpha=0.6,
                             ha='center', va='top')
        except Exception:
            pass

    ax1.set_ylabel('Temperature (°C)', color='white', fontsize=11)
    ax1.set_ylim(40, 100)
    ax1.set_title('GPU Thermal Profile — Jellyfih Evolution Run', color='white', fontsize=13)

    # Legend
    patches = [mpatches.Patch(color=c, label=k) for k, c in colors.items()
               if k in set(throttles)]
    patches += [
        plt.Line2D([0], [0], color='#FF5722', linestyle='--', label='Pause 80°C'),
        plt.Line2D([0], [0], color='#00BCD4', linestyle='--', label='Resume 60°C'),
    ]
    ax1.legend(handles=patches, loc='upper left', fontsize=8,
               facecolor='#1a1a2e', labelcolor='white', framealpha=0.7)

    # ── Clock ratio plot ──────────────────────────────────────────
    ax2.fill_between(elapsed_min, clocks, alpha=0.6, color='#03A9F4')
    ax2.plot(elapsed_min, clocks, color='#03A9F4', linewidth=1)
    ax2.axhline(0.85, color='#FF9800', linestyle='--', linewidth=1, alpha=0.6)
    ax2.set_ylabel('Clock ratio', color='white', fontsize=10)
    ax2.set_xlabel('Elapsed time (minutes)', color='white', fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))

    # Stats annotation
    mean_temp = np.mean(temps)
    max_temp  = np.max(temps)
    pct_throttled = np.mean([t != 'none' and t != 'idle' for t in throttles]) * 100
    fig.text(0.99, 0.01,
             f'mean {mean_temp:.0f}°C  max {max_temp}°C  throttled {pct_throttled:.0f}% of time',
             color='#aaa', fontsize=8, ha='right', va='bottom')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'gpu_thermal_profile.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved to {out_path}")
    print(f"Stats: mean {mean_temp:.1f}°C, max {max_temp}°C, "
          f"{len(temps)} samples over {elapsed_min[-1]:.1f} min")

if __name__ == '__main__':
    main()
