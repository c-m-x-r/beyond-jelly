"""
Continuous GPU temperature poller.
Writes timestamp, temp, clocks, throttle reason to output/gpu_temp_timeline.csv every 2s.
Run alongside evolve.py: uv run python gpu_monitor.py
"""
import subprocess
import csv
import time
import os
import signal
import sys

OUTPUT_DIR = "output"
POLL_INTERVAL = 2.0

def query():
    try:
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=temperature.gpu,clocks.current.graphics,clocks.max.graphics,'
             'clocks_throttle_reasons.active,power.draw,utilization.gpu',
             '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()
        parts = [p.strip() for p in out.split(',')]
        temp      = int(parts[0])
        clock     = int(parts[1])
        max_clock = int(parts[2])
        throttle  = parts[3]   # hex string e.g. 0x0000000000000004
        power     = float(parts[4]) if parts[4] not in ('N/A', '[N/A]') else 0.0
        util      = int(parts[5])
        return temp, clock, max_clock, throttle, power, util
    except Exception:
        return 0, 0, 1, '0x0', 0.0, 0

def throttle_label(hex_str):
    try:
        val = int(hex_str, 16)
    except Exception:
        return 'unknown'
    if val == 0x00:  return 'none'
    if val & 0x01:   return 'idle'
    if val & 0x04:   return 'power_cap'
    if val & 0x08:   return 'hw_slowdown'
    if val & 0x20:   return 'sw_thermal'
    if val & 0x40:   return 'hw_power_brake'
    return f'other({hex_str})'

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, 'gpu_temp_timeline.csv')
    file_exists = os.path.exists(csv_path)
    f = open(csv_path, 'a', newline='')
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(['wall_time', 'temp_c', 'clock_mhz', 'max_clock_mhz',
                         'clock_ratio', 'throttle', 'power_w', 'util_pct'])
    print(f"Logging GPU state to {csv_path} every {POLL_INTERVAL}s  (Ctrl+C to stop)")

    def handle_exit(sig, frame):
        print("\nStopping monitor.")
        f.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    while True:
        t = time.time()
        temp, clock, max_clock, throttle_hex, power, util = query()
        ratio = clock / max_clock if max_clock > 0 else 0.0
        writer.writerow([f'{t:.2f}', temp, clock, max_clock,
                         f'{ratio:.3f}', throttle_label(throttle_hex),
                         f'{power:.1f}', util])
        f.flush()
        elapsed = time.time() - t
        time.sleep(max(0.0, POLL_INTERVAL - elapsed))

if __name__ == '__main__':
    main()
