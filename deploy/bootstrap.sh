#!/usr/bin/env bash
# deploy/bootstrap.sh — runs ON the vast.ai instance after SSH in.
#
# Usage: bash bootstrap.sh <GITHUB_TOKEN> [BRANCH] [N_GENS] [N_PARALLEL]
#
# GITHUB_TOKEN: classic token or fine-grained with repo read access.
#   Create at: https://github.com/settings/tokens
#
# Example:
#   bash bootstrap.sh ghp_xxxxxxxxxxxxxxxxxxxx march 50 4

set -euo pipefail

GITHUB_TOKEN="${1:?Usage: bootstrap.sh <GITHUB_TOKEN> [BRANCH] [N_GENS] [N_PARALLEL]}"
BRANCH="${2:-march}"
REPO="c-m-x-r/jellyfih"
WORKDIR="/root/jellyfih"
N_GENS="${3:-50}"
N_PARALLEL="${4:-4}"   # number of independent evolution loops to run in parallel

echo "======================================================"
echo " Jellyfih vast.ai bootstrap"
echo " branch: $BRANCH  |  gens: $N_GENS  |  parallel runs: $N_PARALLEL"
echo "======================================================"

echo ""
echo "=== [1/7] System deps ==="
apt-get update -qq
apt-get install -y -qq ffmpeg tmux rsync curl git python3 python3-pip htop nvtop 2>/dev/null \
    || apt-get install -y -qq ffmpeg tmux rsync curl git python3 python3-pip htop
# nvtop may not be in older Ubuntu repos; fall back to gpustat
nvtop --version 2>/dev/null || pip install -q gpustat
echo "  OK"

echo ""
echo "=== [2/7] Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"' >> /root/.bashrc
uv --version
echo "  OK"

echo ""
echo "=== [3/7] Installing Node.js 20 + Claude Code ==="
curl -fsSL https://deb.nodesource.com/setup_20.x | bash - 2>/dev/null
apt-get install -y nodejs 2>/dev/null || echo "  WARNING: Node.js install failed — Claude Code skipped"
if command -v node &>/dev/null; then
    node --version && npm --version
    npm install -g @anthropic-ai/claude-code 2>/dev/null && echo "  Claude Code installed" \
        || echo "  WARNING: Claude Code install failed (set ANTHROPIC_API_KEY to use)"
else
    echo "  Skipping Claude Code (no Node.js)"
fi
echo "  OK"

echo ""
echo "=== [4/7] Cloning repo ==="
mkdir -p "$WORKDIR"
cd "$WORKDIR"
if [ -d "$WORKDIR/.git" ]; then
    echo "  Repo already exists — pulling latest"
    git remote set-url origin "https://${GITHUB_TOKEN}@github.com/${REPO}.git"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "  Initialising git in existing directory (preserving output/)"
    git init
    git remote add origin "https://${GITHUB_TOKEN}@github.com/${REPO}.git"
    git fetch origin "$BRANCH"
    git checkout -b "$BRANCH" --track "origin/$BRANCH"
fi
echo "  OK — $(git log --oneline -1)"

echo ""
echo "=== [5/7] Installing Python deps ==="
cd "$WORKDIR"
uv sync
echo "  OK"

echo ""
echo "=== [6/7] GPU setup ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Fix: CUDA 12.x images ship a compat libcuda.so (e.g. 530) but vast.ai hosts
# run newer drivers (e.g. 580). Taichi needs an unversioned libcuda.so that
# resolves to the *host* driver, not the compat stub. Create the symlink if missing.
LIBCUDA_DIR="/lib/x86_64-linux-gnu"
if [ ! -e "$LIBCUDA_DIR/libcuda.so" ] && [ -e "$LIBCUDA_DIR/libcuda.so.1" ]; then
    echo "  Fixing libcuda.so symlink (driver/compat mismatch workaround)"
    ln -sf "$LIBCUDA_DIR/libcuda.so.1" "$LIBCUDA_DIR/libcuda.so"
    ldconfig
fi

# Enable CUDA MPS for fair kernel interleaving across parallel processes
echo "  Starting CUDA MPS daemon..."
nvidia-cuda-mps-control -d 2>/dev/null && echo "  MPS started" \
    || echo "  WARNING: MPS start failed (may already be running, or not supported)"

mkdir -p "$WORKDIR/output"

echo ""
echo "=== [7/7] Launching $N_PARALLEL parallel evolve.py runs ==="

# Write tmux config (WSL clip.exe replaced with copy-selection-and-cancel for server)
cat > /root/.tmux.conf << 'TMUXCONF'
# --- General Settings ---
set -g mouse on               # Enable mouse mode
set -g base-index 1           # Start windows at 1
setw -g pane-base-index 1     # Start panes at 1
set -g renumber-windows on    # Automatically renumber windows
set -g status-interval 5      # Update status bar more often

# --- Key Remaps ---
unbind C-b
set -g prefix `
bind ` send-prefix

bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"

# --- Selection ---
setw -g mode-keys vi
bind-key -T copy-mode-vi MouseDragEnd1Pane send-keys -X copy-selection-and-cancel

# --- Automatic Layout ---
set-hook -g session-created 'split-window -h; split-window -v; select-pane -t 1'
TMUXCONF

# Kill any existing evo session
tmux kill-session -t evo 2>/dev/null || true

# Create session with a monitor window (no command — interactive shell)
TMUX='' tmux new-session -d -s evo -n "monitor"

# Launch N_PARALLEL independent evolution runs, each in its own window
RUN_IDS=("run_a" "run_b" "run_c" "run_d" "run_e" "run_f" "run_g" "run_h")
for i in $(seq 0 $((N_PARALLEL - 1))); do
    ID="${RUN_IDS[$i]}"
    mkdir -p "$WORKDIR/output/$ID"
    WIN_CMD="cd $WORKDIR && export PATH=\"\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH\" && uv run python evolve.py --no-thermal --gens $N_GENS --run-id $ID 2>&1 | tee $WORKDIR/output/$ID/run.log"
    TMUX='' tmux new-window -t evo -n "$ID"
    TMUX='' tmux send-keys -t "evo:$ID" "$WIN_CMD" Enter
done

# Select monitor window for the user to land on
TMUX='' tmux select-window -t evo:monitor

echo ""
echo "======================================================"
echo " Bootstrap complete!"
echo ""
echo " $N_PARALLEL parallel evolution runs starting in tmux windows: run_a..run_$(printf '%s' "${RUN_IDS[$((N_PARALLEL-1))]}")"
echo ""
echo " Attach:         TMUX='' tmux attach -t evo"
echo " Switch window:  tmux switch -t evo   (if inside vast.ai ssh_tmux)"
echo " Monitor GPU:    watch -n2 nvidia-smi          (in monitor window)"
echo " GPU detail:     nvtop  OR  gpustat --interval 1"
echo " Per-run log:    tail -f $WORKDIR/output/run_a/run.log"
echo " Per-run prog:   tail -1 $WORKDIR/output/run_a/evolution_log.csv"
echo " Sync output:    ./deploy/sync_output.sh 38.117.87.51 <PORT>"
echo "======================================================"
