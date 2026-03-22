#!/usr/bin/env bash
# deploy/bootstrap.sh — runs ON the vast.ai instance after SSH in.
#
# Usage: bash bootstrap.sh <GITHUB_TOKEN> [BRANCH]
#
# GITHUB_TOKEN: classic token or fine-grained with repo read access.
#   Create at: https://github.com/settings/tokens
#
# Example:
#   bash bootstrap.sh ghp_xxxxxxxxxxxxxxxxxxxx march

set -euo pipefail

GITHUB_TOKEN="${1:?Usage: bootstrap.sh <GITHUB_TOKEN> [BRANCH]}"
BRANCH="${2:-march}"
REPO="c-m-x-r/jellyfih"
WORKDIR="/root/jellyfih"
N_GENS=50

echo "======================================================"
echo " Jellyfih vast.ai bootstrap"
echo " branch: $BRANCH  |  gens: $N_GENS"
echo "======================================================"

echo ""
echo "=== [1/6] System deps ==="
apt-get update -qq
apt-get install -y -qq ffmpeg tmux rsync curl git
echo "  OK"

echo ""
echo "=== [2/6] Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"' >> /root/.bashrc
uv --version
echo "  OK"

echo ""
echo "=== [3/6] Cloning repo ==="
if [ -d "$WORKDIR/.git" ]; then
    echo "  Repo already exists — pulling latest"
    cd "$WORKDIR"
    git remote set-url origin "https://${GITHUB_TOKEN}@github.com/${REPO}.git"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    git clone --branch "$BRANCH" \
        "https://${GITHUB_TOKEN}@github.com/${REPO}.git" \
        "$WORKDIR"
    cd "$WORKDIR"
fi
echo "  OK — $(git log --oneline -1)"

echo ""
echo "=== [4/6] Installing Python deps ==="
cd "$WORKDIR"
uv sync
echo "  OK"

echo ""
echo "=== [5/6] Checking GPU and checkpoint ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
mkdir -p "$WORKDIR/output"
if [ -f "$WORKDIR/output/checkpoint.pkl" ]; then
    echo "  checkpoint.pkl found — run will RESUME from last checkpoint"
else
    echo "  No checkpoint — starting fresh from gen 0"
fi

echo ""
echo "=== [6/6] Starting evolve.py in tmux session 'evo' ==="
# Kill any existing session to avoid conflicts on re-runs
tmux kill-session -t evo 2>/dev/null || true

tmux new-session -d -s evo "cd $WORKDIR && \
    export PATH=\"\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH\" && \
    uv run python evolve.py --no-thermal --gens $N_GENS 2>&1 | tee output/run.log"

echo ""
echo "======================================================"
echo " Bootstrap complete!"
echo ""
echo " Attach:        tmux attach -t evo"
echo " Follow log:    tail -f $WORKDIR/output/run.log"
echo " GPU status:    watch -n2 nvidia-smi"
echo " Progress:      tail -1 $WORKDIR/output/evolution_log.csv"
echo "======================================================"
