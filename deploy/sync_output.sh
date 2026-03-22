#!/usr/bin/env bash
# deploy/sync_output.sh — runs LOCALLY to pull output/ from remote.
# Safe to run at any point during or after the run.
#
# Usage: ./deploy/sync_output.sh <SSH_HOST> <SSH_PORT>
#
# Find SSH_HOST and SSH_PORT from: vastai show instances
# Or from the vast.ai instance card in the web UI.
#
# Example:
#   ./deploy/sync_output.sh ssh.vast.ai 12345

set -euo pipefail

SSH_HOST="${1:?Usage: sync_output.sh <SSH_HOST> <SSH_PORT>}"
SSH_PORT="${2:?Usage: sync_output.sh <SSH_HOST> <SSH_PORT>}"

REMOTE="root@${SSH_HOST}:/root/jellyfih/output/"
LOCAL="./output/"

echo "Syncing output/ from ${SSH_HOST}:${SSH_PORT} ..."

rsync -avz --progress \
    -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=10" \
    "${REMOTE}" \
    "${LOCAL}"

echo ""
echo "Done. Key files:"
ls -lh ./output/checkpoint.pkl ./output/best_genomes.json ./output/evolution_log.csv \
        ./output/run.log 2>/dev/null || true

# Quick progress summary
if [ -f ./output/evolution_log.csv ]; then
    LINES=$(wc -l < ./output/evolution_log.csv)
    GENS=$(( (LINES - 1) / 16 ))
    echo "Approx generations logged: $GENS"
fi
