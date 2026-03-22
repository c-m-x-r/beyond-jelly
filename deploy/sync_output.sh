#!/usr/bin/env bash
# deploy/sync_output.sh — runs LOCALLY to pull output/ from remote.
# Safe to run at any point during or after the run.
#
# Usage: ./deploy/sync_output.sh <SSH_HOST> <SSH_PORT> [RUN_ID]
#
# Find SSH_HOST and SSH_PORT from: vastai show instances
# Or from the vast.ai instance card in the web UI.
#
# Without RUN_ID: syncs the entire output/ tree (all runs).
# With RUN_ID:    syncs only output/<RUN_ID>/ (e.g. run_a).
#
# Examples:
#   ./deploy/sync_output.sh 38.117.87.51 46231          # all runs
#   ./deploy/sync_output.sh 38.117.87.51 46231 run_a    # single run

set -euo pipefail

SSH_HOST="${1:?Usage: sync_output.sh <SSH_HOST> <SSH_PORT> [RUN_ID]}"
SSH_PORT="${2:?Usage: sync_output.sh <SSH_HOST> <SSH_PORT> [RUN_ID]}"
RUN_ID="${3:-}"

SSH_OPTS="-p ${SSH_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=10"

if [ -n "$RUN_ID" ]; then
    REMOTE="root@${SSH_HOST}:/root/jellyfih/output/${RUN_ID}/"
    LOCAL="./output/${RUN_ID}/"
    mkdir -p "$LOCAL"
    echo "Syncing output/${RUN_ID}/ from ${SSH_HOST}:${SSH_PORT} ..."
else
    REMOTE="root@${SSH_HOST}:/root/jellyfih/output/"
    LOCAL="./output/"
    echo "Syncing entire output/ from ${SSH_HOST}:${SSH_PORT} ..."
fi

rsync -avz --progress \
    -e "ssh $SSH_OPTS" \
    "${REMOTE}" \
    "${LOCAL}"

echo ""
echo "Done. Run directories:"
for d in ./output/run_*/; do
    [ -d "$d" ] || continue
    CSV="$d/evolution_log.csv"
    if [ -f "$CSV" ]; then
        LINES=$(wc -l < "$CSV")
        GENS=$(( (LINES - 1) / 16 ))
        BEST=$(tail -n +2 "$CSV" | sort -t',' -k5 -rn | head -1 | cut -d',' -f5)
        echo "  $d — ~${GENS} gens, best fitness ~${BEST}"
    else
        echo "  $d — no evolution_log.csv yet"
    fi
done
