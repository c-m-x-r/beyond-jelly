#!/usr/bin/env bash
# deploy/destroy.sh — final sync then destroy the vast.ai instance.
# Install vastai CLI first: pip install vastai && vastai set api-key <KEY>
#
# Usage: ./deploy/destroy.sh <INSTANCE_ID> <SSH_HOST> <SSH_PORT>
#
# Find INSTANCE_ID from: vastai show instances
#
# Example:
#   ./deploy/destroy.sh 12345678 ssh.vast.ai 54321

set -euo pipefail

INSTANCE_ID="${1:?Usage: destroy.sh <INSTANCE_ID> <SSH_HOST> <SSH_PORT>}"
SSH_HOST="${2:?Usage: destroy.sh <INSTANCE_ID> <SSH_HOST> <SSH_PORT>}"
SSH_PORT="${3:?Usage: destroy.sh <INSTANCE_ID> <SSH_HOST> <SSH_PORT>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================"
echo " Jellyfih — destroy instance $INSTANCE_ID"
echo "======================================================"

echo ""
echo "=== Final output sync ==="
bash "$SCRIPT_DIR/sync_output.sh" "$SSH_HOST" "$SSH_PORT"  # syncs all run_* dirs

echo ""
read -rp "Sync complete. Destroy instance $INSTANCE_ID? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted. Instance NOT destroyed."
    exit 0
fi

echo ""
echo "=== Destroying instance $INSTANCE_ID ==="
vastai destroy instance "$INSTANCE_ID"

echo ""
echo "Instance $INSTANCE_ID destroyed."
echo "Verify with: vastai show instances"
