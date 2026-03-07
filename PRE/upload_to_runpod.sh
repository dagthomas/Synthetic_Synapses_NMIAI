#!/bin/bash
# Upload all necessary files to RunPod server.
# Usage: bash PRE/upload_to_runpod.sh [HOST] [PORT]
#
# Defaults to the current RunPod pod. Override with positional args:
#   bash PRE/upload_to_runpod.sh 103.196.86.219 16075

set -e

HOST="${1:-103.196.86.219}"
PORT="${2:-16075}"
DEST="root@${HOST}"
REMOTE_DIR="/workspace/AINM/PRE"
KEY="/c/Users/larsh/.ssh/id_ed25519"
SSH_CMD="ssh -o StrictHostKeyChecking=no -p ${PORT} -i ${KEY}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRE_DIR="$SCRIPT_DIR"

echo "=== Upload to RunPod ==="
echo "Host: ${DEST}:${PORT}"
echo "Remote dir: ${REMOTE_DIR}"
echo ""

# 1. Create remote directory structure
echo "--- Creating remote directories ---"
$SSH_CMD "$DEST" "mkdir -p \
    ${REMOTE_DIR}/grocery-bot-gpu/captures \
    ${REMOTE_DIR}/grocery-bot-gpu/cache \
    ${REMOTE_DIR}/grocery-bot-b200 \
    ${REMOTE_DIR}/replay/app/src/lib/components \
    ${REMOTE_DIR}/replay/app/src/routes"

# 2. grocery-bot-gpu (Python solver code)
echo ""
echo "--- Syncing grocery-bot-gpu/*.py ---"
rsync -avz --include='*.py' --exclude='*' \
    -e "$SSH_CMD" \
    "$PRE_DIR/grocery-bot-gpu/" \
    "${DEST}:${REMOTE_DIR}/grocery-bot-gpu/"

# 3. grocery-bot-gpu/captures (JSON captures)
echo ""
echo "--- Syncing grocery-bot-gpu/captures/ ---"
rsync -avz \
    -e "$SSH_CMD" \
    "$PRE_DIR/grocery-bot-gpu/captures/" \
    "${DEST}:${REMOTE_DIR}/grocery-bot-gpu/captures/"

# 4. grocery-bot-gpu/cache (precomputed route tables)
echo ""
echo "--- Syncing grocery-bot-gpu/cache/ ---"
rsync -avz \
    -e "$SSH_CMD" \
    "$PRE_DIR/grocery-bot-gpu/cache/" \
    "${DEST}:${REMOTE_DIR}/grocery-bot-gpu/cache/"

# 5. grocery-bot-b200 (B200 solver code)
echo ""
echo "--- Syncing grocery-bot-b200/ ---"
rsync -avz --include='*.py' --include='requirements.txt' --exclude='*' \
    -e "$SSH_CMD" \
    "$PRE_DIR/grocery-bot-b200/" \
    "${DEST}:${REMOTE_DIR}/grocery-bot-b200/"

# 6. replay/app (SvelteKit GUI) - config files
echo ""
echo "--- Syncing replay/app config files ---"
rsync -avz \
    --include='package.json' \
    --include='svelte.config.js' \
    --include='vite.config.js' \
    --include='.env' \
    --exclude='*' \
    -e "$SSH_CMD" \
    "$PRE_DIR/replay/app/" \
    "${DEST}:${REMOTE_DIR}/replay/app/"

# 7. replay/app/src (app.html + lib + routes)
echo ""
echo "--- Syncing replay/app/src/ ---"
rsync -avz --delete \
    -e "$SSH_CMD" \
    "$PRE_DIR/replay/app/src/" \
    "${DEST}:${REMOTE_DIR}/replay/app/src/"

# 8. docker-compose.yml
echo ""
echo "--- Syncing docker-compose.yml ---"
rsync -avz \
    -e "$SSH_CMD" \
    "$PRE_DIR/replay/docker-compose.yml" \
    "${DEST}:${REMOTE_DIR}/replay/"

# 9. setup_runpod.sh
echo ""
echo "--- Syncing setup_runpod.sh ---"
rsync -avz \
    -e "$SSH_CMD" \
    "$PRE_DIR/setup_runpod.sh" \
    "${DEST}:${REMOTE_DIR}/"

echo ""
echo "============================================="
echo "  Upload complete!"
echo "============================================="
echo ""
echo "Next steps on RunPod:"
echo "  $SSH_CMD $DEST"
echo "  cd ${REMOTE_DIR} && bash setup_runpod.sh"
