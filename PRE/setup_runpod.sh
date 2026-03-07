#!/bin/bash
# RunPod B200 setup script — run this once after cloning the repo.
# Usage: bash PRE/setup_runpod.sh
set -e

echo "=== RunPod B200 Setup ==="

cd "$(dirname "$0")"
PRE_DIR=$(pwd)

# ── 1. Verify GPU ─────────────────────────────────────────────────────
echo ""
echo "--- GPU Check ---"
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    print(f'GPU: {props.name}')
    print(f'VRAM: {vram_gb:.0f} GB')
"

# ── 2. Python dependencies ────────────────────────────────────────────
echo ""
echo "--- Python dependencies ---"
pip install -q numpy websockets psycopg2-binary

# ── 3. Node.js (for GUI) ─────────────────────────────────────────────
echo ""
echo "--- Node.js setup ---"
if ! command -v node &> /dev/null; then
    echo "Installing Node.js 20..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi
echo "Node: $(node --version)"
echo "npm: $(npm --version)"

# ── 4. PostgreSQL via Docker ──────────────────────────────────────────
echo ""
echo "--- PostgreSQL ---"
if command -v docker &> /dev/null; then
    cd "$PRE_DIR/replay"
    docker compose up -d
    echo "Waiting for PostgreSQL to be ready..."
    for i in $(seq 1 30); do
        if docker compose exec -T db pg_isready -U grocery -d grocery_bot > /dev/null 2>&1; then
            echo "PostgreSQL ready!"
            break
        fi
        sleep 1
    done

    # Run GPU solution_store schema setup
    cd "$PRE_DIR/grocery-bot-gpu"
    python3 -c "from solution_store import ensure_schema; ensure_schema()" 2>/dev/null || true
    echo "DB schema ensured"
else
    echo "WARNING: Docker not found. Install PostgreSQL manually or install Docker."
    echo "  Alternative: apt-get install postgresql && ..."
fi

# ── 5. Install GUI (SvelteKit) ────────────────────────────────────────
echo ""
echo "--- GUI dependencies ---"
cd "$PRE_DIR/replay/app"
npm install

# ── 6. Verify B200 config ────────────────────────────────────────────
echo ""
echo "--- B200 Solver Config ---"
cd "$PRE_DIR/grocery-bot-b200"
python3 b200_config.py

# ── Done ──────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo "  Setup complete!"
echo "============================================="
echo ""
echo "Start the GUI:"
echo "  cd $PRE_DIR/replay/app && npm run dev"
echo ""
echo "Then open in your browser:"
echo "  http://localhost:5173/stepladder"
echo ""
echo "On RunPod, access via:"
echo "  https://<POD-ID>-5173.proxy.runpod.net/stepladder"
echo ""
echo "Or use SSH tunnel:"
echo "  ssh -L 5173:localhost:5173 root@<POD-IP>"
echo "  Then open http://localhost:5173/stepladder"
echo ""
