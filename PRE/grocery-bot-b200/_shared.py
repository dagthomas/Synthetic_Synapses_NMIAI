"""Add grocery-bot-gpu/ to sys.path and override solution_store with local file-based version."""
import sys
import os

# Add GPU dir for shared imports (game_engine, precompute, etc.)
GPU_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'grocery-bot-gpu'))
if GPU_DIR not in sys.path:
    sys.path.insert(0, GPU_DIR)

# Override solution_store with local file-based version (no PostgreSQL dependency).
# This MUST happen after GPU_DIR is on sys.path but before any other imports,
# so that all code (including gpu_sequential_solver, nightmare_offline, etc.)
# uses local_store instead of the PostgreSQL-based solution_store.
B200_DIR = os.path.dirname(os.path.abspath(__file__))
import local_store
sys.modules['solution_store'] = local_store
