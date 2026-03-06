"""Add grocery-bot-gpu/ to sys.path for shared imports."""
import sys
import os

GPU_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'grocery-bot-gpu'))
if GPU_DIR not in sys.path:
    sys.path.insert(0, GPU_DIR)
