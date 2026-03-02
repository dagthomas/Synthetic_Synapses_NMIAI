#!/usr/bin/env python
"""Sweep medium difficulty (3 bots, 16x12). Port 9860."""
import subprocess, sys
subprocess.run([sys.executable, "sweep.py", "medium", "--seeds", "40", "--port", "9860"], cwd=__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
