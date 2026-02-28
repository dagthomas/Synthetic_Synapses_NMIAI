#!/usr/bin/env python
"""Sweep expert difficulty (10 bots, 28x18). Port 9880."""
import subprocess, sys
subprocess.run([sys.executable, "sweep.py", "expert", "--seeds", "30", "--port", "9880"], cwd=__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
