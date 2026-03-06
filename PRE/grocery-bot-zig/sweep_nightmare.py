#!/usr/bin/env python
"""Sweep nightmare difficulty (20 bots, 30x18, 3 dropoffs). Port 9890."""
import subprocess, sys  # nosec B404
subprocess.run([sys.executable, "sweep.py", "nightmare", "--seeds", "40", "--port", "9890"], cwd=__import__("os").path.dirname(__import__("os").path.abspath(__file__)), timeout=7200)  # nosec B603 B607
