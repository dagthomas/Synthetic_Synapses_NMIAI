#!/usr/bin/env python
"""Sweep easy difficulty (1 bot, 12x10). Port 9850."""
import subprocess, sys  # nosec B404
subprocess.run([sys.executable, "sweep.py", "easy", "--seeds", "40", "--port", "9850"], cwd=__import__("os").path.dirname(__import__("os").path.abspath(__file__)), timeout=3600)  # nosec B603 B607
