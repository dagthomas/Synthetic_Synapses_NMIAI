#!/usr/bin/env python
"""Sweep hard difficulty (5 bots, 22x14). Port 9870."""
import subprocess, sys  # nosec B404
subprocess.run([sys.executable, "sweep.py", "hard", "--seeds", "40", "--port", "9870"], cwd=__import__("os").path.dirname(__import__("os").path.abspath(__file__)), timeout=3600)  # nosec B603 B607
