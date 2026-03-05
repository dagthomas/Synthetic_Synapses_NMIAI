#!/usr/bin/env python
"""Sweep expert difficulty (10 bots, 28x18). Port 9880."""
import subprocess, sys  # nosec B404
subprocess.run([sys.executable, "sweep.py", "expert", "--seeds", "40", "--port", "9880"], cwd=__import__("os").path.dirname(__import__("os").path.abspath(__file__)), timeout=3600)  # nosec B603 B607
