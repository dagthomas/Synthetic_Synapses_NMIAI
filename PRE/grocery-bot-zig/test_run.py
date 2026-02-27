#!/usr/bin/env python
"""Run sim_server and bot together, capture output."""
import subprocess, sys, time, threading, os

diff = sys.argv[1] if len(sys.argv) > 1 else "easy"
port = 9883

# Start server
srv = subprocess.Popen(
    [sys.executable, "sim_server.py", str(port), diff],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    cwd=os.path.dirname(os.path.abspath(__file__))
)

time.sleep(2)

# Start bot
bot_exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zig-out", "bin", "grocery-bot.exe")
bot = subprocess.Popen(
    [bot_exe, f"ws://localhost:{port}"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    cwd=os.path.dirname(os.path.abspath(__file__))
)

# Wait for bot to finish (game over)
bot_out, _ = bot.communicate(timeout=120)
print("=== BOT OUTPUT ===")
print(bot_out.decode(errors='replace'))

# Wait a bit for server to finish logging
time.sleep(1)
srv.terminate()
srv_out, _ = srv.communicate(timeout=5)
print("=== SERVER OUTPUT ===")
print(srv_out.decode(errors='replace'))
