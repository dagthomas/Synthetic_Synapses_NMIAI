#!/usr/bin/env python
"""Run a single medium game and print server output."""
import subprocess, sys, os, time, threading  # nosec B404

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BOT_EXE = os.path.join(SCRIPT_DIR, "zig-out", "bin", "grocery-bot.exe")
PORT = 9971
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

srv = subprocess.Popen(  # nosec B603 B607
    [sys.executable, "-u", "sim_server.py", str(PORT), "medium", "--seed", str(SEED)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=SCRIPT_DIR,
)
lines = []
def drain():
    for line in srv.stdout:
        text = line.decode(errors="replace").rstrip()
        lines.append(text)
        print(text)
    srv.stdout.close()
t = threading.Thread(target=drain, daemon=True)
t.start()

time.sleep(0.5)
bot = subprocess.Popen([BOT_EXE, f"ws://localhost:{PORT}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=SCRIPT_DIR)  # nosec B603 B607
bot.wait(timeout=30)
time.sleep(0.3)
srv.terminate()
srv.wait(timeout=5)
t.join(timeout=2)
