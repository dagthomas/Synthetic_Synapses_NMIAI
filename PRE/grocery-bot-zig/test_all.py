#!/usr/bin/env python
"""Run sim_server and bot for all difficulties."""
import subprocess, sys, time, os

difficulties = ["easy", "medium", "hard", "expert"]
base_port = 9890

for i, diff in enumerate(difficulties):
    port = base_port + i
    print(f"\n{'='*50}")
    print(f"Testing {diff.upper()} on port {port}")
    print(f"{'='*50}")

    srv = subprocess.Popen(
        [sys.executable, "sim_server.py", str(port), diff],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    time.sleep(2)

    bot_exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zig-out", "bin", "grocery-bot.exe")
    bot = subprocess.Popen(
        [bot_exe, f"ws://localhost:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    bot_out, _ = bot.communicate(timeout=120)
    output = bot_out.decode(errors='replace')

    # Extract final score
    for line in output.split('\n'):
        if 'Game over' in line:
            print(f"  {line.strip()}")
        if 'Score:' in line and '/300' in line:
            last_score_line = line.strip()
    print(f"  Last: {last_score_line}")

    time.sleep(1)
    srv.terminate()
    try:
        srv.communicate(timeout=5)
    except:
        srv.kill()

print(f"\n{'='*50}")
print("ALL TESTS COMPLETE")
print(f"{'='*50}")
