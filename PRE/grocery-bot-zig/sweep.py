#!/usr/bin/env python
"""Sweep runner: runs bot vs sim_server across many seeds, reports statistics.
Usage: python sweep.py [difficulty] [--seeds N] [--port PORT]
"""
import subprocess, sys, os, re, time, statistics, argparse, threading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BOT_EXE = os.path.join(SCRIPT_DIR, "zig-out", "bin", "grocery-bot.exe")


def drain_pipe(pipe, lines):
    """Read lines from a pipe into a list (run in a thread)."""
    for line in pipe:
        lines.append(line.decode(errors="replace").rstrip())
    pipe.close()


def run_one(difficulty, seed, port):
    """Run a single game with given seed. Returns (score, orders_completed) or None on failure."""
    srv = subprocess.Popen(
        [sys.executable, "-u", "sim_server.py", str(port), difficulty, "--seed", str(seed)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=SCRIPT_DIR,
    )
    # Drain server stdout in a thread to prevent pipe buffer deadlock
    srv_lines = []
    srv_thread = threading.Thread(target=drain_pipe, args=(srv.stdout, srv_lines), daemon=True)
    srv_thread.start()

    time.sleep(0.5)

    bot = subprocess.Popen(
        [BOT_EXE, f"ws://localhost:{port}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd=SCRIPT_DIR,
    )

    try:
        bot.wait(timeout=30)
    except subprocess.TimeoutExpired:
        bot.kill()
        bot.wait()
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except:
            srv.kill()
        return None

    time.sleep(0.3)
    srv.terminate()
    try:
        srv.wait(timeout=5)
    except:
        srv.kill()
    srv_thread.join(timeout=2)

    srv_text = "\n".join(srv_lines)

    # Parse score from server output (look for GAME OVER section)
    score_match = re.search(r"Score:\s*(\d+)", srv_text[srv_text.rfind("GAME OVER"):] if "GAME OVER" in srv_text else srv_text)
    orders_match = re.search(r"Orders completed:\s*(\d+)", srv_text)

    if score_match:
        score = int(score_match.group(1))
        orders = int(orders_match.group(1)) if orders_match else 0
        return (score, orders)
    return None


def main():
    parser = argparse.ArgumentParser(description="Sweep runner for grocery bot")
    parser.add_argument("difficulty", nargs="?", default="medium",
                        choices=["easy", "medium", "hard", "expert"])
    parser.add_argument("--seeds", type=int, default=50, help="Number of seeds to test")
    parser.add_argument("--port", type=int, default=9880, help="Base port")
    args = parser.parse_args()

    print(f"Sweeping {args.seeds} seeds on {args.difficulty} (port {args.port})")
    print(f"Bot: {BOT_EXE}")
    print()

    scores = []
    orders = []
    failures = 0

    for i in range(args.seeds):
        seed = 1000 + i  # Deterministic seed sequence
        result = run_one(args.difficulty, seed, args.port)
        if result is None:
            failures += 1
            print(f"  Seed {seed}: FAILED")
        else:
            s, o = result
            scores.append(s)
            orders.append(o)
            print(f"  Seed {seed}: score={s}, orders={o}")

    print()
    print(f"{'='*50}")
    print(f"Results: {args.difficulty}, {len(scores)}/{args.seeds} successful")
    if scores:
        scores_sorted = sorted(scores)
        n = len(scores_sorted)
        p25 = scores_sorted[n // 4]
        p75 = scores_sorted[3 * n // 4]
        print(f"  Mean:   {statistics.mean(scores):.1f}")
        print(f"  Median: {statistics.median(scores):.1f}")
        print(f"  Stdev:  {statistics.stdev(scores):.1f}" if n > 1 else "  Stdev:  N/A")
        print(f"  Min:    {min(scores)}")
        print(f"  Max:    {max(scores)}")
        print(f"  P25:    {p25}")
        print(f"  P75:    {p75}")
        print(f"  Orders: mean={statistics.mean(orders):.1f}, total={sum(orders)}")
    if failures:
        print(f"  Failures: {failures}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
