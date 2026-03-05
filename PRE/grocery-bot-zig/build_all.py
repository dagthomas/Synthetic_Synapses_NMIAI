#!/usr/bin/env python
"""Build all 4 difficulty-specific executables + generic bot."""
import subprocess, sys, os  # nosec B404

ZIG = r"C:\Users\dagth\zig15\zig-x86_64-windows-0.15.2\zig.exe"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

difficulties = ["easy", "medium", "hard", "expert"]

def build_one(difficulty=None):
    cmd = [ZIG, "build", "-Doptimize=ReleaseFast"]
    if difficulty:
        cmd.append(f"-Ddifficulty={difficulty}")
    label = difficulty or "auto"
    print(f"Building {label}...", end=" ", flush=True)
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True, timeout=300)  # nosec B603 B607
    if result.returncode != 0:
        print(f"FAILED")
        print(result.stderr)
        return False
    print("OK")
    return True

if __name__ == "__main__":
    # Build specific difficulty or all
    if len(sys.argv) > 1:
        diff = sys.argv[1]
        if diff == "all":
            ok = True
            for d in difficulties:
                ok = build_one(d) and ok
            build_one(None)  # also build generic
            sys.exit(0 if ok else 1)
        else:
            sys.exit(0 if build_one(diff) else 1)
    else:
        # Build all
        ok = True
        for d in difficulties:
            ok = build_one(d) and ok
        build_one(None)  # also build generic
        sys.exit(0 if ok else 1)
