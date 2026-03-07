"""Shared subprocess orchestration helpers for pipeline scripts.

Extracts common patterns from production_run.py, capture_and_solve_stream.py,
and zig_capture.py:
  - Running bot/solver subprocesses with timeout and stderr capture
  - Capturing game log from stdout (Zig bot writes JSONL to stdout)
  - Parsing GAME_OVER score from stderr output
"""
import os
import re
import subprocess  # nosec B404


GAME_OVER_RE = re.compile(r'GAME_OVER\s+Score[:\s]+(\d+)')
ROUND_RE = re.compile(r'R(\d+)/(\d+)\s+Score:(\d+)')


def run_bot_game(exe_path, ws_url, cwd=None, timeout=180):
    """Start a bot process, capture stdout (game log) and stderr, wait with timeout.

    Args:
        exe_path: Path to the bot executable.
        ws_url: WebSocket URL to pass as first argument.
        cwd: Working directory for the subprocess (default: exe's directory).
        timeout: Max seconds to wait before killing the process.

    Returns:
        (return_code, stderr_output, stdout_output)
        - return_code: Process exit code, or -1 if killed on timeout.
        - stderr_output: Full stderr as a string.
        - stdout_output: Full stdout as a string (game log JSONL lines).
    """
    if cwd is None:
        cwd = os.path.dirname(exe_path) or '.'

    try:
        result = subprocess.run(  # nosec B603 B607
            [exe_path, ws_url],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return_code = result.returncode
        stderr_output = result.stderr
        stdout_output = result.stdout
    except subprocess.TimeoutExpired as e:
        return_code = -1
        stderr_output = (e.stderr or '') if isinstance(e.stderr, str) else ''
        stdout_output = (e.stdout or '') if isinstance(e.stdout, str) else ''

    return return_code, stderr_output, stdout_output


def parse_game_score(stderr_output):
    """Parse GAME_OVER Score:NNN from stderr output.

    Scans all lines and returns the last GAME_OVER score found (in case
    multiple appear due to post-optimize or retry).

    Args:
        stderr_output: Full stderr text from a bot/solver process.

    Returns:
        Integer score, or 0 if no GAME_OVER line was found.
    """
    score = 0
    for line in stderr_output.splitlines():
        m = GAME_OVER_RE.search(line)
        if m:
            score = int(m.group(1))
    return score


def parse_round_progress(line):
    """Parse R<round>/<max> Score:<score> from a stderr line.

    Args:
        line: A single line of stderr output.

    Returns:
        (round, max_rounds, score) tuple, or None if no match.
    """
    m = ROUND_RE.search(line)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None
