"""Shared subprocess orchestration helpers for pipeline scripts.

Extracts common patterns from production_run.py, capture_and_solve_stream.py,
and zig_capture.py:
  - Running bot/solver subprocesses with timeout and stderr capture
  - Finding the latest game log file
  - Parsing GAME_OVER score from stderr output
"""
import glob
import os
import re
import subprocess  # nosec B404


GAME_OVER_RE = re.compile(r'GAME_OVER\s+Score[:\s]+(\d+)')
ROUND_RE = re.compile(r'R(\d+)/(\d+)\s+Score:(\d+)')


def run_bot_game(exe_path, ws_url, cwd=None, timeout=180):
    """Start a bot process, capture stderr, wait with timeout, kill on timeout.

    Args:
        exe_path: Path to the bot executable.
        ws_url: WebSocket URL to pass as first argument.
        cwd: Working directory for the subprocess (default: exe's directory).
        timeout: Max seconds to wait before killing the process.

    Returns:
        (return_code, stderr_output, game_log_path)
        - return_code: Process exit code, or -1 if killed on timeout.
        - stderr_output: Full stderr as a string.
        - game_log_path: Path to the newest game_log_*.jsonl created during
          the run (in cwd), or None if no new log was found.
    """
    if cwd is None:
        cwd = os.path.dirname(exe_path) or '.'

    # Snapshot existing logs so we can detect new ones
    existing_logs = set(glob.glob(os.path.join(cwd, 'game_log_*.jsonl')))

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
    except subprocess.TimeoutExpired as e:
        return_code = -1
        stderr_output = (e.stderr or '') if isinstance(e.stderr, str) else ''

    # Find new game log
    log_path = find_latest_game_log(cwd, existing_logs=existing_logs)

    return return_code, stderr_output, log_path


def find_latest_game_log(directory, existing_logs=None):
    """Find the newest game_log_*.jsonl file in directory.

    Args:
        directory: Directory to search in.
        existing_logs: Optional set of paths to exclude (e.g. logs that
            existed before a subprocess was started). If provided, only
            logs NOT in this set are considered. Falls back to the newest
            log overall if no new logs are found.

    Returns:
        Path to the newest game log, or None if no logs exist.
    """
    all_logs = glob.glob(os.path.join(directory, 'game_log_*.jsonl'))
    if not all_logs:
        return None

    if existing_logs is not None:
        new_logs = [p for p in all_logs if p not in existing_logs]
        if new_logs:
            return max(new_logs, key=os.path.getmtime)
        # No new logs created — do NOT fall back to old logs (wrong difficulty/seed)
        return None
    return max(all_logs, key=os.path.getmtime)


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
