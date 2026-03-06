#!/usr/bin/env python3
"""Fetch a live game token from app.ainm.no using Playwright.

First run: use --setup to open headed browser and log in via Google OAuth.
Subsequent runs: headless, session cookies reused automatically.

Usage:
    python fetch_token.py hard              # headless (after login)
    python fetch_token.py hard --headed     # visible browser window
    python fetch_token.py hard --setup      # first-time login (opens browser)
    python fetch_token.py hard --timeout 90 # custom wait (default 90s)
    python fetch_token.py hard --json       # JSON output

Output (stdout): WSS URL (or JSON with --json).
Errors go to stderr.
"""
import sys
import re
import os
import json
import time
from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(SCRIPT_DIR, '.playwright_profile')
CHALLENGE_URL = 'https://app.ainm.no/challenge'
TOKEN_RE = re.compile(r'wss://game\.ainm\.no/ws\?token=[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+')


def _clean_jwt_url(url: str) -> str:
    """Strip trailing non-JWT characters from extracted URL.

    HS256 JWT signatures are exactly 43 base64url chars (32 bytes).
    Page elements sometimes append digits/text that the greedy regex captures.
    """
    parts = url.split('.')
    if len(parts) >= 3:
        sig = parts[-1]
        # HS256 signature = 32 bytes = 43 base64url chars
        if len(sig) > 43:
            parts[-1] = sig[:43]
            return '.'.join(parts)
    return url
DIFFICULTIES = ['easy', 'medium', 'hard', 'expert', 'nightmare']


def _is_logged_in(page) -> bool:
    """Check if we're on the challenge page (not redirected to login)."""
    url = page.url.lower()
    if '/login' in url or '/auth' in url or 'accounts.google' in url:
        return False
    try:
        btns = page.locator('button:has-text("Easy"), button:has-text("Medium"), '
                            'button:has-text("Hard"), button:has-text("Expert"), '
                            'button:has-text("Nightmare")')
        return btns.count() >= 4
    except Exception:
        return False


def _extract_token(page) -> str | None:
    """Try multiple methods to extract the WSS token URL from the page."""
    # Method 1: JavaScript DOM scan
    try:
        url = page.evaluate('''() => {
            const els = document.querySelectorAll('p, span, div, code, pre, input');
            for (const el of els) {
                const t = el.textContent || el.value || '';
                const m = t.match(/wss:\\/\\/game\\.ainm\\.no\\/ws\\?token=[A-Za-z0-9_\\-]+\\.[A-Za-z0-9_\\-]+\\.[A-Za-z0-9_\\-]+/);
                if (m) return m[0];
            }
            return null;
        }''')
        if url:
            return _clean_jwt_url(url)
    except Exception:
        pass

    # Method 2: Full page HTML regex
    try:
        content = page.content()
        m = TOKEN_RE.search(content)
        if m:
            return _clean_jwt_url(m.group(0))
    except Exception:
        pass

    return None


def _wait_for_token(page, difficulty: str, timeout_s: float) -> str | None:
    """Wait for the toast notification and token to appear, up to timeout_s seconds."""
    deadline = time.time() + timeout_s
    toast_seen = False

    while time.time() < deadline:
        # Check for success toast: "Game token ready for ..."
        if not toast_seen:
            try:
                toast = page.locator('[data-type="success"]:has-text("token ready")')
                if toast.count() > 0:
                    toast_seen = True
                    print(f"  Toast: token ready for {difficulty}", file=sys.stderr)
            except Exception:
                pass

        # Try to extract token
        token = _extract_token(page)
        if token:
            return token

        # Check for error toasts
        try:
            err_toast = page.locator('[data-type="error"]')
            if err_toast.count() > 0:
                err_text = err_toast.first.inner_text()
                print(f"  Error toast: {err_text}", file=sys.stderr)
                if 'cooldown' in err_text.lower() or 'wait' in err_text.lower():
                    print(f"  Cooldown detected, waiting...", file=sys.stderr)
                    time.sleep(5)
                    continue
        except Exception:
            pass

        remaining = deadline - time.time()
        if remaining > 0:
            time.sleep(min(1.0, remaining))

    return None


def fetch_token(difficulty: str = 'hard', headed: bool = False,
                setup: bool = False, timeout_s: float = 90.0) -> str | None:
    """Fetch a single game token for the given difficulty.

    Returns the WSS URL string, or None on failure.
    """
    headless = not (headed or setup)

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            PROFILE_DIR,
            headless=headless,
            args=['--disable-blink-features=AutomationControlled'],
            viewport={'width': 1280, 'height': 800},
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        try:
            page.goto(CHALLENGE_URL, wait_until='networkidle', timeout=30000)
        except PwTimeout:
            print("  Page load timed out, continuing anyway...", file=sys.stderr)

        time.sleep(1)

        # --- Setup mode: manual Google OAuth login ---
        if setup:
            print("Browser opened at app.ainm.no/challenge.", file=sys.stderr)
            print("Please log in with Google, then press ENTER here...", file=sys.stderr)
            input()
            try:
                page.goto(CHALLENGE_URL, wait_until='networkidle', timeout=30000)
            except PwTimeout:
                pass
            time.sleep(1)

        # --- Check if logged in ---
        if not _is_logged_in(page):
            if headless:
                print("ERROR: Not logged in. Run with --setup first to log in via Google.",
                      file=sys.stderr)
                page.screenshot(path=os.path.join(SCRIPT_DIR, 'token_fetch_debug.png'))
                ctx.close()
                return None
            else:
                print("Not logged in. Please log in via the browser...", file=sys.stderr)
                for _ in range(120):
                    time.sleep(1)
                    if _is_logged_in(page):
                        break
                else:
                    print("ERROR: Login timed out.", file=sys.stderr)
                    ctx.close()
                    return None

        # --- Dismiss any overlay/modal ---
        try:
            overlay = page.locator(
                'button:has-text("Close"), button:has-text("Cancel"), '
                'button:has-text("Dismiss"), [aria-label="Close"]'
            )
            if overlay.count() > 0:
                overlay.first.click()
                time.sleep(0.5)
        except Exception:
            pass

        # --- Click the difficulty button ---
        print(f"  Clicking {difficulty}...", file=sys.stderr)
        clicked = False
        btns = page.locator(f'button:has-text("{difficulty.capitalize()}")')
        if btns.count() > 0:
            btns.first.click()
            clicked = True
        else:
            btns = page.get_by_role('button', name=re.compile(difficulty, re.IGNORECASE))
            if btns.count() > 0:
                btns.first.click()
                clicked = True

        if not clicked:
            print(f"ERROR: Could not find {difficulty} button", file=sys.stderr)
            page.screenshot(path=os.path.join(SCRIPT_DIR, 'token_fetch_debug.png'))
            ctx.close()
            return None

        # --- Wait for token (up to timeout_s) ---
        print(f"  Waiting for {difficulty} token (up to {timeout_s:.0f}s)...", file=sys.stderr)
        token = _wait_for_token(page, difficulty, timeout_s)

        if not token:
            page.screenshot(path=os.path.join(SCRIPT_DIR, 'token_fetch_debug.png'))
            print(f"ERROR: Token not found after {timeout_s:.0f}s. "
                  f"Check token_fetch_debug.png", file=sys.stderr)

        ctx.close()
        return token


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fetch AINM game token')
    parser.add_argument('difficulty', nargs='?', default='hard',
                        choices=DIFFICULTIES,
                        help='Difficulty level')
    parser.add_argument('--headed', action='store_true',
                        help='Show browser window')
    parser.add_argument('--setup', action='store_true',
                        help='First-time login: opens browser for Google OAuth')
    parser.add_argument('--timeout', type=float, default=90.0,
                        help='Max seconds to wait for token (default: 90)')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON')
    args = parser.parse_args()

    token = fetch_token(args.difficulty, headed=args.headed,
                        setup=args.setup, timeout_s=args.timeout)
    if token:
        if args.json:
            print(json.dumps({'difficulty': args.difficulty, 'url': token}))
        else:
            print(token)
    else:
        sys.exit(1)
