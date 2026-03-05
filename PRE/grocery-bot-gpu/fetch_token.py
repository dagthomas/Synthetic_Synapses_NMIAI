#!/usr/bin/env python3
"""Fetch a live game token from app.ainm.no using Playwright.

First run: use --setup to open headed browser and log in manually.
Subsequent runs: headless, session reused automatically.

Usage:
    python fetch_token.py hard          # headless (after login)
    python fetch_token.py hard --headed # visible window
    python fetch_token.py hard --setup  # first-time login (opens browser)
"""
import sys
import re
import os
import time
from playwright.sync_api import sync_playwright

PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.playwright_profile')

def fetch_token(difficulty='hard', headed=False, setup=False):
    headless = not (headed or setup)

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            PROFILE_DIR,
            headless=headless,
            args=['--disable-blink-features=AutomationControlled'],
            viewport={'width': 1280, 'height': 800},
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        page.goto('https://app.ainm.no/challenge')
        page.wait_for_load_state('networkidle')
        time.sleep(1)

        if setup:
            print("Browser opened. Please log in, then press ENTER here...", file=sys.stderr)
            input()
            page.goto('https://app.ainm.no/challenge')
            page.wait_for_load_state('networkidle')
            time.sleep(1)

        # Dismiss any overlay/modal (e.g. "Competition Warm Up" dialog)
        try:
            overlay = page.locator('button:has-text("Close"), button:has-text("Cancel"), button:has-text("Dismiss"), [aria-label="Close"]')
            if overlay.count() > 0:
                overlay.first.click()
                time.sleep(0.5)
        except Exception:
            pass  # Overlay may not exist; safe to ignore

        # Click the difficulty button/card
        btns = page.get_by_role('button', name=re.compile(difficulty, re.IGNORECASE))
        if btns.count() > 0:
            btns.first.click()
        else:
            # Try clicking any element with the difficulty text
            page.locator(f'text={difficulty.capitalize()}').first.click()

        # Wait for token to appear (longer wait in headless mode)
        wait_secs = 5 if headless else 2
        time.sleep(wait_secs)

        # Take screenshot for debugging
        page.screenshot(path='token_fetch_debug.png')

        # Also try JavaScript extraction as fallback
        try:
            js_url = page.evaluate('''() => {
                const all = document.querySelectorAll('*');
                for (const el of all) {
                    const t = el.textContent || '';
                    const m = t.match(/wss:\\/\\/game\\.ainm\\.no\\/ws\\?token=[A-Za-z0-9._\\-]+/);
                    if (m) return m[0];
                }
                return null;
            }''')
            if js_url:
                ctx.close()
                return js_url
        except Exception:
            pass  # JS extraction is a fallback; continue to HTML parsing

        # Search page HTML for the token URL
        content = page.content()
        match = re.search(r'wss://game\.ainm\.no/ws\?token=[A-Za-z0-9._\-]+', content)
        if match:
            ctx.close()
            return match.group(0)

        # Try input fields
        for inp in page.query_selector_all('input'):
            val = inp.get_attribute('value') or ''
            if 'wss://' in val:
                ctx.close()
                return val

        # Try visible text
        for el in page.query_selector_all('*'):
            try:
                txt = el.inner_text()
                if 'wss://' in txt and len(txt) < 600:
                    m = re.search(r'wss://game\.ainm\.no/ws\?token=[A-Za-z0-9._\-]+', txt)
                    if m:
                        ctx.close()
                        return m.group(0)
            except Exception:
                pass  # Element may be detached or hidden; skip to next

        ctx.close()
        return None


if __name__ == '__main__':
    diff = sys.argv[1] if len(sys.argv) > 1 else 'hard'
    headed = '--headed' in sys.argv
    setup = '--setup' in sys.argv

    token = fetch_token(diff, headed=headed, setup=setup)
    if token:
        print(token)
    else:
        print("ERROR: Could not find token. Check token_fetch_debug.png", file=sys.stderr)
        sys.exit(1)
