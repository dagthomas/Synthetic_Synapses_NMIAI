#!/usr/bin/env python3
"""Entry point for the Astar Island ADK research agent.

Runs the agent in a loop, each iteration proposing a structural change,
backtesting it, and analyzing results before proposing the next change.

Usage:
    # Autonomous mode (agent iterates on experiments)
    python -m research_agent.run
    python -m research_agent.run --max-iters 20

    # Interactive mode (you chat with the agent)
    python -m research_agent.run --interactive

    # Dry-run (prints prompts without running the agent)
    python -m research_agent.run --dry-run

    # Use ADK CLI directly (interactive chat via ADK)
    adk run research_agent

    # Use ADK web UI
    adk web --port 8000
"""
import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# Load .env from project root
_env_path = _PROJECT_DIR / ".env"
if _env_path.exists():
    for line in _env_path.read_text().strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _get_api_key() -> str:
    """Resolve Google API key from environment."""
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "No API key found. Set GOOGLE_API_KEY in .env or environment.\n"
            "Expected in: " + str(_env_path)
        )
    return key


def _make_user_content(text: str):
    """Create a Content object for a user message."""
    from google.genai import types
    return types.Content(
        role="user",
        parts=[types.Part(text=text)],
    )


def _extract_text_from_events(events) -> str:
    """Extract final agent text from a sequence of events."""
    final_text_parts = []
    for event in events:
        # Only capture agent (model) responses, not tool calls
        if event.author and event.author != "user":
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        final_text_parts.append(part.text)
            # Also capture tool call/response info for logging
            if event.actions and event.actions.tool_calls:
                for tc in event.actions.tool_calls:
                    tool_name = tc.function_call.name if tc.function_call else "unknown"
                    final_text_parts.append(f"\n[Tool call: {tool_name}]")
    return "\n".join(final_text_parts) if final_text_parts else "(no text response)"


async def _run_agent_turn(runner, user_id: str, session_id: str, prompt: str) -> str:
    """Send one message to the agent and collect all response events.

    Returns the concatenated text from all agent response events.
    """
    new_message = _make_user_content(prompt)
    final_text_parts = []

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message,
    ):
        # Print streaming progress indicators
        if event.author and event.author != "user":
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        final_text_parts.append(part.text)
                        # Print as we receive (strip unicode for Windows cp1252)
                        safe_text = part.text.encode("ascii", errors="replace").decode("ascii")
                        print(safe_text, end="", flush=True)
                    elif hasattr(part, "function_call") and part.function_call:
                        tool_name = part.function_call.name
                        print(f"\n  >> Calling tool: {tool_name}...", flush=True)
                    elif hasattr(part, "function_response") and part.function_response:
                        resp_name = part.function_response.name
                        print(f"  << Tool response: {resp_name}", flush=True)

    print()  # newline after streaming
    response = "\n".join(final_text_parts) if final_text_parts else "(no response)"

    # Save agent history to file
    history_path = _PROJECT_DIR / "data" / "adk_agent_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    history_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt[:500],
        "response": response[:5000],
        "session_id": session_id,
    }
    with open(history_path, "a") as f:
        f.write(json.dumps(history_entry) + "\n")

    return response


async def run_agent_loop(max_iters: int = 0, dry_run: bool = False):
    """Run the research agent in an autonomous loop.

    Each iteration sends a prompt that triggers the agent's full workflow:
    read context -> propose code -> backtest -> analyze -> repeat.
    """
    from google.adk.runners import InMemoryRunner
    from .agent import root_agent

    api_key = _get_api_key()
    os.environ["GOOGLE_API_KEY"] = api_key

    runner = InMemoryRunner(agent=root_agent, app_name="astar_research")

    user_id = "researcher"

    # Create session via the runner's session service
    session = await runner.session_service.create_session(
        app_name="astar_research",
        user_id=user_id,
    )
    session_id = session.id

    print("=" * 70)
    print("Astar Island ADK Research Agent")
    print(f"Model: gemini-3.1-pro-preview")
    print(f"Session: {session_id}")
    print(f"Max iterations: {max_iters if max_iters > 0 else 'unlimited'}")
    print(f"Dry run: {dry_run}")
    print("=" * 70)

    # Initial prompt
    initial_prompt = (
        "Start your research workflow. Read the knowledge base and experiment log, "
        "then propose your first structural algorithmic change to improve the "
        "Astar Island prediction pipeline. The current best score is ~91.3 avg. "
        "Focus on the highest-impact improvement opportunity you can identify. "
        "Write the COMPLETE experimental_pred_fn code and run a backtest."
    )

    # Continuation prompt
    continue_prompt = (
        "Analyze the results of your last experiment. Based on what you learned, "
        "propose the next structural change. If the last experiment improved the score, "
        "try to build on that approach. If it failed or made things worse, try a "
        "completely different strategy. Read the experiment log to see all results so far. "
        "Write the COMPLETE experimental_pred_fn code and run a backtest."
    )

    iteration = 0
    prompt = initial_prompt

    try:
        while True:
            if max_iters > 0 and iteration >= max_iters:
                print(f"\nReached max iterations ({max_iters}). Stopping.")
                break

            iteration += 1
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}")
            print(f"{'='*70}")

            if dry_run:
                print(f"[DRY RUN] Would send prompt:\n{prompt[:300]}...")
                prompt = continue_prompt
                continue

            t0 = time.time()
            try:
                response_text = await _run_agent_turn(
                    runner, user_id, session_id, prompt
                )
                elapsed = time.time() - t0
                print(f"\n[Iteration {iteration} completed in {elapsed:.1f}s]")
            except Exception as e:
                print(f"\nAgent error: {e}")
                import traceback
                traceback.print_exc()
                elapsed = time.time() - t0
                print(f"[Iteration {iteration} failed after {elapsed:.1f}s]")

            prompt = continue_prompt

            # Brief pause between iterations to avoid rate limiting
            time.sleep(3)

    except KeyboardInterrupt:
        print(f"\n\nStopped by user after {iteration} iterations.")

    print(f"\nResearch session complete. {iteration} iterations.")
    await runner.close()


async def run_interactive():
    """Run the agent in interactive chat mode."""
    from google.adk.runners import InMemoryRunner
    from .agent import root_agent

    api_key = _get_api_key()
    os.environ["GOOGLE_API_KEY"] = api_key

    runner = InMemoryRunner(agent=root_agent, app_name="astar_research")

    user_id = "researcher"
    session_id = "interactive_session_001"

    print("=" * 70)
    print("Astar Island ADK Research Agent -- Interactive Mode")
    print("Type your message or 'quit' to exit.")
    print("Suggested first message: 'Start researching improvements.'")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        try:
            print("\nAgent: ", end="", flush=True)
            await _run_agent_turn(runner, user_id, session_id, user_input)
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

    await runner.close()


def main():
    parser = argparse.ArgumentParser(
        description="Astar Island ADK Research Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m research_agent.run                      # Autonomous loop (unlimited)
  python -m research_agent.run --max-iters 10       # Run 10 iterations
  python -m research_agent.run --interactive        # Interactive chat mode
  python -m research_agent.run --dry-run            # Print prompts without running
        """,
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=0,
        help="Maximum iterations for autonomous loop (0 = unlimited). Default: 0",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without actually running the agent.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive chat mode instead of autonomous loop.",
    )
    args = parser.parse_args()

    if args.interactive:
        asyncio.run(run_interactive())
    else:
        asyncio.run(run_agent_loop(max_iters=args.max_iters, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
