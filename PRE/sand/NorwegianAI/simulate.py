"""Local simulation — replay saved games to test new bot logic without server."""

from brain import decide_actions
from recorder import list_recordings, load_recording


def run_local(difficulty: str, recording_path: str | None = None):
    """Replay a saved game locally with current bot logic.

    Feeds each recorded game state to decide_actions() and compares
    the new actions against what the bot originally did.
    """
    if recording_path is None:
        recordings = list_recordings(difficulty)
        if not recordings:
            print(f"  No saved games for '{difficulty}'.")
            print(f"  Run a live game first: python main.py {difficulty} <TOKEN>")
            return
        recording_path = recordings[0]  # latest
        print(f"  Using latest recording: {recording_path}")

    data = load_recording(recording_path)
    rounds = data["rounds"]
    original_result = data.get("result", {})

    print(f"\n  Replaying {data['difficulty']} game — {len(rounds)} rounds")
    print(f"  Original score: {original_result.get('score', '?')}")
    print(f"  Original orders completed: {original_result.get('orders_completed', '?')}")
    print("=" * 60)

    action_diffs = 0
    total_actions = 0

    for i, round_data in enumerate(rounds):
        state = round_data["state"]
        old_actions = round_data["actions"]

        # Run current bot logic on the recorded state
        new_actions = decide_actions(state)

        # Compare old vs new
        old_map = {a["bot"]: a for a in old_actions}
        new_map = {a["bot"]: a for a in new_actions}

        round_diffs = []
        for bot_id in sorted(old_map.keys()):
            total_actions += 1
            old_a = old_map.get(bot_id, {})
            new_a = new_map.get(bot_id, {})
            if old_a.get("action") != new_a.get("action") or old_a.get("item_id") != new_a.get("item_id"):
                action_diffs += 1
                round_diffs.append(
                    f"    Bot {bot_id}: {old_a.get('action','?'):10s} -> {new_a.get('action','?'):10s}"
                )

        # Print progress every 10 rounds, or if there are diffs
        rnd = state.get("round", i)
        score = state.get("score", "?")
        if rnd % 50 == 0:
            print(f"  Round {rnd:3d}/300 | Score {score}")
        if round_diffs and rnd % 50 == 0:
            for d in round_diffs:
                print(d)

    print("=" * 60)
    print(f"  Replay complete")
    print(f"  Total actions: {total_actions}")
    print(f"  Actions changed: {action_diffs} ({action_diffs * 100 // max(total_actions, 1)}%)")
    print()
    if action_diffs == 0:
        print("  Bot logic unchanged — same actions as original run.")
    else:
        print(f"  Bot logic differs in {action_diffs} actions.")
        print("  NOTE: Score may differ since game state depends on prior actions.")
        print("  Run a live game to see the real score with new logic.")
