"""WebSocket client to replay pre-computed action sequences.

Connects to sim_server.py or live game server and replays solved games.
"""
import asyncio
import json
import sys
from game_engine import actions_to_ws_format, build_map


async def replay(ws_url, action_sequence, map_state, verbose=True):
    """Replay a pre-computed action sequence over WebSocket.

    Args:
        ws_url: WebSocket URL (e.g., "ws://localhost:9999" or "wss://game-dev.ainm.no/ws?token=...")
        action_sequence: list of per-round actions (each is list of (action_type, item_idx) per bot)
        map_state: MapState for action name translation
        verbose: Print progress

    Returns:
        Final score from server
    """
    try:
        import websockets
    except ImportError:
        print("Install websockets: pip install websockets")
        sys.exit(1)

    async with websockets.connect(ws_url) as ws:
        final_score = 0
        for round_num in range(len(action_sequence)):
            msg = await ws.recv()
            data = json.loads(msg)

            if data.get("type") == "game_over":
                final_score = data.get("score", 0)
                if verbose:
                    print(f"Game over at round {round_num}: score={final_score}")
                return final_score

            # Send pre-computed actions
            server_round = data.get("round", round_num)
            if server_round != round_num:
                if verbose:
                    print(f"WARNING: round mismatch - expected {round_num}, got {server_round}")

            actions = action_sequence[round_num]
            ws_actions = actions_to_ws_format(actions, map_state)
            await ws.send(json.dumps({"actions": ws_actions}))

            if verbose and (round_num < 10 or round_num % 50 == 0):
                score = data.get("score", 0)
                print(f"  R{round_num}: server_score={score}")

        # Receive final game_over
        msg = await ws.recv()
        data = json.loads(msg)
        if data.get("type") == "game_over":
            final_score = data.get("score", 0)
            if verbose:
                print(f"Game over: score={final_score}")

        return final_score


async def replay_with_validation(ws_url, action_sequence, map_state,
                                  expected_score=None, verbose=True):
    """Replay and validate against expected score."""
    score = await replay(ws_url, action_sequence, map_state, verbose)
    if expected_score is not None:
        if score == expected_score:
            print(f"PASS: score {score} matches expected {expected_score}")
        else:
            print(f"FAIL: score {score} != expected {expected_score}")
    return score


def save_actions(action_sequence, filepath):
    """Save action sequence to JSON file for later replay."""
    serializable = []
    for round_actions in action_sequence:
        serializable.append([(int(a), int(i)) for a, i in round_actions])
    with open(filepath, 'w') as f:
        json.dump(serializable, f)
    print(f"Saved {len(action_sequence)} rounds to {filepath}")


def load_actions(filepath):
    """Load action sequence from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return [[(a, i) for a, i in round_actions] for round_actions in data]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ws_url', help='WebSocket URL')
    parser.add_argument('actions_file', help='JSON file with pre-computed actions')
    parser.add_argument('--difficulty', default='easy')
    parser.add_argument('--capture', default=None,
                        help='Capture JSON file (uses real server item IDs)')
    args = parser.parse_args()

    if args.capture:
        import json
        from game_engine import build_map_from_capture
        with open(args.capture) as f:
            capture = json.load(f)
        ms = build_map_from_capture(capture)
        print(f"Using captured map ({len(capture['items'])} items)")
    else:
        ms = build_map(args.difficulty)

    actions = load_actions(args.actions_file)
    asyncio.run(replay(args.ws_url, actions, ms))
