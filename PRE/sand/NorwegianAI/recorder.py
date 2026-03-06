"""Record and load game sessions for offline replay."""

import json
import os
from datetime import datetime

SIMULATION_DIR = os.path.join(os.path.dirname(__file__), "simulation")


class GameRecorder:
    """Records every game state and action to a JSON file."""

    def __init__(self, difficulty: str):
        self.difficulty = difficulty
        self.rounds = []
        self.result = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def record_round(self, state: dict, actions: list):
        self.rounds.append({"state": state, "actions": actions})

    def record_result(self, game_over: dict):
        self.result = game_over

    def save(self) -> str:
        """Save recording to simulation/<difficulty>/ and return filepath."""
        folder = os.path.join(SIMULATION_DIR, self.difficulty)
        os.makedirs(folder, exist_ok=True)

        filename = f"game_{self.timestamp}.json"
        filepath = os.path.join(folder, filename)

        data = {
            "difficulty": self.difficulty,
            "timestamp": self.timestamp,
            "total_rounds": len(self.rounds),
            "result": self.result,
            "rounds": self.rounds,
        }

        with open(filepath, "w") as f:
            json.dump(data, f)

        size_kb = os.path.getsize(filepath) / 1024
        print(f"  Saved recording: {filepath} ({size_kb:.0f} KB, {len(self.rounds)} rounds)")
        return filepath


def list_recordings(difficulty: str) -> list[str]:
    """List all saved recordings for a difficulty, newest first."""
    folder = os.path.join(SIMULATION_DIR, difficulty)
    if not os.path.exists(folder):
        return []

    files = [f for f in os.listdir(folder) if f.startswith("game_") and f.endswith(".json")]
    files.sort(reverse=True)
    return [os.path.join(folder, f) for f in files]


def load_recording(filepath: str) -> dict:
    """Load a recording from disk."""
    with open(filepath) as f:
        return json.load(f)
