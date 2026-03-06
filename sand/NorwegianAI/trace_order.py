"""Trace a specific order round-by-round to find inefficiencies."""
import json
from simulator import load_game_data, LocalSimulator
from recorder import list_recordings
from brain import decide_actions


class DetailedTracer(LocalSimulator):
    def __init__(self, game_data, track_rounds):
        super().__init__(game_data)
        self.track_start = track_rounds[0]
        self.track_end = track_rounds[1]

    def _try_dropoff(self, bot):
        bx, by = bot["position"]
        if (bx, by) != self.drop_off:
            return
        if not bot["inventory"]:
            return
        active = self.orders[0] if self.orders else None
        if not active or active["status"] != "active":
            return
        inv_before = list(bot["inventory"])
        self._deliver_items(bot, active)
        if self.track_start <= self.round <= self.track_end:
            print(f"    ** DROPOFF: inv={inv_before} → delivered to {active['id']}")


def main():
    recordings = list_recordings("easy")
    game_data = load_game_data(recordings[0])

    # Track slowest orders
    for name, rng in [("Order 0", (0, 35)), ("Order 8", (170, 210)), ("Order 6", (122, 155))]:
        print(f"\n{'='*60}")
        print(f"  TRACING: {name} (rounds {rng[0]}-{rng[1]})")
        print(f"{'='*60}")

        sim = DetailedTracer(game_data, rng)

        for rnd in range(sim.max_rounds):
            sim.round = rnd
            state = sim.get_state()
            actions = decide_actions(state)

            if rng[0] <= rnd <= rng[1]:
                bot = state["bots"][0]
                pos = tuple(bot["position"])
                inv = list(bot["inventory"])
                act = actions[0]["action"] if actions else "?"
                item_id = actions[0].get("item_id", "")

                active = next((o for o in state["orders"] if o["status"] == "active"), None)
                if active:
                    delivered = len(active["items_delivered"])
                    required = len(active["items_required"])
                    needs = active["items_required"]
                else:
                    delivered = required = 0
                    needs = []

                extra = f" item={item_id}" if item_id else ""
                print(f"  R{rnd:3d}: pos={pos} inv={inv} → {act}{extra}  [{delivered}/{required}]")

            sim.apply_actions(actions)

        print(f"  Final score: {sim.score}")


if __name__ == "__main__":
    main()
