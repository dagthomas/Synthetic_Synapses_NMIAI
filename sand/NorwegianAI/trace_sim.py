"""Trace simulation drop-offs and cascade in detail."""
import copy
import json
from simulator import load_game_data, LocalSimulator
from recorder import list_recordings
from brain import decide_actions


class TracingSimulator(LocalSimulator):
    """Simulator that traces drop-offs and cascading."""

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
        delivered_before = list(active["items_delivered"])
        prev_orders = self.orders_completed

        self._deliver_items(bot, active)

        if self.orders_completed > prev_orders:
            # Order completed! Check cascade
            new_active = self.orders[0] if self.orders else None
            cascade_items = list(new_active["items_delivered"]) if new_active else []
            print(f"  R{self.round}: DROP-OFF → ORDER COMPLETE! "
                  f"inv={inv_before} delivered_before={delivered_before} "
                  f"cascade_to_next={cascade_items} "
                  f"new_order_needs={new_active['items_required'] if new_active else 'none'}")
        else:
            items_added = len(active["items_delivered"]) - len(delivered_before)
            inv_after = list(bot["inventory"])
            print(f"  R{self.round}: DROP-OFF: inv={inv_before} → delivered {items_added} items, "
                  f"remaining_inv={inv_after}, order progress={len(active['items_delivered'])}/{len(active['items_required'])}")


def main():
    recordings = list_recordings("easy")
    if not recordings:
        print("No recordings found")
        return

    game_data = load_game_data(recordings[0])
    sim = TracingSimulator(game_data)

    for rnd in range(sim.max_rounds):
        sim.round = rnd
        state = sim.get_state()
        actions = decide_actions(state)
        sim.apply_actions(actions)

    print(f"\nScore: {sim.score} | Orders: {sim.orders_completed}")


if __name__ == "__main__":
    main()
