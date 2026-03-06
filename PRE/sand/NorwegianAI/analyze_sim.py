"""Detailed simulation analysis — tracks per-order timing, cascading, and trip efficiency."""

import copy
import json
from simulator import load_game_data, LocalSimulator
from recorder import list_recordings
from brain import decide_actions


class AnalyzingSimulator(LocalSimulator):
    """Simulator with detailed order tracking."""

    def __init__(self, game_data):
        super().__init__(game_data)
        self.order_log = []  # [{id, start_round, end_round, items_required, cascaded, trips}]
        self._current_order_start = 0
        self._current_trips = 0
        self._cascade_count = 0
        self._items_per_trip = []
        self._prev_active_id = None
        self._trip_items = 0

    def _try_dropoff(self, bot):
        """Override to track deliveries and cascading."""
        bx, by = bot["position"]
        if (bx, by) != self.drop_off:
            return
        if not bot["inventory"]:
            return

        active = self.orders[0] if self.orders else None
        if not active or active["status"] != "active":
            return

        prev_delivered = len(active["items_delivered"])
        prev_orders = self.orders_completed
        prev_active_id = active["id"]

        self._deliver_items_tracked(bot, active)

        # Check if order completed
        if self.orders_completed > prev_orders:
            # Log the completed order
            new_delivered = len([o for o in self.order_log if True])  # just count
            self.order_log.append({
                "id": prev_active_id,
                "start_round": self._current_order_start,
                "end_round": self.round,
                "duration": self.round - self._current_order_start,
                "cascaded": self._cascade_count,
                "trips": self._current_trips + 1,  # this dropoff is a trip
                "items_per_trip": self._items_per_trip + [self._trip_items],
            })

            # Reset for new order
            self._current_order_start = self.round
            self._current_trips = 0
            self._items_per_trip = []
            self._trip_items = 0

            # Count cascade items for new active order
            new_active = self.orders[0] if self.orders else None
            if new_active:
                self._cascade_count = len(new_active["items_delivered"])
            else:
                self._cascade_count = 0
        else:
            # Just a delivery, not completing the order
            new_delivered = len(active["items_delivered"])
            items_this_drop = new_delivered - prev_delivered
            self._trip_items = items_this_drop
            self._items_per_trip.append(items_this_drop)
            self._current_trips += 1
            self._trip_items = 0

    def _deliver_items_tracked(self, bot, order):
        """Same as _deliver_items but tracks cascade."""
        remaining_inv = []
        delivered_count = 0
        for item_type in bot["inventory"]:
            needed = order["items_required"].count(item_type) - order["items_delivered"].count(item_type)
            if needed > 0:
                order["items_delivered"].append(item_type)
                self.score += 1
                self.items_delivered += 1
                delivered_count += 1
            else:
                remaining_inv.append(item_type)

        bot["inventory"] = remaining_inv
        self._trip_items = delivered_count

        if sorted(order["items_delivered"]) == sorted(order["items_required"]):
            order["complete"] = True
            self.score += 5
            self.orders_completed += 1

            self.orders.pop(0)
            if self.orders:
                self.orders[0]["status"] = "active"
            self._activate_next_order()

            new_active = self.orders[0] if self.orders else None
            if new_active and new_active["status"] == "active" and bot["inventory"]:
                self._deliver_items_tracked(bot, new_active)


def main():
    recordings = list_recordings("easy")
    if not recordings:
        print("No recordings found")
        return

    recording_path = recordings[0]
    print(f"Using: {recording_path}\n")

    game_data = load_game_data(recording_path)
    sim = AnalyzingSimulator(game_data)

    # Track bot actions per round
    action_log = []
    pickup_log = []

    for rnd in range(sim.max_rounds):
        sim.round = rnd
        state = sim.get_state()
        actions = decide_actions(state)

        # Log actions
        for a in actions:
            if a["action"] == "pick_up":
                pickup_log.append({"round": rnd, "item_id": a.get("item_id"), "bot": a["bot"]})

        sim.apply_actions(actions)

    print(f"{'='*60}")
    print(f"SCORE: {sim.score} | Orders: {sim.orders_completed} | Items: {sim.items_delivered}")
    print(f"{'='*60}\n")

    print(f"{'Order':<10} {'Start':>6} {'End':>6} {'Dur':>5} {'Casc':>5} {'Trips':>6} {'Items/Trip'}")
    print("-" * 60)
    total_cascaded = 0
    for o in sim.order_log:
        ipt = str(o["items_per_trip"]) if o["items_per_trip"] else "[]"
        total_cascaded += o["cascaded"]
        print(f"{o['id']:<10} {o['start_round']:>6} {o['end_round']:>6} {o['duration']:>5} {o['cascaded']:>5} {o['trips']:>6} {ipt}")

    print(f"\nTotal cascaded items: {total_cascaded}")
    print(f"Average order duration: {sum(o['duration'] for o in sim.order_log) / len(sim.order_log):.1f} rounds")

    # Analyze where bot spends time
    print(f"\nPickup count: {len(pickup_log)}")


if __name__ == "__main__":
    main()
