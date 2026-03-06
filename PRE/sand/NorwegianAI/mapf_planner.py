"""MAPF-based full-game planner for Grocery Bot.

Simulates the game step-by-step with full order management.
Uses coordinated task assignment and multi-step pathfinding.

Usage:
    python mapf_planner.py hard
"""

import heapq
import json
import sys
import time
from collections import defaultdict

from distance import DistanceMatrix
from recorder import list_recordings
from simulator import LocalSimulator, SyntheticOrderGenerator, load_game_data


# ============================================================
# Time-Space A* (for single query — not full MAPF)
# ============================================================

def plan_path(start, goal, blocked, width, height, occupied_now=None, max_depth=60):
    """BFS shortest path avoiding blocked cells and optionally current occupants.

    Returns list of positions [start, ..., goal] or None.
    Simple BFS without temporal reservations (for now).
    """
    if start == goal:
        return [start]

    from collections import deque
    DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    q = deque([(start, [start])])
    visited = {start}

    while q:
        pos, path = q.popleft()
        if len(path) > max_depth:
            continue
        for dx, dy in DIRS:
            nx, ny = pos[0] + dx, pos[1] + dy
            npos = (nx, ny)
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if npos in blocked:
                continue
            if npos in visited:
                continue
            if occupied_now and npos in occupied_now:
                continue
            visited.add(npos)
            new_path = path + [npos]
            if npos == goal:
                return new_path
            q.append((npos, new_path))
    return None


DELTA_TO_ACTION = {
    (1, 0): "move_right", (-1, 0): "move_left",
    (0, 1): "move_down", (0, -1): "move_up",
    (0, 0): "wait",
}


# ============================================================
# Simulating Planner — plans by simulating the full game
# ============================================================

class SimulatingPlanner:
    """Plan by running the game with a coordinated brain.

    Uses the LocalSimulator to produce real scores, with a custom brain
    that has full knowledge of the order sequence and does coordinated planning.
    """

    def __init__(self, game_data):
        self.game_data = game_data
        self.width = game_data["grid"]["width"]
        self.height = game_data["grid"]["height"]
        self.max_rounds = game_data["max_rounds"]
        self.n_bots = len(game_data["bots"])
        self.drop_off = tuple(game_data["drop_off"])

        # Build blocked set
        self.blocked = set()
        for w in game_data["grid"]["walls"]:
            self.blocked.add((w[0], w[1]))
        for item in game_data["items"]:
            self.blocked.add((item["position"][0], item["position"][1]))

        # Distance matrix
        state0 = {
            "grid": game_data["grid"],
            "items": game_data["items"],
            "bots": game_data["bots"],
            "orders": [],
            "drop_off": list(game_data["drop_off"]),
        }
        self.dm = DistanceMatrix(state0)

        # Item lookup
        self.shelves_by_type = defaultdict(list)
        for item in game_data["items"]:
            shelf = tuple(item["position"])
            adj, _ = self.dm.best_adjacent(self.drop_off, shelf)
            if adj:
                self.shelves_by_type[item["type"]].append({
                    "shelf": shelf,
                    "item_id": item["id"],
                    "adj": adj,
                })

        # Full order sequence
        self.order_sequence = game_data["order_sequence"]

        # Bot goals and state
        self._bot_goals = {}   # bid -> {"type": "pickup"|"deliver"|"idle", ...}
        self._bot_paths = {}   # bid -> [positions remaining in planned path]
        self._assigned = {}    # bid -> [item_types to pick for current order]
        self._idle_assigned = {}  # bid -> [item_types for speculative/preview]
        self._order_id = None

    def plan(self):
        """Run simulation with coordinated brain, return optimized plan."""
        sim = LocalSimulator(self.game_data)
        all_actions = []

        for rnd in range(self.max_rounds):
            sim.round = rnd
            state = sim.get_state()
            actions = self._decide(state)
            all_actions.append(actions)
            sim.apply_actions(actions)

        result = {
            "score": sim.score,
            "items_delivered": sim.items_delivered,
            "orders_completed": sim.orders_completed,
        }
        return all_actions, result

    def _decide(self, state):
        """Coordinated decision-making for all bots."""
        bots = state["bots"]
        items = state["items"]
        orders = state["orders"]
        drop_off = tuple(state["drop_off"])

        active = next((o for o in orders if o.get("status") == "active" and not o["complete"]), None)
        preview = next((o for o in orders if o.get("status") == "preview" and not o["complete"]), None)

        if not active:
            return [{"bot": b["id"], "action": "wait"} for b in bots]

        # Detect order change
        if active["id"] != self._order_id:
            self._order_id = active["id"]
            self._assigned = {}
            self._bot_paths = {}
            self._bot_goals = {}
            self._redistribute(bots, active, preview, items, drop_off)

        # Compute what active still needs
        active_needed = self._get_needed(active)
        from_map = self._from_map(active, bots)

        # Compute locked set (process bots in ID order, lock targets)
        locked = set(tuple(b["position"]) for b in bots)
        actions = []
        sorted_bots = sorted(bots, key=lambda b: b["id"])

        for bot in sorted_bots:
            bid = bot["id"]
            pos = tuple(bot["position"])
            locked.discard(pos)

            action = self._bot_step(bot, bots, items, active, preview, active_needed,
                                     from_map, drop_off, state, locked)
            actions.append(action)

            # Lock target cell
            act_str = action["action"]
            if act_str.startswith("move_"):
                dx, dy = {"move_right": (1, 0), "move_left": (-1, 0),
                           "move_down": (0, 1), "move_up": (0, -1)}[act_str]
                locked.add((pos[0] + dx, pos[1] + dy))
            else:
                locked.add(pos)

        # Swap/same-dest conflict resolution (same as brain.py)
        _MOVE_DELTA = {"move_right": (1, 0), "move_left": (-1, 0),
                       "move_down": (0, 1), "move_up": (0, -1)}
        bot_pos = {b["id"]: tuple(b["position"]) for b in bots}
        bot_dest = {}
        for a in actions:
            bid_a = a["bot"]
            act_a = a["action"]
            if act_a in _MOVE_DELTA:
                dx, dy = _MOVE_DELTA[act_a]
                bot_dest[bid_a] = (bot_pos[bid_a][0] + dx, bot_pos[bid_a][1] + dy)
            else:
                bot_dest[bid_a] = bot_pos[bid_a]

        action_map = {a["bot"]: a for a in actions}
        for i, a1 in enumerate(actions):
            for a2 in actions[i+1:]:
                b1, b2 = a1["bot"], a2["bot"]
                if bot_dest[b1] == bot_pos[b2] and bot_dest[b2] == bot_pos[b1]:
                    waiter = max(b1, b2)
                    action_map[waiter]["action"] = "wait"
                    action_map[waiter].pop("item_id", None)

        dest_to_bots = {}
        for a in actions:
            bid_a = a["bot"]
            if a["action"] in _MOVE_DELTA:
                dx, dy = _MOVE_DELTA[a["action"]]
                dest = (bot_pos[bid_a][0] + dx, bot_pos[bid_a][1] + dy)
                dest_to_bots.setdefault(dest, []).append(bid_a)
        for dest, bids in dest_to_bots.items():
            if len(bids) > 1:
                bids.sort()
                for bid_a in bids[1:]:
                    action_map[bid_a]["action"] = "wait"
                    action_map[bid_a].pop("item_id", None)

        return actions

    def _redistribute(self, bots, active, preview, items, drop_off):
        """Assign active items to bots."""
        needed = self._from_map(active, bots)
        if not needed:
            return

        # Items to assign
        item_list = []
        for itype, count in needed.items():
            item_list.extend([itype] * count)

        # Greedy assignment: assign each item to nearest bot with capacity
        self._assigned = {b["id"]: [] for b in bots}
        for item_type in item_list:
            best_bid = None
            best_cost = float("inf")
            for b in bots:
                bid = b["id"]
                if len(self._assigned[bid]) + len(b["inventory"]) >= 3:
                    continue
                # Find nearest shelf
                cost = self._pickup_cost(item_type, tuple(b["position"]))
                if cost < best_cost:
                    best_cost = cost
                    best_bid = bid
            if best_bid is not None:
                self._assigned[best_bid].append(item_type)

    def _bot_step(self, bot, all_bots, items, active, preview, active_needed,
                   from_map, drop_off, state, locked):
        """Decide action for one bot."""
        bid = bot["id"]
        pos = tuple(bot["position"])
        inventory = list(bot["inventory"])

        has_useful = any(active_needed.get(it, 0) > 0 for it in inventory)
        assigned = self._assigned.get(bid, [])

        # Phase 1: At dropoff with useful items -> deliver
        if has_useful and pos == drop_off:
            return {"bot": bid, "action": "drop_off"}

        # Phase 2: Clear dropoff if useless items
        if not has_useful and inventory and pos == drop_off:
            # Move north to clear
            for dx, dy in [(0, -1), (1, 0), (-1, 0), (0, 1)]:
                nxt = (pos[0] + dx, pos[1] + dy)
                if nxt not in self.blocked and nxt not in locked:
                    if self.dm.dist(pos, nxt) == 1:
                        return {"bot": bid, "action": DELTA_TO_ACTION[(dx, dy)]}
            return {"bot": bid, "action": "wait"}

        # Phase 3: Has assigned items -> go pick them up
        if assigned:
            item_type = assigned[0]
            # Find shelf for this item
            shelf_info = self._best_shelf(item_type, pos)
            if shelf_info:
                target = shelf_info["adj"]
                if abs(pos[0] - target[0]) + abs(pos[1] - target[1]) == 0:
                    # Adjacent to shelf -> pick up
                    shelf_pos = shelf_info["shelf"]
                    if abs(pos[0] - shelf_pos[0]) + abs(pos[1] - shelf_pos[1]) == 1:
                        self._assigned[bid] = assigned[1:]  # remove from assignment
                        return {"bot": bid, "action": "pick_up", "item_id": shelf_info["item_id"]}
                # Navigate to shelf adjacency
                return self._navigate(bid, pos, target, locked)

        # Phase 4: Has useful items -> go deliver
        if has_useful and (len(inventory) >= 3 or not assigned):
            return self._navigate(bid, pos, drop_off, locked)

        # Phase 5: Pick preview items or speculative items
        if len(inventory) < 3:
            pick = self._find_speculative_pick(bot, all_bots, items, preview,
                                                active_needed, drop_off, pos, locked)
            if pick:
                return pick

        # Phase 6: Park
        return {"bot": bid, "action": "wait"}

    def _find_speculative_pick(self, bot, all_bots, items, preview, active_needed,
                                drop_off, pos, locked):
        """Find a speculative item to pick up."""
        bid = bot["id"]

        # Preview items first
        if preview:
            prev_needed = self._get_needed(preview)
            for b in all_bots:
                for it in b["inventory"]:
                    if it in prev_needed and prev_needed[it] > 0:
                        prev_needed[it] -= 1
            prev_needed = {k: v for k, v in prev_needed.items() if v > 0}
            if prev_needed:
                return self._pick_from_set(bid, pos, items, prev_needed, drop_off, locked)

        # Speculative: pick cheap items not held by anyone
        held_types = set()
        for b in all_bots:
            for it in b["inventory"]:
                held_types.add(it)

        spec_needed = {}
        for item in items:
            itype = item["type"]
            if itype not in held_types:
                shelf = tuple(item["position"])
                cost = self.dm.trip_cost(drop_off, shelf)
                if cost <= 28 and (itype not in spec_needed or cost < spec_needed[itype]):
                    spec_needed[itype] = cost
        if spec_needed:
            return self._pick_from_set(bid, pos, items, {k: 1 for k in spec_needed},
                                        drop_off, locked)
        return None

    def _pick_from_set(self, bid, pos, items, needed_types, drop_off, locked):
        """Find and navigate to best item matching needed_types."""
        best_item = None
        best_cost = float("inf")
        for item in items:
            if item["type"] not in needed_types:
                continue
            shelf = tuple(item["position"])
            cost = self.dm.trip_cost(pos, shelf)
            if cost < best_cost:
                best_cost = cost
                best_item = item

        if best_item:
            shelf = tuple(best_item["position"])
            if abs(pos[0] - shelf[0]) + abs(pos[1] - shelf[1]) == 1:
                return {"bot": bid, "action": "pick_up", "item_id": best_item["id"]}
            adj, _ = self.dm.best_adjacent(pos, shelf)
            if adj:
                return self._navigate(bid, pos, adj, locked)
        return None

    def _navigate(self, bid, pos, target, locked):
        """Navigate toward target, avoiding locked cells."""
        if pos == target:
            return {"bot": bid, "action": "wait"}

        # Gradient descent
        nxt = self.dm.next_step(pos, target)
        if nxt and nxt != pos and nxt not in locked:
            dx = nxt[0] - pos[0]
            dy = nxt[1] - pos[1]
            return {"bot": bid, "action": DELTA_TO_ACTION.get((dx, dy), "wait")}

        # BFS fallback
        from pathfinding import bfs
        blocked_for_bfs = set(self.blocked) | locked
        path = bfs(pos, target, blocked_for_bfs, self.width, self.height)
        if path and len(path) >= 2:
            dx = path[1][0] - pos[0]
            dy = path[1][1] - pos[1]
            return {"bot": bid, "action": DELTA_TO_ACTION.get((dx, dy), "wait")}

        # Deadlock: try any free adjacent cell
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt not in self.blocked and nxt not in locked:
                if self.dm.dist(pos, nxt) == 1:
                    return {"bot": bid, "action": DELTA_TO_ACTION[(dx, dy)]}

        return {"bot": bid, "action": "wait"}

    def _get_needed(self, order):
        needed = {}
        for it in order["items_required"]:
            needed[it] = needed.get(it, 0) + 1
        for it in order["items_delivered"]:
            needed[it] = needed.get(it, 0) - 1
        return {k: v for k, v in needed.items() if v > 0}

    def _from_map(self, order, bots):
        needed = self._get_needed(order)
        for b in bots:
            for it in b["inventory"]:
                if it in needed and needed[it] > 0:
                    needed[it] -= 1
        return {k: v for k, v in needed.items() if v > 0}

    def _pickup_cost(self, item_type, from_pos):
        shelves = self.shelves_by_type.get(item_type, [])
        best = float("inf")
        for s in shelves:
            d = self.dm.dist(from_pos, s["adj"])
            if d < best:
                best = d
        return best

    def _best_shelf(self, item_type, from_pos):
        shelves = self.shelves_by_type.get(item_type, [])
        best = None
        best_d = float("inf")
        for s in shelves:
            d = self.dm.dist(from_pos, s["adj"])
            if d < best_d:
                best_d = d
                best = s
        return best


def run_mapf_planning(difficulty):
    """Run planning for a difficulty."""
    recordings = list_recordings(difficulty)
    if not recordings:
        print(f"No recordings for '{difficulty}'.")
        return

    recording_path = recordings[0]
    print(f"Recording: {recording_path}")

    game_data = load_game_data(recording_path)
    print(f"Grid: {game_data['grid']['width']}x{game_data['grid']['height']}")
    print(f"Bots: {len(game_data['bots'])}")
    print(f"Orders: {len(game_data['order_sequence'])}")
    print()

    planner = SimulatingPlanner(game_data)
    t0 = time.time()
    all_actions, result = planner.plan()
    elapsed = time.time() - t0

    print(f"\nPlanning complete in {elapsed:.1f}s")
    print(f"Score: {result['score']}")
    print(f"Items: {result['items_delivered']}")
    print(f"Orders: {result['orders_completed']}")


if __name__ == "__main__":
    diff = sys.argv[1] if len(sys.argv) > 1 else "hard"
    run_mapf_planning(diff)
