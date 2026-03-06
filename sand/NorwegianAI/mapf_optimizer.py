"""Forward MAPF optimizer: simulation-in-the-loop planning with perfect order knowledge.

Runs the real LocalSimulator step by step. At each round, decides what each bot
should do using known order sequence, distance matrix, and lock-based collision avoidance.

Key differences from the reactive brain:
1. Knows ALL future orders (not just preview)
2. Pre-assigns items for future orders to idle bots (targeted, not random speculative)
3. Coordinates delivery timing (doesn't deliver speculative items)
4. Uses the same navigation and collision avoidance as the reactive brain

Usage:
    python mapf_optimizer.py hard
    python main.py replay hard <URL>
"""

import copy
import json
import sys
import time
from collections import defaultdict

from distance import DistanceMatrix
from pathfinding import bfs, path_to_action
from recorder import list_recordings
from simulator import LocalSimulator, SyntheticOrderGenerator, load_game_data


def get_needed_items(order):
    needed = {}
    for item in order["items_required"]:
        needed[item] = needed.get(item, 0) + 1
    for item in order["items_delivered"]:
        needed[item] = needed.get(item, 0) - 1
    return {k: v for k, v in needed.items() if v > 0}


def _pos_to_action(from_pos, to_pos):
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    if dx == 1: return "move_right"
    if dx == -1: return "move_left"
    if dy == 1: return "move_down"
    if dy == -1: return "move_up"
    return "wait"


class ForwardPlanner:
    """Simulation-in-the-loop MAPF planner."""

    def __init__(self, game_data):
        self.game_data = game_data
        self.max_rounds = game_data["max_rounds"]
        self.drop_off = tuple(game_data["drop_off"])
        self.n_bots = len(game_data["bots"])

        state0 = {
            "grid": game_data["grid"],
            "items": game_data["items"],
            "drop_off": game_data["drop_off"],
        }
        self.dm = DistanceMatrix(state0)
        self.items_template = game_data["items"]

        # Build order sequence (known + synthetic)
        self.order_seq = list(game_data["order_sequence"])
        item_types = list(set(i["type"] for i in self.items_template))
        gen = SyntheticOrderGenerator(item_types=item_types, seed=42, min_items=3, max_items=5)
        while len(self.order_seq) < 30:
            self.order_seq.append(gen.generate())

        # Shelf info
        self.shelves_by_type = defaultdict(list)
        for item in self.items_template:
            pos = (item["position"][0], item["position"][1])
            if pos not in self.shelves_by_type[item["type"]]:
                self.shelves_by_type[item["type"]].append(pos)

        # Detect aisle corridors
        shelf_xs = sorted(set(item["position"][0] for item in self.items_template))
        self.aisle_xs = []
        for i in range(len(shelf_xs) - 1):
            if shelf_xs[i + 1] - shelf_xs[i] == 2:
                self.aisle_xs.append(shelf_xs[i] + 1)

        # Static blocked cells
        self.walls = set(tuple(w) for w in game_data["grid"]["walls"])
        self.shelf_positions = set(
            (item["position"][0], item["position"][1]) for item in self.items_template
        )
        self.static_blocked = self.walls | self.shelf_positions

    def plan(self):
        """Run forward planning. Returns plan dict for _execute_optimized_plan."""
        sim = LocalSimulator(copy.deepcopy(self.game_data))

        all_actions = []
        all_positions = [[] for _ in range(self.n_bots)]

        # State tracking
        assignments = {}  # bot_id -> [item_types to pick]
        order_idx = 0
        last_order_id = None
        bot_targets = {}  # bot_id -> (target_pos, purpose)  purpose: "pickup", "deliver", "clear", "park"
        # Track which future orders have been pre-assigned
        preassigned_orders = set()

        for rnd in range(self.max_rounds):
            sim.round = rnd
            state = sim.get_state()

            bots = state["bots"]
            items = state["items"]
            orders = state["orders"]
            drop_off = self.drop_off

            active = next((o for o in orders if o.get("status") == "active" and not o["complete"]), None)
            preview = next((o for o in orders if o.get("status") == "preview" and not o["complete"]), None)

            # Detect order change
            if active and active["id"] != last_order_id:
                last_order_id = active["id"]
                # Find order index
                for i, os in enumerate(self.order_seq):
                    if os.get("id") == active["id"]:
                        order_idx = i
                        break
                # Clear old assignments and re-distribute
                assignments = self._distribute(bots, active, items, drop_off)
                bot_targets = {}

            if not active:
                actions = [{"bot": b["id"], "action": "wait"} for b in bots]
                for b in bots:
                    all_positions[b["id"]].append(tuple(b["position"]))
                all_actions.append(actions)
                sim.apply_actions(actions)
                continue

            active_needed = get_needed_items(active)

            # Lock-based collision avoidance (same as reactive brain)
            locked = set(tuple(b["position"]) for b in bots)
            actions = []

            for bot in sorted(bots, key=lambda b: b["id"]):
                bid = bot["id"]
                pos = tuple(bot["position"])
                locked.discard(pos)

                action = self._decide_bot(
                    bot, bots, items, active, active_needed, preview,
                    drop_off, state, locked, assignments, bot_targets,
                    order_idx, preassigned_orders
                )
                actions.append(action)

                # Lock next position
                act_str = action["action"]
                if act_str.startswith("move_"):
                    dx, dy = {"move_right": (1, 0), "move_left": (-1, 0),
                              "move_down": (0, 1), "move_up": (0, -1)}[act_str]
                    locked.add((pos[0] + dx, pos[1] + dy))
                else:
                    locked.add(pos)

            # Swap/same-dest conflict resolution
            actions = self._resolve_conflicts(actions, bots)

            for b in bots:
                all_positions[b["id"]].append(tuple(b["position"]))
            all_actions.append(actions)
            sim.apply_actions(actions)

        # Build plan
        plan = {"type": "optimized", "n_bots": self.n_bots, "bot_actions": {}, "bot_positions": {}}
        for bid in range(self.n_bots):
            plan["bot_actions"][str(bid)] = [
                {k: v for k, v in a.items() if k != "bot"}
                for round_actions in all_actions
                for a in round_actions
                if a["bot"] == bid
            ][:self.max_rounds]
            plan["bot_positions"][str(bid)] = all_positions[bid][:self.max_rounds]

        return plan, sim.score

    def _distribute(self, bots, active, items, drop_off):
        """Min-makespan assignment of active order items to bots."""
        needed = get_needed_items(active)
        for bot in bots:
            for itype in bot["inventory"]:
                if itype in needed and needed[itype] > 0:
                    needed[itype] -= 1
        needed = {k: v for k, v in needed.items() if v > 0}

        item_list = []
        for itype, count in needed.items():
            item_list.extend([itype] * count)

        if not item_list:
            return {b["id"]: [] for b in bots}

        bot_ids = [b["id"] for b in bots]
        bot_map = {b["id"]: b for b in bots}

        # Greedy min-makespan
        assignment = {bid: [] for bid in bot_ids}
        for item_type in item_list:
            best_bid = None
            best_cost = float("inf")
            for bid in bot_ids:
                if len(bot_map[bid]["inventory"]) + len(assignment[bid]) >= 3:
                    continue
                # Find cheapest shelf for this type from bot's position
                bpos = tuple(bot_map[bid]["position"])
                cost = self._cheapest_shelf_cost(item_type, bpos, items)
                # Balance: penalize bots that already have items assigned
                cost += len(assignment[bid]) * 5
                if cost < best_cost:
                    best_cost = cost
                    best_bid = bid
            if best_bid is not None:
                assignment[best_bid].append(item_type)

        return assignment

    def _cheapest_shelf_cost(self, item_type, from_pos, items):
        best = float("inf")
        for item in items:
            if item["type"] == item_type:
                shelf = (item["position"][0], item["position"][1])
                adj, _ = self.dm.best_adjacent(from_pos, shelf)
                if adj:
                    cost = self.dm.dist(from_pos, adj) + 1 + self.dm.dist(adj, self.drop_off)
                    if cost < best:
                        best = cost
        return best

    def _decide_bot(self, bot, all_bots, items, active, active_needed, preview,
                     drop_off, state, locked, assignments, bot_targets,
                     order_idx, preassigned_orders):
        """Decide one bot's action."""
        bid = bot["id"]
        pos = tuple(bot["position"])
        inventory = bot["inventory"]
        assigned = assignments.get(bid, [])

        has_useful = any(active_needed.get(it, 0) > 0 for it in inventory)

        # --- Clear dropoff area if carrying useless items ---
        if not has_useful and inventory and pos == drop_off:
            # Check preview utility
            preview_useful = False
            if preview:
                pn = get_needed_items(preview)
                preview_useful = any(pn.get(it, 0) > 0 for it in inventory)
            if not preview_useful:
                for dx, dy in [(0, -1), (1, 0), (-1, 0), (0, 1)]:
                    nxt = (pos[0] + dx, pos[1] + dy)
                    if self.dm.dist(pos, nxt) == 1 and nxt not in locked:
                        return {"bot": bid, "action": _pos_to_action(pos, nxt)}
                return {"bot": bid, "action": "wait"}

        # --- At dropoff with useful items → deliver ---
        if has_useful and pos == drop_off:
            return {"bot": bid, "action": "drop_off"}

        # --- Has useful items + (full OR no active assigned) → go deliver ---
        if has_useful and (len(inventory) >= 3 or not assigned):
            return self._navigate(bid, pos, drop_off, state, locked)

        # --- Pick up assigned active items ---
        if assigned and len(inventory) < 3:
            # Adjacent to any assigned item? Pick it up
            for ai, atype in enumerate(assigned):
                for item in items:
                    if item["type"] == atype:
                        shelf = (item["position"][0], item["position"][1])
                        if abs(pos[0] - shelf[0]) + abs(pos[1] - shelf[1]) == 1:
                            assigned.pop(ai)
                            return {"bot": bid, "action": "pick_up", "item_id": item["id"]}

            # Navigate to cheapest assigned item
            target_type = assigned[0]
            best_item = None
            best_cost = float("inf")
            for item in items:
                if item["type"] == target_type:
                    shelf = (item["position"][0], item["position"][1])
                    trip = self.dm.trip_cost(pos, shelf)
                    if trip < best_cost:
                        best_cost = trip
                        best_item = item
            if best_item:
                shelf = (best_item["position"][0], best_item["position"][1])
                adj, _ = self.dm.best_adjacent(pos, shelf)
                if adj:
                    return self._navigate(bid, pos, adj, state, locked)

        # --- Idle: preview picking / speculative ---
        if not assigned and not has_useful and len(inventory) < 3:
            # Preview items
            if preview:
                prev_needed = get_needed_items(preview)
                for b in all_bots:
                    for it in b["inventory"]:
                        if it in prev_needed and prev_needed[it] > 0:
                            prev_needed[it] -= 1
                prev_needed = {k: v for k, v in prev_needed.items() if v > 0}
                if prev_needed:
                    return self._pick_cheapest(bid, pos, items, prev_needed, locked, state)

            # Future order pre-picking: pick items for order N+2 (if known)
            # But ONLY from home aisle (cheap, no extra travel)
            if order_idx + 2 < len(self.order_seq) and self.aisle_xs:
                fo = self.order_seq[order_idx + 2]
                fo_types = set(fo["items_required"])
                # Find items on my home aisle that match future order
                home_x = self.aisle_xs[bid % len(self.aisle_xs)]
                for item in items:
                    if item["type"] in fo_types:
                        shelf = (item["position"][0], item["position"][1])
                        # Is this shelf on my home aisle?
                        if shelf[0] == home_x - 1 or shelf[0] == home_x + 1:
                            if abs(pos[0] - shelf[0]) + abs(pos[1] - shelf[1]) == 1:
                                return {"bot": bid, "action": "pick_up", "item_id": item["id"]}
                            adj, _ = self.dm.best_adjacent(pos, shelf)
                            if adj:
                                return self._navigate(bid, pos, adj, state, locked)

            # Regular speculative: cheap un-held types
            held_types = set()
            for b in all_bots:
                for it in b["inventory"]:
                    held_types.add(it)
            spec_needed = {}
            for item in items:
                itype = item["type"]
                if itype not in held_types:
                    shelf = (item["position"][0], item["position"][1])
                    cost = self.dm.trip_cost(drop_off, shelf)
                    if cost <= 28 and (itype not in spec_needed or cost < spec_needed[itype]):
                        spec_needed[itype] = cost
            if spec_needed:
                return self._pick_cheapest(bid, pos, items, {k: 1 for k in spec_needed}, locked, state,
                                            home_aisle_penalty=4)

        # --- Park ---
        home_x = self.aisle_xs[bid % len(self.aisle_xs)] if self.aisle_xs else pos[0]
        park_pos = (home_x, 1)
        if pos != park_pos:
            return self._navigate(bid, pos, park_pos, state, locked)
        return {"bot": bid, "action": "wait"}

    def _pick_cheapest(self, bid, pos, items, needed, locked, state, home_aisle_penalty=0):
        """Navigate to and pick up the cheapest needed item."""
        best_item = None
        best_cost = float("inf")
        home_x = self.aisle_xs[bid % len(self.aisle_xs)] if self.aisle_xs else None

        for item in items:
            if item["type"] not in needed:
                continue
            shelf = (item["position"][0], item["position"][1])
            cost = self.dm.trip_cost(pos, shelf)
            if home_aisle_penalty and home_x:
                on_home = (shelf[0] == home_x - 1 or shelf[0] == home_x + 1)
                if not on_home:
                    cost += home_aisle_penalty
            if cost < best_cost:
                best_cost = cost
                best_item = item

        if best_item:
            shelf = (best_item["position"][0], best_item["position"][1])
            if abs(pos[0] - shelf[0]) + abs(pos[1] - shelf[1]) == 1:
                return {"bot": bid, "action": "pick_up", "item_id": best_item["id"]}
            adj, _ = self.dm.best_adjacent(pos, shelf)
            if adj:
                return self._navigate(bid, pos, adj, state, locked)

        return {"bot": bid, "action": "wait"}

    def _navigate(self, bid, pos, target, state, locked):
        """Navigate toward target with lock-based collision avoidance."""
        if pos == target:
            return {"bot": bid, "action": "wait"}

        # DM gradient descent
        nxt = self.dm.next_step(pos, target)
        if nxt and nxt != pos and nxt not in locked:
            return {"bot": bid, "action": _pos_to_action(pos, nxt)}

        # BFS fallback
        blocked = set(self.static_blocked)
        blocked.update(locked)
        path = bfs(pos, target, blocked, state["grid"]["width"], state["grid"]["height"])
        if path and len(path) >= 2:
            direct = self.dm.dist(pos, target)
            if len(path) - 1 <= direct + 14:
                return {"bot": bid, "action": path_to_action(pos, path[1])}

        # Deadlock breaker
        cur_dist = self.dm.dist(pos, target)
        options = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nxt = (pos[0] + dx, pos[1] + dy)
            if self.dm.dist(pos, nxt) == 1 and nxt not in locked:
                d = self.dm.dist(nxt, target)
                priority = 0 if d < cur_dist else (1 if d == cur_dist else 2)
                options.append((priority, d, nxt))
        if options:
            options.sort()
            if options[0][0] <= 1:
                return {"bot": bid, "action": _pos_to_action(pos, options[0][2])}
            if state.get("round", 0) % 2 == 0:
                return {"bot": bid, "action": _pos_to_action(pos, options[0][2])}

        return {"bot": bid, "action": "wait"}

    def _resolve_conflicts(self, actions, bots):
        """Fix swap and same-destination conflicts."""
        _MOVE_DELTA = {"move_right": (1, 0), "move_left": (-1, 0),
                       "move_down": (0, 1), "move_up": (0, -1)}
        bot_pos = {b["id"]: tuple(b["position"]) for b in bots}
        bot_dest = {}
        for a in actions:
            bid = a["bot"]
            act = a["action"]
            if act in _MOVE_DELTA:
                dx, dy = _MOVE_DELTA[act]
                bot_dest[bid] = (bot_pos[bid][0] + dx, bot_pos[bid][1] + dy)
            else:
                bot_dest[bid] = bot_pos[bid]

        action_map = {a["bot"]: a for a in actions}

        # Swap detection
        for i, a1 in enumerate(actions):
            for a2 in actions[i + 1:]:
                b1, b2 = a1["bot"], a2["bot"]
                if bot_dest[b1] == bot_pos[b2] and bot_dest[b2] == bot_pos[b1]:
                    waiter = max(b1, b2)
                    action_map[waiter]["action"] = "wait"
                    action_map[waiter].pop("item_id", None)

        # Same-destination
        dest_to_bots = {}
        for a in actions:
            bid = a["bot"]
            if a["action"] in _MOVE_DELTA:
                dx, dy = _MOVE_DELTA[a["action"]]
                dest = (bot_pos[bid][0] + dx, bot_pos[bid][1] + dy)
                dest_to_bots.setdefault(dest, []).append(bid)
        for dest, bids in dest_to_bots.items():
            if len(bids) > 1:
                bids.sort()
                for bid in bids[1:]:
                    action_map[bid]["action"] = "wait"
                    action_map[bid].pop("item_id", None)

        return actions


def run_forward_planning(difficulty):
    """Run forward MAPF planning on latest recording."""
    recordings = list_recordings(difficulty)
    if not recordings:
        print(f"No recordings for '{difficulty}'.")
        return

    game_data_path = recordings[0]
    print(f"  Recording: {game_data_path}")

    game_data = load_game_data(game_data_path)
    planner = ForwardPlanner(game_data)

    t0 = time.time()
    plan, score = planner.plan()
    elapsed = time.time() - t0

    print(f"  Forward MAPF score: {score} ({elapsed:.1f}s)")

    # Save plan
    plan_path = f"simulation/{difficulty}/mapf_plan.json"
    with open(plan_path, "w") as f:
        json.dump(plan, f)
    print(f"  Saved plan to {plan_path}")

    return plan, score


if __name__ == "__main__":
    difficulty = sys.argv[1] if len(sys.argv) > 1 else "hard"
    run_forward_planning(difficulty)
