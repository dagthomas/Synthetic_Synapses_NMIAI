"""Local game simulator — runs a full game without the server.

Uses recorded data (grid, items, orders) to simulate the game engine locally.
Produces real scores so we can iterate on brain logic without burning tokens.
"""

import copy
import json
import random

from recorder import list_recordings


class SyntheticOrderGenerator:
    """Generates random orders when recorded orders are exhausted."""

    def __init__(self, item_types=None, seed=42, min_items=3, max_items=4):
        self._rng = random.Random(seed)
        self._next_id = 1000  # Start high to avoid collision with real IDs
        self._item_types = item_types or ["cheese", "butter", "yogurt", "milk"]
        self._min_items = min_items
        self._max_items = max_items

    def generate(self):
        """Generate a random order."""
        n_items = self._rng.randint(self._min_items, self._max_items)
        items = [self._rng.choice(self._item_types) for _ in range(n_items)]
        order = {
            "id": f"synth_{self._next_id}",
            "items_required": items,
        }
        self._next_id += 1
        return order


def load_game_data(recording_path):
    """Extract game setup from a recording."""
    with open(recording_path) as f:
        data = json.load(f)

    state0 = data["rounds"][0]["state"]

    # Extract order sequence from recording
    seen_ids = set()
    order_sequence = []
    for rnd in data["rounds"]:
        for order in rnd["state"]["orders"]:
            if order["id"] not in seen_ids:
                seen_ids.add(order["id"])
                order_sequence.append({
                    "id": order["id"],
                    "items_required": list(order["items_required"]),
                })

    return {
        "grid": state0["grid"],
        "items": state0["items"],
        "drop_off": state0["drop_off"],
        "bots": state0["bots"],
        "order_sequence": order_sequence,
        "max_rounds": state0["max_rounds"],
    }


class LocalSimulator:
    """Simulates the grocery bot game locally."""

    def __init__(self, game_data):
        self.grid = game_data["grid"]
        self.width = self.grid["width"]
        self.height = self.grid["height"]
        self.walls = set(tuple(w) for w in self.grid["walls"])
        self.items_template = game_data["items"]  # Items never deplete
        self.drop_off = tuple(game_data["drop_off"])
        self.max_rounds = game_data["max_rounds"]
        self.order_sequence = game_data["order_sequence"]

        # Shelf positions (impassable even without items)
        self.shelves = set()
        for item in self.items_template:
            self.shelves.add((item["position"][0], item["position"][1]))

        self.blocked = self.walls | self.shelves

        # Initialize bots
        self.bots = []
        for bot in game_data["bots"]:
            self.bots.append({
                "id": bot["id"],
                "position": list(bot["position"]),
                "inventory": [],
            })

        # Synthetic order generator using actual item types from the map
        actual_types = sorted(set(item["type"] for item in self.items_template))
        # Detect order size range from bot count (difficulty proxy)
        n_bots = len(self.bots)
        if n_bots <= 1:
            min_items, max_items = 3, 4    # easy
        elif n_bots <= 3:
            min_items, max_items = 3, 5    # medium
        elif n_bots <= 5:
            min_items, max_items = 3, 5    # hard
        else:
            min_items, max_items = 4, 6    # expert
        self._synth_gen = SyntheticOrderGenerator(
            item_types=actual_types, min_items=min_items, max_items=max_items
        )

        # Initialize orders
        self.order_idx = 0
        self.orders = []
        self._activate_next_order()  # active
        self._activate_next_order()  # preview

        # Score tracking
        self.score = 0
        self.items_delivered = 0
        self.orders_completed = 0
        self.round = 0

    def _activate_next_order(self):
        """Add next order from sequence, generating synthetic ones if exhausted."""
        if self.order_idx < len(self.order_sequence):
            order_data = self.order_sequence[self.order_idx]
        else:
            order_data = self._synth_gen.generate()

        status = "active" if not self.orders else "preview"
        if self.orders and self.orders[0].get("status") == "active":
            status = "preview"
        self.orders.append({
            "id": order_data["id"],
            "items_required": list(order_data["items_required"]),
            "items_delivered": [],
            "complete": False,
            "status": status,
        })
        self.order_idx += 1

    def get_state(self):
        """Build current game state (same format as server sends)."""
        return {
            "type": "game_state",
            "round": self.round,
            "max_rounds": self.max_rounds,
            "grid": self.grid,
            "bots": copy.deepcopy(self.bots),
            "items": copy.deepcopy(self.items_template),
            "orders": copy.deepcopy(self.orders),
            "drop_off": list(self.drop_off),
            "score": self.score,
            "active_order_index": 0,
            "total_orders": len(self.order_sequence),
        }

    def apply_actions(self, actions):
        """Apply bot actions.

        Simultaneous mode (matches likely server behavior):
        1. Compute intended destinations for all bots
        2. Resolve conflicts (swaps, same-destination)
        3. Apply non-conflicting moves + pickups/dropoffs
        """
        actions_by_bot = {a["bot"]: a for a in actions}
        _DELTAS = {"move_up": (0, -1), "move_down": (0, 1),
                   "move_left": (-1, 0), "move_right": (1, 0)}

        # Phase 1: compute intended destinations
        intents = {}  # bot_id -> (nx, ny) or None (non-move action)
        for bot in self.bots:
            action = actions_by_bot.get(bot["id"], {"action": "wait"})
            act = action.get("action", "wait")
            if act in _DELTAS:
                dx, dy = _DELTAS[act]
                bx, by = bot["position"]
                nx, ny = bx + dx, by + dy
                # Basic validity (bounds, walls/shelves)
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in self.blocked:
                    intents[bot["id"]] = (nx, ny)
                else:
                    intents[bot["id"]] = None  # invalid move, treat as wait
            else:
                intents[bot["id"]] = None  # non-move action

        # Phase 2: resolve conflicts
        # 2a: Swap detection — if A→B_pos and B→A_pos, both fail
        bot_map = {b["id"]: b for b in self.bots}
        pos_to_bot = {tuple(b["position"]): b["id"] for b in self.bots}
        failed = set()

        for bid, dest in intents.items():
            if dest is None:
                continue
            # Is there a bot at dest?
            occupant = pos_to_bot.get(dest)
            if occupant is not None and occupant != bid:
                # Check if occupant is moving to our position (swap)
                occ_dest = intents.get(occupant)
                if occ_dest == tuple(bot_map[bid]["position"]):
                    failed.add(bid)
                    failed.add(occupant)

        # 2b: Same-destination — if multiple bots move to same cell, all fail
        dest_to_bids = {}
        for bid, dest in intents.items():
            if dest is not None and bid not in failed:
                dest_to_bids.setdefault(dest, []).append(bid)
        for dest, bids in dest_to_bids.items():
            if len(bids) > 1:
                failed.update(bids)

        # 2c: Moving into occupied cell — cascade until stable
        # If occupant can't move (staying, failed, or blocked by cascade), we can't move there either
        changed = True
        while changed:
            changed = False
            for bid, dest in intents.items():
                if dest is None or bid in failed:
                    continue
                occupant = pos_to_bot.get(dest)
                if occupant is not None and occupant != bid:
                    # Occupant staying (non-move, failed, or blocked)?
                    if intents.get(occupant) is None or occupant in failed:
                        failed.add(bid)
                        changed = True

        # Phase 3: apply actions
        # First: apply successful moves
        for bot in sorted(self.bots, key=lambda b: b["id"]):
            bid = bot["id"]
            dest = intents.get(bid)
            if dest is not None and bid not in failed:
                bot["position"] = list(dest)

        # Then: apply non-move actions (pickup, dropoff)
        for bot in sorted(self.bots, key=lambda b: b["id"]):
            action = actions_by_bot.get(bot["id"], {"action": "wait"})
            act = action.get("action", "wait")
            if act not in _DELTAS:
                self._apply_single_action(bot, action)

    def _apply_single_action(self, bot, action):
        """Apply one bot's action."""
        act = action.get("action", "wait")
        bx, by = bot["position"]

        if act == "move_up":
            self._try_move(bot, bx, by - 1)
        elif act == "move_down":
            self._try_move(bot, bx, by + 1)
        elif act == "move_left":
            self._try_move(bot, bx - 1, by)
        elif act == "move_right":
            self._try_move(bot, bx + 1, by)
        elif act == "pick_up":
            self._try_pickup(bot, action.get("item_id"))
        elif act == "drop_off":
            self._try_dropoff(bot)

    def _try_move(self, bot, nx, ny):
        """Try to move bot to (nx, ny)."""
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            return
        if (nx, ny) in self.blocked:
            return
        # Check collision with other bots
        for other in self.bots:
            if other["id"] != bot["id"] and other["position"] == [nx, ny]:
                return
        bot["position"] = [nx, ny]

    def _try_pickup(self, bot, item_id):
        """Try to pick up an item."""
        if not item_id or len(bot["inventory"]) >= 3:
            return

        # Find the item
        target = None
        for item in self.items_template:
            if item["id"] == item_id:
                target = item
                break

        if not target:
            return

        # Must be adjacent (Manhattan distance 1)
        ix, iy = target["position"]
        bx, by = bot["position"]
        if abs(bx - ix) + abs(by - iy) != 1:
            return

        bot["inventory"].append(target["type"])

    def _try_dropoff(self, bot):
        """Try to drop off items."""
        bx, by = bot["position"]
        if (bx, by) != self.drop_off:
            return
        if not bot["inventory"]:
            return

        active = self.orders[0] if self.orders else None
        if not active or active["status"] != "active":
            return

        self._deliver_items(bot, active)

    def _deliver_items(self, bot, order):
        """Deliver matching items from bot to order. Handle order completion."""
        remaining_inv = []
        for item_type in bot["inventory"]:
            # Check if order needs this item
            needed = order["items_required"].count(item_type) - order["items_delivered"].count(item_type)
            if needed > 0:
                order["items_delivered"].append(item_type)
                self.score += 1
                self.items_delivered += 1
            else:
                remaining_inv.append(item_type)

        bot["inventory"] = remaining_inv

        # Check if order is complete
        if sorted(order["items_delivered"]) == sorted(order["items_required"]):
            order["complete"] = True
            self.score += 5
            self.orders_completed += 1

            # Transition: remove completed, promote preview to active
            self.orders.pop(0)
            if self.orders:
                self.orders[0]["status"] = "active"
            self._activate_next_order()

            # Re-check: bot's remaining items might match new active order
            new_active = self.orders[0] if self.orders else None
            if new_active and new_active["status"] == "active" and bot["inventory"]:
                self._deliver_items(bot, new_active)

    def run(self, brain_fn, verbose=True):
        """Run full game simulation.

        Args:
            brain_fn: function(state) -> actions list
            verbose: print progress

        Returns:
            Final game result dict.
        """
        for rnd in range(self.max_rounds):
            self.round = rnd
            state = self.get_state()
            actions = brain_fn(state)
            self.apply_actions(actions)

            if verbose and rnd % 10 == 0:
                active = self.orders[0] if self.orders else None
                delivered = len(active["items_delivered"]) if active else "?"
                required = len(active["items_required"]) if active else "?"
                print(
                    f"  Round {rnd:3d}/300 | Score {self.score:3d} | "
                    f"Orders {self.orders_completed} | Active {delivered}/{required}"
                )

        result = {
            "score": self.score,
            "rounds_used": self.max_rounds,
            "items_delivered": self.items_delivered,
            "orders_completed": self.orders_completed,
        }

        if verbose:
            print(f"\n{'=' * 40}")
            print(f"  SIMULATION COMPLETE")
            print(f"  Score:            {result['score']}")
            print(f"  Items delivered:  {result['items_delivered']}")
            print(f"  Orders completed: {result['orders_completed']}")
            print(f"{'=' * 40}")

        return result


def run_simulation(difficulty):
    """Run local simulation using latest recording data."""
    recordings = list_recordings(difficulty)
    if not recordings:
        print(f"  No recordings for '{difficulty}'. Play a live game first.")
        return

    recording_path = recordings[0]
    print(f"  Using recording: {recording_path}")

    game_data = load_game_data(recording_path)
    print(f"  Orders discovered: {len(game_data['order_sequence'])}")
    print(f"  Items on map: {len(game_data['items'])}")
    print(f"  Bots: {len(game_data['bots'])}")
    print()

    from brain import decide_actions
    order_sequence = game_data["order_sequence"]
    sim = LocalSimulator(game_data)
    return sim.run(lambda state: decide_actions(state, game_plan=order_sequence))
