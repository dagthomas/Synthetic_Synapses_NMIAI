"""Full-Game Optimizer — Sequential DP with Beam Search.

Plans the entire 300-round game offline for all bots, then replays
pre-computed actions live. Uses sequential beam search: plan each bot
one at a time, locking previous bots' trajectories as moving walls.

Usage:
    python main.py optimize hard    # Optimize latest hard recording
    python main.py replay hard URL  # Replay optimized plan live
"""

import copy
import json
import os
import time
from collections import Counter, defaultdict

from distance import DistanceMatrix
from recorder import list_recordings
from simulator import LocalSimulator, load_game_data


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ActionNode:
    """Linked list node for memory-efficient action history."""
    __slots__ = ('action', 'parent', 'round')

    def __init__(self, action, parent, round_num):
        self.action = action
        self.parent = parent
        self.round = round_num

    def to_list(self):
        actions = []
        node = self
        while node is not None:
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        return actions


class BeamState:
    """State tracked during beam search for a single bot."""
    __slots__ = ('pos', 'inventory', 'order_idx', 'own_delivered',
                 'score', 'items_delivered', 'orders_completed',
                 'action_tail', 'heuristic')

    def __init__(self, pos, inventory, order_idx, own_delivered,
                 score, items_delivered, orders_completed,
                 action_tail, heuristic=0):
        self.pos = pos
        self.inventory = inventory      # tuple of item types
        self.order_idx = order_idx      # global order index
        self.own_delivered = own_delivered  # tuple of (order_idx, item_type) delivered by THIS bot
        self.score = score              # THIS bot's marginal score contribution
        self.items_delivered = items_delivered
        self.orders_completed = orders_completed
        self.action_tail = action_tail
        self.heuristic = heuristic

    def state_key(self):
        return (self.pos, self.inventory, self.order_idx, self.own_delivered)


class LockedTrajectory:
    """A fully planned bot's positions and actions per round."""
    def __init__(self, bot_id, positions, actions):
        self.bot_id = bot_id
        self.positions = positions
        self.actions = actions


# ---------------------------------------------------------------------------
# Order Timeline — pre-simulates locked bots
# ---------------------------------------------------------------------------

class OrderTimeline:
    """Pre-simulates locked bots to determine per-round order state.

    Key outputs:
    - round_order_idx[t]: which order is active after round t
    - round_delivered[t]: {type: count} delivered by locked bots to active order
    - order_needs[oidx]: {type: count} items STILL needed from the planning bot
      (after subtracting what locked bots will eventually deliver)
    """

    def __init__(self, all_orders, locked_trajectories, drop_off, max_rounds):
        self.all_orders = all_orders
        self.drop_off = tuple(drop_off)
        self.max_rounds = max_rounds

        self.round_order_idx = []
        self.round_delivered = []

        # What items the planning bot needs to deliver for each order
        # (total required minus what locked bots handle)
        self.order_needs = {}  # order_idx -> {type: count}

        self._simulate(locked_trajectories)

    def _simulate(self, locked_trajs):
        """Simulate locked bots and compute what each order still needs."""
        if not locked_trajs:
            # No locked bots: all orders need everything
            for i, order in enumerate(self.all_orders):
                self.order_needs[i] = dict(Counter(order["items_required"]))
            for rnd in range(self.max_rounds):
                self.round_order_idx.append(0)
                self.round_delivered.append({})
            return

        order_idx = 0
        delivered = {}
        inventories = {lt.bot_id: [] for lt in locked_trajs}

        # Track what each locked bot delivers to each order over the full game
        order_deliveries = defaultdict(Counter)  # order_idx -> Counter of delivered types

        for rnd in range(self.max_rounds):
            if order_idx >= len(self.all_orders):
                self.round_order_idx.append(order_idx)
                self.round_delivered.append({})
                continue

            current = self.all_orders[order_idx]
            required = current["items_required"]

            for lt in sorted(locked_trajs, key=lambda x: x.bot_id):
                if rnd >= len(lt.actions):
                    continue
                action = lt.actions[rnd]
                act = action.get("action", "wait")
                pre_pos = tuple(lt.positions[rnd])

                if act == "pick_up":
                    item_type = action.get("item_type")
                    if item_type and len(inventories[lt.bot_id]) < 3:
                        inventories[lt.bot_id].append(item_type)
                elif act == "drop_off":
                    if pre_pos == self.drop_off and inventories[lt.bot_id]:
                        new_inv = []
                        for itype in inventories[lt.bot_id]:
                            needed = required.count(itype) - delivered.get(itype, 0)
                            if needed > 0:
                                delivered[itype] = delivered.get(itype, 0) + 1
                                order_deliveries[order_idx][itype] += 1
                            else:
                                new_inv.append(itype)
                        inventories[lt.bot_id] = new_inv

                        if self._is_complete(required, delivered):
                            order_idx += 1
                            delivered = {}
                            if order_idx < len(self.all_orders):
                                next_req = self.all_orders[order_idx]["items_required"]
                                # Cascade deliveries
                                for lt2 in sorted(locked_trajs, key=lambda x: x.bot_id):
                                    if tuple(lt2.positions[rnd]) != self.drop_off:
                                        continue
                                    remaining = []
                                    for it in inventories[lt2.bot_id]:
                                        n = next_req.count(it) - delivered.get(it, 0)
                                        if n > 0:
                                            delivered[it] = delivered.get(it, 0) + 1
                                            order_deliveries[order_idx][it] += 1
                                        else:
                                            remaining.append(it)
                                    inventories[lt2.bot_id] = remaining
                                required = next_req

            self.round_order_idx.append(order_idx)
            self.round_delivered.append(dict(delivered))

        # Compute what each order still needs from the planning bot
        for i, order in enumerate(self.all_orders):
            req = Counter(order["items_required"])
            locked_del = order_deliveries.get(i, Counter())
            remaining = {}
            for itype, count in req.items():
                left = count - locked_del.get(itype, 0)
                if left > 0:
                    remaining[itype] = left
            self.order_needs[i] = remaining

    def _is_complete(self, required, delivered):
        for item, count in Counter(required).items():
            if delivered.get(item, 0) < count:
                return False
        return True

    def get_state_at(self, rnd):
        """Get (order_idx, delivered) at start of round rnd."""
        if rnd <= 0:
            return 0, {}
        idx = min(rnd - 1, len(self.round_order_idx) - 1)
        return self.round_order_idx[idx], dict(self.round_delivered[idx])


# ---------------------------------------------------------------------------
# Full Game Optimizer
# ---------------------------------------------------------------------------

class FullGameOptimizer:
    """Plans optimal actions for all bots using sequential beam search."""

    def __init__(self, game_data, beam_width=1000):
        self.game_data = game_data
        self.beam_width = beam_width
        self.max_rounds = game_data["max_rounds"]
        self.drop_off = tuple(game_data["drop_off"])
        self.n_bots = len(game_data["bots"])

        state0 = {
            "grid": game_data["grid"],
            "items": game_data["items"],
            "drop_off": game_data["drop_off"],
        }
        self.dm = DistanceMatrix(state0)

        self.shelf_map = defaultdict(list)
        for item in game_data["items"]:
            pos = (item["position"][0], item["position"][1])
            self.shelf_map[pos].append(item)

        self.shelves_by_type = defaultdict(list)
        for item in game_data["items"]:
            pos = (item["position"][0], item["position"][1])
            if pos not in self.shelves_by_type[item["type"]]:
                self.shelves_by_type[item["type"]].append(pos)

        self.all_orders = game_data["order_sequence"]
        self.bot_starts = [tuple(b["position"]) for b in game_data["bots"]]

        self.walls = set(tuple(w) for w in game_data["grid"]["walls"])
        self.shelf_positions = set()
        for item in game_data["items"]:
            self.shelf_positions.add((item["position"][0], item["position"][1]))
        self.static_blocked = self.walls | self.shelf_positions

        self.width = game_data["grid"]["width"]
        self.height = game_data["grid"]["height"]

        self.walkable = set()
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in self.static_blocked:
                    self.walkable.add((x, y))

    def optimize(self, passes=3):
        """Run sequential beam search with refinement.

        Uses simulation-in-the-loop: after each pass, verify the combined
        score and try random beam width variations to find the best config.
        """
        total_start = time.time()

        best_score = 0
        best_locked = None

        # Try multiple beam widths to find the best one
        # (collision sensitivity means different widths give very different results)
        beam_widths = [self.beam_width]
        if self.n_bots > 1:
            # Multi-bot: sweep around the base beam width
            base = self.beam_width
            beam_widths = sorted(set([
                max(50, base // 4), max(50, base // 2),
                max(50, base * 3 // 4), base,
                base * 3 // 2, base * 2
            ]))

        for bw in beam_widths:
            self.beam_width = bw
            locked = []

            for bot_id in range(self.n_bots):
                traj = self.plan_single_bot(bot_id, locked)
                locked.append(traj)

            # Refine
            for _ in range(passes - 1):
                improved = False
                for bot_id in range(self.n_bots):
                    others = [lt for lt in locked if lt.bot_id != bot_id]
                    new_traj = self.plan_single_bot(bot_id, others)
                    if new_traj.score > locked[bot_id].score:
                        locked[bot_id] = new_traj
                        improved = True
                if not improved:
                    break

            plan = self._build_plan(locked)
            score = self._quick_verify(plan)
            individual = sum(lt.score for lt in locked)
            print(f"    beam={bw:4d}: individual={individual:3d}, verified={score:3d}")

            if score > best_score:
                best_score = score
                best_locked = list(locked)

        elapsed = time.time() - total_start
        print(f"\n  Best verified score: {best_score}")
        print(f"  Optimization complete in {elapsed:.1f}s")

        plan = self._build_plan(best_locked)
        combined_score = self._verify_plan(plan)
        return plan, combined_score

    def _quick_verify(self, plan):
        """Quick verification — return score only."""
        sim = LocalSimulator(copy.deepcopy(self.game_data))
        for rnd in range(self.max_rounds):
            sim.round = rnd
            actions = []
            for bot_id in range(self.n_bots):
                bot_actions = plan["bot_actions"][str(bot_id)]
                if rnd < len(bot_actions):
                    action = dict(bot_actions[rnd])
                    action["bot"] = bot_id
                    actions.append(action)
                else:
                    actions.append({"bot": bot_id, "action": "wait"})
            sim.apply_actions(actions)
        return sim.score

    def plan_single_bot(self, bot_id, locked_trajectories, max_orders=None):
        """Plan a single bot using beam search DP."""
        t0 = time.time()

        timeline = OrderTimeline(
            self.all_orders, locked_trajectories, self.drop_off, self.max_rounds
        )
        self._current_timeline = timeline
        self._max_orders = max_orders

        start_pos = self.bot_starts[bot_id]

        initial = BeamState(
            pos=start_pos, inventory=(), order_idx=0, own_delivered=(),
            score=0, items_delivered=0, orders_completed=0,
            action_tail=None, heuristic=0,
        )
        beam = [initial]

        for rnd in range(self.max_rounds):
            blocked = self._get_blocked_at_round(rnd, bot_id, locked_trajectories)
            locked_oidx, locked_del = timeline.get_state_at(rnd)

            candidates = []
            for state in beam:
                synced = self._sync_order(state, locked_oidx, locked_del, timeline)

                for action in self._valid_actions(synced, blocked, rnd, timeline):
                    new_state = self._apply_action(synced, action, rnd, timeline)
                    if new_state is not None:
                        candidates.append(new_state)

            if not candidates:
                break
            beam = self._select_beam(candidates)

        best = max(beam, key=lambda s: (s.score, s.items_delivered,
                                         -self.dm.dist(s.pos, self.drop_off)))

        actions = best.action_tail.to_list() if best.action_tail else []
        while len(actions) < self.max_rounds:
            actions.append({"action": "wait"})

        positions = [start_pos]
        pos = start_pos
        for a in actions:
            pos = self._next_pos(pos, a)
            positions.append(pos)

        traj = LockedTrajectory(bot_id, positions, actions)
        traj.score = best.score
        traj.items_delivered = best.items_delivered
        traj.orders_completed = best.orders_completed
        return traj

    def _sync_order(self, state, locked_oidx, locked_del, timeline):
        """If locked bots completed orders, advance the planning bot's order_idx."""
        if state.order_idx >= locked_oidx:
            return state

        new_idx = locked_oidx
        new_score = state.score
        new_items = state.items_delivered
        new_orders = state.orders_completed
        new_inv = state.inventory
        new_own_del = tuple(d for d in state.own_delivered if d[0] >= new_idx)

        # If at dropoff, cascade inventory to new active order
        if state.pos == self.drop_off and new_inv and new_idx < len(self.all_orders):
            needs = dict(timeline.order_needs.get(new_idx, {}))
            # Subtract own prior deliveries to this order
            for (d_oidx, itype) in new_own_del:
                if d_oidx == new_idx:
                    needs[itype] = needs.get(itype, 0) - 1

            remaining = []
            for itype in new_inv:
                if needs.get(itype, 0) > 0:
                    needs[itype] -= 1
                    new_own_del = new_own_del + ((new_idx, itype),)
                    new_score += 1
                    new_items += 1
                else:
                    remaining.append(itype)
            new_inv = tuple(remaining)

            # Check if this bot just completed the order
            if self._check_order_complete(new_idx, new_own_del, timeline):
                new_score += 5
                new_orders += 1
                new_idx += 1
                new_own_del = tuple(d for d in new_own_del if d[0] >= new_idx)

        return BeamState(
            pos=state.pos, inventory=new_inv,
            order_idx=new_idx, own_delivered=new_own_del,
            score=new_score, items_delivered=new_items,
            orders_completed=new_orders,
            action_tail=state.action_tail, heuristic=state.heuristic,
        )

    def _get_blocked_at_round(self, rnd, planning_bot_id, locked_trajectories):
        """Get blocked positions from locked bots.

        In the simulator, bots processed in ID order. When planning bot k acts:
        - Bots j < k already acted → at post-action position
        - Bots j > k haven't acted → at pre-action position
        """
        blocked = set()
        for lt in locked_trajectories:
            if lt.bot_id < planning_bot_id:
                idx = min(rnd + 1, len(lt.positions) - 1)
            else:
                idx = min(rnd, len(lt.positions) - 1)
            blocked.add(lt.positions[idx])
        return blocked

    def _valid_actions(self, state, bot_blocked, rnd, timeline):
        """Generate valid actions."""
        actions = []
        px, py = state.pos

        # Move actions
        for act, (dx, dy) in [("move_up", (0, -1)), ("move_down", (0, 1)),
                               ("move_left", (-1, 0)), ("move_right", (1, 0))]:
            target = (px + dx, py + dy)
            if target in self.walkable and target not in bot_blocked:
                actions.append({"action": act})

        # Pickup actions
        if len(state.inventory) < 3:
            picked_types = set()
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                shelf_pos = (px + dx, py + dy)
                if shelf_pos in self.shelf_map:
                    for item in self.shelf_map[shelf_pos]:
                        itype = item["type"]
                        if itype not in picked_types:
                            if self._is_useful(state, itype, timeline):
                                actions.append({
                                    "action": "pick_up",
                                    "item_id": item["id"],
                                    "item_type": itype,
                                })
                                picked_types.add(itype)

        # Drop-off
        if state.pos == self.drop_off and state.inventory:
            if self._has_deliverable(state, timeline):
                actions.append({"action": "drop_off"})

        actions.append({"action": "wait"})
        return actions

    def _is_useful(self, state, item_type, timeline):
        """Check if item_type is needed by this bot for current or future orders."""
        oidx = state.order_idx
        # If max_orders set and we've completed enough, stop picking
        if self._max_orders and state.orders_completed >= self._max_orders:
            return False
        for look in range(3):
            check_idx = oidx + look
            if check_idx >= len(self.all_orders):
                break
            needs = timeline.order_needs.get(check_idx, {})
            if needs.get(item_type, 0) > 0:
                # Subtract what we already have / delivered
                own_count = sum(1 for it in state.inventory if it == item_type)
                own_del = sum(1 for (d_oidx, it) in state.own_delivered
                             if d_oidx == check_idx and it == item_type)
                if needs[item_type] - own_del - own_count > 0:
                    return True
                # Even if this order is covered, might be useful for next
                continue
        return False

    def _has_deliverable(self, state, timeline):
        """Check if inventory has items matching current order needs."""
        oidx = state.order_idx
        if oidx >= len(self.all_orders):
            return False
        needs = dict(timeline.order_needs.get(oidx, {}))
        for (d_oidx, itype) in state.own_delivered:
            if d_oidx == oidx:
                needs[itype] = needs.get(itype, 0) - 1
        for itype in state.inventory:
            if needs.get(itype, 0) > 0:
                return True
        return False

    def _apply_action(self, state, action, rnd, timeline):
        """Apply action to beam state."""
        act = action["action"]
        new_pos = state.pos
        new_inv = state.inventory
        new_oidx = state.order_idx
        new_own_del = state.own_delivered
        new_score = state.score
        new_items = state.items_delivered
        new_orders = state.orders_completed

        if act.startswith("move_"):
            new_pos = self._next_pos(state.pos, action)
            if new_pos == state.pos:
                return None
        elif act == "pick_up":
            new_inv = state.inventory + (action["item_type"],)
        elif act == "drop_off":
            if new_oidx >= len(self.all_orders):
                return None

            # What does this bot still need to deliver for current order?
            needs = dict(timeline.order_needs.get(new_oidx, {}))
            for (d_oidx, itype) in new_own_del:
                if d_oidx == new_oidx:
                    needs[itype] = needs.get(itype, 0) - 1

            remaining = []
            for itype in new_inv:
                if needs.get(itype, 0) > 0:
                    needs[itype] -= 1
                    new_own_del = new_own_del + ((new_oidx, itype),)
                    new_score += 1
                    new_items += 1
                else:
                    remaining.append(itype)
            new_inv = tuple(remaining)

            # Check if order is now complete (locked + this bot's deliveries)
            if self._check_order_complete(new_oidx, new_own_del, timeline):
                new_score += 5
                new_orders += 1
                new_oidx += 1
                new_own_del = tuple(d for d in new_own_del if d[0] >= new_oidx)

                # Cascade to next order
                if new_oidx < len(self.all_orders) and new_inv:
                    next_needs = dict(timeline.order_needs.get(new_oidx, {}))
                    still_remaining = []
                    for itype in new_inv:
                        if next_needs.get(itype, 0) > 0:
                            next_needs[itype] -= 1
                            new_own_del = new_own_del + ((new_oidx, itype),)
                            new_score += 1
                            new_items += 1
                        else:
                            still_remaining.append(itype)
                    new_inv = tuple(still_remaining)

                    if self._check_order_complete(new_oidx, new_own_del, timeline):
                        new_score += 5
                        new_orders += 1
                        new_oidx += 1
                        new_own_del = tuple(d for d in new_own_del if d[0] >= new_oidx)

        action_record = {"action": act}
        if act == "pick_up":
            action_record["item_id"] = action["item_id"]
            action_record["item_type"] = action["item_type"]
        node = ActionNode(action_record, state.action_tail, rnd)

        heuristic = self._evaluate(
            new_pos, new_inv, new_oidx, new_own_del,
            new_score, new_items, new_orders, rnd, timeline
        )

        return BeamState(
            pos=new_pos, inventory=new_inv,
            order_idx=new_oidx, own_delivered=new_own_del,
            score=new_score, items_delivered=new_items,
            orders_completed=new_orders,
            action_tail=node, heuristic=heuristic,
        )

    def _check_order_complete(self, order_idx, own_delivered, timeline):
        """Check if order is complete (locked deliveries + own deliveries)."""
        if order_idx >= len(self.all_orders):
            return False
        needs = dict(timeline.order_needs.get(order_idx, {}))
        for (d_oidx, itype) in own_delivered:
            if d_oidx == order_idx:
                needs[itype] = needs.get(itype, 0) - 1
        return all(v <= 0 for v in needs.values())

    def _evaluate(self, pos, inventory, order_idx, own_delivered,
                  score, items_delivered, orders_completed, rnd, timeline):
        """Heuristic evaluation."""
        ev = score * 100000

        if order_idx >= len(self.all_orders):
            return ev + (self.max_rounds - rnd) * 10

        # What does this bot still need for current order?
        needs = dict(timeline.order_needs.get(order_idx, {}))
        for (d_oidx, itype) in own_delivered:
            if d_oidx == order_idx:
                needs[itype] = needs.get(itype, 0) - 1
        needs = {k: v for k, v in needs.items() if v > 0}

        # Count useful inventory items
        inv_counts = Counter(inventory)
        useful_active = 0
        useful_preview = 0
        for itype, count in inv_counts.items():
            active_needed = needs.get(itype, 0)
            active_match = min(count, max(0, active_needed))
            useful_active += active_match
            remaining = count - active_match
            if remaining > 0 and order_idx + 1 < len(self.all_orders):
                preview_needs = timeline.order_needs.get(order_idx + 1, {})
                useful_preview += min(remaining, preview_needs.get(itype, 0))

        dist_drop = self.dm.dist(pos, self.drop_off)
        total_still_needed = sum(max(0, v) for v in needs.values())

        # Active items value
        ev += useful_active * (30000 - dist_drop * 800)
        # Preview pipeline
        ev += useful_preview * 12000

        # Near completion
        remaining_to_pick = total_still_needed - useful_active
        if remaining_to_pick <= 0 and useful_active > 0:
            ev += 50000  # Just deliver!
        elif remaining_to_pick == 1:
            ev += 30000

        # Dead inventory penalty
        dead = len(inventory) - useful_active - useful_preview
        if dead > 0:
            ev -= dead * 25000

        # Dropoff congestion penalty
        if pos == self.drop_off and useful_active == 0:
            ev -= 15000

        # Time tiebreaker
        ev += (self.max_rounds - rnd) * 10

        # Proximity to needed shelves
        if remaining_to_pick > 0:
            min_d = 999
            for itype, count in needs.items():
                in_inv = inv_counts.get(itype, 0)
                if count - in_inv > 0:
                    for shelf_pos in self.shelves_by_type.get(itype, []):
                        _, d = self.dm.best_adjacent(pos, shelf_pos)
                        if d < min_d:
                            min_d = d
            if min_d < 999:
                ev += (30 - min(30, min_d)) * 500

        return ev

    def _select_beam(self, candidates):
        """Deduplicate and select top beam_width candidates."""
        seen = {}
        for c in candidates:
            key = c.state_key()
            if key not in seen or c.heuristic > seen[key].heuristic:
                seen[key] = c
        deduped = sorted(seen.values(), key=lambda s: s.heuristic, reverse=True)
        return deduped[:self.beam_width]

    def _next_pos(self, pos, action):
        act = action["action"]
        px, py = pos
        if act == "move_up": return (px, py - 1)
        if act == "move_down": return (px, py + 1)
        if act == "move_left": return (px - 1, py)
        if act == "move_right": return (px + 1, py)
        return pos

    def _build_plan(self, locked_trajectories):
        plan = {
            "type": "optimized",
            "n_bots": self.n_bots,
            "max_rounds": self.max_rounds,
            "bot_actions": {},
            "bot_positions": {},
        }
        for lt in locked_trajectories:
            plan["bot_actions"][str(lt.bot_id)] = lt.actions
            plan["bot_positions"][str(lt.bot_id)] = [list(p) for p in lt.positions]
        return plan

    def _verify_plan(self, plan):
        """Verify plan by running through LocalSimulator."""
        print("\n  Verifying plan via LocalSimulator...")
        sim = LocalSimulator(copy.deepcopy(self.game_data))

        for rnd in range(self.max_rounds):
            sim.round = rnd
            actions = []
            for bot_id in range(self.n_bots):
                bot_actions = plan["bot_actions"][str(bot_id)]
                if rnd < len(bot_actions):
                    action = dict(bot_actions[rnd])
                    action["bot"] = bot_id
                    actions.append(action)
                else:
                    actions.append({"bot": bot_id, "action": "wait"})
            sim.apply_actions(actions)

            if rnd % 50 == 0:
                active = sim.orders[0] if sim.orders else None
                delivered = len(active["items_delivered"]) if active else "?"
                required = len(active["items_required"]) if active else "?"
                print(f"    Round {rnd:3d}/300 | Score {sim.score:3d} | "
                      f"Orders {sim.orders_completed} | Active {delivered}/{required}")

        print(f"\n  Verified score: {sim.score}")
        print(f"  Items delivered: {sim.items_delivered}")
        print(f"  Orders completed: {sim.orders_completed}")
        return sim.score


# ---------------------------------------------------------------------------
# Plan I/O
# ---------------------------------------------------------------------------

def save_optimized_plan(plan, difficulty):
    folder = os.path.join(os.path.dirname(__file__), "simulation", difficulty)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "optimized_plan.json")
    with open(path, "w") as f:
        json.dump(plan, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved optimized plan: {path} ({size_kb:.0f} KB)")
    return path


def load_optimized_plan(difficulty):
    path = os.path.join(os.path.dirname(__file__), "simulation", difficulty,
                        "optimized_plan.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_optimization(difficulty, beam_width=1000, passes=3):
    """Run full-game optimization on latest recording."""
    from simulator import SyntheticOrderGenerator

    recordings = list_recordings(difficulty)
    if not recordings:
        print(f"  No recordings for '{difficulty}'. Play a live game first.")
        return None

    recording_path = recordings[0]
    print(f"  Using recording: {recording_path}")

    game_data = load_game_data(recording_path)

    # Extend orders with synthetic ones to cover the full 300 rounds
    n_bots = len(game_data["bots"])
    item_types = sorted(set(it["type"] for it in game_data["items"]))
    if n_bots <= 1:
        min_items, max_items = 3, 4
    elif n_bots <= 3:
        min_items, max_items = 3, 5
    elif n_bots <= 5:
        min_items, max_items = 3, 5
    else:
        min_items, max_items = 4, 6

    synth = SyntheticOrderGenerator(item_types, seed=42,
                                     min_items=min_items, max_items=max_items)
    # Add enough synthetic orders to cover 300 rounds
    while len(game_data["order_sequence"]) < 50:
        game_data["order_sequence"].append(synth.generate())

    print(f"  Orders: {len(game_data['order_sequence'])} (including synthetic)")
    print(f"  Bots: {n_bots}")
    print(f"  Grid: {game_data['grid']['width']}x{game_data['grid']['height']}")
    print(f"  Beam width: {beam_width}, Passes: {passes}")

    optimizer = FullGameOptimizer(game_data, beam_width=beam_width)
    plan, score = optimizer.optimize(passes=passes)

    save_optimized_plan(plan, difficulty)
    return score
