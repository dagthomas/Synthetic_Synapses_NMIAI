"""Probe game that captures server state for offline optimization.

Usage:
    python capture_game.py <wss://game.ainm.no/ws?token=...> <difficulty>

Saves capture to captures/<difficulty>_<date>.json.
"""
import asyncio
import json
import sys
import os
import time
from datetime import datetime, timezone
from collections import deque


def bfs_first_step(walls_set, width, height, start, goal):
    if start == goal:
        return "wait"
    q = deque([(start, None)])
    visited = {start}
    directions = [(0, -1, "move_up"), (0, 1, "move_down"),
                  (-1, 0, "move_left"), (1, 0, "move_right")]
    while q:
        (x, y), first = q.popleft()
        for dx, dy, action in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < width and 0 <= ny < height and
                (nx, ny) not in walls_set and (nx, ny) not in visited):
                fa = first if first else action
                if (nx, ny) == goal:
                    return fa
                visited.add((nx, ny))
                q.append(((nx, ny), fa))
    return "wait"


def bfs_distance(walls_set, width, height, start, goal):
    if start == goal:
        return 0
    q = deque([(start, 0)])
    visited = {start}
    while q:
        (x, y), dist = q.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < width and 0 <= ny < height and
                (nx, ny) not in walls_set and (nx, ny) not in visited):
                if (nx, ny) == goal:
                    return dist + 1
                visited.add((nx, ny))
                q.append(((nx, ny), dist + 1))
    return 9999


def find_adj_cells(ix, iy, walls_set, width, height):
    adj = []
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        ax, ay = ix + dx, iy + dy
        if 0 <= ax < width and 0 <= ay < height and (ax, ay) not in walls_set:
            adj.append((ax, ay))
    return adj


def get_order_needs(order):
    needed = {}
    for item in order["items_required"]:
        needed[item] = needed.get(item, 0) + 1
    for item in order.get("items_delivered", []):
        needed[item] = needed.get(item, 0) - 1
    return {k: v for k, v in needed.items() if v > 0}


def get_drops(data):
    dzones = data.get("drop_off_zones")
    if dzones:
        return [tuple(z) for z in dzones]
    return [tuple(data["drop_off"])]


def nearest_drop(bx, by, drops):
    return min(drops, key=lambda d: abs(bx - d[0]) + abs(by - d[1]))


# ── Stateful bot controller for capture ──────────────────────────────

ST_IDLE = 0
ST_PICKING = 1
ST_DELIVERING = 2

class CaptureBot:
    """Persistent state for a single bot across rounds."""
    __slots__ = ('bid', 'state', 'target', 'item_id', 'stuck_count', 'last_pos',
                 'blacklist')

    def __init__(self, bid):
        self.bid = bid
        self.state = ST_IDLE
        self.target = None
        self.item_id = None
        self.stuck_count = 0
        self.last_pos = None
        self.blacklist = {}  # item_id -> expires_at_round

    def set_idle(self):
        self.state = ST_IDLE
        self.target = None
        self.item_id = None

    def assign_pick(self, target, item_id):
        self.state = ST_PICKING
        self.target = target
        self.item_id = item_id

    def assign_deliver(self, drop):
        self.state = ST_DELIVERING
        self.target = drop
        self.item_id = None


class CaptureController:
    """Stateful multi-bot controller for capture games.

    Simple strategy: pick 1 item → deliver → repeat.
    Max 6 active bots, others park. Bots use nearest drop-off.
    """

    def __init__(self):
        self.bots = {}
        self.claimed_items = {}
        self.max_active = 2  # set dynamically in decide() based on bot count
        self.spawn = None
        self._active_set = False

    def decide(self, data, walls_set, width, height):
        import random
        drops = get_drops(data)
        rnd = data.get("round", 0)

        # Init bots on first call
        if not self.bots:
            for bot in data["bots"]:
                self.bots[bot["id"]] = CaptureBot(bot["id"])
            # Scale max_active by bot count: 2 for <=5 bots, ~1/3 for more
            num_bots = len(data["bots"])
            if not self._active_set:
                if num_bots <= 5:
                    self.max_active = 2
                elif num_bots <= 10:
                    self.max_active = 4
                else:
                    self.max_active = max(6, num_bots // 3)
                self._active_set = True
            # Detect spawn: all bots start there
            positions = [tuple(b["position"]) for b in data["bots"]]
            if len(set(positions)) == 1:
                self.spawn = positions[0]

        # Find active/preview orders
        active = preview = None
        for o in data["orders"]:
            if o.get("status") == "active" and not o["complete"]:
                active = o
            elif o.get("status") == "preview" and not o["complete"]:
                preview = o

        if not active:
            return [{"bot": b["id"], "action": "wait"} for b in data["bots"]]

        needed = get_order_needs(active)

        # Items on map
        items_by_type = {}
        items_by_id = {}
        for item in data["items"]:
            items_by_type.setdefault(item["type"], []).append(item)
            items_by_id[item["id"]] = item

        # Stuck detection + cleanup
        for bot in data["bots"]:
            bid = bot["id"]
            cb = self.bots[bid]
            pos = tuple(bot["position"])

            if cb.last_pos == pos and cb.state != ST_IDLE:
                cb.stuck_count += 1
            else:
                if cb.state != ST_IDLE:
                    cb.stuck_count = 0
            cb.last_pos = pos

            # Stuck too long → blacklist item, random walk
            if cb.stuck_count > 6:
                if cb.item_id:
                    cb.blacklist[cb.item_id] = rnd + 50
                cb.set_idle()
                cb.stuck_count = -8  # longer random walk

            # Item already picked by someone else
            if cb.state == ST_PICKING and cb.item_id:
                if cb.item_id not in items_by_id:
                    self.claimed_items.pop(cb.item_id, None)
                    cb.set_idle()

        self.claimed_items = {
            iid: bid for iid, bid in self.claimed_items.items()
            if iid in items_by_id
        }

        # What's still needed (minus bot inventories and in-transit items)
        global_remaining = dict(needed)
        for bot in data["bots"]:
            for item in bot["inventory"]:
                if item in global_remaining and global_remaining[item] > 0:
                    global_remaining[item] -= 1
        still_needed = {k: v for k, v in global_remaining.items() if v > 0}

        picking_types = {}
        for cb in self.bots.values():
            if cb.state == ST_PICKING and cb.item_id and cb.item_id in items_by_id:
                t = items_by_id[cb.item_id]["type"]
                picking_types[t] = picking_types.get(t, 0) + 1

        needs_picking = {}
        for t, c in still_needed.items():
            already = picking_types.get(t, 0)
            if c - already > 0:
                needs_picking[t] = c - already

        # Phase 1: Any bot with inventory → deliver to nearest drop-off
        for bot in data["bots"]:
            cb = self.bots[bot["id"]]
            if bot["inventory"] and cb.state == ST_IDLE and cb.stuck_count >= 0:
                bx, by = bot["position"]
                drop = nearest_drop(bx, by, drops)
                cb.assign_deliver(drop)

        # Phase 2: Assign idle bots to pick needed items
        n_active = sum(1 for cb in self.bots.values() if cb.state != ST_IDLE)
        for item_type, count in list(needs_picking.items()):
            if n_active >= self.max_active:
                break
            available = [it for it in items_by_type.get(item_type, [])
                         if it["id"] not in self.claimed_items]
            for item_obj in available[:count]:
                if n_active >= self.max_active:
                    break
                ix, iy = item_obj["position"]
                adj = find_adj_cells(ix, iy, walls_set, width, height)
                if not adj:
                    continue

                best_bid = None
                best_dist = 9999
                best_target = None
                for bot in data["bots"]:
                    bid = bot["id"]
                    cb = self.bots[bid]
                    if cb.state != ST_IDLE or cb.stuck_count < 0:
                        continue
                    if bot["inventory"]:
                        continue
                    # Skip if this bot blacklisted this item
                    if item_obj["id"] in cb.blacklist and cb.blacklist[item_obj["id"]] > rnd:
                        continue
                    bx, by = bot["position"]
                    for ax, ay in adj:
                        d = abs(bx - ax) + abs(by - ay)
                        if d < best_dist:
                            best_dist = d
                            best_bid = bid
                            best_target = (ax, ay)

                if best_bid is not None:
                    self.bots[best_bid].assign_pick(best_target, item_obj["id"])
                    self.claimed_items[item_obj["id"]] = best_bid
                    n_active += 1

        # Phase 3: Preview items
        if preview and not needs_picking and n_active < self.max_active:
            pneeded = get_order_needs(preview)
            for bot in data["bots"]:
                for item in bot["inventory"]:
                    if item in pneeded and pneeded[item] > 0:
                        pneeded[item] -= 1
            for item_type, count in list(pneeded.items()):
                if count <= 0 or n_active >= self.max_active:
                    continue
                available = [it for it in items_by_type.get(item_type, [])
                             if it["id"] not in self.claimed_items]
                for item_obj in available[:count]:
                    if n_active >= self.max_active:
                        break
                    ix, iy = item_obj["position"]
                    adj = find_adj_cells(ix, iy, walls_set, width, height)
                    if not adj:
                        continue
                    best_bid = None
                    best_dist = 9999
                    best_target = None
                    for bot in data["bots"]:
                        bid = bot["id"]
                        cb = self.bots[bid]
                        if cb.state != ST_IDLE or cb.stuck_count < 0:
                            continue
                        if bot["inventory"]:
                            continue
                        if item_obj["id"] in cb.blacklist and cb.blacklist[item_obj["id"]] > rnd:
                            continue
                        bx, by = bot["position"]
                        for ax, ay in adj:
                            d = abs(bx - ax) + abs(by - ay)
                            if d < best_dist:
                                best_dist = d
                                best_bid = bid
                                best_target = (ax, ay)
                    if best_bid is not None:
                        self.bots[best_bid].assign_pick(best_target, item_obj["id"])
                        self.claimed_items[item_obj["id"]] = best_bid
                        n_active += 1

        # Generate actions
        actions = []
        for bot in data["bots"]:
            bid = bot["id"]
            cb = self.bots[bid]
            bx, by = bot["position"]
            pos = (bx, by)

            if cb.state == ST_DELIVERING:
                if pos == cb.target:
                    if bot["inventory"]:
                        actions.append({"bot": bid, "action": "drop_off"})
                        cb.set_idle()
                    else:
                        cb.set_idle()
                        actions.append({"bot": bid, "action": "wait"})
                else:
                    # Stuck delivering → try different drop-off
                    if cb.stuck_count > 3 and len(drops) > 1:
                        other = [d for d in drops if d != cb.target]
                        if other:
                            cb.target = min(other, key=lambda d: abs(bx-d[0])+abs(by-d[1]))
                            cb.stuck_count = 0
                    act = bfs_first_step(walls_set, width, height, pos, cb.target)
                    actions.append({"bot": bid, "action": act})

            elif cb.state == ST_PICKING:
                item_obj = items_by_id.get(cb.item_id)
                if not item_obj:
                    cb.set_idle()
                    actions.append({"bot": bid, "action": "wait"})
                else:
                    ix, iy = item_obj["position"]
                    if abs(bx - ix) + abs(by - iy) == 1:
                        actions.append({"bot": bid, "action": "pick_up",
                                        "item_id": cb.item_id})
                        self.claimed_items.pop(cb.item_id, None)
                        inv_after = len(bot["inventory"]) + 1

                        # Chain-pick: if inventory not full, find another needed item
                        next_item = None
                        if inv_after < 3:
                            best_d = 9999
                            for t, c in needs_picking.items():
                                if c <= 0:
                                    continue
                                for it in items_by_type.get(t, []):
                                    if it["id"] in self.claimed_items:
                                        continue
                                    if it["id"] == cb.item_id:
                                        continue
                                    if it["id"] in cb.blacklist and cb.blacklist[it["id"]] > rnd:
                                        continue
                                    jx, jy = it["position"]
                                    jadj = find_adj_cells(jx, jy, walls_set, width, height)
                                    for ax, ay in jadj:
                                        d = abs(bx - ax) + abs(by - ay)
                                        if d < best_d:
                                            best_d = d
                                            next_item = (it, (ax, ay))

                        if next_item and best_d < 30:
                            nit, ntarget = next_item
                            cb.assign_pick(ntarget, nit["id"])
                            self.claimed_items[nit["id"]] = bid
                            # Reduce needs_picking
                            t = nit["type"]
                            if t in needs_picking:
                                needs_picking[t] -= 1
                        else:
                            cb.set_idle()
                    elif pos == cb.target:
                        cb.set_idle()
                        actions.append({"bot": bid, "action": "wait"})
                    else:
                        act = bfs_first_step(walls_set, width, height, pos, cb.target)
                        actions.append({"bot": bid, "action": act})

            else:  # ST_IDLE
                # Random walk mode
                if cb.stuck_count < 0:
                    moves = []
                    for dx, dy, act in [(0,-1,"move_up"),(0,1,"move_down"),
                                        (-1,0,"move_left"),(1,0,"move_right")]:
                        nx, ny = bx+dx, by+dy
                        if (0 <= nx < width and 0 <= ny < height and
                            (nx,ny) not in walls_set):
                            moves.append(act)
                    if moves:
                        actions.append({"bot": bid, "action": random.choice(moves)})
                    else:
                        actions.append({"bot": bid, "action": "wait"})
                    cb.stuck_count += 1
                    continue

                # Idle with inventory → deliver
                if bot["inventory"]:
                    drop = nearest_drop(bx, by, drops)
                    cb.assign_deliver(drop)
                    act = bfs_first_step(walls_set, width, height, pos, drop)
                    actions.append({"bot": bid, "action": act})
                else:
                    # Park at spawn (collision-exempt, won't block corridors)
                    park = self.spawn or (width - 2, height - 2)
                    if pos == park:
                        actions.append({"bot": bid, "action": "wait"})
                    else:
                        act = bfs_first_step(walls_set, width, height, pos, park)
                        actions.append({"bot": bid, "action": act})

        return actions


def get_capture_path(difficulty, date_str=None):
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    os.makedirs('captures', exist_ok=True)
    return f'captures/{difficulty}_{date_str}.json'


def load_capture(difficulty, date_str=None):
    path = get_capture_path(difficulty, date_str)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


async def capture_and_play(ws_url, difficulty, save_path=None):
    """Play a probe game and capture server state for offline optimization."""
    import websockets

    if save_path is None:
        save_path = get_capture_path(difficulty)

    captured = {
        'difficulty': difficulty,
        'captured_at': datetime.now(timezone.utc).isoformat(),
        'grid': None,
        'items': None,
        'drop_off': None,
        'num_bots': 0,
        'orders': [],
    }

    seen_order_ids = set()
    walls_set = None
    width = height = 0
    controller = CaptureController()

    print(f"Probe game: {difficulty}")
    print(f"Connecting to {ws_url[:60]}...")

    async with websockets.connect(ws_url) as ws:
        async for message in ws:
            data = json.loads(message)

            if data["type"] == "game_over":
                captured['probe_score'] = data['score']
                captured['rounds_used'] = data['rounds_used']
                captured['orders_completed'] = data['orders_completed']
                captured['items_delivered'] = data['items_delivered']
                print(f"\nProbe complete! Score={data['score']}, "
                      f"Orders={data['orders_completed']}, "
                      f"Items={data['items_delivered']}")
                break

            if data["type"] != "game_state":
                continue

            rnd = data["round"]

            if rnd == 0:
                width = data["grid"]["width"]
                height = data["grid"]["height"]
                captured['grid'] = data['grid']
                captured['items'] = data['items']
                captured['drop_off'] = data['drop_off']
                if 'drop_off_zones' in data:
                    captured['drop_off_zones'] = data['drop_off_zones']
                captured['num_bots'] = len(data['bots'])

                walls_set = set()
                for w_pos in data["grid"]["walls"]:
                    walls_set.add((w_pos[0], w_pos[1]))
                for item in data["items"]:
                    ix, iy = item["position"]
                    walls_set.add((ix, iy))

                print(f"Map: {width}x{height}, {len(data['bots'])} bots, "
                      f"{len(data['items'])} items")

            # Capture new orders
            for order in data["orders"]:
                oid = order["id"]
                if oid not in seen_order_ids:
                    seen_order_ids.add(oid)
                    captured['orders'].append({
                        'id': oid,
                        'items_required': list(order['items_required']),
                    })

            actions = controller.decide(data, walls_set, width, height)
            await ws.send(json.dumps({"actions": actions}))

            if rnd < 5 or rnd % 50 == 0 or rnd >= 495:
                n_pick = sum(1 for cb in controller.bots.values() if cb.state == ST_PICKING)
                n_del = sum(1 for cb in controller.bots.values() if cb.state == ST_DELIVERING)
                print(f"  R{rnd}: score={data['score']} orders_seen={len(captured['orders'])} "
                      f"pick={n_pick} del={n_del}")

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(captured, f, indent=2)

    print(f"\nCapture saved: {save_path}")
    print(f"  Items: {len(captured['items'])}")
    print(f"  Orders captured: {len(captured['orders'])}")

    return captured


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python capture_game.py <ws_url> <difficulty>")
        print("       python capture_game.py --check <difficulty> [date]")
        sys.exit(1)

    if sys.argv[1] == '--check':
        difficulty = sys.argv[2]
        date = sys.argv[3] if len(sys.argv) > 3 else None
        cap = load_capture(difficulty, date)
        if cap:
            print(f"Capture exists for {difficulty} ({cap.get('captured_at', '?')})")
            print(f"  Items: {len(cap['items'])}")
            print(f"  Orders: {len(cap['orders'])}")
            print(f"  Probe score: {cap.get('probe_score', 'N/A')}")
        else:
            path = get_capture_path(difficulty, date)
            print(f"No capture at {path}")
    else:
        ws_url = sys.argv[1]
        difficulty = sys.argv[2]
        asyncio.run(capture_and_play(ws_url, difficulty))
