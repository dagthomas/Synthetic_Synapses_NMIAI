"""WebSocket client for Grocery Bot — probe and replay modes.

Usage:
    python client.py probe <ws_url> [--save orders.json]
    python client.py replay <ws_url> --plan plan.txt
"""
import asyncio
import json
import sys
import time
import argparse
from pathlib import Path

try:
    import websockets
except ImportError:
    print("pip install websockets", file=sys.stderr)
    sys.exit(1)


# ---- Game state helpers ----

def parse_game_state(data):
    """Parse a game_state message into a clean dict."""
    return {
        'round': data['round'],
        'max_rounds': data['max_rounds'],
        'width': data['grid']['width'],
        'height': data['grid']['height'],
        'walls': data['grid']['walls'],
        'bots': data['bots'],
        'items': data['items'],
        'orders': data['orders'],
        'drop_off': data.get('drop_off', data.get('drop_off_zones', [[1, 1]])[0]),
        'drop_off_zones': data.get('drop_off_zones', [data.get('drop_off', [1, 1])]),
        'score': data.get('score', 0),
        'active_order_index': data.get('active_order_index', 0),
        'total_orders': data.get('total_orders', 0),
    }


def bfs_path(grid_w, grid_h, walls_set, shelves_set, start, goal, occupied=None):
    """Simple BFS returning list of actions (strings) from start to goal."""
    if start == goal:
        return []
    if occupied is None:
        occupied = set()

    dirs = [('move_up', 0, -1), ('move_down', 0, 1),
            ('move_left', -1, 0), ('move_right', 1, 0)]

    visited = {start}
    queue = [(start, [])]
    qi = 0
    while qi < len(queue):
        (cx, cy), path = queue[qi]
        qi += 1
        for name, dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= grid_w or ny < 0 or ny >= grid_h:
                continue
            if (nx, ny) in walls_set or (nx, ny) in shelves_set:
                continue
            if (nx, ny) in occupied and (nx, ny) != goal:
                continue
            if (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            new_path = path + [name]
            if (nx, ny) == goal:
                return new_path
            queue.append(((nx, ny), new_path))
    return None  # no path


# ---- Greedy bot (for probing) ----

class GreedyBot:
    """Simple greedy bot that completes orders one at a time."""

    def __init__(self):
        self.discovered_orders = []
        self.seen_order_ids = set()
        self.walls_set = None
        self.shelves_set = None
        self.item_map = {}  # (x,y) -> [item_info, ...]
        self.item_by_id = {}
        self.grid_w = 0
        self.grid_h = 0

    def init_map(self, state):
        self.grid_w = state['width']
        self.grid_h = state['height']
        self.walls_set = set()
        self.shelves_set = set()
        for w in state['walls']:
            self.walls_set.add(tuple(w))
        for item in state['items']:
            pos = tuple(item['position'])
            self.shelves_set.add(pos)
            self.item_by_id[item['id']] = item
            if pos not in self.item_map:
                self.item_map[pos] = []
            self.item_map[pos].append(item)

    def record_orders(self, orders):
        for o in orders:
            oid = o.get('id', f"order_{len(self.discovered_orders)}")
            if oid not in self.seen_order_ids:
                self.seen_order_ids.add(oid)
                self.discovered_orders.append({
                    'id': oid,
                    'items_required': o['items_required'],
                    'status': o.get('status', 'unknown'),
                })

    def get_actions(self, state):
        """Compute greedy actions for all bots."""
        bots = state['bots']
        orders = state['orders']
        self.record_orders(orders)

        active = None
        for o in orders:
            if o.get('status') == 'active':
                active = o
                break
        if active is None and orders:
            active = orders[0]

        # What items does the active order still need?
        needed = list(active['items_required']) if active else []
        delivered = active.get('items_delivered', []) if active else []
        for d in delivered:
            if d in needed:
                needed.remove(d)

        # Remove items already in bot inventories (being carried)
        carrying = {}
        for bot in bots:
            for item_type in bot.get('inventory', []):
                carrying[item_type] = carrying.get(item_type, 0) + 1

        for bot in bots:
            for item_type in bot.get('inventory', []):
                if item_type in needed:
                    needed.remove(item_type)

        # Build occupied set (bot positions)
        occupied = set()
        for bot in bots:
            occupied.add(tuple(bot['position']))

        actions = []
        assigned_types = set()  # avoid multiple bots going for same type

        for bot in bots:
            bx, by = bot['position']
            inv = bot.get('inventory', [])
            drop_zones = [tuple(dz) for dz in state['drop_off_zones']]

            # If carrying items matching active order, go deliver
            has_active_items = active and any(
                item_type in [t for t in active['items_required']]
                for item_type in inv
            )

            if has_active_items and len(inv) > 0:
                # Check if at dropoff
                if (bx, by) in drop_zones:
                    actions.append({'bot': bot['id'], 'action': 'drop_off'})
                    continue
                # Navigate to nearest dropoff
                best_dz = min(drop_zones, key=lambda dz: abs(dz[0]-bx)+abs(dz[1]-by))
                path = bfs_path(self.grid_w, self.grid_h, self.walls_set,
                                self.shelves_set, (bx, by), best_dz, occupied - {(bx, by)})
                if path:
                    actions.append({'bot': bot['id'], 'action': path[0]})
                    continue

            # If inventory not full, go pick up a needed item
            if len(inv) < 3 and needed:
                # Find nearest item of a needed type
                best_item = None
                best_dist = 999999

                for item in state['items']:
                    itype = item['type']
                    if itype not in needed:
                        continue
                    if itype in assigned_types:
                        continue
                    ix, iy = item['position']
                    # Find adjacent walkable cell
                    for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
                        ax, ay = ix+dx, iy+dy
                        if (ax, ay) in self.walls_set or (ax, ay) in self.shelves_set:
                            continue
                        d = abs(ax-bx) + abs(ay-by)
                        if d < best_dist:
                            best_dist = d
                            best_item = (item, (ax, ay))

                if best_item:
                    item, adj = best_item
                    ix, iy = item['position']
                    # Check if already adjacent
                    if abs(bx - ix) + abs(by - iy) == 1:
                        actions.append({'bot': bot['id'], 'action': 'pick_up',
                                        'item_id': item['id']})
                        needed.remove(item['type'])
                        assigned_types.add(item['type'])
                        continue
                    # Navigate to adjacent cell
                    path = bfs_path(self.grid_w, self.grid_h, self.walls_set,
                                    self.shelves_set, (bx, by), adj, occupied - {(bx, by)})
                    if path:
                        assigned_types.add(item['type'])
                        actions.append({'bot': bot['id'], 'action': path[0]})
                        continue

            # Default: wait
            actions.append({'bot': bot['id'], 'action': 'wait'})

        return actions


# ---- Replay bot (executes pre-computed plan) ----

class ReplayBot:
    """Replays a pre-computed plan from the C++ solver."""

    def __init__(self, plan_file):
        self.plan = self._load_plan(plan_file)
        self.item_list = []  # set on first game state

    def _load_plan(self, plan_file):
        """Load plan from text file (solver output format)."""
        lines = Path(plan_file).read_text().strip().split('\n')
        header = lines[0].split()
        num_rounds = int(header[0])
        num_bots = int(header[1])

        plan = []
        for i in range(1, num_rounds + 1):
            if i >= len(lines):
                break
            vals = list(map(int, lines[i].split()))
            round_actions = []
            for b in range(num_bots):
                act = vals[b * 2]
                arg = vals[b * 2 + 1]
                round_actions.append((act, arg))
            plan.append(round_actions)
        return plan

    def init_items(self, state):
        # Sort items same way as solver (by position)
        items = sorted(state['items'], key=lambda it: (it['position'][0], it['position'][1]))
        self.item_list = items

    def get_actions(self, state, round_num):
        """Get WebSocket actions for this round."""
        if round_num >= len(self.plan):
            return [{'bot': b['id'], 'action': 'wait'} for b in state['bots']]

        action_names = ['wait', 'move_up', 'move_down', 'move_left', 'move_right', 'pick_up', 'drop_off']
        actions = []
        for bid, (act, arg) in enumerate(self.plan[round_num]):
            a = {'bot': bid, 'action': action_names[act]}
            if act == 5 and 0 <= arg < len(self.item_list):
                a['item_id'] = self.item_list[arg]['id']
            actions.append(a)
        return actions


# ---- WebSocket game loop ----

async def play_game(ws_url, bot, mode='probe', timeout=130):
    """Connect to game and play until game_over."""
    print(f"Connecting to {ws_url[:60]}...", file=sys.stderr)

    async with websockets.connect(ws_url, close_timeout=5,
                                   max_size=10 * 1024 * 1024) as ws:
        round_num = 0
        final_score = 0
        map_initialized = False
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                print(f"Timeout after {timeout}s", file=sys.stderr)
                break

            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=10)
            except asyncio.TimeoutError:
                print("No message for 10s, disconnecting", file=sys.stderr)
                break

            data = json.loads(msg)

            if data['type'] == 'game_over':
                final_score = data.get('score', 0)
                print(f"Game over: score={final_score}, "
                      f"items={data.get('items_delivered', 0)}, "
                      f"orders={data.get('orders_completed', 0)}, "
                      f"rounds={data.get('rounds_used', 0)}", file=sys.stderr)
                break

            if data['type'] != 'game_state':
                continue

            state = parse_game_state(data)
            round_num = state['round']

            if not map_initialized:
                if mode == 'probe':
                    bot.init_map(state)
                else:
                    bot.init_items(state)
                map_initialized = True

            # Get actions
            if mode == 'probe':
                ws_actions = bot.get_actions(state)
            else:
                ws_actions = bot.get_actions(state, round_num)

            await ws.send(json.dumps({'actions': ws_actions}))

            if round_num % 50 == 0:
                print(f"  Round {round_num}: score={state['score']}", file=sys.stderr)

    return final_score


async def probe_game(ws_url, save_file=None):
    """Play a game with greedy bot to discover orders."""
    bot = GreedyBot()
    score = await play_game(ws_url, bot, mode='probe')

    print(f"\nDiscovered {len(bot.discovered_orders)} orders, score={score}", file=sys.stderr)

    if save_file:
        Path(save_file).write_text(json.dumps({
            'orders': bot.discovered_orders,
            'score': score,
            'num_orders': len(bot.discovered_orders),
        }, indent=2))
        print(f"Saved to {save_file}", file=sys.stderr)

    return bot.discovered_orders, score


async def replay_game(ws_url, plan_file):
    """Replay a pre-computed plan."""
    bot = ReplayBot(plan_file)
    score = await play_game(ws_url, bot, mode='replay')
    return score


# ---- Capture: extract full game data for solver ----

async def capture_game(ws_url, save_file='capture.json'):
    """Play greedy and capture full game data for offline solver."""
    bot = GreedyBot()

    print(f"Connecting to {ws_url[:60]}...", file=sys.stderr)
    capture_data = None

    async with websockets.connect(ws_url, close_timeout=5,
                                   max_size=10 * 1024 * 1024) as ws:
        round_num = 0
        map_initialized = False
        start_time = time.time()

        while True:
            if time.time() - start_time > 130:
                break

            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=10)
            except asyncio.TimeoutError:
                break

            data = json.loads(msg)

            if data['type'] == 'game_over':
                print(f"Game over: score={data.get('score', 0)}, "
                      f"orders={data.get('orders_completed', 0)}", file=sys.stderr)
                break

            if data['type'] != 'game_state':
                continue

            state = parse_game_state(data)
            round_num = state['round']

            if not map_initialized:
                bot.init_map(state)
                # Capture map data
                capture_data = {
                    'width': state['width'],
                    'height': state['height'],
                    'walls': state['walls'],
                    'items': state['items'],
                    'drop_off': state['drop_off'],
                    'drop_off_zones': state['drop_off_zones'],
                    'max_rounds': state['max_rounds'],
                    'num_bots': len(state['bots']),
                }
                map_initialized = True

            # Record orders
            bot.record_orders(state['orders'])

            # Play greedy
            ws_actions = bot.get_actions(state)
            await ws.send(json.dumps({'actions': ws_actions}))

            if round_num % 50 == 0:
                print(f"  Round {round_num}: score={state['score']}, "
                      f"orders_known={len(bot.discovered_orders)}", file=sys.stderr)

    if capture_data:
        capture_data['orders'] = [
            {'items_required': o['items_required']}
            for o in bot.discovered_orders
        ]
        Path(save_file).write_text(json.dumps(capture_data, indent=2))
        print(f"Captured {len(bot.discovered_orders)} orders → {save_file}", file=sys.stderr)

    return capture_data


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser(description='Grocery Bot Client')
    sub = parser.add_subparsers(dest='mode')

    p_probe = sub.add_parser('probe', help='Play greedy, discover orders')
    p_probe.add_argument('ws_url')
    p_probe.add_argument('--save', default='orders.json')

    p_capture = sub.add_parser('capture', help='Capture full game data')
    p_capture.add_argument('ws_url')
    p_capture.add_argument('--save', default='capture.json')

    p_replay = sub.add_parser('replay', help='Replay pre-computed plan')
    p_replay.add_argument('ws_url')
    p_replay.add_argument('--plan', required=True)

    args = parser.parse_args()

    if args.mode == 'probe':
        asyncio.run(probe_game(args.ws_url, args.save))
    elif args.mode == 'capture':
        asyncio.run(capture_game(args.ws_url, args.save))
    elif args.mode == 'replay':
        asyncio.run(replay_game(args.ws_url, args.plan))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
