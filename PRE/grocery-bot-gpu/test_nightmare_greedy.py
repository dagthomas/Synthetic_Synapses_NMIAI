"""Test the nightmare greedy logic in a full simulation."""
import json, sys, random
import numpy as np
from collections import deque

sys.path.insert(0, '.')
from game_engine import (
    build_map_from_capture, init_game_from_capture, step,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY, CELL_FLOOR, CELL_DROPOFF
)
from live_solver import ws_to_capture


def bfs_avoid(start, goal, walkable, blocked, exempt):
    if start == goal:
        return ACT_WAIT
    def ok(pos):
        if pos not in walkable:
            return False
        if pos in blocked and pos != goal and pos not in exempt:
            return False
        return True
    visited = {start: None}
    queue = deque()
    for act in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
        nx, ny = start[0] + DX[act], start[1] + DY[act]
        if ok((nx, ny)) and (nx, ny) not in visited:
            visited[(nx, ny)] = act
            if (nx, ny) == goal:
                return act
            queue.append((nx, ny))
    while queue:
        cx, cy = queue.popleft()
        first_act = visited[(cx, cy)]
        for act in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
            nx, ny = cx + DX[act], cy + DY[act]
            if ok((nx, ny)) and (nx, ny) not in visited:
                visited[(nx, ny)] = first_act
                if (nx, ny) == goal:
                    return first_act
                queue.append((nx, ny))
    # Fallback: ignore blocks, go manhattan
    best_act, best_d = ACT_WAIT, 9999
    for act in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
        nx, ny = start[0] + DX[act], start[1] + DY[act]
        if (nx, ny) in walkable:
            d = abs(nx - goal[0]) + abs(ny - goal[1])
            if d < best_d:
                best_d = d
                best_act = act
    return best_act


def nightmare_greedy(gs, ms, walkable, drop_zones, items_by_type, stall_count, pos_history, exempt_unused):
    """Match the live_gpu_stream._nightmare_greedy logic exactly."""
    n_bots = gs.bot_positions.shape[0]

    # Parse orders
    order_need, preview_need = {}, {}
    for o in gs.orders:
        d = order_need if o.status == 'active' else preview_need if o.status == 'preview' else None
        if d is None:
            continue
        for i in range(len(o.required)):
            t = ms.item_type_names[int(o.required[i])]
            if o.delivered[i] == 0:
                d[t] = d.get(t, 0) + 1

    spawn = (ms.width - 2, ms.height - 2)

    # --- Zone definitions ---
    dz_list = sorted(drop_zones, key=lambda dz: dz[0])
    if len(dz_list) >= 3:
        zone_defs = [
            {'dropoff': dz_list[0], 'x_max': 10},
            {'dropoff': dz_list[1], 'x_max': 19},
            {'dropoff': dz_list[2], 'x_max': 30},
        ]
    else:
        zone_defs = [{'dropoff': dz_list[0], 'x_max': 30}]
    n_zones = len(zone_defs)
    bot_zone = {}
    if n_zones == 3:
        for bid_i in range(n_bots):
            if bid_i < 7: bot_zone[bid_i] = 0
            elif bid_i < 14: bot_zone[bid_i] = 1
            else: bot_zone[bid_i] = 2
    else:
        for bid_i in range(n_bots):
            bot_zone[bid_i] = 0
    zone_items_by_type = [{} for _ in range(n_zones)]
    for item_idx_z, item_z in enumerate(ms.items):
        tz = item_z.get('type', '')
        ix = item_z['position'][0]
        zi = 0
        for zi_c in range(n_zones):
            if ix < zone_defs[zi_c]['x_max']:
                zi = zi_c
                break
        if tz not in zone_items_by_type[zi]:
            zone_items_by_type[zi][tz] = []
        for adj_z in ms.item_adjacencies.get(item_idx_z, []):
            zone_items_by_type[zi][tz].append((item_idx_z, adj_z))

    # Per-zone type carrying and routing
    zone_type_carrying = [{} for _ in range(n_zones)]
    for bid in range(n_bots):
        bzi = bot_zone.get(bid, 0)
        for slot in range(INV_CAP):
            tid = int(gs.bot_inventories[bid, slot])
            if tid >= 0:
                t = ms.item_type_names[tid]
                zone_type_carrying[bzi][t] = zone_type_carrying[bzi].get(t, 0) + 1
    zone_type_routed = [{} for _ in range(n_zones)]
    zone_bot_count = [0] * n_zones
    for bid in range(n_bots):
        zone_bot_count[bot_zone.get(bid, 0)] += 1

    def _type_cap(needed, is_active, zone_bots):
        return needed + max(1, zone_bots // 4)

    # --- Collision tracking ---
    bot_pos = {}
    for bid in range(n_bots):
        bot_pos[bid] = (int(gs.bot_positions[bid, 0]), int(gs.bot_positions[bid, 1]))

    remaining = {}  # pos -> count of unprocessed bots
    for pos in bot_pos.values():
        remaining[pos] = remaining.get(pos, 0) + 1
    decided = set()

    # Exempt: spawn + all dropoff zones (BFS can path through them like spawn)
    exempt = {spawn} | drop_zones

    def _blocked():
        return set(remaining.keys()) | decided

    actions_by_bid = {}
    picks, drops, moves, waits = 0, 0, 0, 0

    # --- Phase 0: Pre-evacuate useless bots from dropoffs ---
    pre_evacuated = {}
    for bid in range(n_bots):
        bpos = bot_pos[bid]
        if bpos not in drop_zones:
            continue
        inv = [ms.item_type_names[int(gs.bot_inventories[bid, s])]
               for s in range(INV_CAP) if gs.bot_inventories[bid, s] >= 0]
        has_useful = any(t in order_need for t in inv)
        has_preview_evac = any(t in preview_need for t in inv)
        if has_useful or has_preview_evac:
            continue  # will deliver useful/preview items, don't evacuate
        bx, by = bpos
        best_act = ACT_WAIT
        for try_act in [ACT_MOVE_LEFT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_RIGHT]:
            nx, ny = bx + DX[try_act], by + DY[try_act]
            if (nx, ny) in walkable and (nx, ny) not in _blocked():
                best_act = try_act
                break
        if best_act != ACT_WAIT:
            pre_evacuated[bid] = best_act
            nx, ny = bx + DX[best_act], by + DY[best_act]
            remaining[bpos] -= 1
            if remaining[bpos] <= 0:
                del remaining[bpos]
            decided.add((nx, ny))

    # --- Phase 1: Main loop (ID order) ---
    for bid in range(n_bots):
        bpos = bot_pos[bid]
        bx, by = bpos
        inv = [ms.item_type_names[int(gs.bot_inventories[bid, s])]
               for s in range(INV_CAP) if gs.bot_inventories[bid, s] >= 0]
        inv_full = len(inv) >= INV_CAP
        has_useful = any(t in order_need for t in inv)
        has_preview = any(t in preview_need for t in inv)

        # Zone assignment
        zi = bot_zone.get(bid, 0)
        my_items = zone_items_by_type[zi]
        my_dropoff = zone_defs[zi]['dropoff']

        # Per-bot inventory type counts (prevent hoarding same type)
        bot_type_count = {}
        for t_inv in inv:
            bot_type_count[t_inv] = bot_type_count.get(t_inv, 0) + 1

        if bid in pre_evacuated:
            act = pre_evacuated[bid]
            item_idx = -1
            # stall tracking
            hist = pos_history.get(bid, [])
            hist.append(bpos)
            if len(hist) > 6: hist = hist[-6:]
            pos_history[bid] = hist
            stall_count[bid] = 0
            actions_by_bid[bid] = (act, item_idx)
            moves += 1
            continue

        # Remove from remaining
        if bpos in remaining:
            remaining[bpos] -= 1
            if remaining[bpos] <= 0:
                del remaining[bpos]

        blocked = _blocked()
        act, item_idx = ACT_WAIT, -1

        # 1. At dropoff with useful -> deliver
        if bpos in drop_zones and has_useful:
            act = ACT_DROPOFF
        # 2. At dropoff without useful -> evacuate
        elif bpos in drop_zones:
            for try_act in [ACT_MOVE_LEFT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_RIGHT]:
                nx, ny = bx + DX[try_act], by + DY[try_act]
                if (nx, ny) in walkable and (nx, ny) not in blocked:
                    act = try_act
                    break
        # 3. Adjacent to needed item -> pickup (diversify: max 1 per type unless order needs more)
        elif not inv_full:
            for idx_nd, need_dict in enumerate([order_need, preview_need]):
                if act == ACT_PICKUP:
                    break
                is_active = (idx_nd == 0)
                if not is_active and not has_useful:
                    continue
                for t in need_dict:
                    if need_dict[t] <= 0:
                        continue
                    if bot_type_count.get(t, 0) >= need_dict[t]:
                        continue
                    ztc = zone_type_carrying[zi]
                    ztr = zone_type_routed[zi]
                    total = ztc.get(t, 0) + ztr.get(t, 0)
                    if total >= _type_cap(need_dict[t], is_active, zone_bot_count[zi]):
                        continue
                    for idx_c, adj in my_items.get(t, []):
                        if adj == bpos:
                            act = ACT_PICKUP
                            item_idx = idx_c
                            ztc[t] = ztc.get(t, 0) + 1
                            bot_type_count[t] = bot_type_count.get(t, 0) + 1
                            break
                    if act == ACT_PICKUP:
                        break

        # 4. Has useful -> go to zone's dropoff
        if act == ACT_WAIT and has_useful:
            a = bfs_avoid(bpos, my_dropoff, walkable, blocked, exempt)
            if a != ACT_WAIT:
                act = a

        # 5. Not full -> route to needed item (respecting per-bot diversity)
        if act == ACT_WAIT and not inv_full:
            best_dist, best_adj, best_idx, best_type = 9999, None, -1, None
            for idx_nd, need_dict in enumerate([order_need, preview_need]):
                if best_adj is not None:
                    break
                is_active = (idx_nd == 0)
                if not is_active and not has_useful:
                    continue
                for t in need_dict:
                    if need_dict[t] <= 0:
                        continue
                    if bot_type_count.get(t, 0) >= need_dict[t]:
                        continue
                    ztc5 = zone_type_carrying[zi]
                    ztr5 = zone_type_routed[zi]
                    total = ztc5.get(t, 0) + ztr5.get(t, 0)
                    if total >= _type_cap(need_dict[t], is_active, zone_bot_count[zi]):
                        continue
                    for idx_c, adj in my_items.get(t, []):
                        d = abs(bpos[0]-adj[0]) + abs(bpos[1]-adj[1])
                        if d < best_dist:
                            best_dist = d
                            best_adj = adj
                            best_idx = idx_c
                            best_type = t
            if best_adj is not None:
                if bpos == best_adj:
                    act = ACT_PICKUP
                    item_idx = best_idx
                    zone_type_carrying[zi][best_type] = zone_type_carrying[zi].get(best_type, 0) + 1
                else:
                    act = bfs_avoid(bpos, best_adj, walkable, blocked, exempt)
                    zone_type_routed[zi][best_type] = zone_type_routed[zi].get(best_type, 0) + 1

        # 6. Has useful or preview -> deliver at zone's dropoff
        if act == ACT_WAIT and inv and (has_useful or has_preview):
            a = bfs_avoid(bpos, my_dropoff, walkable, blocked, exempt)
            if a != ACT_WAIT:
                act = a

        # 6b. Dead-inv near dropoff -> flee to clear corridor
        if act == ACT_WAIT and inv_full and not has_useful and not has_preview:
            near_any_dz = any(abs(bpos[0]-dz[0])+abs(bpos[1]-dz[1]) <= 3 for dz in drop_zones)
            if near_any_dz:
                ndz = min(drop_zones, key=lambda dz: abs(bpos[0]-dz[0])+abs(bpos[1]-dz[1]))
                best_flee, best_d = ACT_WAIT, abs(bpos[0]-ndz[0])+abs(bpos[1]-ndz[1])
                for try_act in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]:
                    nx, ny = bpos[0] + DX[try_act], bpos[1] + DY[try_act]
                    if (nx, ny) in walkable and (nx, ny) not in blocked:
                        d = abs(nx-ndz[0])+abs(ny-ndz[1])
                        if d > best_d:
                            best_d = d
                            best_flee = try_act
                if best_flee != ACT_WAIT:
                    act = best_flee

        # 7. Idle -> approach nearest active item
        if act == ACT_WAIT and not inv_full:
            best_dist, best_adj = 9999, None
            for t in order_need:
                if order_need[t] <= 0:
                    continue
                for idx_c, adj in my_items.get(t, []):
                    d = abs(bpos[0]-adj[0]) + abs(bpos[1]-adj[1])
                    if d < best_dist:
                        best_dist = d
                        best_adj = adj
            if best_adj is not None:
                act = bfs_avoid(bpos, best_adj, walkable, blocked, exempt)

        # 8. Anti-stall
        sc = stall_count.get(bid, 0)
        if act == ACT_WAIT and sc >= 4:
            for try_act in random.sample([ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT], 4):
                nx, ny = bpos[0] + DX[try_act], bpos[1] + DY[try_act]
                if (nx, ny) in walkable and (nx, ny) not in blocked:
                    act = try_act
                    break
            if act == ACT_WAIT:
                for try_act in random.sample([ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT], 4):
                    nx, ny = bpos[0] + DX[try_act], bpos[1] + DY[try_act]
                    if (nx, ny) in walkable:
                        act = try_act
                        break

        # Update stall
        hist = pos_history.get(bid, [])
        hist.append(bpos)
        if len(hist) > 6:
            hist = hist[-6:]
        pos_history[bid] = hist
        if len(hist) >= 2 and hist[-1] == hist[-2]:
            stall_count[bid] = stall_count.get(bid, 0) + 1
        else:
            stall_count[bid] = 0

        # Update collision tracking
        if act in (ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT):
            nx, ny = bx + DX[act], by + DY[act]
            if (nx, ny) in walkable:
                decided.add((nx, ny))
            else:
                decided.add(bpos)
                act = ACT_WAIT
        else:
            decided.add(bpos)

        actions_by_bid[bid] = (act, item_idx)
        if act == ACT_PICKUP: picks += 1
        elif act == ACT_DROPOFF: drops += 1
        elif ACT_MOVE_UP <= act <= ACT_MOVE_RIGHT: moves += 1
        else: waits += 1

    actions_list = [actions_by_bid[bid] for bid in range(n_bots)]
    return actions_list, picks, drops, moves, waits


def main():
    # Use nightmare log
    import os, glob
    # Find a nightmare game log (20 bots)
    candidates = sorted(glob.glob('../grocery-bot-zig/game_log_*.jsonl'), reverse=True)
    f = None
    for c in candidates:
        with open(c) as fh:
            d = json.loads(fh.readline())
            if len(d.get('bots', [])) == 20:
                f = c
                break
    if f is None:
        print("No nightmare game log found")
        return
    with open(f) as fh:
        first = json.loads(fh.readline())

    capture = ws_to_capture(first)
    gs, all_orders = init_game_from_capture(capture, num_orders=50)
    ms = gs.map_state

    walkable = set()
    for y in range(ms.height):
        for x in range(ms.width):
            if ms.grid[y, x] in (CELL_FLOOR, CELL_DROPOFF):
                walkable.add((x, y))

    drop_zones = {tuple(dz) for dz in ms.drop_off_zones}

    items_by_type = {}
    for item_idx, item in enumerate(ms.items):
        t = item.get('type', '')
        if t not in items_by_type:
            items_by_type[t] = []
        for adj in ms.item_adjacencies.get(item_idx, []):
            items_by_type[t].append((item_idx, adj))

    spawn = (ms.width - 2, ms.height - 2)
    exempt = drop_zones | {spawn}

    stall_count = {}
    pos_history = {}

    n_bots = gs.bot_positions.shape[0]
    max_rounds = 500 if n_bots >= 20 else 300
    print(f"Simulating {n_bots} bots on {ms.width}x{ms.height} grid, {len(walkable)} walkable cells")
    print(f"Drop-off zones: {drop_zones}")
    print(f"Spawn: {ms.spawn}, Rounds: {max_rounds}")

    # Zone info
    dz_list = sorted(drop_zones, key=lambda dz: dz[0])
    if len(dz_list) >= 3:
        zone_names = ['LEFT', 'MIDDLE', 'RIGHT']
        zone_bot_ranges = ['0-6', '7-13', '14-19']
        for i, dz in enumerate(dz_list):
            print(f"  Zone {zone_names[i]}: dropoff={dz}, bots={zone_bot_ranges[i]}")
    print()

    for rnd in range(max_rounds):
        actions_list, picks, drops, moves, waits = nightmare_greedy(
            gs, ms, walkable, drop_zones, items_by_type, stall_count, pos_history, exempt)

        if rnd < 20 or (picks > 0 and rnd < 80) or (drops > 0 and rnd < 80):
            print(f"R{rnd:3d}: score={gs.score:3d} moves={moves} picks={picks} drops={drops} waits={waits}")
            for bid in range(n_bots):
                bpos = (int(gs.bot_positions[bid, 0]), int(gs.bot_positions[bid, 1]))
                inv = [ms.item_type_names[int(gs.bot_inventories[bid, s])]
                       for s in range(INV_CAP) if gs.bot_inventories[bid, s] >= 0]
                act_type, act_item = actions_list[bid]
                act_name = ['wait','up','down','left','right','pickup','dropoff'][act_type]
                if inv or act_type in (ACT_PICKUP, ACT_DROPOFF):
                    item_name = ms.items[act_item]['type'] if act_item >= 0 else ''
                    print(f"  B{bid}: {bpos} inv={inv} -> {act_name} {item_name}")
        elif rnd % 50 == 0:
            print(f"R{rnd:3d}: score={gs.score:3d} moves={moves} picks={picks} drops={drops} waits={waits}")
            for o in gs.orders:
                if o.status in ('active', 'preview'):
                    need = [ms.item_type_names[int(o.required[i])]
                            for i in range(len(o.required)) if o.delivered[i] == 0]
                    print(f"  {o.status.upper()} order needs: {need}")
            if rnd <= 200:
                zone_names = ['L', 'M', 'R']
                for bid in range(n_bots):
                    bpos = (int(gs.bot_positions[bid, 0]), int(gs.bot_positions[bid, 1]))
                    inv = [ms.item_type_names[int(gs.bot_inventories[bid, s])]
                           for s in range(INV_CAP) if gs.bot_inventories[bid, s] >= 0]
                    act_type, _ = actions_list[bid]
                    act_name = ['wait','up','down','left','right','pickup','dropoff'][act_type]
                    zi = 0 if bid < 7 else (1 if bid < 14 else 2)
                    print(f"  B{bid}({zone_names[zi]}): {bpos} inv={inv} -> {act_name}")

        step(gs, actions_list, all_orders)

    print(f"\nFINAL SCORE: {gs.score} (orders={gs.orders_completed}, items={gs.items_delivered})")


if __name__ == '__main__':
    main()
