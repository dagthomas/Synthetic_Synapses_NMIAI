"""Nightmare solver: zone-based assignment, space-time collision avoidance.

Usage:
    python nightmare_solver.py [--seeds 1000-1009] [--verbose] [--active-bots 10]
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from game_engine import (
    init_game, step, GameState, Order, MapState,
    ACT_WAIT, ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT,
    ACT_PICKUP, ACT_DROPOFF, INV_CAP, DX, DY,
    CELL_FLOOR, CELL_DROPOFF,
)
from configs import DIFF_ROUNDS, CONFIGS


def build_walkable(ms):
    w = set()
    for y in range(ms.height):
        for x in range(ms.width):
            if ms.grid[y, x] in (CELL_FLOOR, CELL_DROPOFF):
                w.add((x, y))
    return w


def bfs_dist(start, walkable):
    dist = {start: 0}
    q = deque([start])
    while q:
        x, y = q.popleft()
        d = dist[(x, y)]
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in walkable and (nx, ny) not in dist:
                dist[(nx, ny)] = d + 1
                q.append((nx, ny))
    return dist


def bfs_path(start, goal, walkable):
    """Return full BFS path from start to goal, or [] if unreachable."""
    if start == goal:
        return [start]
    q = deque([start])
    parent = {start: None}
    while q:
        cx, cy = q.popleft()
        for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + ddx, cy + ddy
            if (nx, ny) not in walkable or (nx, ny) in parent:
                continue
            parent[(nx, ny)] = (cx, cy)
            if (nx, ny) == goal:
                path = [(nx, ny)]
                cur = (cx, cy)
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            q.append((nx, ny))
    return []


def spacetime_path(start, goal, walkable, reservations, t_start, max_t=50,
                   static_avoid=None):
    """Space-time BFS: find path avoiding reserved (cell, time) pairs.

    reservations: set of (x, y, t) tuples that are blocked.
    static_avoid: set of (x, y) cells to always avoid (e.g., blocked aisles).
    Returns (path, first_action) or ([], ACT_WAIT).
    """
    if start == goal:
        return [start], ACT_WAIT
    if static_avoid is None:
        static_avoid = set()

    q = deque()
    vis = {}
    sx, sy = start

    for act in [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT, ACT_WAIT]:
        if act == ACT_WAIT:
            nx, ny = sx, sy
        else:
            nx, ny = sx + DX[act], sy + DY[act]
        if (nx, ny) not in walkable or (nx, ny) in static_avoid:
            continue
        t1 = t_start + 1
        if (nx, ny, t1) in reservations:
            continue
        if (nx, ny) == goal:
            return [start, (nx, ny)], act
        key = (nx, ny, t1)
        if key not in vis:
            vis[key] = act
            q.append((nx, ny, t1))

    while q:
        cx, cy, ct = q.popleft()
        first = vis[(cx, cy, ct)]
        if ct - t_start >= max_t:
            continue

        for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]:
            nx, ny = cx + ddx, cy + ddy
            nt = ct + 1
            if (nx, ny) not in walkable or (nx, ny) in static_avoid:
                continue
            if (nx, ny, nt) in reservations:
                continue
            key = (nx, ny, nt)
            if key in vis:
                continue
            vis[key] = first
            if (nx, ny) == goal:
                return [], first
            q.append((nx, ny, nt))

    # Fallback 1: regular BFS with static_avoid
    if static_avoid:
        safe_walkable = walkable - static_avoid
        path = bfs_path(start, goal, safe_walkable)
        if len(path) >= 2:
            nx, ny = path[1]
            dx, dy = nx - start[0], ny - start[1]
            if dy == -1: return path, ACT_MOVE_UP
            if dy == 1: return path, ACT_MOVE_DOWN
            if dx == -1: return path, ACT_MOVE_LEFT
            if dx == 1: return path, ACT_MOVE_RIGHT
    # Fallback 2: regular BFS, ignore all avoidance
    path = bfs_path(start, goal, walkable)
    if len(path) >= 2:
        nx, ny = path[1]
        dx, dy = nx - start[0], ny - start[1]
        if dy == -1: return path, ACT_MOVE_UP
        if dy == 1: return path, ACT_MOVE_DOWN
        if dx == -1: return path, ACT_MOVE_LEFT
        if dx == 1: return path, ACT_MOVE_RIGHT
    return [], ACT_WAIT


class NightmareSolver:
    def __init__(self, state, all_orders, seed=0, max_active=10):
        self.ms = state.map_state
        self.walkable = build_walkable(self.ms)
        self.all_orders = all_orders
        self.num_bots = CONFIGS['nightmare']['bots']
        self.drop_zones = [tuple(dz) for dz in self.ms.drop_off_zones]
        self.drop_set = set(self.drop_zones)
        self.spawn = self.ms.spawn
        self.max_active = max_active

        # Precompute all-pairs distances
        self.cell_dist = {}
        for cell in self.walkable:
            self.cell_dist[cell] = bfs_dist(cell, self.walkable)

        # Item type -> list of (item_idx, [adj_cells])
        self.type_items = {}
        for idx in range(self.ms.num_items):
            tid = int(self.ms.item_types[idx])
            adjs = list(self.ms.item_adjacencies.get(idx, []))
            self.type_items.setdefault(tid, []).append((idx, adjs))

        # Zone infrastructure: LEFT=0, MID=1, RIGHT=2
        self.num_zones = 3
        self.zone_x_ranges = [(0, 9), (10, 18), (19, 30)]

        # Per-zone items
        self.zone_type_items = [{} for _ in range(self.num_zones)]
        for idx in range(self.ms.num_items):
            tid = int(self.ms.item_types[idx])
            ix = int(self.ms.item_positions[idx, 0])
            adjs = list(self.ms.item_adjacencies.get(idx, []))
            for z, (lo, hi) in enumerate(self.zone_x_ranges):
                if lo <= ix <= hi:
                    self.zone_type_items[z].setdefault(tid, []).append((idx, adjs))
                    break

        # Bot zone assignment
        self.bot_zone = [-1] * self.num_bots
        self._zone_counts = [0] * self.num_zones

        # Per-bot state
        self.stall = [0] * self.num_bots
        self.prev = [None] * self.num_bots
        self.activated = set()
        self.bot_trip = [[] for _ in range(self.num_bots)]

        # Role system: 'active' picks active items, 'preview' picks preview items
        self.bot_role = ['active'] * self.num_bots
        self._zone_active = [0] * self.num_zones
        self._zone_preview = [0] * self.num_zones

        # Precompute aisle cells for blocking
        self.aisle_x = {2, 6, 10, 14, 18, 22, 26}
        self.aisle_cells_by_x = {}
        for ax in self.aisle_x:
            cells = set()
            for y in range(self.ms.height):
                if (ax, y) in self.walkable and y not in (1, 9, 16):
                    cells.add((ax, y))
            self.aisle_cells_by_x[ax] = cells

        # Staging cells: adjacent to each dropoff but NOT on it
        self.staging = {}
        for dz in self.drop_zones:
            dx, dy = dz
            best, best_d = None, 999
            # Prefer cells in the y=9 corridor above the dropoff
            for nx, ny in [(dx, dy - 1), (dx - 1, dy), (dx + 1, dy), (dx, dy + 1)]:
                if (nx, ny) in self.walkable and (nx, ny) not in self.drop_set:
                    # Prefer cells that don't block the main corridor
                    d = abs(ny - 9)  # Closer to mid-corridor is better
                    if d < best_d:
                        best, best_d = (nx, ny), d
            self.staging[dz] = best or dz

    def d(self, a, b):
        dm = self.cell_dist.get(a)
        return dm.get(b, 999) if dm else 999

    def _assign_zone(self, bid):
        if self.bot_zone[bid] >= 0:
            return
        zone_order = [2, 1, 0]
        best_z = min(zone_order, key=lambda z: self._zone_counts[z])
        self.bot_zone[bid] = best_z
        self._zone_counts[best_z] += 1
        self.bot_role[bid] = 'active'
        self._zone_active[best_z] += 1

    def bot_drop(self, bid, pos=None):
        if len(self.activated) <= 1 and pos is not None:
            return self.nearest_drop(pos)
        z = self.bot_zone[bid]
        if z < 0:
            return self.drop_zones[2]
        return self.drop_zones[z]

    def nearest_drop(self, pos):
        return min(self.drop_zones, key=lambda dz: self.d(dz, pos))

    def drop_d(self, pos):
        return min(self.d(dz, pos) for dz in self.drop_zones)

    def plan_trip(self, bot_pos, needs, claimed_adjs, max_items=3, zone=-1):
        """Plan optimal multi-item trip."""
        item_source = self.zone_type_items[zone] if zone >= 0 else self.type_items
        drop = self.drop_zones[zone] if zone >= 0 else self.nearest_drop(bot_pos)

        candidates = []
        for t, count in needs.items():
            items_for_type = []
            for idx, adjs in item_source.get(t, []):
                for adj in adjs:
                    if adj in claimed_adjs:
                        continue
                    dd = self.d(bot_pos, adj)
                    items_for_type.append((dd, idx, adj, t))
            items_for_type.sort()
            for entry in items_for_type[:3]:
                candidates.append(entry)

        if not candidates and zone >= 0:
            return self.plan_trip(bot_pos, needs, claimed_adjs, max_items, zone=-1)

        if not candidates:
            return []

        n_pick = min(max_items, len(candidates), sum(needs.values()))

        if n_pick == 1:
            candidates.sort()
            dd, idx, adj, t = candidates[0]
            return [(idx, adj, t)]

        candidates.sort()
        top = candidates[:8]
        best_cost = 999
        best_plan = []

        if n_pick >= 2:
            for i in range(len(top)):
                for j in range(len(top)):
                    if j == i:
                        continue
                    t_i, t_j = top[i][3], top[j][3]
                    if t_i == t_j and needs.get(t_i, 0) < 2:
                        continue
                    a1, a2 = top[i][2], top[j][2]
                    d1 = self.d(bot_pos, a1)
                    d12 = self.d(a1, a2)
                    d2drop = self.d(a2, drop)
                    cost = d1 + d12 + d2drop
                    if cost < best_cost:
                        best_cost = cost
                        best_plan = [(top[i][1], a1, t_i), (top[j][1], a2, t_j)]

        if n_pick >= 3:
            for i in range(len(top)):
                for j in range(len(top)):
                    if j == i:
                        continue
                    for k in range(len(top)):
                        if k == i or k == j:
                            continue
                        t_i, t_j, t_k = top[i][3], top[j][3], top[k][3]
                        type_counts = {}
                        for t in [t_i, t_j, t_k]:
                            type_counts[t] = type_counts.get(t, 0) + 1
                        skip = False
                        for t, c in type_counts.items():
                            if c > needs.get(t, 0):
                                skip = True
                                break
                        if skip:
                            continue
                        a1, a2, a3 = top[i][2], top[j][2], top[k][2]
                        d1 = self.d(bot_pos, a1)
                        d12 = self.d(a1, a2)
                        d23 = self.d(a2, a3)
                        d3drop = self.d(a3, drop)
                        cost = d1 + d12 + d23 + d3drop
                        if cost < best_cost:
                            best_cost = cost
                            best_plan = [
                                (top[i][1], a1, t_i),
                                (top[j][1], a2, t_j),
                                (top[k][1], a3, t_k),
                            ]

        return best_plan

    def _find_parking(self, pos):
        best, best_d = None, 999
        for cy in [1, 9]:
            for dx in range(15):
                for cx in [pos[0] + dx, pos[0] - dx]:
                    if 0 <= cx < self.ms.width and (cx, cy) in self.walkable:
                        dd = self.d(pos, (cx, cy))
                        if 0 < dd < best_d:
                            best_d = dd
                            best = (cx, cy)
        return best

    def _move_toward(self, p, target, reservations, rnd, static_avoid=None):
        """Move toward target using space-time BFS to avoid reservations."""
        _, act = spacetime_path(p, target, self.walkable, reservations, rnd,
                                static_avoid=static_avoid)
        return act

    def _get_blocked_aisles(self, bid, zone_aisles):
        """Get aisle cells blocked by same-zone bots (not self)."""
        z = self.bot_zone[bid]
        if z < 0:
            return set()
        blocked = set()
        own_x = self.prev[bid][0] if self.prev[bid] else -1
        for ax in zone_aisles[z]:
            if ax != own_x:  # Don't block own aisle
                blocked |= self.aisle_cells_by_x.get(ax, set())
        return blocked

    def _reserve_path(self, start, target, reservations, rnd, max_steps=8):
        """Add path cells to reservations for future time steps."""
        path = bfs_path(start, target, self.walkable)
        for i, cell in enumerate(path[:max_steps]):
            reservations.add((cell[0], cell[1], rnd + i))
            reservations.add((cell[0], cell[1], rnd + i + 1))  # Also reserve arrival time

    def action(self, state, all_orders, rnd):
        active = state.get_active_order()
        preview = state.get_preview_order()

        active_needs = {}
        if active:
            for t in active.needs():
                active_needs[t] = active_needs.get(t, 0) + 1

        preview_needs = {}
        if preview:
            for t in preview.needs():
                preview_needs[t] = preview_needs.get(t, 0) + 1

        pos = {}
        inv = {}
        for bid in range(self.num_bots):
            pos[bid] = (int(state.bot_positions[bid, 0]), int(state.bot_positions[bid, 1]))
            inv[bid] = state.bot_inv_list(bid)

        for bid in range(self.num_bots):
            if self.prev[bid] == pos[bid]:
                self.stall[bid] += 1
            else:
                self.stall[bid] = 0
            self.prev[bid] = pos[bid]

        # Active carry and shortage
        active_carry = {}
        for bid in range(self.num_bots):
            for t in inv[bid]:
                if t in active_needs:
                    active_carry[t] = active_carry.get(t, 0) + 1
        active_short = {}
        for t, need in active_needs.items():
            s = need - active_carry.get(t, 0)
            if s > 0:
                active_short[t] = s
        total_short = sum(active_short.values())

        # Preview shortage
        preview_carry = {}
        for bid in range(self.num_bots):
            for t in inv[bid]:
                if t in preview_needs and t not in active_needs:
                    preview_carry[t] = preview_carry.get(t, 0) + 1
        preview_short = {}
        for t, need in preview_needs.items():
            if t in active_needs:
                continue
            s = need - preview_carry.get(t, 0)
            if s > 0:
                preview_short[t] = s

        # Activation with zone assignment
        for bid in range(self.num_bots):
            if inv[bid]:
                self.activated.add(bid)
                self._assign_zone(bid)
        need_workers = total_short + (sum(preview_short.values()) if total_short == 0 else 0)
        idle_workers = sum(1 for b in self.activated
                          if not inv[b] and pos[b] not in self.drop_set)
        to_act = max(0, min(need_workers - idle_workers,
                            self.max_active - len(self.activated)))
        for bid in range(self.num_bots):
            if to_act <= 0:
                break
            if bid not in self.activated and pos[bid] == self.spawn:
                self.activated.add(bid)
                self._assign_zone(bid)
                to_act -= 1

        actions = [(ACT_WAIT, -1)] * self.num_bots

        # Space-time reservations: set of (x, y, t) blocked cells
        reservations = set()
        claimed_adjs = set()
        type_assigned = {}

        # Pre-reserve current positions of ALL bots at time rnd+1
        # (unprocessed bots block cells until they move)
        # Spawn tile is exempt (multiple bots can coexist there)
        for bid in range(self.num_bots):
            px, py = pos[bid]
            if (px, py) != self.spawn:
                reservations.add((px, py, rnd + 1))

        # Track which bots are "in aisle" (not in corridor)
        # A bot is "in aisle" if it's on a non-corridor row
        corridors = {1, 9, 16}
        bot_in_aisle = {}
        zone_aisle_locked = [False] * self.num_zones
        for bid in range(self.num_bots):
            if bid in self.activated:
                by = pos[bid][1]
                in_aisle = by not in corridors
                bot_in_aisle[bid] = in_aisle
                if in_aisle:
                    bz = self.bot_zone[bid]
                    if bz >= 0:
                        zone_aisle_locked[bz] = True

        # Track occupied aisles per zone: zone -> set of aisle x-values with bots
        zone_aisles = [set() for _ in range(self.num_zones)]
        for bid in range(self.num_bots):
            if bid in self.activated:
                bx = pos[bid][0]
                bz = self.bot_zone[bid]
                if bx in self.aisle_x and bz >= 0:
                    zone_aisles[bz].add(bx)

        for bid in range(self.num_bots):
            if bid not in self.activated:
                continue

            # Release this bot's current position from reservations
            # (it's about to move, so it won't block this cell)
            if pos[bid] != self.spawn:
                reservations.discard((pos[bid][0], pos[bid][1], rnd + 1))

            p = pos[bid]
            botinv = inv[bid]
            n_inv = len(botinv)
            free = INV_CAP - n_inv
            z = self.bot_zone[bid]

            has_active = any(t in active_needs for t in botinv)
            has_preview = any(t in preview_needs for t in botinv) and not has_active
            has_dead = n_inv > 0 and not has_active and not has_preview

            # Blocked aisle cells: aisles occupied by same-zone bots
            blocked = self._get_blocked_aisles(bid, zone_aisles)

            # Stall escape
            if self.stall[bid] >= 6:
                dirs = [ACT_MOVE_UP, ACT_MOVE_DOWN, ACT_MOVE_LEFT, ACT_MOVE_RIGHT]
                h = (bid * 7 + rnd * 13) % 4
                dirs = dirs[h:] + dirs[:h]
                for a in dirs:
                    nx, ny = p[0] + DX[a], p[1] + DY[a]
                    if (nx, ny) in self.walkable and (nx, ny, rnd + 1) not in reservations:
                        actions[bid] = (a, -1)
                        reservations.add((nx, ny, rnd + 1))
                        break
                continue

            # At dropoff
            if p in self.drop_set:
                if has_active:
                    actions[bid] = (ACT_DROPOFF, -1)
                    reservations.add((p[0], p[1], rnd + 1))
                    continue
                if has_preview:
                    if total_short > 0 and free > 0:
                        pass  # Fall through to pick active items
                    else:
                        # Move to staging cell to not block active deliveries
                        stg = self.staging[p]
                        act = self._move_toward(p, stg, reservations, rnd, blocked)
                        if act != ACT_WAIT:
                            nx, ny = p[0] + DX[act], p[1] + DY[act]
                            reservations.add((nx, ny, rnd + 1))
                        else:
                            reservations.add((p[0], p[1], rnd + 1))
                        actions[bid] = (act, -1)
                        continue
                if has_dead:
                    if free > 0 and (active_short or (total_short == 0 and preview_short)):
                        pass  # Fall through to trip planning
                    else:
                        # Move off dropoff to avoid blocking
                        tgt = self._find_parking(p)
                        if tgt:
                            act = self._move_toward(p, tgt, reservations, rnd, blocked)
                            if act != ACT_WAIT:
                                nx, ny = p[0] + DX[act], p[1] + DY[act]
                                reservations.add((nx, ny, rnd + 1))
                            actions[bid] = (act, -1)
                        else:
                            reservations.add((p[0], p[1], rnd + 1))
                        continue
                # Empty at dropoff — check if there's work, otherwise evacuate
                if n_inv == 0:
                    if active_short or (total_short == 0 and preview_short):
                        pass  # Fall through to trip planning (will pick items)
                    else:
                        # Nothing to pick — move off dropoff
                        stg = self.staging.get(p, self.spawn)
                        act = self._move_toward(p, stg, reservations, rnd, blocked)
                        if act != ACT_WAIT:
                            nx, ny = p[0] + DX[act], p[1] + DY[act]
                            reservations.add((nx, ny, rnd + 1))
                        else:
                            reservations.add((p[0], p[1], rnd + 1))
                        actions[bid] = (act, -1)
                        continue

            # Adjacent pickup
            if free > 0:
                best_idx = None
                best_pri = -1
                for item_idx in range(self.ms.num_items):
                    tid = int(self.ms.item_types[item_idx])
                    is_active = tid in active_short
                    is_preview = (tid in preview_short and total_short == 0)
                    if not is_active and not is_preview:
                        continue
                    for adj in self.ms.item_adjacencies.get(item_idx, []):
                        if adj == p:
                            pri = 2 if is_active else 1
                            if pri > best_pri:
                                best_idx = item_idx
                                best_pri = pri
                                break
                if best_idx is not None:
                    actions[bid] = (ACT_PICKUP, best_idx)
                    reservations.add((p[0], p[1], rnd + 1))
                    tid = int(self.ms.item_types[best_idx])
                    if tid in active_short:
                        type_assigned[tid] = type_assigned.get(tid, 0) + 1
                    continue

            # Carrying active → pick more or deliver
            if has_active:
                drop = self.bot_drop(bid, p)
                drop_dd = self.d(p, drop)
                z_for_items = z if len(self.activated) > 1 else -1

                if free > 0 and total_short > 0:
                    item_src = (self.zone_type_items[z_for_items] if z_for_items >= 0
                                else self.type_items)
                    best_t, best_idx, best_adj, best_dd = -1, -1, None, 999
                    for t, s in active_short.items():
                        if type_assigned.get(t, 0) >= s + 1:
                            continue
                        for idx, adjs in item_src.get(t, []):
                            for adj in adjs:
                                if adj in claimed_adjs:
                                    continue
                                dd = self.d(p, adj)
                                if dd < best_dd:
                                    best_t, best_idx, best_adj, best_dd = t, idx, adj, dd
                    if best_adj is not None:
                        item_to_drop = self.d(best_adj, drop)
                        detour = best_dd + item_to_drop - drop_dd
                        max_detour = 20 if len(self.activated) <= 2 else 10
                        if detour < max_detour:
                            if p == best_adj:
                                actions[bid] = (ACT_PICKUP, best_idx)
                                reservations.add((p[0], p[1], rnd + 1))
                            else:
                                act = self._move_toward(p, best_adj, reservations, rnd, blocked)
                                if act != ACT_WAIT:
                                    nx, ny = p[0] + DX[act], p[1] + DY[act]
                                    reservations.add((nx, ny, rnd + 1))
                                    self._reserve_path(p, best_adj, reservations, rnd)
                                else:
                                    reservations.add((p[0], p[1], rnd + 1))
                                actions[bid] = (act, -1)
                            claimed_adjs.add(best_adj)
                            type_assigned[best_t] = type_assigned.get(best_t, 0) + 1
                            continue

                # Preview detour (only if room beyond active needs)
                if free > total_short and preview_short:
                    p_item_src = (self.zone_type_items[z_for_items] if z_for_items >= 0
                                  else self.type_items)
                    max_preview_det = 15 if total_short <= 1 else (8 if total_short <= 2 else 3)
                    best_t, best_idx, best_adj = -1, -1, None
                    best_detour = 999
                    for t in list(preview_short):
                        for idx, adjs in p_item_src.get(t, []):
                            for adj in adjs:
                                if adj in claimed_adjs:
                                    continue
                                dd = self.d(p, adj)
                                item_to_drop = self.d(adj, drop)
                                det = dd + item_to_drop - drop_dd
                                if det < max_preview_det and det < best_detour:
                                    best_t, best_idx, best_adj = t, idx, adj
                                    best_detour = det
                    if best_adj is not None:
                        if p == best_adj:
                            actions[bid] = (ACT_PICKUP, best_idx)
                            reservations.add((p[0], p[1], rnd + 1))
                        else:
                            act = self._move_toward(p, best_adj, reservations, rnd, blocked)
                            if act != ACT_WAIT:
                                nx, ny = p[0] + DX[act], p[1] + DY[act]
                                reservations.add((nx, ny, rnd + 1))
                            else:
                                reservations.add((p[0], p[1], rnd + 1))
                            actions[bid] = (act, -1)
                        claimed_adjs.add(best_adj)
                        preview_short[best_t] = preview_short.get(best_t, 1) - 1
                        if preview_short.get(best_t, 0) <= 0:
                            preview_short.pop(best_t, None)
                        continue

                # Deliver to dropoff
                act = self._move_toward(p, drop, reservations, rnd, blocked)
                if act != ACT_WAIT:
                    nx, ny = p[0] + DX[act], p[1] + DY[act]
                    reservations.add((nx, ny, rnd + 1))
                    self._reserve_path(p, drop, reservations, rnd)
                else:
                    reservations.add((p[0], p[1], rnd + 1))
                actions[bid] = (act, -1)
                continue

            # Carrying preview → stage NEAR dropoff (not on it)
            if has_preview:
                drop = self.bot_drop(bid, p)
                stg = self.staging[drop]
                act = self._move_toward(p, stg, reservations, rnd, blocked)
                if act != ACT_WAIT:
                    nx, ny = p[0] + DX[act], p[1] + DY[act]
                    reservations.add((nx, ny, rnd + 1))
                    self._reserve_path(p, stg, reservations, rnd)
                else:
                    reservations.add((p[0], p[1], rnd + 1))
                actions[bid] = (act, -1)
                continue

            # Dead inventory: if free slots and items needed, fall through
            if has_dead:
                if free > 0 and (active_short or (total_short == 0 and preview_short)):
                    pass  # Fall through to trip planning
                else:
                    dd = self.drop_d(p)
                    if dd <= 6:
                        tgt = self._find_parking(p)
                        if tgt:
                            act = self._move_toward(p, tgt, reservations, rnd, blocked)
                            if act != ACT_WAIT:
                                nx, ny = p[0] + DX[act], p[1] + DY[act]
                                reservations.add((nx, ny, rnd + 1))
                            actions[bid] = (act, -1)
                    reservations.add((p[0], p[1], rnd + 1))
                    continue

            # Empty: pick items using trip planner
            # Role-based: active bots pick active items, preview bots pick preview items
            role = self.bot_role[bid]
            if role == 'preview' and preview_short:
                pick_needs = preview_short
            elif active_short:
                pick_needs = active_short
            elif total_short == 0 and preview_short:
                pick_needs = preview_short
            else:
                pick_needs = {}
            if pick_needs and free > 0:
                if self.bot_trip[bid]:
                    trip_valid = all(t in pick_needs for _, _, t in self.bot_trip[bid])
                    if not trip_valid:
                        self.bot_trip[bid] = []

                if not self.bot_trip[bid]:
                    avail_needs = {}
                    for t, s in pick_needs.items():
                        cap = s + 1 if t in active_short else s
                        if type_assigned.get(t, 0) < cap:
                            avail_needs[t] = s - type_assigned.get(t, 0)
                    if avail_needs:
                        use_zone = self.bot_zone[bid] if len(self.activated) > 1 else -1
                        self.bot_trip[bid] = self.plan_trip(
                            p, avail_needs, claimed_adjs, max_items=free,
                            zone=use_zone)

                if self.bot_trip[bid]:
                    idx, adj, t = self.bot_trip[bid][0]
                    if p == adj:
                        actions[bid] = (ACT_PICKUP, idx)
                        reservations.add((p[0], p[1], rnd + 1))
                        self.bot_trip[bid].pop(0)
                    else:
                        act = self._move_toward(p, adj, reservations, rnd, blocked)
                        if act != ACT_WAIT:
                            nx, ny = p[0] + DX[act], p[1] + DY[act]
                            reservations.add((nx, ny, rnd + 1))
                            self._reserve_path(p, adj, reservations, rnd)
                        else:
                            reservations.add((p[0], p[1], rnd + 1))
                        actions[bid] = (act, -1)
                    claimed_adjs.add(adj)
                    type_assigned[t] = type_assigned.get(t, 0) + 1
                    continue

            # Idle: if on dropoff, move to staging; otherwise park at spawn
            if p in self.drop_set:
                stg = self.staging.get(p, self.spawn)
                act = self._move_toward(p, stg, reservations, rnd, blocked)
                if act != ACT_WAIT:
                    nx, ny = p[0] + DX[act], p[1] + DY[act]
                    reservations.add((nx, ny, rnd + 1))
                else:
                    reservations.add((p[0], p[1], rnd + 1))
                actions[bid] = (act, -1)
            elif p != self.spawn:
                act = self._move_toward(p, self.spawn, reservations, rnd, blocked)
                if act != ACT_WAIT:
                    nx, ny = p[0] + DX[act], p[1] + DY[act]
                    reservations.add((nx, ny, rnd + 1))
                actions[bid] = (act, -1)
            else:
                reservations.add((p[0], p[1], rnd + 1))

        return actions


def solve_nightmare(seed, verbose=False, max_active=10):
    state, all_orders = init_game(seed, 'nightmare', num_orders=100)
    solver = NightmareSolver(state, all_orders, seed=seed, max_active=max_active)
    action_log = []
    num_rounds = DIFF_ROUNDS['nightmare']
    chains = 0

    for rnd in range(num_rounds):
        state.round = rnd
        actions = solver.action(state, all_orders, rnd)
        action_log.append(actions)
        o_before = state.orders_completed
        step(state, actions, all_orders)
        c = state.orders_completed - o_before
        if c > 1:
            chains += c - 1

        if verbose and (rnd < 30 or rnd % 50 == 0 or c > 0):
            active = state.get_active_order()
            extra = f" +{c}!" if c > 1 else ""
            bot_info = []
            for bid in range(min(6, solver.num_bots)):
                bx, by = int(state.bot_positions[bid, 0]), int(state.bot_positions[bid, 1])
                bi = state.bot_inv_list(bid)
                a = actions[bid]
                act_name = {0:'W',1:'U',2:'D',3:'L',4:'R',5:'P',6:'DO'}[a[0]]
                z = solver.bot_zone[bid]
                bot_info.append(f"b{bid}({bx},{by})z{z}i{len(bi)}{act_name}")
            print(f"R{rnd:3d} S={state.score:3d} Ord={state.orders_completed:2d}"
                  f" Act={len(solver.activated):2d}"
                  + (f" Need={len(active.needs())}" if active else " DONE")
                  + extra + " | " + " ".join(bot_info))

    if verbose:
        print(f"\nFinal: Score={state.score} Ord={state.orders_completed}"
              f" Items={state.items_delivered} Chains={chains}")
    return state.score, action_log


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='1000-1009')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--active-bots', type=int, default=10)
    args = parser.parse_args()

    from configs import parse_seeds
    seeds = parse_seeds(args.seeds)

    scores = []
    t0 = time.time()
    for seed in seeds:
        score, _ = solve_nightmare(seed, verbose=args.verbose, max_active=args.active_bots)
        scores.append(score)
        print(f"Seed {seed}: {score}")

    elapsed = time.time() - t0
    print(f"\n{'='*40}")
    print(f"Seeds: {len(seeds)}")
    print(f"Mean: {np.mean(scores):.1f}")
    print(f"Max:  {max(scores)}")
    print(f"Min:  {min(scores)}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(seeds):.1f}s/seed)")


if __name__ == '__main__':
    main()
