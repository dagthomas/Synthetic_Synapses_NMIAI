"""Decision engine - fully optimized with flat arrays, tuples, fast NeedList."""
from __future__ import annotations
import sys

from types_ import (
    NeedList, PersistentBot,
    FLOOR, WALL, SHELF, DROPOFF,
    UP, DOWN, LEFT, RIGHT,
    DIR_NAMES, DIR_DX, DIR_DY, DIR_REVERSE,
    HIST_LEN, INV_CAP, MAX_BOTS, UNREACHABLE,
)
from pathfinding import bfs_dist_map, bfs_path, find_best_adj
from trip import plan_best_trip

# ── Module-level persistent state ─────────────────────────────────────
pbots: list[PersistentBot] = []
pbots_initialized = False

last_score = 0
rounds_since_score_change = 0

expected_next_pos: list[tuple[int, int]] = [(0, 0)] * MAX_BOTS
expected_count = 0

pending_dirs: list[int] = [-1] * MAX_BOTS  # -1 = None
pending_is_move: list[bool] = [False] * MAX_BOTS
offset_detected = False
offset_check_mismatches = 0


def init_pbots():
    global pbots, pbots_initialized, last_score, rounds_since_score_change
    global offset_detected, offset_check_mismatches
    global pending_dirs, pending_is_move, expected_count

    pbots = [PersistentBot() for _ in range(MAX_BOTS)]
    last_score = 0
    rounds_since_score_change = 0
    offset_detected = False
    offset_check_mismatches = 0
    pending_dirs = [-1] * MAX_BOTS
    pending_is_move = [False] * MAX_BOTS
    expected_count = 0
    pbots_initialized = True


def _push_hist(pb, x, y):
    pos = (x, y)
    if len(pb.pos_hist) < HIST_LEN:
        pb.pos_hist.append(pos)
    else:
        pb.pos_hist[pb.pos_hist_idx] = pos
    pb.pos_hist_idx = (pb.pos_hist_idx + 1) % HIST_LEN


def _is_oscillating(pb, x, y):
    count = 0
    for p in pb.pos_hist:
        if p[0] == x and p[1] == y:
            count += 1
            if count >= 4:
                return True
    return False


def _flee_dropoff(grid, w, h, px, py, bot_id, bot_positions, bot_count):
    for d in range(4):
        nx = px + DIR_DX[d]
        ny = py + DIR_DY[d]
        if nx < 0 or ny < 0 or nx >= w or ny >= h:
            continue
        cell = grid[ny * w + nx]
        if cell == WALL or cell == SHELF:
            continue
        blocked = False
        for bi in range(bot_count):
            if bi == bot_id:
                continue
            if bot_positions[bi][0] == nx and bot_positions[bi][1] == ny:
                blocked = True
                break
        if not blocked:
            return d
    return -1


def _escape_dir(grid, w, h, px, py, pb, bot_id, bot_positions, bot_count):
    reverse = DIR_REVERSE[pb.last_dir] if pb.last_dir >= 0 else -1
    best_dir = -1
    best_score = -100
    cx = w // 2
    cy = h // 2
    cur_cdist = abs(px - cx) + abs(py - cy)

    for d in range(4):
        nx = px + DIR_DX[d]
        ny = py + DIR_DY[d]
        if nx < 0 or ny < 0 or nx >= w or ny >= h:
            continue
        cell = grid[ny * w + nx]
        if cell == WALL or cell == SHELF:
            continue
        blocked = False
        for bi in range(bot_count):
            if bi == bot_id:
                continue
            if bot_positions[bi][0] == nx and bot_positions[bi][1] == ny:
                blocked = True
                break
        if blocked:
            continue

        score = 0
        if d == reverse:
            score -= 10
        for p in pb.pos_hist:
            if p[0] == nx and p[1] == ny:
                score -= 3
        if abs(nx - cx) + abs(ny - cy) < cur_cdist:
            score += 2
        if score > best_score:
            best_score = score
            best_dir = d
    return best_dir


def build_needs(orders):
    active = NeedList()
    preview = NeedList()
    for req, delivered, is_active, complete in orders:
        if complete:
            continue
        need: dict[str, int] = {}
        for rt in req:
            need[rt] = need.get(rt, 0) + 1
        for dt in delivered:
            if dt in need:
                need[dt] -= 1
        target = active if is_active else preview
        for t, c in need.items():
            for _ in range(max(0, c)):
                target.add(t)
    return active, preview


def _find_item_by_id(item_ids, item_id):
    try:
        return item_ids.index(item_id)
    except ValueError:
        return -1


# ── Decision Engine ────────────────────────────────────────────────────
def decide_actions(state) -> str:
    global pbots_initialized, last_score, rounds_since_score_change
    global offset_detected, offset_check_mismatches
    global expected_count

    if not pbots_initialized:
        init_pbots()

    grid = state.grid
    w = state.width
    h = state.height
    bc = state.bot_count
    ic = state.item_count
    bot_ids = state.bot_ids
    bot_pos = state.bot_positions
    bot_inv = state.bot_inventories
    item_ids = state.item_ids
    item_types = state.item_types
    item_pos = state.item_positions
    drop = state.dropoff

    active_orig, preview_orig = build_needs(state.orders)

    # pick_remaining: active items still needing pickup
    pick_remaining = active_orig.copy()
    for bi in range(bc):
        for item in bot_inv[bi]:
            pick_remaining.remove(item)

    preview = preview_orig.copy()
    rounds_left = max(0, state.max_rounds - state.round)

    # Count bots with only preview items
    bots_with_preview_only = 0
    for bi in range(bc):
        if not bot_inv[bi]:
            continue
        has_any = False
        ca = active_orig.copy()
        for item in bot_inv[bi]:
            if ca.contains(item):
                has_any = True
                ca.remove(item)
        if not has_any:
            bots_with_preview_only += 1

    max_preview_carriers = bc if bc <= 2 else 1

    # Score tracking
    if state.score != last_score:
        rounds_since_score_change = 0
        last_score = state.score
    else:
        rounds_since_score_change += 1

    # ── Offset detection - FAST: detect from round 5, threshold 2 rounds ──
    if expected_count > 0 and state.round >= 5 and not offset_detected:
        moving_mm = 0
        moving_c = 0
        ck = min(expected_count, bc)
        for bi in range(ck):
            if pending_is_move[bi]:
                moving_c += 1
                if expected_next_pos[bi] != bot_pos[bi]:
                    moving_mm += 1
        # Lower thresholds: 3+ moving bots with 50%+ mismatches
        thresh = max(2, (moving_c + 1) // 2)
        if moving_c >= 3 and moving_mm >= thresh:
            offset_check_mismatches += 1
        else:
            offset_check_mismatches = max(0, offset_check_mismatches - 1)
        if offset_check_mismatches >= 2:
            offset_detected = True
            print(f"R{state.round} OFFSET MODE ENABLED ({moving_mm}/{moving_c})", file=sys.stderr)

    # Compute effective positions
    eff_pos = list(bot_pos)  # shallow copy of tuples
    if offset_detected:
        for bi in range(bc):
            if pending_is_move[bi] and pending_dirs[bi] >= 0:
                pd = pending_dirs[bi]
                fx = bot_pos[bi][0] + DIR_DX[pd]
                fy = bot_pos[bi][1] + DIR_DY[pd]
                if 0 <= fx < w and 0 <= fy < h:
                    cell = grid[fy * w + fx]
                    if cell == FLOOR or cell == DROPOFF:
                        eff_pos[bi] = (fx, fy)

    # Distance from dropoff
    dm_drop = bfs_dist_map(grid, w, h, drop[0], drop[1])

    # Detect stuck order
    order_stuck = False
    temp_needs = active_orig.copy()
    for bi in range(bc):
        for item in bot_inv[bi]:
            temp_needs.remove(item)
    for t, c in temp_needs.unique_with_counts():
        found = False
        for ii in range(ic):
            if item_types[ii] == t:
                ix, iy = item_pos[ii]
                if find_best_adj(grid, w, h, ix, iy, dm_drop) is not None:
                    found = True
                    break
        if not found:
            order_stuck = True
            break

    # Bot positions for collision
    coll_pos = list(eff_pos)

    # Claimed items
    claimed = [-1] * ic

    # Orchestrator assignments
    orch_active_assigned = [0] * bc
    current_aoi = state.active_order_idx

    # ── Update persistent state ──
    for bi in range(bc):
        bx, by = bot_pos[bi]
        pb = pbots[bi]

        _push_hist(pb, bx, by)

        if pb.last_pos == (bx, by):
            pb.stall_count += 1
        else:
            pb.stall_count = 0
            dx = bx - pb.last_pos[0]
            dy = by - pb.last_pos[1]
            if dx == 1: pb.last_dir = RIGHT
            elif dx == -1: pb.last_dir = LEFT
            elif dy == 1: pb.last_dir = DOWN
            elif dy == -1: pb.last_dir = UP
        pb.last_pos = (bx, by)

        if pb.stall_count > 8:
            pb.has_trip = False
            pb.delivering = False
            pb.stall_count = 0
            pb.osc_count += 1

        if _is_oscillating(pb, bx, by):
            pb.osc_count += 1
            if pb.osc_count >= 6:
                pb.has_trip = False
                pb.delivering = False
                pb.escape_rounds = 4
                pb.osc_count = 0
        elif pb.osc_count > 0:
            pb.osc_count -= 1

        if pb.escape_rounds > 0:
            pb.escape_rounds -= 1

        if pb.last_tried_pickup:
            pb.last_tried_pickup = False

        if pb.last_active_order_idx != current_aoi:
            pb.has_trip = False
            pb.delivering = False
            pb.osc_count = 0
            pb.rounds_on_order = 0
        pb.last_active_order_idx = current_aoi

        pb.rounds_on_order = min(pb.rounds_on_order + 1, 65535)

        if pb.rounds_on_order > 30 and pb.rounds_on_order % 15 == 0:
            pb.has_trip = False
            pb.delivering = False
            pb.osc_count = 0

        if pb.has_trip:
            valid = True
            for ti in range(pb.trip_pos, pb.trip_count):
                idx = _find_item_by_id(item_ids, pb.trip_ids[ti])
                if idx < 0:
                    valid = False
                    break
                claimed[idx] = bi
            if not valid:
                pb.has_trip = False

    # ── Pre-compute BFS for all bots ──
    all_dm = [bfs_dist_map(grid, w, h, eff_pos[bi][0], eff_pos[bi][1]) for bi in range(bc)]

    # ── Dropoff priority ──
    MAX_DA = 3 if bc >= 6 else (2 if bc > 1 else 1)
    dp = [False] * bc
    if bc > 1:
        dbots = []
        for bi in range(bc):
            if not pbots[bi].delivering:
                continue
            ba = active_orig.copy()
            ha = False
            for item in bot_inv[bi]:
                if ba.contains(item):
                    ha = True
                    ba.remove(item)
            if not ha:
                continue
            ex, ey = eff_pos[bi]
            d = dm_drop[ey * w + ex]
            dbots.append((bi, d))
        dbots.sort(key=lambda x: x[1])
        for i in range(min(len(dbots), MAX_DA)):
            dp[dbots[i][0]] = True
        for bi in range(bc):
            ex, ey = eff_pos[bi]
            if dm_drop[ey * w + ex] <= 1:
                dp[bi] = True
    else:
        dp[0] = True

    # ── Orchestrator ──
    if bc > 1:
        ba_count = [0] * bc
        for ii in range(ic):
            if 0 <= claimed[ii] < bc:
                ba_count[claimed[ii]] += 1

        type_track = pick_remaining.unique_with_counts()

        for tt_idx in range(len(type_track)):
            t, needed = type_track[tt_idx]
            assigned = 0
            while assigned < needed:
                best_d = UNREACHABLE
                best_ii = -1
                best_bk = -1

                for ii in range(ic):
                    if claimed[ii] >= 0:
                        continue
                    if item_types[ii] != t:
                        continue
                    ix, iy = item_pos[ii]
                    for bk in range(bc):
                        if pbots[bk].delivering and dp[bk]:
                            continue
                        free = max(0, INV_CAP - len(bot_inv[bk]))
                        if ba_count[bk] >= free:
                            continue
                        adj = find_best_adj(grid, w, h, ix, iy, all_dm[bk])
                        if adj is None:
                            continue
                        d = all_dm[bk][adj[1] * w + adj[0]]
                        if d < best_d:
                            best_d = d
                            best_ii = ii
                            best_bk = bk

                if best_ii < 0:
                    break
                claimed[best_ii] = best_bk
                ba_count[best_bk] += 1
                assigned += 1

            if assigned >= needed:
                for ii in range(ic):
                    if claimed[ii] >= 0:
                        continue
                    if item_types[ii] == t:
                        claimed[ii] = MAX_BOTS

        orch_active_assigned = ba_count[:]

        # Phase 2: preview
        total_pa = 0
        max_op = 4 if bc <= 2 else 2 if bc <= 4 else min(preview.count, bc // 2)
        if preview.count > 0 and bots_with_preview_only < max_preview_carriers:
            for t, needed in preview.unique_with_counts():
                if total_pa >= max_op:
                    break
                assigned = 0
                while assigned < needed and total_pa < max_op:
                    best_d = UNREACHABLE
                    best_ii = -1
                    best_bk = -1
                    for ii in range(ic):
                        if claimed[ii] >= 0:
                            continue
                        if item_types[ii] != t:
                            continue
                        ix, iy = item_pos[ii]
                        for bk in range(bc):
                            if pbots[bk].delivering:
                                continue
                            if orch_active_assigned[bk] > 0:
                                continue
                            free = max(0, INV_CAP - len(bot_inv[bk]))
                            if ba_count[bk] >= free:
                                continue
                            adj = find_best_adj(grid, w, h, ix, iy, all_dm[bk])
                            if adj is None:
                                continue
                            d = all_dm[bk][adj[1] * w + adj[0]]
                            if d < best_d:
                                best_d = d
                                best_ii = ii
                                best_bk = bk
                    if best_ii < 0:
                        break
                    claimed[best_ii] = best_bk
                    ba_count[best_bk] += 1
                    assigned += 1
                    total_pa += 1

    # ── Per-bot decisions - build JSON string directly ──
    parts = []

    for bi in range(bc):
        inv = bot_inv[bi]
        inv_len = len(inv)
        pb = pbots[bi]
        bx, by = eff_pos[bi]
        bid = bot_ids[bi]
        dm_bot = all_dm[bi]

        pb.last_tried_pickup = False

        # Per-bot needs
        ba = active_orig.copy()
        has_active = False
        for item in inv:
            if ba.contains(item):
                has_active = True
                ba.remove(item)
        bp = preview_orig.copy()
        for item in inv:
            bp.remove(item)

        allow_preview = (
            orch_active_assigned[bi] == 0
            and inv_len == 0
            and (bc <= 2 or bots_with_preview_only < max_preview_carriers)
        )

        # Validate trip
        if pb.has_trip and pb.trip_pos < pb.trip_count:
            ca = ba.copy()
            cp = bp.copy()
            valid = True
            for ti in range(pb.trip_pos, pb.trip_count):
                idx = _find_item_by_id(item_ids, pb.trip_ids[ti])
                if idx < 0:
                    valid = False
                    break
                it = item_types[idx]
                if ca.contains(it):
                    ca.remove(it)
                elif allow_preview and cp.contains(it):
                    cp.remove(it)
                else:
                    valid = False
                    break
            if not valid:
                pb.has_trip = False

        # Helper to append move action
        def _move(d):
            parts.append(f'{{"bot":{bid},"action":"{DIR_NAMES[d]}"}}')
            cx, cy = coll_pos[bi]
            coll_pos[bi] = (cx + DIR_DX[d], cy + DIR_DY[d])
            pending_dirs[bi] = d
            pending_is_move[bi] = True

        def _wait():
            parts.append(f'{{"bot":{bid},"action":"wait"}}')
            pending_is_move[bi] = False
            pending_dirs[bi] = -1

        def _pickup(iid):
            parts.append(f'{{"bot":{bid},"action":"pick_up","item_id":"{iid}"}}')
            pending_is_move[bi] = False
            pending_dirs[bi] = -1

        def _dropoff():
            parts.append(f'{{"bot":{bid},"action":"drop_off"}}')
            pending_is_move[bi] = False
            pending_dirs[bi] = -1

        # ─── 1. At dropoff -> drop off ───
        if bx == drop[0] and by == drop[1] and inv_len > 0 and has_active:
            _dropoff()
            pb.has_trip = False
            pb.delivering = False
            continue

        # ─── 1b. At dropoff, no active -> evacuate ──
        if bx == drop[0] and by == drop[1] and not has_active:
            fd = _flee_dropoff(grid, w, h, bx, by, bi, coll_pos, bc)
            if fd >= 0:
                _move(fd)
            else:
                _wait()
            continue

        # ─── 1c. Escape ────
        if pb.escape_rounds > 0:
            esc_picked = False
            for ii in range(ic):
                ix, iy = item_pos[ii]
                if abs(bx - ix) + abs(by - iy) != 1:
                    continue
                if inv_len >= INV_CAP:
                    break
                if claimed[ii] >= 0 and claimed[ii] != bi:
                    continue
                it = item_types[ii]
                esc_act = pick_remaining.contains(it)
                esc_prev = allow_preview and bp.contains(it)
                if not esc_act and not esc_prev:
                    continue
                _pickup(item_ids[ii])
                if esc_act:
                    pick_remaining.remove(it)
                else:
                    preview.remove(it)
                claimed[ii] = bi
                esc_picked = True
                break

            if not esc_picked:
                ed = _escape_dir(grid, w, h, bx, by, pb, bi, coll_pos, bc)
                if ed >= 0:
                    _move(ed)
                    continue
            else:
                continue

        # ─── 2. Adjacent pickup ──
        picked = False
        for pass_num in range(2):
            if picked:
                break
            for ii in range(ic):
                ix, iy = item_pos[ii]
                if abs(bx - ix) + abs(by - iy) != 1:
                    continue
                if inv_len >= INV_CAP:
                    break
                if claimed[ii] >= 0 and claimed[ii] != bi:
                    continue
                it = item_types[ii]
                if pass_num == 0:
                    if not pick_remaining.contains(it):
                        continue
                else:
                    if not bp.contains(it):
                        continue
                    # Allow preview pickup with 2+ free slots (keeps 1 reserved for active)
                    free_for_preview = max(0, INV_CAP - inv_len)
                    if not allow_preview and free_for_preview < 2:
                        continue

                _pickup(item_ids[ii])
                pb.last_tried_pickup = True
                pb.last_pickup_pos = (bx, by)
                pb.last_pickup_ipos = (ix, iy)
                pb.last_inv_len = inv_len
                pick_remaining.remove(it)
                preview.remove(it)
                claimed[ii] = bi

                if pb.has_trip:
                    # Advance trip
                    if pb.trip_pos < pb.trip_count and pb.trip_ids[pb.trip_pos] == item_ids[ii]:
                        pb.trip_pos += 1
                        if pb.trip_pos >= pb.trip_count:
                            pb.has_trip = False
                    else:
                        for ti in range(pb.trip_pos, pb.trip_count):
                            tidx = _find_item_by_id(item_ids, pb.trip_ids[ti])
                            if tidx >= 0 and item_types[tidx] == it:
                                pb.trip_ids.pop(ti)
                                pb.trip_adjs.pop(ti)
                                if pb.trip_pos >= pb.trip_count:
                                    pb.has_trip = False
                                break

                picked = True
                break
        if picked:
            continue

        # ─── 3. Should deliver? ──
        active_on_map = 0
        tpr = pick_remaining.copy()
        for ii in range(ic):
            if claimed[ii] >= 0 and claimed[ii] != bi:
                continue
            it = item_types[ii]
            if tpr.contains(it):
                active_on_map += 1
                tpr.remove(it)

        inv_full = inv_len >= INV_CAP
        all_active_got = pick_remaining.count == 0
        no_active = active_on_map == 0
        trip_done = pb.has_trip and pb.trip_pos >= pb.trip_count
        eff_slots = max(0, INV_CAP - inv_len)
        dtd = dm_drop[by * w + bx]
        endgame = rounds_left <= dtd + 3

        far_few = (bc >= 5 and not inv_full and dtd > 8
                   and inv_len < 2 and all_active_got and not endgame)

        should_del = (has_active and not far_few
                      and (inv_full or all_active_got or no_active or order_stuck or endgame or trip_done))

        if should_del:
            pb.delivering = True
            pb.has_trip = False
        # (Delay delivery for preview fill-up removed — caused dead inventory cascade)
        if far_few and pb.delivering:
            pb.delivering = False
        if not has_active:
            pb.delivering = False

        # ─── 4. Delivering -> dropoff ──
        if pb.delivering and has_active:
            direct_dist = dm_drop[by * w + bx]

            # Detour pickup: while delivering, scan for needed items within 2 extra rounds
            # Guards: must have free slot, direct_dist > 3, not endgame
            if eff_slots > 0 and direct_dist > 3 and not endgame:
                best_detour_ii = -1
                best_detour_cost = direct_dist + 3  # max 2 extra rounds
                for ii in range(ic):
                    if claimed[ii] >= 0 and claimed[ii] != bi:
                        continue
                    it = item_types[ii]
                    is_active_det = pick_remaining.contains(it)
                    is_preview_det = bp.contains(it)
                    if not is_active_det and not is_preview_det:
                        continue
                    ix, iy = item_pos[ii]
                    adj = find_best_adj(grid, w, h, ix, iy, dm_bot)
                    if adj is None:
                        continue
                    d_to = dm_bot[adj[1] * w + adj[0]]
                    if d_to >= UNREACHABLE:
                        continue
                    d_back = dm_drop[adj[1] * w + adj[0]]
                    detour_cost = d_to + d_back
                    if is_active_det and detour_cost > 0:
                        detour_cost -= 1  # Active items get 1-cost discount
                    if detour_cost < best_detour_cost:
                        best_detour_cost = detour_cost
                        best_detour_ii = ii
                if best_detour_ii >= 0:
                    det_ix, det_iy = item_pos[best_detour_ii]
                    det_adj = find_best_adj(grid, w, h, det_ix, det_iy, dm_bot)
                    if det_adj is not None:
                        det_dist, det_fd = bfs_path(grid, w, h, bx, by, det_adj[0], det_adj[1], bi, coll_pos, bc)
                        if det_dist < UNREACHABLE and det_fd >= 0:
                            _move(det_fd)
                            claimed[best_detour_ii] = bi
                            continue

            dist, fd = bfs_path(grid, w, h, bx, by, drop[0], drop[1], bi, coll_pos, bc)
            if dist < UNREACHABLE and fd >= 0:
                # Anti-oscillation near dropoff
                cur_md = abs(bx - drop[0]) + abs(by - drop[1])
                if cur_md <= 2 and bc > 1 and pb.stall_count < 3:
                    nx = bx + DIR_DX[fd]
                    ny = by + DIR_DY[fd]
                    if abs(nx - drop[0]) + abs(ny - drop[1]) > cur_md:
                        _wait()
                        continue
                _move(fd)
                continue

        # ─── 5. Follow trip ──
        if pb.has_trip and pb.trip_pos < pb.trip_count:
            tid = pb.trip_ids[pb.trip_pos]
            idx = _find_item_by_id(item_ids, tid)
            if idx >= 0:
                adj = pb.trip_adjs[pb.trip_pos]
                if bx == adj[0] and by == adj[1]:
                    if inv_len < INV_CAP:
                        _pickup(item_ids[idx])
                        pb.last_tried_pickup = True
                        pb.last_pickup_pos = (bx, by)
                        pb.last_pickup_ipos = item_pos[idx]
                        pb.last_inv_len = inv_len
                        pick_remaining.remove(item_types[idx])
                        preview.remove(item_types[idx])
                        claimed[idx] = bi
                        pb.trip_pos += 1
                        if pb.trip_pos >= pb.trip_count:
                            pb.has_trip = False
                        continue
                else:
                    dist, fd = bfs_path(grid, w, h, bx, by, adj[0], adj[1], bi, coll_pos, bc)
                    if dist < UNREACHABLE and fd >= 0:
                        _move(fd)
                        claimed[idx] = bi
                        continue
            pb.has_trip = False

        # ─── 6. Plan new trip ──
        if eff_slots > 0 and (not pb.delivering or not dp[bi]):
            tp = plan_best_trip(
                grid, w, h, dm_bot, dm_drop,
                ba, bp, claimed, bi,
                eff_slots, allow_preview, rounds_left,
                item_types, item_pos, ic,
            )
            if tp is not None:
                t_items, t_adjs, t_cost, t_ac, t_pc, t_comp = tp
                pb.has_trip = True
                pb.trip_ids = [item_ids[t_items[i]] for i in range(len(t_items))]
                pb.trip_adjs = list(t_adjs)
                pb.trip_pos = 0
                pb.osc_count = 0
                for i in range(len(t_items)):
                    claimed[t_items[i]] = bi
                    preview.remove(item_types[t_items[i]])

                fa = t_adjs[0]
                fi = t_items[0]
                if bx == fa[0] and by == fa[1]:
                    if inv_len < INV_CAP:
                        _pickup(item_ids[fi])
                        pb.last_tried_pickup = True
                        pb.last_pickup_pos = (bx, by)
                        pb.last_pickup_ipos = item_pos[fi]
                        pb.last_inv_len = inv_len
                        pb.trip_pos += 1
                        if pb.trip_pos >= pb.trip_count:
                            pb.has_trip = False
                        continue
                else:
                    dist, fd = bfs_path(grid, w, h, bx, by, fa[0], fa[1], bi, coll_pos, bc)
                    if dist < UNREACHABLE and fd >= 0:
                        _move(fd)
                        continue

        # ─── 7. Emergency deliver ──
        if inv_len > 0 and has_active:
            pb.delivering = True
            dist, fd = bfs_path(grid, w, h, bx, by, drop[0], drop[1], bi, coll_pos, bc)
            if dist < UNREACHABLE and fd >= 0:
                cur_md = abs(bx - drop[0]) + abs(by - drop[1])
                if cur_md <= 2 and bc > 1 and pb.stall_count < 3:
                    nx = bx + DIR_DX[fd]
                    ny = by + DIR_DY[fd]
                    if abs(nx - drop[0]) + abs(ny - drop[1]) > cur_md:
                        _wait()
                        continue
                _move(fd)
                continue

        # ─── 8. Pre-position ──
        if not has_active and inv_len == 0:
            targets = []
            for ii in range(ic):
                if len(targets) >= 12:
                    break
                if not pick_remaining.contains(item_types[ii]):
                    continue
                if claimed[ii] >= 0 and claimed[ii] != bi:
                    continue
                ix, iy = item_pos[ii]
                adj = find_best_adj(grid, w, h, ix, iy, dm_bot)
                if adj is None:
                    continue
                d = dm_bot[adj[1] * w + adj[0]]
                if d < UNREACHABLE:
                    targets.append((adj, d))

            for ii in range(ic):
                if len(targets) >= 16:
                    break
                if not preview_orig.contains(item_types[ii]):
                    continue
                ix, iy = item_pos[ii]
                adj = find_best_adj(grid, w, h, ix, iy, dm_bot)
                if adj is None:
                    continue
                d = dm_bot[adj[1] * w + adj[0]]
                if d < UNREACHABLE:
                    targets.append((adj, d))

            if targets:
                targets.sort(key=lambda x: x[1])
                idle_rank = sum(1 for pbi in range(bi) if not pbots[pbi].delivering and not bot_inv[pbi])
                pick_i = idle_rank % len(targets)
                tgt = targets[pick_i][0]
                if bx != tgt[0] or by != tgt[1]:
                    dist, fd = bfs_path(grid, w, h, bx, by, tgt[0], tgt[1], bi, coll_pos, bc)
                    if dist < UNREACHABLE and fd >= 0:
                        _move(fd)
                        continue

        # Dead inventory
        if not has_active and inv_len > 0:
            has_pm = any(preview_orig.contains(item) for item in inv)
            dd = dm_drop[by * w + bx]
            if has_pm:
                if dd > 3:
                    dist, fd = bfs_path(grid, w, h, bx, by, drop[0], drop[1], bi, coll_pos, bc)
                    if dist < UNREACHABLE and fd >= 0:
                        _move(fd)
                        continue
            elif dd <= 2:
                fd = _flee_dropoff(grid, w, h, bx, by, bi, coll_pos, bc)
                if fd >= 0:
                    _move(fd)
                    continue

        # ─── 9. Wait ──
        _wait()

    # Record expected positions
    for bi in range(bc):
        expected_next_pos[bi] = eff_pos[bi] if offset_detected else coll_pos[bi]
    expected_count = bc

    return '{"actions":[' + ','.join(parts) + ']}'
