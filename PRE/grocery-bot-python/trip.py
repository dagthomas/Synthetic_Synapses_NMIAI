"""Trip planning - optimized with flat distance maps and minimal object creation."""
from __future__ import annotations

from types_ import NeedList, INV_CAP, UNREACHABLE
from pathfinding import bfs_dist_map, find_best_adj


def trip_score(cost, ac, pc, count, completes_order, rounds_left=300):
    if cost == 0:
        return 10**18
    value = ac * 20 + pc * 3
    if completes_order:
        value += 80
    if completes_order and rounds_left < 60:
        value += 20
    value += count * 2
    if rounds_left < 60 and cost * 2 > rounds_left:
        value = value // 2
    return value * 10000 // cost


def plan_best_trip(
    grid, w, h,
    dm_bot, dm_drop,
    bot_active, bot_preview,
    claimed, bi,
    slots_free, allow_preview,
    rounds_left,
    item_types, item_positions, item_count,
):
    """Plan best trip. Returns (items, adjs, cost, ac, pc, completes) or None.
    items/adjs are lists of indices/tuples."""
    active_remaining = bot_active.count

    # Count needed types
    active_tc = {}
    for t, c in bot_active._counts.items():
        if c > 0:
            active_tc[t] = c
    preview_tc = {}
    if allow_preview:
        for t, c in bot_preview._counts.items():
            if c > 0:
                preview_tc[t] = c

    # Build candidates sorted by distance
    temp = []
    for ii in range(item_count):
        if claimed[ii] >= 0 and claimed[ii] != bi:
            continue
        it = item_types[ii]

        is_active = it in active_tc and active_tc[it] > 0
        is_preview = False
        if not is_active and allow_preview:
            is_preview = it in preview_tc and preview_tc[it] > 0
        if not is_active and not is_preview:
            continue

        ix, iy = item_positions[ii]
        adj = find_best_adj(grid, w, h, ix, iy, dm_bot)
        if adj is None:
            continue
        ax, ay = adj
        d_to = dm_bot[ay * w + ax]
        if d_to >= UNREACHABLE:
            continue
        d_back = dm_drop[ay * w + ax]
        temp.append((ii, d_to, is_active, ax, ay, d_back))

    temp.sort(key=lambda c: c[1])

    # Pick closest per type-slot
    sel_a: dict[str, int] = {}
    sel_p: dict[str, int] = {}
    # Candidate data as parallel arrays for speed
    c_idx = []
    c_adj = []  # (ax, ay) tuples
    c_active = []
    c_dbot = []
    c_ddrop = []

    for ii, dist, is_act, ax, ay, d_back in temp:
        if len(c_idx) >= 16:
            break
        it = item_types[ii]
        if is_act:
            used = sel_a.get(it, 0)
            if used >= active_tc.get(it, 0):
                continue
            sel_a[it] = used + 1
        else:
            used = sel_p.get(it, 0)
            if used >= preview_tc.get(it, 0):
                continue
            sel_p[it] = used + 1

        c_idx.append(ii)
        c_adj.append((ax, ay))
        c_active.append(is_act)
        c_dbot.append(dist)
        c_ddrop.append(d_back)

    if not c_idx:
        return None

    n = min(len(c_idx), 16)

    # Pre-compute BFS from each candidate adj
    cand_dms = [bfs_dist_map(grid, w, h, c_adj[i][0], c_adj[i][1]) for i in range(n)]

    best = None
    best_sc = 0

    # Single-item trips
    for a in range(n):
        cost = c_dbot[a] + c_ddrop[a]
        if cost + 3 > rounds_left:
            continue
        ac = 1 if c_active[a] else 0
        pc = 1 - ac
        if ac == 0 and active_remaining > 0:
            continue
        comp = ac >= active_remaining and active_remaining > 0
        sc = trip_score(cost, ac, pc, 1, comp, rounds_left)
        if sc > best_sc:
            best = ([c_idx[a]], [c_adj[a]], cost, ac, pc, comp)
            best_sc = sc

    if slots_free < 2:
        return best

    # 2-item trips
    for a in range(n):
        for b in range(a + 1, n):
            if not c_active[a] and not c_active[b] and not allow_preview:
                continue

            bax, bay = c_adj[b]
            aax, aay = c_adj[a]
            ab_dist = cand_dms[a][bay * w + bax]
            ba_dist = cand_dms[b][aay * w + aax]

            cost_ab = c_dbot[a] + ab_dist + c_ddrop[b]
            cost_ba = c_dbot[b] + ba_dist + c_ddrop[a]

            ac = (1 if c_active[a] else 0) + (1 if c_active[b] else 0)
            pc = 2 - ac
            if ac == 0 and active_remaining > 0:
                continue
            comp = ac >= active_remaining and active_remaining > 0

            if cost_ab + 4 <= rounds_left:
                sc = trip_score(cost_ab, ac, pc, 2, comp, rounds_left)
                if sc > best_sc:
                    bix, biy = item_positions[c_idx[b]]
                    adj_b = find_best_adj(grid, w, h, bix, biy, cand_dms[a]) or c_adj[b]
                    best = ([c_idx[a], c_idx[b]], [c_adj[a], adj_b], cost_ab, ac, pc, comp)
                    best_sc = sc

            if cost_ba + 4 <= rounds_left:
                sc = trip_score(cost_ba, ac, pc, 2, comp, rounds_left)
                if sc > best_sc:
                    aix, aiy = item_positions[c_idx[a]]
                    adj_a = find_best_adj(grid, w, h, aix, aiy, cand_dms[b]) or c_adj[a]
                    best = ([c_idx[b], c_idx[a]], [c_adj[b], adj_a], cost_ba, ac, pc, comp)
                    best_sc = sc

    if slots_free < 3:
        return best

    # 3-item trips
    n3 = min(n, 10)
    for a in range(n3):
        for b in range(a + 1, n3):
            for c in range(b + 1, n3):
                ac = (1 if c_active[a] else 0) + (1 if c_active[b] else 0) + (1 if c_active[c] else 0)
                pc = 3 - ac
                if ac == 0 and active_remaining > 0:
                    continue
                comp = ac >= active_remaining and active_remaining > 0

                for p0, p1, p2 in ((a,b,c),(a,c,b),(b,a,c),(b,c,a),(c,a,b),(c,b,a)):
                    ax1, ay1 = c_adj[p1]
                    ax2, ay2 = c_adj[p2]
                    cost = (c_dbot[p0]
                            + cand_dms[p0][ay1 * w + ax1]
                            + cand_dms[p1][ay2 * w + ax2]
                            + c_ddrop[p2])
                    if cost + 5 > rounds_left:
                        continue
                    sc = trip_score(cost, ac, pc, 3, comp, rounds_left)
                    if sc > best_sc:
                        ix1, iy1 = item_positions[c_idx[p1]]
                        ix2, iy2 = item_positions[c_idx[p2]]
                        adj1 = find_best_adj(grid, w, h, ix1, iy1, cand_dms[p0]) or c_adj[p1]
                        adj2 = find_best_adj(grid, w, h, ix2, iy2, cand_dms[p1]) or c_adj[p2]
                        best = ([c_idx[p0], c_idx[p1], c_idx[p2]],
                                [c_adj[p0], adj1, adj2], cost, ac, pc, comp)
                        best_sc = sc

    return best
