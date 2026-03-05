const std = @import("std");
const types = @import("types.zig");
const pathfinding = @import("pathfinding.zig");

const Pos = types.Pos;
const Dir = types.Dir;
const Cell = types.Cell;
const GameState = types.GameState;
const DistMap = types.DistMap;
const AisleDir = types.AisleDir;
const STResult = types.STResult;
const MAX_H = types.MAX_H;
const MAX_W = types.MAX_W;
const MAX_BOTS = types.MAX_BOTS;
const MAX_TIME_HORIZON = types.MAX_TIME_HORIZON;
const WRONG_WAY_PENALTY = types.WRONG_WAY_PENALTY;
const UNREACHABLE = types.UNREACHABLE;

// ── Reservation Table ─────────────────────────────────────────────────
// reserved[t][y][x] = bot_id + 1 (0 = free)
var res_table: [MAX_TIME_HORIZON][MAX_H][MAX_W]u8 = undefined;
var res_max_t: u16 = 0;

pub fn clearReservations() void {
    for (0..MAX_TIME_HORIZON) |t| {
        for (0..MAX_H) |y| {
            @memset(&res_table[t][y], 0);
        }
    }
    res_max_t = 0;
}

pub fn reserve(x: u16, y: u16, t: u16, bot_id: u8) void {
    if (t >= MAX_TIME_HORIZON or y >= MAX_H or x >= MAX_W) return;
    res_table[t][y][x] = bot_id + 1;
    if (t + 1 > res_max_t) res_max_t = t + 1;
}

pub fn isReserved(x: u16, y: u16, t: u16, bot_id: u8) bool {
    if (t >= MAX_TIME_HORIZON) return false;
    if (y >= MAX_H or x >= MAX_W) return true;
    const v = res_table[t][y][x];
    return v != 0 and v != bot_id + 1;
}

fn reservePath(path: []const Pos, bot_id: u8, max_steps: u16) void {
    const steps = @min(@as(u16, @intCast(path.len)), max_steps);
    for (0..steps) |t| {
        const p = path[t];
        if (p.x >= 0 and p.y >= 0) {
            reserve(@intCast(p.x), @intCast(p.y), @intCast(t), bot_id);
        }
    }
    if (steps > 0) {
        const last = path[steps - 1];
        if (last.x >= 0 and last.y >= 0) {
            for (steps..MAX_TIME_HORIZON) |t| {
                reserve(@intCast(last.x), @intCast(last.y), @intCast(t), bot_id);
            }
        }
    }
}

pub fn reserveStationary(x: u16, y: u16, t_start: u16, t_end: u16, bot_id: u8) void {
    var t = t_start;
    while (t < t_end and t < MAX_TIME_HORIZON) : (t += 1) {
        reserve(x, y, t, bot_id);
    }
}

// ── Stored Plans (Multi-Step Commitment) ─────────────────────────────
// Instead of replanning every round, follow committed ST-A* paths for
// multiple steps. Other bots see committed paths via re-reserved entries.
const MAX_PLAN_LEN = MAX_TIME_HORIZON; // 12

const StoredPlan = struct {
    positions: [MAX_PLAN_LEN + 1]Pos, // [0]=start, [i]=after step i
    dirs: [MAX_PLAN_LEN]?Dir, // direction at each step (null=wait)
    goal: Pos,
    len: u16, // total steps in plan
    step: u16, // next step to take (0=first step)
    valid: bool,
};

var stored_plans: [MAX_BOTS]StoredPlan = undefined;
var plans_initialized: bool = false;

pub fn initPlans() void {
    for (0..MAX_BOTS) |i| {
        stored_plans[i].valid = false;
    }
    plans_initialized = true;
}

fn dirBetween(from: Pos, to: Pos) ?Dir {
    if (to.x > from.x) return .right;
    if (to.x < from.x) return .left;
    if (to.y > from.y) return .down;
    if (to.y < from.y) return .up;
    return null; // same position = wait
}

var max_commit_len: u16 = MAX_PLAN_LEN;

/// Set max commitment length. Disable for Expert (10 bots) where
/// reservation pressure from committed paths hurts mean score.
pub fn setMaxCommitLen(bot_count: u8) void {
    // 5 bots (Hard): full 12 steps — tested +9.5% mean, +5% max.
    // 10 bots (Expert): disabled — commitment hurts mean by ~4%.
    max_commit_len = if (bot_count >= 8) 0 else MAX_PLAN_LEN;
}

fn storePlan(bot_id: u8, path: []const Pos, goal: Pos) void {
    if (!plans_initialized or bot_id >= MAX_BOTS or max_commit_len == 0) return;
    const plan = &stored_plans[bot_id];
    const path_steps = @as(u16, @intCast(path.len)) -| 1;
    if (path_steps == 0) {
        plan.valid = false;
        return;
    }
    const len = @min(path_steps, max_commit_len);
    for (0..len + 1) |i| {
        plan.positions[i] = path[i];
    }
    for (0..len) |i| {
        plan.dirs[i] = dirBetween(path[i], path[i + 1]);
    }
    plan.goal = goal;
    plan.len = len;
    plan.step = 0;
    plan.valid = true;
}

pub const CommittedResult = struct {
    dir: ?Dir,
    remaining: u16,
};

/// Get committed direction for a bot going to a specific goal.
/// Returns null if no committed plan exists for this goal.
pub fn getCommitted(bot_id: u8, goal: Pos) ?CommittedResult {
    if (!plans_initialized or bot_id >= MAX_BOTS) return null;
    const plan = &stored_plans[bot_id];
    if (!plan.valid) return null;
    if (!plan.goal.eql(goal)) return null;
    if (plan.step >= plan.len) return null;
    return CommittedResult{
        .dir = plan.dirs[plan.step],
        .remaining = plan.len - plan.step,
    };
}

/// Invalidate a bot's committed plan (goal changed, new order, etc.)
pub fn invalidatePlan(bot_id: u8) void {
    if (bot_id < MAX_BOTS) stored_plans[bot_id].valid = false;
}

/// Called after clearReservations(). Checks if bots followed their
/// committed paths, advances step counters, and re-reserves remaining
/// future positions so subsequently-planning bots see them.
///
/// Semantics: step = index of the NEXT action to take.
/// After taking dirs[step], bot ends up at positions[step+1].
/// So on the next round, we check positions[step+1] to verify the bot moved correctly.
pub fn advanceAndReReserve(bot_positions: *const [MAX_BOTS]Pos, bot_count: u8) void {
    if (!plans_initialized) return;
    for (0..bot_count) |bi| {
        const plan = &stored_plans[bi];
        if (!plan.valid) continue;

        // After last round, bot should have taken dirs[step] and arrived at positions[step+1]
        if (plan.step + 1 > plan.len) {
            plan.valid = false; // Plan exhausted
            continue;
        }
        const expected = plan.positions[plan.step + 1];
        if (!bot_positions[bi].eql(expected)) {
            plan.valid = false; // Bot deviated (collision or different action)
            continue;
        }

        // Advance: the bot took dirs[step], now ready for dirs[step+1]
        plan.step += 1;

        // Plan exhausted? (no more steps to take)
        if (plan.step >= plan.len) {
            plan.valid = false;
            continue;
        }

        // Re-reserve remaining path from current position onward
        // positions[step] = current pos, positions[step+1..len] = future
        var path_buf: [MAX_PLAN_LEN + 1]Pos = undefined;
        var pi: u16 = 0;
        var si = plan.step;
        while (si <= plan.len) : (si += 1) {
            path_buf[pi] = plan.positions[si];
            pi += 1;
        }
        reservePath(path_buf[0..pi], @intCast(bi), MAX_TIME_HORIZON);
    }
}

// ── Aisle Detection ───────────────────────────────────────────────────
var aisle_dir: [MAX_H][MAX_W]?AisleDir = undefined;
var aisles_initialized: bool = false;

// Module-level aisle/corridor maps (promoted for use in occupancy penalty)
var aisle_columns: [MAX_W]bool = [_]bool{false} ** MAX_W;
var corridor_rows: [MAX_H]bool = [_]bool{false} ** MAX_H;
var init_width: u16 = 0;
var init_height: u16 = 0;

pub fn isInAisle(x: u16) bool {
    if (x >= MAX_W) return false;
    return aisle_columns[x];
}

pub fn isCorridorRow(y: u16) bool {
    if (y >= MAX_H) return false;
    return corridor_rows[y];
}

pub fn init(state: *const GameState) void {
    const w = state.width;
    const h = state.height;
    init_width = w;
    init_height = h;

    for (0..MAX_H) |y| {
        for (0..MAX_W) |x| {
            aisle_dir[y][x] = null;
        }
    }

    @memset(&corridor_rows, false);
    for (0..h) |y| {
        var floor_count: u16 = 0;
        for (0..w) |x| {
            const cell = state.grid[y][x];
            if (cell == .floor or cell == .dropoff) floor_count += 1;
        }
        if (floor_count * 10 > w * 6) {
            corridor_rows[y] = true;
        }
    }

    @memset(&aisle_columns, false);
    var aisle_col_count: u16 = 0;

    for (1..w - 1) |x| {
        var aisle_cells: u16 = 0;
        var total_floor: u16 = 0;
        for (0..h) |y| {
            if (corridor_rows[y]) continue;
            const cell = state.grid[y][x];
            if (cell != .floor and cell != .dropoff) continue;
            total_floor += 1;
            const left = state.grid[y][x - 1];
            const right = state.grid[y][x + 1];
            if ((left == .shelf or left == .wall) and (right == .shelf or right == .wall)) {
                aisle_cells += 1;
            }
        }
        // 75% threshold — only clear 1-tile aisles
        if (total_floor > 2 and aisle_cells * 4 > total_floor * 3) {
            aisle_columns[x] = true;
            aisle_col_count += 1;
        }
    }

    var aisle_idx: u16 = 0;
    for (0..w) |x| {
        if (!aisle_columns[x]) continue;
        const dir: AisleDir = if (aisle_idx % 2 == 0) .down else .up;
        for (0..h) |y| {
            if (corridor_rows[y]) continue;
            const cell = state.grid[y][x];
            if (cell == .floor or cell == .dropoff) {
                aisle_dir[y][x] = dir;
            }
        }
        aisle_idx += 1;
    }

    aisles_initialized = true;
    std.debug.print("MAPF: Detected {d} aisle columns\n", .{aisle_col_count});
}

// ── Space-Time A* ────────────────────────────────────────────────────
const MAX_OPEN = 4000; // Smaller budget = faster per call
const MAX_PATH = MAX_TIME_HORIZON + 1;

const STNode = struct {
    x: u16,
    y: u16,
    t: u16,
    g: u16,
    f: u16,
    parent_idx: u16, // Index in closed list (0xFFFF = none)
    dir: ?Dir,
};

/// Run Space-Time A* from start to goal. Only effective for short-range goals
/// (distance <= MAX_TIME_HORIZON). Returns null if goal not reached.
fn spaceTimeAStar(state: *const GameState, start: Pos, goal: Pos, bot_id: u8, bot_positions: ?*const [MAX_BOTS]Pos) ?STResult {
    if (start.eql(goal)) return STResult{ .first_dir = null, .path_len = 0 };

    const w = state.width;
    const h = state.height;

    const dm_goal = pathfinding.getPrecomputedDm(state, goal);

    // Quick reject: if goal is too far, ST-A* will exhaust budget — skip
    const start_h = dm_goal[@intCast(start.y)][@intCast(start.x)];
    if (start_h >= UNREACHABLE) return null;
    if (start_h > MAX_TIME_HORIZON) return null; // Too far for horizon

    var visited: [MAX_TIME_HORIZON][MAX_H][MAX_W]bool = undefined;
    for (0..MAX_TIME_HORIZON) |t| {
        for (0..h) |y| {
            @memset(visited[t][y][0..w], false);
        }
    }

    var open: [MAX_OPEN]STNode = undefined;
    var open_len: u16 = 0;

    var closed: [MAX_OPEN]STNode = undefined;
    var closed_len: u16 = 0;

    const sx: u16 = @intCast(start.x);
    const sy: u16 = @intCast(start.y);

    open[0] = STNode{
        .x = sx,
        .y = sy,
        .t = 0,
        .g = 0,
        .f = start_h,
        .parent_idx = 0xFFFF,
        .dir = null,
    };
    open_len = 1;

    const ddx = [4]i16{ 0, 0, -1, 1 };
    const ddy = [4]i16{ -1, 1, 0, 0 };
    const dirs = [4]Dir{ .up, .down, .left, .right };

    var expansions: u16 = 0;
    const max_expansions: u16 = 3000; // Conservative budget

    while (open_len > 0 and expansions < max_expansions) {
        // Find min f in open list
        var best_idx: u16 = 0;
        for (1..open_len) |i| {
            if (open[i].f < open[best_idx].f or
                (open[i].f == open[best_idx].f and open[i].g > open[best_idx].g))
            {
                best_idx = @intCast(i);
            }
        }

        const cur = open[best_idx];
        open_len -= 1;
        if (best_idx < open_len) {
            open[best_idx] = open[open_len];
        }

        if (cur.t < MAX_TIME_HORIZON and visited[cur.t][cur.y][cur.x]) continue;
        if (cur.t < MAX_TIME_HORIZON) {
            visited[cur.t][cur.y][cur.x] = true;
        }

        const cur_closed_idx = closed_len;
        if (closed_len < MAX_OPEN) {
            closed[closed_len] = cur;
            closed_len += 1;
        } else {
            break;
        }

        expansions += 1;

        // Check if goal reached
        if (cur.x == @as(u16, @intCast(goal.x)) and cur.y == @as(u16, @intCast(goal.y))) {
            // Reconstruct first direction
            var first_dir: ?Dir = cur.dir;
            var trace_idx = cur.parent_idx;
            var prev_dir = cur.dir;
            while (trace_idx != 0xFFFF and trace_idx < closed_len) {
                const parent = closed[trace_idx];
                if (parent.parent_idx == 0xFFFF) {
                    first_dir = prev_dir;
                    break;
                }
                prev_dir = parent.dir;
                trace_idx = parent.parent_idx;
            }

            // Reserve path
            var rev_buf: [MAX_PATH]struct { x: u16, y: u16 } = undefined;
            var rev_len: u16 = 0;
            rev_buf[0] = .{ .x = cur.x, .y = cur.y };
            rev_len = 1;
            var ri = cur.parent_idx;
            while (ri != 0xFFFF and ri < closed_len and rev_len < MAX_PATH) {
                rev_buf[rev_len] = .{ .x = closed[ri].x, .y = closed[ri].y };
                rev_len += 1;
                ri = closed[ri].parent_idx;
            }
            var path_buf: [MAX_PATH]Pos = undefined;
            var pi: u16 = 0;
            while (pi < rev_len) : (pi += 1) {
                const rpi = rev_len - 1 - pi;
                path_buf[pi] = .{
                    .x = @intCast(rev_buf[rpi].x),
                    .y = @intCast(rev_buf[rpi].y),
                };
            }
            reservePath(path_buf[0..pi], bot_id, MAX_TIME_HORIZON);

            // Store full path as committed plan for multi-step following
            storePlan(bot_id, path_buf[0..pi], goal);

            return STResult{ .first_dir = first_dir, .path_len = cur.t };
        }

        if (cur.t + 1 >= MAX_TIME_HORIZON) continue;

        const nt: u16 = cur.t + 1;

        // 4 cardinal moves
        for (0..4) |di| {
            const nx_i = @as(i32, cur.x) + ddx[di];
            const ny_i = @as(i32, cur.y) + ddy[di];
            if (nx_i < 0 or ny_i < 0 or nx_i >= w or ny_i >= h) continue;
            const nx: u16 = @intCast(nx_i);
            const ny: u16 = @intCast(ny_i);

            const cell = state.grid[ny][nx];
            const is_goal = (nx == @as(u16, @intCast(goal.x)) and ny == @as(u16, @intCast(goal.y)));
            if (!is_goal and (cell == .wall or cell == .shelf)) continue;

            if (isReserved(nx, ny, nt, bot_id)) continue;

            // Swap conflict
            if (isReserved(nx, ny, cur.t, bot_id) and isReserved(cur.x, cur.y, nt, bot_id)) continue;

            if (nt < MAX_TIME_HORIZON and visited[nt][ny][nx]) continue;

            var move_cost: u16 = 1;

            // Soft aisle penalty
            if (aisles_initialized and ny < MAX_H and nx < MAX_W) {
                if (aisle_dir[ny][nx]) |preferred| {
                    const moving_dir = dirs[di];
                    const is_wrong_way = switch (preferred) {
                        .down => moving_dir == .up,
                        .up => moving_dir == .down,
                    };
                    if (is_wrong_way) move_cost += WRONG_WAY_PENALTY;
                }

                // Aisle occupancy penalty: avoid entering aisle column with another bot
                if (aisle_columns[nx] and bot_positions != null) {
                    const bps = bot_positions.?;
                    var occupied = false;
                    for (0..state.bot_count) |bk| {
                        if (bk == bot_id) continue;
                        if (bps[bk].x < 0 or bps[bk].y < 0) continue;
                        const bkx: u16 = @intCast(bps[bk].x);
                        const bky: u16 = @intCast(bps[bk].y);
                        if (bkx == nx and !corridor_rows[bky]) {
                            occupied = true;
                            break;
                        }
                    }
                    if (occupied) {
                        // +3 for Hard (5 bots), +2 for Expert (8+ bots) to avoid over-penalizing
                        move_cost += if (state.bot_count >= 8) 2 else 3;
                    }
                }
            }

            const ng = cur.g + move_cost;
            const nh = dm_goal[ny][nx];
            if (nh >= UNREACHABLE) continue;
            const nf = ng + nh;

            if (open_len < MAX_OPEN) {
                open[open_len] = STNode{
                    .x = nx,
                    .y = ny,
                    .t = nt,
                    .g = ng,
                    .f = nf,
                    .parent_idx = cur_closed_idx,
                    .dir = dirs[di],
                };
                open_len += 1;
            }
        }

        // Wait action
        {
            if (!isReserved(cur.x, cur.y, nt, bot_id)) {
                if (!(nt < MAX_TIME_HORIZON and visited[nt][cur.y][cur.x])) {
                    const wg = cur.g + 1;
                    const wh = dm_goal[cur.y][cur.x];
                    if (wh < UNREACHABLE) {
                        if (open_len < MAX_OPEN) {
                            open[open_len] = STNode{
                                .x = cur.x,
                                .y = cur.y,
                                .t = nt,
                                .g = wg,
                                .f = wg + wh,
                                .parent_idx = cur_closed_idx,
                                .dir = null,
                            };
                            open_len += 1;
                        }
                    }
                }
            }
        }
    }

    return null;
}

// ── Public API ────────────────────────────────────────────────────────

pub fn planAndReserve(state: *const GameState, start: Pos, goal: Pos, bot_id: u8, bot_positions: ?*const [MAX_BOTS]Pos) ?STResult {
    return spaceTimeAStar(state, start, goal, bot_id, bot_positions);
}
