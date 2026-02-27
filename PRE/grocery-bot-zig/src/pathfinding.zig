const std = @import("std");
const types = @import("types.zig");

const Pos = types.Pos;
const Dir = types.Dir;
const Cell = types.Cell;
const GameState = types.GameState;
const DistMap = types.DistMap;
const MAX_H = types.MAX_H;
const MAX_W = types.MAX_W;
const MAX_CELLS = types.MAX_CELLS;
const MAX_BOTS = types.MAX_BOTS;
const UNREACHABLE = types.UNREACHABLE;

// ── BFS Distance Map ──────────────────────────────────────────────────
pub fn bfsDistMap(state: *const GameState, start: Pos, dm: *DistMap) void {
    const w = state.width;
    const h = state.height;
    for (0..h) |y| for (0..w) |x| {
        dm[y][x] = UNREACHABLE;
    };
    if (start.x < 0 or start.y < 0 or start.x >= w or start.y >= h) return;
    const cell_s = state.grid[@intCast(start.y)][@intCast(start.x)];
    if (cell_s == .wall or cell_s == .shelf) return;

    var queue: [MAX_CELLS]struct { x: u16, y: u16 } = undefined;
    var head: u16 = 0;
    var tail: u16 = 0;
    const sx: u16 = @intCast(start.x);
    const sy: u16 = @intCast(start.y);
    dm[sy][sx] = 0;
    queue[tail] = .{ .x = sx, .y = sy };
    tail += 1;

    const ddx = [4]i16{ 0, 0, -1, 1 };
    const ddy = [4]i16{ -1, 1, 0, 0 };
    while (head < tail) {
        const cur = queue[head];
        head += 1;
        const cd = dm[cur.y][cur.x];
        for (0..4) |i| {
            const nx_i = @as(i32, cur.x) + ddx[i];
            const ny_i = @as(i32, cur.y) + ddy[i];
            if (nx_i < 0 or ny_i < 0 or nx_i >= w or ny_i >= h) continue;
            const nx: u16 = @intCast(nx_i);
            const ny: u16 = @intCast(ny_i);
            if (dm[ny][nx] != UNREACHABLE) continue;
            const cell = state.grid[ny][nx];
            if (cell == .wall or cell == .shelf) continue;
            dm[ny][nx] = cd + 1;
            queue[tail] = .{ .x = nx, .y = ny };
            tail += 1;
        }
    }
}

// ── Distance lookup from a precomputed DistMap ────────────────────────
pub fn distFromMap(dm: *const DistMap, pos: Pos) u16 {
    if (pos.x < 0 or pos.y < 0) return UNREACHABLE;
    return dm[@intCast(pos.y)][@intCast(pos.x)];
}

// ── BFS with first-step collision avoidance ────────────────────────────
pub const BfsResult = struct { dist: u16, first_dir: ?Dir };

pub fn bfs(state: *const GameState, start: Pos, target: Pos, bot_id: u8, bot_positions: *const [MAX_BOTS]Pos) BfsResult {
    if (start.eql(target)) return .{ .dist = 0, .first_dir = null };
    const w = state.width;
    const h = state.height;
    var visited: [MAX_CELLS]bool = undefined;
    @memset(visited[0 .. w * h], false);
    const QEntry = struct { x: u16, y: u16, dist: u16, first_dir: ?Dir };
    var queue: [MAX_CELLS]QEntry = undefined;
    var head: u16 = 0;
    var tail: u16 = 0;
    const sx: u16 = @intCast(start.x);
    const sy: u16 = @intCast(start.y);
    visited[sy * w + sx] = true;
    queue[tail] = .{ .x = sx, .y = sy, .dist = 0, .first_dir = null };
    tail += 1;

    const ddx = [4]i16{ 0, 0, -1, 1 };
    const ddy = [4]i16{ -1, 1, 0, 0 };
    const dirs = [4]Dir{ .up, .down, .left, .right };
    while (head < tail) {
        const cur = queue[head];
        head += 1;
        for (0..4) |i| {
            const nx_i = @as(i32, cur.x) + ddx[i];
            const ny_i = @as(i32, cur.y) + ddy[i];
            if (nx_i < 0 or ny_i < 0 or nx_i >= w or ny_i >= h) continue;
            const nx: u16 = @intCast(nx_i);
            const ny: u16 = @intCast(ny_i);
            const idx = ny * w + nx;
            if (visited[idx]) continue;
            const cell = state.grid[ny][nx];
            const is_target = (nx == @as(u16, @intCast(target.x)) and ny == @as(u16, @intCast(target.y)));
            if (!is_target and (cell == .wall or cell == .shelf)) continue;
            if (cur.dist < 2) {
                var blocked = false;
                for (0..state.bot_count) |bi| {
                    if (bi == bot_id) continue;
                    if (bot_positions[bi].x == @as(i16, @intCast(nx)) and bot_positions[bi].y == @as(i16, @intCast(ny))) {
                        blocked = true;
                        break;
                    }
                }
                if (blocked) continue;
            }
            visited[idx] = true;
            const first = if (cur.first_dir) |d| d else dirs[i];
            if (is_target) return .{ .dist = cur.dist + 1, .first_dir = first };
            queue[tail] = .{ .x = nx, .y = ny, .dist = cur.dist + 1, .first_dir = first };
            tail += 1;
        }
    }
    // BFS failed - try to find any walkable direction as fallback
    return .{ .dist = UNREACHABLE, .first_dir = safeGreedyDir(state, start, target) };
}

pub fn safeGreedyDir(state: *const GameState, from: Pos, to: Pos) ?Dir {
    const ddx2 = to.x - from.x;
    const ddy2 = to.y - from.y;
    const ordered_dirs: [4]Dir = blk: {
        if (@abs(ddx2) > @abs(ddy2)) {
            if (ddx2 > 0) break :blk [4]Dir{ .right, .down, .up, .left };
            break :blk [4]Dir{ .left, .down, .up, .right };
        }
        if (ddy2 > 0) break :blk [4]Dir{ .down, .right, .left, .up };
        if (ddy2 < 0) break :blk [4]Dir{ .up, .right, .left, .down };
        break :blk [4]Dir{ .right, .down, .left, .up };
    };
    const offx = [4]i16{ 0, 0, -1, 1 };
    const offy = [4]i16{ -1, 1, 0, 0 };
    for (ordered_dirs) |d| {
        const di: usize = switch (d) { .up => 0, .down => 1, .left => 2, .right => 3 };
        const nx = from.x + offx[di];
        const ny = from.y + offy[di];
        if (nx < 0 or ny < 0 or nx >= state.width or ny >= state.height) continue;
        const cell = state.grid[@intCast(ny)][@intCast(nx)];
        if (cell == .floor or cell == .dropoff) return d;
    }
    return null;
}

// ── Find best adjacent floor cell to an item ──────────────────────────
pub fn findBestAdj(state: *const GameState, item_pos: Pos, dm: *const DistMap) ?Pos {
    const offsets = [4][2]i16{ .{ 0, -1 }, .{ 0, 1 }, .{ -1, 0 }, .{ 1, 0 } };
    var best: ?Pos = null;
    var best_d: u16 = UNREACHABLE;
    for (offsets) |off| {
        const nx = item_pos.x + off[0];
        const ny = item_pos.y + off[1];
        if (nx < 0 or ny < 0 or nx >= state.width or ny >= state.height) continue;
        const cell = state.grid[@intCast(ny)][@intCast(nx)];
        if (cell == .floor or cell == .dropoff) {
            const d = dm[@intCast(ny)][@intCast(nx)];
            if (d < best_d) { best_d = d; best = .{ .x = nx, .y = ny }; }
        }
    }
    return best;
}
