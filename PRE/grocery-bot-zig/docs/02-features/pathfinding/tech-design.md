# Pathfinding - Technical Design

BFS-based pathfinding with collision avoidance for navigating the grid.

---

## Algorithms

### BFS Distance Map (`bfsDistMap`)

Computes shortest distance from a start position to every reachable cell.

- **Input**: Grid, start position
- **Output**: `DistMap` (32x20 array of u16 distances)
- **Obstacles**: Walls and shelves are impassable
- **Complexity**: O(W * H) time and space
- **Unreachable cells**: Marked as `UNREACHABLE` (9999)

Used for: trip cost estimation, delivery distance calculation, pre-positioning.

### Collision-Aware BFS (`bfs`)

Single-target pathfinding that avoids first-step collisions with other bots.

- **Input**: Grid, start, target, array of bot positions to avoid
- **Output**: `BfsResult { dist: u16, first_dir: ?Dir }`
- **Collision rule**: First step cannot land on another bot's position
- **Fallback**: `safeGreedyDir()` if BFS finds no path

### Safe Greedy Direction (`safeGreedyDir`)

Heuristic fallback when BFS fails (e.g., all paths blocked by bots).

- Tries primary axis first (axis with larger delta)
- Validates: within bounds, walkable cell, no bot collision
- Prefers floor/dropoff cells

### Find Best Adjacent (`findBestAdj`)

Finds the best floor cell adjacent to a shelf item for pickup.

- Checks all 4 neighbors of item position
- Filters: must be floor or dropoff, must be reachable (not UNREACHABLE in dist map)
- Tiebreak: closest to dropoff (encourages efficient routing)

---

## Grid Cell Types

| Cell | Walkable | Notes |
|------|----------|-------|
| Floor | Yes | Standard movement |
| Dropoff | Yes | Delivery point |
| Wall | No | Impassable |
| Shelf | No | Contains items, pick from adjacent floor |

---

## Files

- `src/pathfinding.zig` - All pathfinding algorithms
- `src/types.zig` - `Pos`, `Dir`, `Cell`, `DistMap` types
