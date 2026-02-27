# Architecture

High-level architecture of the grocery-bot-zig game client.

---

## System Overview

```mermaid
flowchart TB
    GS[Game Server] <-->|WebSocket| WS[ws.zig]
    WS -->|JSON frame| P[parser.zig]
    P -->|GameState| S[strategy.zig]
    S -->|queries| PF[pathfinding.zig]
    S -->|queries| TP[trip.zig]
    S -->|actions| WS
    M[main.zig] -->|orchestrates| WS
    M -->|calls| S
    T[types.zig] -.->|shared types| P
    T -.->|shared types| S
    T -.->|shared types| PF
    T -.->|shared types| TP
```

---

## Data Flow Per Round

```mermaid
sequenceDiagram
    participant Server
    participant Main
    participant Parser
    participant Strategy
    participant Pathfinding
    participant Trip

    Server->>Main: WebSocket frame (game state JSON)
    Main->>Parser: parseGameState(payload)
    Parser-->>Main: GameState struct
    Main->>Strategy: decideActions(state, round)
    Strategy->>Pathfinding: bfsDistMap() for each bot
    Strategy->>Trip: planBestTrip() for idle bots
    Trip->>Pathfinding: findBestAdj() per candidate
    Strategy-->>Main: action strings per bot
    Main->>Server: WebSocket frame (JSON actions)
```

---

## Memory Model

All allocations are fixed at compile time:

| Buffer | Size | Purpose |
|--------|------|---------|
| Grid | 32x20 cells | Map layout |
| Distance maps | 32x20 x u16 per bot | BFS results |
| Bot state | 10 bots | Persistent state across rounds |
| Item list | 512 items | All shelf items |
| JSON buffer | 1 MB | Incoming frame parsing |
| Candidate list | 64 entries | Trip planning candidates |

No heap allocations occur during the game loop.

---

## Key Design Decisions

1. **Fixed buffers over dynamic allocation**: Predictable memory, no GC pauses, suitable for real-time response
2. **BFS over A-star**: Grid is small enough (max 28x18) that BFS is fast and guarantees optimal paths
3. **Greedy trip planning**: Full TSP is NP-hard; capped to 10 candidates with all permutations (max 6 for 3-item trips)
4. **Centralized orchestrator**: Single `decideActions()` call coordinates all bots, preventing duplicate item assignments
5. **Offset detection**: Position mismatch tracking compensates for server's 1-round action delay
