# Documentation Structure

```
docs/
├── 00-context/              # WHY and WHAT EXISTS
│   ├── vision.md            # Project purpose and goals
│   ├── system-state.md      # Current build state and scores
│   └── architecture.md      # Technical architecture overview
├── 01-product/
│   └── prd.md               # Game rules and scoring targets
├── 02-features/
│   ├── websocket-client/    # WebSocket protocol and game loop
│   ├── pathfinding/         # BFS pathfinding and collision avoidance
│   ├── trip-planning/       # Trip optimization (mini-TSP)
│   ├── strategy-engine/     # Per-bot decision engine
│   └── multi-bot-coordination/  # Orchestrator and dropoff priority
├── 04-process/
│   ├── dev-workflow.md      # Build, run, and debug workflow
│   └── getting-started.md   # Onboarding and setup
└── DOCS_STRUCTURE.md        # This file
```

## Mermaid Chart Guidelines

- **>3 nodes**: Use vertical layout (`flowchart TB`). Horizontal charts with many boxes become unreadable.
- **≤3 nodes**: Horizontal (`flowchart LR`) is fine.
- **Sequence diagrams**: Keep as-is (vertical by nature).
- Prefer simple node labels — move details into text below the chart.
