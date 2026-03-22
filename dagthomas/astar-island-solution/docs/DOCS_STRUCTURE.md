# Documentation Structure

```
docs/
├── 00-context/                        # WHY and WHAT EXISTS
│   ├── vision.md                      # Case, problem, and solution overview
│   ├── architecture.md                # System architecture and data flow
│   ├── system-state.md                # Current scores and pipeline status
│   └── war-stories.md                 # Failures, breakthroughs, and lessons
├── 01-product/
│   └── prd.md                         # Game rules, scoring, constraints, targets
├── 02-features/
│   ├── autonomous-research/           # 3-agent auto-research system
│   │   ├── feature-spec.md            # Agents, compound effect, speed comparison
│   │   └── tech-design.md             # Parameter space, search strategy, coordination
│   ├── daemon-pipeline/               # 24/7 daemon with iterative re-submission
│   │   ├── feature-spec.md            # Pipeline flow, regime detection, re-submission
│   │   └── tech-design.md             # 8-stage prediction, re-submit strategy
│   ├── gpu-simulator/                 # CUDA Monte Carlo simulator + CMA-ES
│   │   ├── feature-spec.md            # CMA-ES fitting, Monte Carlo, ensemble
│   │   └── tech-design.md             # Distance decay, GPU parallelism, performance
│   ├── statistical-model/             # The 8-stage prediction pipeline
│   │   └── feature-spec.md            # Calibration, FK pooling, multipliers, vectorization
│   ├── island-explorer/               # SvelteKit 3D island viewer
│   │   └── feature-spec.md            # Three.js, weather, celestials, wildlife, Imagen
│   ├── terminal-ui/                   # Go Bubble Tea terminal dashboard
│   │   └── feature-spec.md            # 10 tabs, multi-pane, scramble animation
│   └── research-agent/                # Google ADK + Gemini autonomous agent
│       └── feature-spec.md            # ADK tools, backtest harness, execution modes
├── 04-process/
│   └── getting-started.md             # Setup, dependencies, how to run
└── DOCS_STRUCTURE.md                  # This file
```

## Mermaid Chart Guidelines

- **>3 nodes**: Use vertical layout (`flowchart TB`). Horizontal charts with many boxes become unreadable.
- **<=3 nodes**: Horizontal (`flowchart LR`) is fine.
- **Sequence diagrams**: Keep as-is (vertical by nature).
- Prefer simple node labels — move details into text below the chart.
