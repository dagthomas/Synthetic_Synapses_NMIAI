# Documentation Structure

```
docs/
├── 00-context/
│   ├── vision.md                      # Challenge overview and approach
│   ├── architecture.md                # System architecture and data flow
│   └── war-stories.md                 # Failures, breakthroughs, scoring analysis
├── 01-product/
│   └── prd.md                         # Competition rules, scoring, task types
├── 02-features/
│   ├── agent-pipeline/                # Gemini LLM agent with Google ADK
│   ├── tool-system/                   # 28 modules, 137 tool functions
│   ├── task-router/                   # Deterministic 30-type classifier
│   ├── simulator/                     # Prompt generation + verification + scoring
│   ├── dashboard/                     # React + FastAPI eval dashboard
│   └── auto-fixer/                    # LLM-driven self-repair from competition logs
├── 04-process/
│   └── getting-started.md             # Setup, running, testing
└── DOCS_STRUCTURE.md                  # This file
```

## Mermaid Chart Guidelines

- **>3 nodes**: Use vertical layout (`flowchart TB`).
- **<=3 nodes**: Horizontal (`flowchart LR`) is fine.
- **Sequence diagrams**: Keep as-is (vertical by nature).
