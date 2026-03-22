# Vision

NM i AI (Norwegian AI Championship) is a competitive AI platform at [app.ainm.no](https://app.ainm.no). This repository contains solutions for all four challenges, with active development on two.

---

## Purpose

Build competition-grade AI systems across four distinct domains: computer vision, probabilistic prediction, agentic accounting, and game AI. The goal is to maximize leaderboard scores through algorithmic innovation, hyperparameter optimization, and autonomous experimentation infrastructure.

---

## Boundaries

- **In scope:** All four challenges, shared infrastructure, autonomous optimization systems
- **Out of scope:** Production deployment, multi-user support, long-term maintenance

---

## Challenge Portfolio

| Challenge | Domain | Status | Approach |
|-----------|--------|--------|----------|
| NorgesGruppen Data | Object Detection | Active, submitted | YOLOv8x + ConvNeXt-V2 ensemble |
| Astar Island | Probabilistic Prediction | Active, autonomous | Bayesian calibration + simulator + autoloop |
| Tripletex | AI Accounting Agent | Not started | LLM agent + API integration |
| Grocery Bot | Game AI | Skipped | Not scored, warm-up only |

---

## Key Design Principles

1. **Autonomous optimization over manual tuning** -- autoloop runs 160k+ experiments/hour
2. **Ensemble over single model** -- WBF-fused YOLO detectors + classifier re-ranking
3. **Bayesian over frequentist** -- hierarchical priors with regime-conditional calibration
4. **Iterate within round window** -- re-submit improved predictions during ~165 min rounds
