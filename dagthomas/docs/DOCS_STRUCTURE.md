# Documentation Structure -- NM i AI

Challenge documentation served via MCP at `https://mcp-docs.ainm.no/mcp`.

## Challenge Documentation (`/docs`)

### astar-island/
*Challenge: Viking civilisation prediction (40x40x6 probability tensor)*
- `overview.md`: Challenge description & rules
- `mechanics.md`: Simulator mechanics & cell types
- `endpoint.md`: API reference (`https://api.ainm.no/astar-island`)
- `scoring.md`: Entropy-weighted KL divergence scoring (0-100)
- `quickstart.md`: Getting started guide

### norgesgruppen-data/
*Challenge: Grocery product detection on store shelves*
- `overview.md`: Challenge description & rules
- `submission.md`: Submission format, Docker sandbox, blocked imports
- `scoring.md`: 70% detection mAP + 30% classification mAP (mAP@0.5)
- `examples.md`: Example predictions & edge cases

### tripletex/
*Challenge: AI accounting agent via Tripletex API*
- `overview.md`: Challenge description & rules
- `endpoint.md`: HTTPS `/solve` endpoint spec
- `scoring.md`: Correctness x tier multiplier + efficiency bonus
- `examples.md`: Example tasks & responses
- `sandbox.md`: Sandbox environment details

---

## Solution Documentation (`/docs/0*`)

### 00-context/
*WHY and WHAT EXISTS -- system-level understanding*
- `vision.md`: Project purpose, challenge portfolio, design principles
- `system-state.md`: Running processes, data volumes, scores, external dependencies
- `architecture.md`: Repository layout, system diagrams, technology stack

### 01-product/
*WHAT the product must do -- requirements*
- `prd.md`: Per-challenge objectives, scoring, constraints, requirements checklist

### 02-features/
*HOW features are designed & built*

#### astar-island-prediction/
- `feature-spec.md`: System overview, subsystems, user flows, edge cases
- `tech-design.md`: Data models, API endpoints, pipeline diagram, file map
- `autoloop-deep-dive.md`: 44-parameter space, FastHarness pre-computation, evaluation pipeline, optimization loop, perturbation strategy, experiment log format
- `gpu-simulator.md`: GPUSimulator internals (batched CUDA), 17 simulator params, CMA-ES fitting (GT + observations), regime warm starts, transfer learning, adaptive alpha
- `prediction-pipeline.md`: 11-stage step-by-step walkthrough of `predict_gemini.py`
- `calibration-model.md`: Hierarchical prior (fine/coarse/base/global), feature key definition, training, inference, regime-conditional weighting
- `daemon-system.md`: Round monitor, submission pipeline, GPU resubmission loop, calibration download, error handling
- `exploration-system.md`: Grid/entropy strategies, data structures (ObservationAccumulator, GlobalMultipliers, FeatureKeyBuckets), overlay functions, terrain openness diffusion
- `multi-researcher.md`: LLM research agent (Gemini Pro+Flash), orchestration loop, idea evaluation, knowledge base, integration path

#### norgesgruppen-detection/
- `feature-spec.md`: Two-stage pipeline overview, training pipeline, acceptance criteria
- `tech-design.md`: Inference stages (YOLO -> WBF -> ConvNeXt-V2 -> blend), training architecture, sandbox compatibility, timing budget, file map

#### tripletex-agent/
- `feature-spec.md`: Challenge overview, scoring, task execution flow (not started)

### 04-process/
*HOW to work with this system*
- `operations.md`: Full system startup, training, submission, monitoring, troubleshooting
- `dev-workflow.md`: Daily development loops, pre-submission checklist, general rules
