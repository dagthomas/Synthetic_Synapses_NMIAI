---
name: docs-generator
description: Automatically generate and maintain project documentation following the /docs hierarchy. Use when creating new feature docs, updating context docs, scaffolding doc structures, or auditing documentation completeness.
user_invocable: true
invocation: /docs
argument_description: "Command and optional target, e.g. 'feature leave-management', 'context system-state', 'log decision', 'audit'"
---

# Documentation Generator

Generate and maintain documentation following the `docs/DOCS_STRUCTURE.md` hierarchy. All docs live under `docs/` with five numbered tiers.

## Directory Structure

```
docs/
├── 00-context/       # WHY and WHAT EXISTS — system-level understanding
├── 01-product/       # WHAT the product must do — requirements
├── 02-features/      # HOW features are designed & built — per-feature folders
├── 04-process/       # HOW to work with this system — workflows
└── DOCS_STRUCTURE.md # This standard
```

> **Note:** `03-logs/` is maintained manually and is excluded from this generator.

## Commands

Parse the user's argument to determine the action:

| Argument Pattern | Action |
|---|---|
| `feature <name>` | Create/update a feature folder in `02-features/` |
| `context <topic>` | Create/update a file in `00-context/` |
| `product` | Create/update `01-product/prd.md` |
| `process <topic>` | Create/update a file in `04-process/` |
| `audit` | Scan all docs and report gaps/staleness |
| *(no argument)* | Ask user what they want to document |

---

## 1. Feature Documentation (`feature <name>`)

Create `docs/02-features/feature-<name>/` with up to 4 files. Always read the corresponding backend service code and frontend routes BEFORE writing docs — never fabricate content.

### Research Phase

Before writing any feature doc, gather real implementation details:

1. **Find the backend service**: Search `backend/internal/<domain>/` for models, service files, and handlers
2. **Find the frontend routes**: Search `frontend/src/routes/` for the feature's pages and `+page.server.ts` loaders
3. **Find proto definitions**: Check `proto/hrm/v1/` for relevant `.proto` files
4. **Check existing docs**: Read any existing files in the feature folder first

### File Templates

#### `feature-spec.md` — User intent & acceptance criteria

```markdown
# <Feature Name>

<1-2 sentence description of what this feature does for users.>

---

## Features

- <Bullet list of capabilities>

---

## User Flows

### <Flow Name>

```mermaid
<stateDiagram, sequenceDiagram, or flowchart showing the primary user flow>
```

---

## Acceptance Criteria

- [ ] <Testable criterion>
- [ ] <Testable criterion>

---

## Edge Cases

- <Edge case and expected behavior>

---

## Norwegian Compliance

<Only include if the feature has AML/labor-law implications. Otherwise omit this section.>
```

#### `tech-design.md` — Architecture & implementation

```markdown
# <Feature Name> - Technical Design

<Brief description of the technical approach.>

---

## Data Models

<Show actual GORM model structs or JSON schema from the codebase. Do NOT invent fields.>

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/<resource>` | List |
| POST | `/api/v1/<resource>` | Create |

---

## Service Architecture

```mermaid
<Sequence or flowchart diagram showing service interactions>
```

---

## Multi-Tenancy

<How tenant isolation is enforced for this feature — automatic via BaseTenantModel, RLS, storage prefixing, etc.>

---

## Security Considerations

- <RBAC roles that can access this feature>
- <Data validation rules>
```

#### `dev-tasks.md` — LLM-executable tasks

```markdown
# <Feature Name> - Development Tasks

## Prerequisites
- <Dependencies on other features/services>

## Tasks

### Task 1: <Atomic task title>
- **Files**: `backend/internal/<domain>/<file>.go`
- **Action**: <What to implement>
- **Depends on**: None

### Task 2: <Atomic task title>
- **Files**: `frontend/src/routes/<path>/+page.svelte`
- **Action**: <What to implement>
- **Depends on**: Task 1
```

#### `test-plan.md` — Validation strategy

```markdown
# <Feature Name> - Test Plan

## Unit Tests
- <What to test at the function level>

## Integration Tests
- <API endpoint tests with expected responses>

## E2E Tests
- <User flow tests>

## Multi-Tenancy Tests
- <Cross-tenant isolation verification>
```

### Behavior Rules for Features

- **New feature**: Create all 4 files. Skip `dev-tasks.md` and `test-plan.md` if the feature is already fully implemented.
- **Existing feature**: Read existing docs first. Only update files that are stale or missing. Do NOT overwrite manual edits.
- **Folder naming**: Use kebab-case, no `feature-` prefix in the folder name (e.g., `leave-management/`, not `feature-leave-management/`).

---

## 2. Context Documentation (`context <topic>`)

Files in `docs/00-context/` describe WHY the system exists and WHAT is currently built.

### Core Files

| File | Purpose | Source of truth |
|---|---|---|
| `vision.md` | Product purpose & boundaries | Product owner decisions |
| `assumptions.md` | Risks, unknowns, constraints | Team discussions |
| `system-state.md` | What is actually built & running | Running Docker services + code |

### Additional Context Files

Create topic-specific files for architectural deep-dives (e.g., `multi-tenancy.md`, `redis-caching.md`). These should:

- Describe the current state, not aspirational design
- Include Mermaid diagrams for complex flows
- Reference actual file paths and config values from the codebase
- Stay under 300 lines — split into multiple files if longer

---

## 3. Product Documentation (`product`)

`docs/01-product/prd.md` is the **single source of truth** for requirements.

When updating:
- Read the current PRD first
- Add new requirements in the appropriate section
- Mark completed requirements with status indicators
- Never remove requirements — mark them as deprecated if no longer needed

---

## 4. Process Documentation (`process <topic>`)

Files in `docs/04-process/` describe how to work with the system.

| File | Purpose |
|---|---|
| `dev-workflow.md` | Daily development loop |
| `definition-of-done.md` | When docs/code are "done" |
| `llm-prompts.md` | Canonical prompts per doc type |
| `getting-started.md` | Onboarding and setup |
| `testing.md` | Testing strategies |

---

## 5. Audit (`audit`)

Scan the documentation and report:

1. **Missing feature docs**: Check each `docs/02-features/*/` folder — which ones lack `feature-spec.md` or `tech-design.md`?
2. **Empty/stub files**: Files with fewer than 10 lines of actual content
3. **Orphaned features**: Backend services with no corresponding feature folder
4. **Stale context docs**: `00-context/` files that don't reflect current `docker-compose.yml` or service list
Output the audit as a markdown table with columns: File | Status | Issue.

---

## General Rules

1. **Read before write**: Always read existing docs and source code before generating. Never fabricate API endpoints, model fields, or configurations.
2. **Mermaid diagrams**: Use `stateDiagram-v2` for lifecycles, `sequenceDiagram` for API flows, `flowchart TB` for architecture. Keep diagrams under 40 lines.
3. **Horizontal rules**: Use `---` between major sections for visual clarity.
4. **No emoji**: Do not use emoji in documentation unless the user requests it.
5. **File paths**: When referencing code, use relative paths from project root (e.g., `backend/internal/leave/service.go`).
6. **Multi-tenancy**: Every feature doc must address tenant isolation — it is a core architectural concern.
7. **Norwegian compliance**: Note AML/labor-law implications when relevant, but don't force this section into features where it doesn't apply.
8. **Concise**: Aim for clarity over length. Feature specs: 50-150 lines. Tech designs: 80-200 lines. Dev tasks: 30-100 lines. Test plans: 30-80 lines.