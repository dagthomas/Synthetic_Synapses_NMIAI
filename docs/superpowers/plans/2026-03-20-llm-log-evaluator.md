# LLM Log Evaluator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an "Evaluate" button to each Solve Log entry that uses an LLM to judge whether the logs accomplished exactly what the prompt asked, and if not, auto-fixes code and re-runs in a loop until the LLM says pass.

**Architecture:** New SSE endpoint `POST /api/logs/evaluate` streams evaluation + auto-fix loop events. The LLM evaluator is a new function in `auto_fixer.py` that sends prompt + tool_calls + api_log + agent_response to Gemini and gets a pass/fail verdict. The loop: evaluate → if fail → analyze + fix code → rerun prompt via `/solve-debug` → re-evaluate → repeat (max 5 iterations). Frontend adds an "Evaluate" button per LogCard that streams results inline.

**Tech Stack:** Python/FastAPI (SSE streaming), Gemini 2.5 Flash (LLM eval), React/TypeScript (frontend)

---

### Task 1: Add LLM evaluation function to auto_fixer.py

**Files:**
- Modify: `LARS/Tripletex/auto_fixer.py` (append after `apply_fixes()`, ~line 402)

- [ ] **Step 1: Add `llm_evaluate_logs()` function**

Append this function after `apply_fixes()` at line 402:

```python
def llm_evaluate_logs(prompt: str, tool_calls: list, api_log: list,
                      agent_response: str) -> dict:
    """Use Gemini to evaluate whether logs accomplished what the prompt asked.

    Returns {"passed": bool, "reasoning": str, "issues": list[str]}.
    """
    from google import genai
    from config import GOOGLE_API_KEY

    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Build a structured summary of what happened
    tc_summary = []
    for i, tc in enumerate(tool_calls[:30], 1):
        ok = tc.get("result", {}).get("ok", "?")
        status = "OK" if ok else "ERROR"
        args_str = json.dumps(tc.get("args", {}), ensure_ascii=False)[:300]
        result_str = ""
        if tc.get("result"):
            if tc["result"].get("error"):
                result_str = f" → Error: {tc['result']['error'][:200]}"
            elif tc["result"].get("data"):
                result_str = f" → {str(tc['result']['data'])[:200]}"
        tc_summary.append(f"  #{i} [{status}] {tc.get('tool', '?')}({args_str}){result_str}")

    api_summary = []
    for e in api_log[:50]:
        status = e.get("status", "?")
        api_summary.append(f"  {e.get('method', '?')} {status} {e.get('url', '?')}")

    context = f"""\
=== ORIGINAL PROMPT ===
{prompt}

=== TOOL CALLS ({len(tool_calls)} total) ===
{chr(10).join(tc_summary) if tc_summary else '  (none)'}

=== API CALLS ({len(api_log)} total) ===
{chr(10).join(api_summary) if api_summary else '  (none)'}

=== AGENT FINAL RESPONSE ===
{agent_response[:1000] if agent_response else '(empty)'}
"""

    system_prompt = """\
You are a strict QA evaluator for a Tripletex accounting API integration agent.

Given an original task prompt and the complete execution logs (tool calls, API calls, agent response), determine whether the agent CORRECTLY and COMPLETELY accomplished what the prompt asked.

Check for:
1. Did the agent create/update/delete the correct entity type?
2. Were all fields from the prompt set correctly (names, amounts, dates, etc.)?
3. Were all required sub-steps completed (e.g., for an invoice: customer + product + order + invoice)?
4. Did any tool calls or API calls fail that prevented completion?
5. Does the agent response confirm successful completion?

Be STRICT: if the prompt asked for specific values and you can't confirm they were set correctly from the logs, mark it as FAIL.

Respond in this exact JSON format (no markdown, no code fences):
{"passed": true/false, "reasoning": "2-3 sentence explanation", "issues": ["issue 1", "issue 2"]}

If passed, issues should be an empty list.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=context,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
        ),
    )

    # Parse JSON response
    text = response.text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {"passed": False, "reasoning": f"LLM returned unparseable response: {text[:200]}", "issues": ["parse_error"]}

    return {
        "passed": bool(result.get("passed", False)),
        "reasoning": str(result.get("reasoning", "")),
        "issues": list(result.get("issues", [])),
    }
```

- [ ] **Step 2: Verify the import is available**

Run: `cd /c/Users/larsh/source/repos/AINM/LARS/Tripletex && python -c "from auto_fixer import llm_evaluate_logs; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add LARS/Tripletex/auto_fixer.py
git commit -m "feat: add LLM log evaluation function to auto_fixer"
```

---

### Task 2: Add SSE endpoint for evaluate + auto-fix loop

**Files:**
- Modify: `LARS/Tripletex/dashboard/app.py` (add new endpoint after the auto-fix section, ~line 435)

- [ ] **Step 1: Add request model and endpoint**

After the `/api/auto-fix/apply` endpoint (line 434), add:

```python
class LogEvalRequest(BaseModel):
    solve_log_id: int
    max_iterations: int = Field(5, ge=1, le=10)
    auto_apply: bool = True


@app.post("/api/logs/evaluate")
async def evaluate_log(req: LogEvalRequest):
    """SSE stream: LLM-evaluate a solve log, auto-fix + rerun loop until pass."""
    base_url, token, agent_url = _get_credentials()
    if not base_url or not token:
        return JSONResponse(
            {"error": "Set TRIPLETEX_BASE_URL and TRIPLETEX_SESSION_TOKEN env vars"},
            status_code=400,
        )

    # Fetch the solve log
    solve_log = db.get_solve_log_by_id(req.solve_log_id)
    if not solve_log:
        return JSONResponse({"error": f"Solve log {req.solve_log_id} not found"}, 404)

    def _generate():
        import time as _time
        from auto_fixer import (
            llm_evaluate_logs, build_error_report, get_fix_suggestions,
            parse_fixes, apply_fixes, _find_relevant_sources,
        )

        prompt = solve_log["prompt"]
        tool_calls = json.loads(solve_log.get("tool_calls_json") or "[]")
        api_log = json.loads(solve_log.get("api_log_json") or "[]")
        agent_response = solve_log.get("agent_response", "")
        task_type = solve_log.get("task_type", "unknown")

        for iteration in range(1, req.max_iterations + 1):
            # Phase: evaluating
            yield f"data: {json.dumps({'type': 'phase', 'phase': 'evaluating', 'message': f'LLM evaluating logs (iteration {iteration}/{req.max_iterations})...'})}\n\n"

            try:
                verdict = llm_evaluate_logs(prompt, tool_calls, api_log, agent_response)
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'LLM evaluation failed: {e}'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            yield f"data: {json.dumps({'type': 'eval_verdict', 'iteration': iteration, 'passed': verdict['passed'], 'reasoning': verdict['reasoning'], 'issues': verdict['issues'], 'tool_calls': tool_calls[:20], 'api_log_summary': [{'method': e.get('method'), 'status': e.get('status'), 'url': e.get('url', '')[-80:]} for e in api_log[:30]], 'agent_response': agent_response[:500]})}\n\n"

            # If passed, we're done
            if verdict["passed"]:
                yield f"data: {json.dumps({'type': 'phase', 'phase': 'done', 'message': f'PASS — LLM confirmed logs match prompt (iteration {iteration}).'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # If last iteration, report failure
            if iteration == req.max_iterations:
                yield f"data: {json.dumps({'type': 'phase', 'phase': 'done', 'message': f'FAIL — Max iterations ({req.max_iterations}) reached. Issues remain.'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Phase: analyzing for fixes
            yield f"data: {json.dumps({'type': 'phase', 'phase': 'analyzing', 'message': 'Analyzing failures and generating code fixes...'})}\n\n"

            try:
                # Build an error-report-like context from the LLM verdict
                report_lines = [
                    f"=== LLM EVAL REPORT: {task_type} (iteration {iteration}) ===",
                    f"VERDICT: FAIL",
                    f"REASONING: {verdict['reasoning']}",
                    f"ISSUES: {'; '.join(verdict['issues'])}",
                    "",
                    f"PROMPT: {prompt}",
                    "",
                    "=== TOOL CALLS ===",
                ]
                for i, tc in enumerate(tool_calls, 1):
                    ok = tc.get("result", {}).get("ok", "?")
                    report_lines.append(f"  #{i} [{'OK' if ok else 'ERROR'}] {tc.get('tool', '?')}({json.dumps(tc.get('args', {}), ensure_ascii=False)[:200]})")
                    if tc.get("result") and not tc.get("result", {}).get("ok"):
                        report_lines.append(f"       Error: {tc.get('result', {}).get('error', '')[:300]}")
                report_lines.append("")
                report_lines.append(f"AGENT RESPONSE: {agent_response[:500]}")
                report = "\n".join(report_lines)

                sources = _find_relevant_sources(tool_calls, task_type)
                fix_text = get_fix_suggestions(report, sources)
                fixes = parse_fixes(fix_text)
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Analysis failed: {e}'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            yield f"data: {json.dumps({'type': 'fixes', 'iteration': iteration, 'raw_text': fix_text, 'parsed_fixes': fixes, 'report': report})}\n\n"

            if not fixes:
                yield f"data: {json.dumps({'type': 'phase', 'phase': 'done', 'message': f'No actionable fixes found. Stopping after iteration {iteration}.'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Phase: applying fixes
            if req.auto_apply:
                yield f"data: {json.dumps({'type': 'phase', 'phase': 'applying', 'message': f'Applying {len(fixes)} fix(es)...'})}\n\n"
                try:
                    import os as _os
                    base_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
                    apply_results = apply_fixes(fixes, base_dir)
                    yield f"data: {json.dumps({'type': 'applied', 'iteration': iteration, 'results': apply_results})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Apply failed: {e}'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

            # Phase: rerunning
            yield f"data: {json.dumps({'type': 'phase', 'phase': 'rerunning', 'message': f'Re-running prompt via /solve-debug (iteration {iteration + 1})...'})}\n\n"

            try:
                import requests as _requests
                payload = {
                    "prompt": prompt,
                    "files": [],
                    "tripletex_credentials": {
                        "base_url": base_url,
                        "session_token": token,
                    },
                }
                headers = {"Content-Type": "application/json"}
                api_key = os.environ.get("AGENT_API_KEY", "")
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                resp = _requests.post(
                    f"{agent_url}/solve-debug",
                    params={"source": "log_eval"},
                    json=payload, headers=headers, timeout=300,
                )

                if resp.status_code == 200:
                    result = resp.json()
                    tool_calls = result.get("tool_calls", [])
                    api_log = result.get("api_log", [])
                    agent_response = result.get("agent_response", "")
                else:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Agent returned {resp.status_code}: {resp.text[:300]}'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                yield f"data: {json.dumps({'type': 'rerun_result', 'iteration': iteration + 1, 'api_calls': result.get('api_calls', 0), 'api_errors': result.get('api_errors', 0), 'tool_count': len(tool_calls), 'agent_response': agent_response[:500]})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Rerun failed: {e}'})}\n\n"
                yield "data: [DONE]\n\n"
                return

        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")
```

- [ ] **Step 2: Add `get_solve_log_by_id()` to db.py**

Open `LARS/Tripletex/dashboard/db.py` and add this function:

```python
def get_solve_log_by_id(log_id: int) -> dict | None:
    """Fetch a single solve_log by its ID."""
    with closing(get_conn()) as conn:
        row = conn.execute("SELECT * FROM solve_logs WHERE id = ?", (log_id,)).fetchone()
        if not row:
            return None
        return dict(row)
```

- [ ] **Step 3: Verify the endpoint loads**

Run: `cd /c/Users/larsh/source/repos/AINM/LARS/Tripletex && python -c "from dashboard.app import app; print([r.path for r in app.routes if 'evaluate' in str(r.path)])"`
Expected: `['/api/logs/evaluate']`

- [ ] **Step 4: Commit**

```bash
git add LARS/Tripletex/dashboard/app.py LARS/Tripletex/dashboard/db.py
git commit -m "feat: add SSE endpoint for LLM log evaluation + auto-fix loop"
```

---

### Task 3: Add TypeScript types for evaluation events

**Files:**
- Modify: `LARS/Tripletex/dashboard/frontend/src/types/api.ts` (append at end)

- [ ] **Step 1: Add evaluation event types**

Append at end of file (after the `SolveLog` interface, line 229):

```typescript

// Log evaluation types
export type LogEvalEvent =
  | { type: "phase"; phase: string; message: string }
  | { type: "eval_verdict"; iteration: number; passed: boolean; reasoning: string; issues: string[]; tool_calls: ToolCall[]; api_log_summary: { method: string; status: number; url: string }[]; agent_response: string }
  | { type: "fixes"; iteration: number; raw_text: string; parsed_fixes: AutoFixParsedFix[]; report: string }
  | { type: "applied"; iteration: number; results: AutoFixApplyResult[] }
  | { type: "rerun_result"; iteration: number; api_calls: number; api_errors: number; tool_count: number; agent_response: string }
  | { type: "error"; message: string }
```

- [ ] **Step 2: Commit**

```bash
git add LARS/Tripletex/dashboard/frontend/src/types/api.ts
git commit -m "feat: add TypeScript types for log evaluation events"
```

---

### Task 4: Add streaming API function for log evaluation

**Files:**
- Modify: `LARS/Tripletex/dashboard/frontend/src/lib/api.ts` (append at end)

- [ ] **Step 1: Add `streamLogEval()` function**

Append at end of file (after the `applyFixes` export, line 169):

```typescript

// Log Evaluation
export function streamLogEval(
  solveLogId: number,
  maxIterations: number,
  onEvent: (event: import("@/types/api").LogEvalEvent) => void,
  onDone: () => void,
  onError: (err: string) => void,
): AbortController {
  const controller = new AbortController()
  fetch("/api/logs/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ solve_log_id: solveLogId, max_iterations: maxIterations, auto_apply: true }),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        const body = await res.json().catch(() => ({ error: res.statusText }))
        onError(body.error || res.statusText)
        return
      }
      const reader = res.body?.getReader()
      if (!reader) { onError("No response body"); return }
      const decoder = new TextDecoder()
      let buffer = ""
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6).trim()
            if (data === "[DONE]") { onDone(); return }
            try { onEvent(JSON.parse(data)) } catch { /* skip malformed */ }
          }
        }
      }
      onDone()
    })
    .catch((err) => {
      if (err.name !== "AbortError") onError(String(err))
    })
  return controller
}
```

- [ ] **Step 2: Commit**

```bash
git add LARS/Tripletex/dashboard/frontend/src/lib/api.ts
git commit -m "feat: add streaming API function for log evaluation"
```

---

### Task 5: Add "Evaluate" button and inline results to Solve Logs panel

**Files:**
- Modify: `LARS/Tripletex/dashboard/frontend/src/components/panels/logs-panel.tsx`

- [ ] **Step 1: Add imports**

At line 1, update imports to include the new types and API:

```typescript
import { useState, useCallback, useRef } from "react"
```

Add to the lucide imports (line 11-28), add after `Globe`:
```typescript
  Play,
  RotateCcw,
  Brain,
  Loader2,
  Square,
```

Add after the existing import of `deleteAllLogs` (line 3):
```typescript
import { deleteAllLogs, streamLogEval } from "@/lib/api"
import type { SolveLog, ToolCall, LogEvalEvent, AutoFixParsedFix, AutoFixApplyResult } from "@/types/api"
```

Remove the now-duplicate individual imports of `SolveLog` and `ToolCall` from line 4.

- [ ] **Step 2: Add evaluation state and handler to LogCard**

Replace the `LogCard` function definition (line 204-337) with the version below. The key additions are: evaluation state, the `handleEvaluate` callback, a "Evaluate" button in the header, and an `EvalResults` section in the expanded view.

```typescript
function LogCard({ log, index }: { log: SolveLog; index: number }) {
  const [expanded, setExpanded] = useState(false)
  const toolCalls = parseToolCalls(log.tool_calls_json)
  const files = parseFiles(log.files_json)
  const hasErrors = (log.api_errors || 0) > 0
  const failedTools = toolCalls.filter(tc => tc.result?.ok === false)

  // Evaluation state
  const [evalRunning, setEvalRunning] = useState(false)
  const [evalPhase, setEvalPhase] = useState("")
  const [evalPhaseMsg, setEvalPhaseMsg] = useState("")
  const [verdicts, setVerdicts] = useState<Array<{ iteration: number; passed: boolean; reasoning: string; issues: string[] }>>([])
  const [fixes, setFixes] = useState<Array<{ iteration: number; raw_text: string; parsed_fixes: AutoFixParsedFix[]; report: string }>>([])
  const [applied, setApplied] = useState<Array<{ iteration: number; results: AutoFixApplyResult[] }>>([])
  const [rerunResults, setRerunResults] = useState<Array<{ iteration: number; api_calls: number; api_errors: number; tool_count: number; agent_response: string }>>([])
  const [evalError, setEvalError] = useState("")
  const controllerRef = useRef<AbortController | null>(null)

  const handleEvaluate = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    if (!log.id) return

    // Reset state
    setEvalRunning(true)
    setEvalPhase("")
    setEvalPhaseMsg("")
    setVerdicts([])
    setFixes([])
    setApplied([])
    setRerunResults([])
    setEvalError("")
    setExpanded(true)

    controllerRef.current = streamLogEval(
      log.id,
      5,
      (event: LogEvalEvent) => {
        switch (event.type) {
          case "phase":
            setEvalPhase(event.phase)
            setEvalPhaseMsg(event.message)
            break
          case "eval_verdict":
            setVerdicts(prev => [...prev, { iteration: event.iteration, passed: event.passed, reasoning: event.reasoning, issues: event.issues }])
            break
          case "fixes":
            setFixes(prev => [...prev, { iteration: event.iteration, raw_text: event.raw_text, parsed_fixes: event.parsed_fixes, report: event.report }])
            break
          case "applied":
            setApplied(prev => [...prev, { iteration: event.iteration, results: event.results }])
            break
          case "rerun_result":
            setRerunResults(prev => [...prev, { iteration: event.iteration, api_calls: event.api_calls, api_errors: event.api_errors, tool_count: event.tool_count, agent_response: event.agent_response }])
            break
          case "error":
            setEvalError(event.message)
            break
        }
      },
      () => setEvalRunning(false),
      (err) => { setEvalError(err); setEvalRunning(false) },
    )
  }, [log.id])

  const handleStopEval = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    controllerRef.current?.abort()
    setEvalRunning(false)
  }, [])

  const lastVerdict = verdicts[verdicts.length - 1]
  const hasEvalResults = verdicts.length > 0

  return (
    <Card
      className={cn(
        "shadow-sm transition-all duration-200 overflow-hidden",
        hasErrors || failedTools.length > 0
          ? "border-l-4 border-l-amber-500"
          : "border-l-4 border-l-emerald-500"
      )}
      style={{ animationDelay: `${index * 30}ms` }}
    >
      <CardContent className="p-3">
        {/* Header row */}
        <div
          className="flex items-center gap-2 cursor-pointer"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />
          ) : (
            <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0" />
          )}

          {hasErrors || failedTools.length > 0 ? (
            <div className="h-6 w-6 rounded-full bg-amber-50 dark:bg-amber-900/20 flex items-center justify-center shrink-0">
              <AlertTriangle className="h-3.5 w-3.5 text-amber-500" />
            </div>
          ) : (
            <div className="h-6 w-6 rounded-full bg-emerald-50 dark:bg-emerald-900/20 flex items-center justify-center shrink-0">
              <CheckCircle className="h-3.5 w-3.5 text-emerald-500" />
            </div>
          )}

          <span className="text-[12px] text-muted-foreground font-mono shrink-0">
            {formatTime(log.created_at)}
          </span>

          {log.task_type && (
            <Badge variant="outline" className="text-[10px] shrink-0">
              <Tag className="h-2.5 w-2.5 mr-1" />
              {log.task_type}
            </Badge>
          )}

          <span className="text-[12px] truncate flex-1">{log.prompt}</span>

          {/* Eval verdict badge */}
          {hasEvalResults && !evalRunning && (
            <Badge
              variant={lastVerdict?.passed ? "default" : "destructive"}
              className={cn("text-[10px] shrink-0", lastVerdict?.passed && "bg-emerald-600")}
            >
              <Brain className="h-2.5 w-2.5 mr-1" />
              {lastVerdict?.passed ? "PASS" : `FAIL (${verdicts.length}x)`}
            </Badge>
          )}

          {/* Evaluate / Stop button */}
          {log.id && (
            evalRunning ? (
              <button
                onClick={handleStopEval}
                className="flex items-center gap-1 px-2 py-1 text-[10px] font-medium rounded bg-red-500/10 text-red-600 hover:bg-red-500/20 transition-colors shrink-0"
              >
                <Square className="h-2.5 w-2.5" />
                Stop
              </button>
            ) : (
              <button
                onClick={handleEvaluate}
                className="flex items-center gap-1 px-2 py-1 text-[10px] font-medium rounded bg-violet-500/10 text-violet-600 hover:bg-violet-500/20 transition-colors shrink-0"
              >
                {hasEvalResults ? <RotateCcw className="h-2.5 w-2.5" /> : <Play className="h-2.5 w-2.5" />}
                {hasEvalResults ? "Re-eval" : "Evaluate"}
              </button>
            )
          )}

          <Badge variant="secondary" className="text-[10px] tabular-nums shrink-0">
            <Clock className="h-2.5 w-2.5 mr-1" />
            {(log.elapsed_seconds || 0).toFixed(1)}s
          </Badge>
          <Badge variant="secondary" className="text-[10px] tabular-nums shrink-0">
            <Zap className="h-2.5 w-2.5 mr-1" />
            {log.api_calls || 0}
          </Badge>
          {log.tool_count != null && (
            <Badge variant="secondary" className="text-[10px] tabular-nums shrink-0">
              <Wrench className="h-2.5 w-2.5 mr-1" />
              {log.tool_count}
            </Badge>
          )}
          {(log.api_errors || 0) > 0 && (
            <Badge variant="destructive" className="text-[10px] tabular-nums shrink-0">
              {log.api_errors} err
            </Badge>
          )}
        </div>

        {/* Eval progress spinner */}
        {evalRunning && (
          <div className="mt-2 ml-4 flex items-center gap-2 text-[12px] text-violet-600">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            <span>{evalPhaseMsg || evalPhase || "Starting evaluation..."}</span>
          </div>
        )}

        {/* Expanded details */}
        {expanded && (
          <div className="mt-3 ml-4 space-y-3">
            {/* Eval results section */}
            {(hasEvalResults || evalError) && (
              <EvalResultsSection
                verdicts={verdicts}
                fixes={fixes}
                applied={applied}
                rerunResults={rerunResults}
                error={evalError}
              />
            )}

            {/* Full prompt */}
            <Section icon={<MessageSquare className="h-3 w-3" />} title="Prompt" copyText={log.prompt}>
              <div className="bg-muted/30 rounded-lg p-3 text-[12px] whitespace-pre-wrap break-words">
                {log.prompt}
              </div>
            </Section>

            {/* Files */}
            {files.length > 0 && (
              <div className="flex gap-1 flex-wrap">
                {files.map((f, i) => (
                  <Badge key={i} variant="outline" className="text-[10px]">
                    <FileText className="h-2.5 w-2.5 mr-1" />
                    {f}
                  </Badge>
                ))}
              </div>
            )}

            {/* Agent response */}
            {log.agent_response && (
              <Section icon={<CheckCircle className="h-3 w-3" />} title="Agent Response" copyText={log.agent_response}>
                <div className="bg-muted/30 rounded-lg p-3 text-[12px] whitespace-pre-wrap break-words max-h-[300px] overflow-y-auto">
                  {log.agent_response}
                </div>
              </Section>
            )}

            {/* Tool calls */}
            {toolCalls.length > 0 && (
              <Section icon={<Wrench className="h-3 w-3" />} title={`Tool Calls (${toolCalls.length})`}>
                <div className="space-y-2">
                  {toolCalls.map((tc, j) => (
                    <ToolCallCard key={j} tc={tc} index={j} />
                  ))}
                </div>
              </Section>
            )}

            {/* API log */}
            {log.api_log_json && <ApiLogSection json={log.api_log_json} />}

            <Separator />

            {/* Meta */}
            <div className="flex gap-4 text-[10px] text-muted-foreground flex-wrap">
              <span>ID: {log.request_id?.slice(0, 8)}</span>
              <span>Base: {log.base_url}</span>
              {log.source && <span>Source: {log.source}</span>}
              {log.task_type && <span>Task: {log.task_type}</span>}
              {log.tool_count != null && <span>Tools loaded: {log.tool_count}</span>}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 3: Add EvalResultsSection component**

Add this component before the `Section` component (before line 339):

```typescript
function EvalResultsSection({
  verdicts,
  fixes,
  applied,
  rerunResults,
  error,
}: {
  verdicts: Array<{ iteration: number; passed: boolean; reasoning: string; issues: string[] }>
  fixes: Array<{ iteration: number; raw_text: string; parsed_fixes: AutoFixParsedFix[]; report: string }>
  applied: Array<{ iteration: number; results: AutoFixApplyResult[] }>
  rerunResults: Array<{ iteration: number; api_calls: number; api_errors: number; tool_count: number; agent_response: string }>
  error: string
}) {
  const [expandedIter, setExpandedIter] = useState<number | null>(null)

  return (
    <Section icon={<Brain className="h-3 w-3" />} title="LLM Evaluation">
      <div className="space-y-2">
        {verdicts.map((v) => {
          const iterFixes = fixes.find(f => f.iteration === v.iteration)
          const iterApplied = applied.find(a => a.iteration === v.iteration)
          const iterRerun = rerunResults.find(r => r.iteration === v.iteration + 1)
          const isExpanded = expandedIter === v.iteration

          return (
            <div
              key={v.iteration}
              className={cn(
                "rounded-lg border text-[12px] overflow-hidden",
                v.passed ? "border-emerald-200 dark:border-emerald-900" : "border-red-200 dark:border-red-900"
              )}
            >
              {/* Iteration header */}
              <div
                className={cn(
                  "flex items-center gap-2 px-3 py-1.5 cursor-pointer",
                  v.passed ? "bg-emerald-50 dark:bg-emerald-950/30" : "bg-red-50 dark:bg-red-950/30"
                )}
                onClick={() => setExpandedIter(isExpanded ? null : v.iteration)}
              >
                {isExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                <Badge variant={v.passed ? "default" : "destructive"} className={cn("text-[10px]", v.passed && "bg-emerald-600")}>
                  {v.passed ? "PASS" : "FAIL"}
                </Badge>
                <span className="text-[10px] text-muted-foreground">Iteration {v.iteration}</span>
                <span className="flex-1 text-[11px] truncate">{v.reasoning}</span>
                {iterFixes && (
                  <Badge variant="outline" className="text-[9px]">{iterFixes.parsed_fixes.length} fixes</Badge>
                )}
                {iterRerun && (
                  <Badge variant="secondary" className="text-[9px]">rerun: {iterRerun.api_calls} calls</Badge>
                )}
              </div>

              {/* Expanded details */}
              {isExpanded && (
                <div className="px-3 py-2 space-y-2 border-t border-border/30">
                  {/* Reasoning */}
                  <div>
                    <span className="text-[9px] uppercase tracking-wide text-muted-foreground font-semibold">Reasoning</span>
                    <p className="text-[11px] mt-0.5">{v.reasoning}</p>
                  </div>

                  {/* Issues */}
                  {v.issues.length > 0 && (
                    <div>
                      <span className="text-[9px] uppercase tracking-wide text-muted-foreground font-semibold">Issues</span>
                      <ul className="mt-0.5 space-y-0.5">
                        {v.issues.map((issue, i) => (
                          <li key={i} className="text-[11px] text-red-600 flex items-start gap-1">
                            <XCircle className="h-3 w-3 mt-0.5 shrink-0" />
                            {issue}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Fixes */}
                  {iterFixes && iterFixes.parsed_fixes.length > 0 && (
                    <div>
                      <span className="text-[9px] uppercase tracking-wide text-muted-foreground font-semibold">Code Fixes</span>
                      <div className="mt-1 space-y-1">
                        {iterFixes.parsed_fixes.map((fix, i) => (
                          <div key={i} className="rounded border border-border/50 p-2 bg-muted/20">
                            <div className="flex items-center gap-2 mb-1">
                              <Badge variant="outline" className="text-[9px] font-mono">{fix.file}</Badge>
                              <span className="text-[10px] text-muted-foreground">{fix.reason}</span>
                            </div>
                            <div className="grid grid-cols-2 gap-1 text-[10px] font-mono">
                              <pre className="bg-red-50 dark:bg-red-950/20 rounded p-1.5 whitespace-pre-wrap break-words max-h-[100px] overflow-y-auto">{fix.old}</pre>
                              <pre className="bg-emerald-50 dark:bg-emerald-950/20 rounded p-1.5 whitespace-pre-wrap break-words max-h-[100px] overflow-y-auto">{fix.new}</pre>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Apply results */}
                  {iterApplied && (
                    <div className="flex gap-1 flex-wrap">
                      {iterApplied.results.map((r, i) => (
                        <Badge key={i} variant={r.applied ? "default" : "destructive"} className={cn("text-[9px]", r.applied && "bg-emerald-600")}>
                          {r.file}: {r.applied ? "applied" : r.error}
                        </Badge>
                      ))}
                    </div>
                  )}

                  {/* Rerun result */}
                  {iterRerun && (
                    <div className="rounded bg-muted/30 p-2">
                      <span className="text-[9px] uppercase tracking-wide text-muted-foreground font-semibold">Rerun Result</span>
                      <div className="flex gap-3 mt-1 text-[11px]">
                        <span>API calls: {iterRerun.api_calls}</span>
                        <span>Errors: {iterRerun.api_errors}</span>
                        <span>Tools: {iterRerun.tool_count}</span>
                      </div>
                      {iterRerun.agent_response && (
                        <p className="text-[10px] text-muted-foreground mt-1 truncate">{iterRerun.agent_response}</p>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )
        })}

        {error && (
          <div className="rounded-lg border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950/30 px-3 py-2 text-[11px] text-red-600">
            <AlertTriangle className="h-3 w-3 inline mr-1" />
            {error}
          </div>
        )}
      </div>
    </Section>
  )
}
```

- [ ] **Step 4: Commit**

```bash
git add LARS/Tripletex/dashboard/frontend/src/components/panels/logs-panel.tsx
git commit -m "feat: add Evaluate button + inline LLM eval results to Solve Logs panel"
```

---

### Task 6: Build frontend and verify

**Files:**
- No new files — just build the existing frontend

- [ ] **Step 1: Build the React frontend**

Run: `cd /c/Users/larsh/source/repos/AINM/LARS/Tripletex/dashboard/frontend && npm run build`
Expected: Build succeeds with no TypeScript errors

- [ ] **Step 2: Fix any build errors**

If TypeScript errors, fix them and rebuild.

- [ ] **Step 3: Commit built assets**

```bash
git add LARS/Tripletex/dashboard/static/dist/
git commit -m "build: compile frontend with log evaluation feature"
```

---

### Task 7: End-to-end smoke test

- [ ] **Step 1: Start the dashboard**

Run: `cd /c/Users/larsh/source/repos/AINM/LARS/Tripletex && python -m uvicorn dashboard.app:app --port 8080`

- [ ] **Step 2: Verify the endpoint exists**

Run: `curl -s http://localhost:8080/api/logs?limit=1 | python -m json.tool | head -5`
Expected: Returns a solve log entry with an `id` field

- [ ] **Step 3: Test the evaluate endpoint**

Run: `curl -sN -X POST http://localhost:8080/api/logs/evaluate -H 'Content-Type: application/json' -d '{"solve_log_id": 1, "max_iterations": 1}' | head -20`
Expected: SSE events stream back with `eval_verdict` type

- [ ] **Step 4: Verify UI in browser**

Open http://localhost:8080, navigate to Solve Logs. Each log entry should have a purple "Evaluate" button. Click it and verify the evaluation streams inline.
