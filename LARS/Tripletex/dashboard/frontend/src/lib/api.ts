import type {
  TaskDef,
  EvalRun,
  SandboxHealth,
  SeedResult,
  CleanResult,
  Payload,
  ReplayResult,
  ToolTestData,
  ToolInfo,
  RunStats,
  Languages,
  CoverageCategory,
  SolveLog,
  TaskLiveSummary,
} from "@/types/api"

async function request<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(path, opts)
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(body.error || res.statusText)
  }
  return res.json()
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
}

// Tasks
export const fetchTasks = () => request<TaskDef[]>("/api/tasks")
export const fetchLanguages = () => request<Languages>("/api/languages")

// Runs
export const fetchRuns = (params?: { status?: string; source?: string; limit?: number }) => {
  const sp = new URLSearchParams()
  if (params?.status && params.status !== "all") sp.set("status", params.status)
  if (params?.source && params.source !== "all") sp.set("source", params.source)
  sp.set("limit", String(params?.limit ?? 200))
  return request<EvalRun[]>(`/api/runs?${sp}`)
}

export const deleteRun = (id: number) =>
  request<{ ok: boolean }>(`/api/runs/${id}`, { method: "DELETE" })

export const deleteRuns = (runIds: number[]) =>
  post<{ ok: boolean; deleted: number }>("/api/runs/delete", { run_ids: runIds })

export const cleanupStale = () =>
  post<{ ok: boolean; cleaned: number }>("/api/runs/cleanup-stale", {})

// Eval
export const startBatch = (
  taskNames: string[],
  languages: string[],
  countPerCombo: number
) =>
  post<{ message: string; total: number; skipped?: string[] }>(
    "/api/eval/batch",
    { task_names: taskNames, languages, count_per_combo: countPerCombo }
  )

// Stats
export const fetchStats = () => request<RunStats[]>("/api/stats")

// Sandbox
export const fetchSandboxHealth = () =>
  request<SandboxHealth>("/api/sandbox/health")

export const seedSandbox = (types: string[], clean = false) =>
  post<SeedResult>("/api/sandbox/seed", { types, clean })

export const cleanSandbox = () =>
  post<CleanResult>("/api/sandbox/clean", {})

// Payloads / Replay
export const fetchPayloads = (limit = 100) =>
  request<Payload[]>(`/api/payloads?limit=${limit}`)

export const replayPayloads = (filenames: string[], timeout = 180) =>
  post<ReplayResult[]>("/api/replay", { filenames, timeout })

// Tool Tests
export const runToolTests = () =>
  post<ToolTestData>("/api/test-tools", {})

// Tool Catalog
export const fetchToolCatalog = () =>
  request<ToolInfo[]>("/api/tools")

// Coverage
export const fetchCoverage = () =>
  request<CoverageCategory[]>("/api/coverage")

// Tasks Live Summary
export const fetchTasksLiveSummary = () =>
  request<TaskLiveSummary[]>("/api/tasks/live-summary")

// Seed Data (export/import)
export const exportSeedData = () =>
  post<{ ok: boolean; total: number; tables: Record<string, number> }>("/api/seed/export", {})

export const importSeedData = (reset = false) =>
  post<{ ok: boolean; total: number; tables: Record<string, number>; reset: boolean }>(
    `/api/seed/import?reset=${reset}`, {}
  )

// Solve Logs
export const fetchLogs = (limit = 100) =>
  request<SolveLog[]>(`/api/logs?limit=${limit}`)

export const deleteAllLogs = () =>
  request<{ ok: boolean; deleted_db: number; deleted_files: number }>("/api/logs", { method: "DELETE" })

export const fetchLogJson = (logId: number) =>
  request<Record<string, unknown>>(`/api/logs/${logId}/json`)

// Auto Fix — Last Results
export const fetchLastEvalResults = () =>
  request<import("@/types/api").LastEvalResult[]>("/api/auto-fix/last-results")

// Auto Fix — Batch Task Fix
export function streamBatchTaskFix(
  taskNames: string[],
  language: string,
  maxRetries: number,
  autoApply: boolean,
  onEvent: (event: import("@/types/api").BatchTaskFixEvent) => void,
  onDone: () => void,
  onError: (err: string) => void,
): AbortController {
  const controller = new AbortController()
  fetch("/api/auto-fix/batch-tasks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task_names: taskNames, language, max_retries: maxRetries, auto_apply: autoApply }),
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

// Auto Fix — Single Task
export function streamAutoFix(
  taskName: string,
  language: string,
  autoApply: boolean,
  onEvent: (event: import("@/types/api").AutoFixEvent) => void,
  onDone: () => void,
  onError: (err: string) => void,
): AbortController {
  const controller = new AbortController()
  fetch("/api/auto-fix/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task_name: taskName, language, auto_apply: autoApply }),
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

export const applyFixes = (fixes: import("@/types/api").AutoFixParsedFix[]) =>
  post<{ ok: boolean; results: import("@/types/api").AutoFixApplyResult[] }>("/api/auto-fix/apply", fixes)

// Batch Auto Fix (real logs)
export function streamBatchAutoFix(
  logIds: number[],
  limit: number,
  maxIterations: number,
  onEvent: (event: import("@/types/api").BatchAutoFixEvent) => void,
  onDone: () => void,
  onError: (err: string) => void,
): AbortController {
  const controller = new AbortController()
  fetch("/api/auto-fix/batch-loop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ log_ids: logIds, limit, max_iterations: maxIterations }),
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

// Live Activity Events (SSE)
export function subscribeLiveEvents(
  onEvent: (event: import("@/types/api").LiveEvent) => void,
  onConnect: () => void,
  onDisconnect: () => void,
): { close: () => void } {
  let source: EventSource | null = null
  let closed = false

  const connect = () => {
    if (closed) return
    source = new EventSource("/api/agent/events")
    source.onopen = () => onConnect()
    source.onmessage = (e) => {
      try { onEvent(JSON.parse(e.data)) } catch { /* skip malformed */ }
    }
    source.onerror = () => {
      onDisconnect()
      source?.close()
      // Reconnect after 3s
      if (!closed) setTimeout(connect, 3000)
    }
  }

  connect()
  return {
    close: () => {
      closed = true
      source?.close()
    },
  }
}

// Auto Test
export const fetchAutoTestResults = (limit = 200) =>
  request<import("@/types/api").AutoTestResult[]>(`/api/auto-test/results?limit=${limit}`)

export const fetchAutoTestLogs = (limit = 200) =>
  request<import("@/types/api").SolveLog[]>(`/api/auto-test/logs?limit=${limit}`)

export const scoreAutoTestLog = (solveLogId: number) =>
  post<Record<string, unknown>>("/api/auto-test/score-log", { solve_log_id: solveLogId })

export const deleteAutoTestResults = () =>
  request<{ ok: boolean; deleted: number }>("/api/auto-test/results", { method: "DELETE" })

export function streamAutoTestBatch(
  logIds: number[],
  limit: number,
  onEvent: (event: import("@/types/api").AutoTestEvent) => void,
  onDone: () => void,
  onError: (err: string) => void,
): AbortController {
  const controller = new AbortController()
  fetch("/api/auto-test/batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ log_ids: logIds, limit }),
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

// Live Eval (real competition submissions)
export function streamLiveEval(
  sinceId: number,
  limit: number,
  autoFix: boolean,
  onEvent: (event: import("@/types/api").LiveEvalEvent) => void,
  onDone: () => void,
  onError: (err: string) => void,
): AbortController {
  const controller = new AbortController()
  fetch("/api/auto-fix/live-eval", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ since_id: sinceId, limit, auto_fix: autoFix }),
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

// Score Tracking
export const fetchLatestScores = () =>
  request<import("@/types/api").ScoreSnapshot>("/api/scores/latest")

export const submitScoreSnapshot = (data: { submissions: unknown[]; total_score: number; rank?: number }) =>
  post<{ ok: boolean; snapshot_id: number; tasks_stored: number; new_mappings: { task_number: number; task_type: string }[] }>(
    "/api/scores/snapshot", data
  )

export const mapTaskNumber = (taskNumber: number, taskType: string) =>
  post<{ ok: boolean; task_number: number; task_type: string }>(
    "/api/scores/map-task", { task_number: taskNumber, task_type: taskType }
  )

export const fetchScoresFromApi = (cookie?: string) =>
  post<{ ok: boolean; snapshot_id: number; tasks_stored: number; new_mappings: { task_number: number; task_type: string }[]; total_score: number; rank: number | null }>(
    "/api/scores/fetch", { cookie: cookie || "" }
  )

export const setScoreAuth = (cookie: string) =>
  post<{ ok: boolean }>("/api/scores/set-auth", { cookie })

export const fetchScoreAuthStatus = () =>
  request<{ has_cookie: boolean; preview: string }>("/api/scores/auth-status")

export const fetchScoreMappings = () =>
  request<Record<string, import("@/types/api").TaskMapping>>("/api/scores/mappings")

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
