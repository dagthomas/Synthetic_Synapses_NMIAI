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
export const fetchRuns = (params?: { status?: string; limit?: number }) => {
  const sp = new URLSearchParams()
  if (params?.status && params.status !== "all") sp.set("status", params.status)
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

// Solve Logs
export const fetchLogs = (limit = 100) =>
  request<SolveLog[]>(`/api/logs?limit=${limit}`)

export const deleteAllLogs = () =>
  request<{ ok: boolean; deleted_db: number; deleted_files: number }>("/api/logs", { method: "DELETE" })
