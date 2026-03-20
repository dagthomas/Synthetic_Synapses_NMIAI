export interface TaskDef {
  name: string
  tier: number
  description: string
  baseline_calls: number
  field_count: number
  max_points: number
}

export interface EvalRun {
  id: number
  task_name: string
  tier: number
  language: string
  prompt: string
  expected_json: string | null
  status: "pending" | "running" | "completed" | "failed"
  agent_url: string
  api_calls: number
  api_errors: number
  elapsed_seconds: number
  correctness: number | null
  base_score: number | null
  efficiency_bonus: number | null
  final_score: number | null
  max_possible: number | null
  checks_json: string | null
  error_message: string | null
  created_at: string
  completed_at: string | null
}

export interface FieldCheck {
  field: string
  passed: boolean
  detail: string
  points: number
  max: number
}

export interface SandboxEntityInfo {
  count: number
  ok: boolean
}

export interface SandboxHealth {
  connected: boolean
  ready: boolean
  base_url: string
  entities: Record<string, SandboxEntityInfo>
  bank_account_1920: boolean
  error?: string
}

export interface SeedResult {
  results: Record<string, { created: number; errors: string[] }>
  bank_account?: { ok: boolean; already_set?: boolean; error?: string }
  total_created: number
  total_errors: number
}

export interface CleanResult {
  results: Record<string, { deleted: number; errors: string[]; skipped?: string }>
  total_deleted: number
}

export interface Payload {
  filename: string
  prompt: string
  files: string[]
  base_url: string
}

export interface ToolCallResult {
  ok: boolean | null
  error?: string
}

export interface ToolCall {
  tool: string
  args: Record<string, unknown>
  result: ToolCallResult
}

export interface ApiLogEntry {
  method: string
  url: string
  status: number
  ok: boolean
  error?: string
}

export interface ReplayResult {
  filename: string
  prompt: string
  status: "OK" | "FAIL"
  tool_calls: ToolCall[]
  api_calls: number
  api_errors: number
  api_log: ApiLogEntry[]
  agent_response: string
  elapsed: number
  error: string
}

export interface ToolTestResult {
  tool: string
  status: "OK" | "FAIL" | "EXCEPTION"
  elapsed: number
  status_code?: number
  error?: string
  result_preview?: string
}

export interface ToolTestData {
  total: number
  ok: number
  fail: number
  exception: number
  total_time: string
  api_calls: number
  api_errors: number
  results: ToolTestResult[]
}

export interface RunStats {
  task_name: string
  tier: number
  language: string
  run_count: number
  avg_score: number | null
  avg_correctness: number | null
  avg_elapsed: number | null
  completed: number
  failed: number
}

export type Languages = Record<string, string>

export interface CoverageCategory {
  category: string
  endpoint_count: number
  tools: string[]
  covered: string[]
  uncovered: string[]
}
