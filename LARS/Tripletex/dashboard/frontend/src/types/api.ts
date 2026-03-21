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
  source: string | null
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
  data?: string
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
  request_body?: Record<string, unknown>
  request_params?: Record<string, unknown>
  response_body?: unknown
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

export interface ToolParam {
  name: string
  type: string
  required: boolean
  default: string | null
}

export interface ToolInfo {
  name: string
  module: string
  summary: string
  docstring: string
  params: ToolParam[]
}

export interface TaskLiveSummary {
  name: string
  tier: number
  description: string
  baseline_calls: number
  field_count: number
  max_points: number
  live_runs: number
  avg_api_calls: number | null
  avg_api_errors: number | null
  avg_elapsed: number | null
  min_api_calls: number | null
  max_api_calls: number | null
  last_run: string | null
  sample_prompt: string | null
}

export type Languages = Record<string, string>

export interface CoverageCategory {
  category: string
  endpoint_count: number
  tools: string[]
  covered: string[]
  uncovered: string[]
}

// Auto-fix types
export interface AutoFixScore {
  correctness: number
  tier_multiplier: number
  base_score: number
  efficiency_bonus: number
  final_score: number
  max_possible: number
}

export interface AutoFixParsedFix {
  file: string
  old: string
  new: string
  reason: string
}

export interface AutoFixApplyResult {
  file: string
  applied: boolean
  reason?: string
  error?: string
}

export type AutoFixEvent =
  | { type: "phase"; phase: string; message: string }
  | { type: "eval_result"; score: AutoFixScore; prompt: string; expected: Record<string, unknown>; language: string; api_calls: number; api_errors: number; elapsed: number; checks: FieldCheck[]; tool_calls: ToolCall[]; agent_response: string }
  | { type: "fixes"; raw_text: string; parsed_fixes: AutoFixParsedFix[]; report: string }
  | { type: "applied"; results: AutoFixApplyResult[] }
  | { type: "error"; message: string }

export interface SolveLog {
  id: number | null
  request_id: string
  prompt: string
  files_json: string
  base_url: string
  api_calls: number
  api_errors: number
  elapsed_seconds: number
  agent_response: string
  tool_calls_json?: string
  api_log_json?: string
  task_type?: string
  tool_count?: number
  created_at: string
  source?: string
}

// Live activity event types (SSE from agent)
export type LiveEvent =
  | { type: "request_start"; request_id: string; prompt: string; files: string[]; source: string; ts: string }
  | { type: "classify"; request_id: string; task_type: string; classification_level: string; tool_count: number; total_tools: number; tools: string[]; ts: string }
  | { type: "agent_start"; request_id: string; ts: string }
  | { type: "tool_call"; request_id: string; turn: number; tool: string; args: Record<string, unknown>; ts: string }
  | { type: "tool_result"; request_id: string; turn: number; tool: string; ok: boolean; error?: string; ts: string }
  | { type: "api_call"; request_id: string; method: string; url: string; status: number; ok: boolean; elapsed: number; error?: string; ts: string }
  | { type: "text"; request_id: string; text: string; ts: string }
  | { type: "request_done"; request_id: string; elapsed: number; api_calls: number; api_errors: number; response: string; task_type: string; turns: number; ts: string }
  | { type: "request_error"; request_id: string; error: string; ts: string }
  | { type: "error"; message: string; ts?: string }

// Batch auto-fix types (real logs)
export type BatchAutoFixEvent =
  | { type: "batch_start"; total_logs: number; log_ids: number[]; log_summaries: { id: number; prompt: string; task_type: string }[]; max_iterations: number }
  | { type: "iteration_start"; iteration: number; logs_remaining: number; total_logs: number }
  | { type: "replaying"; iteration: number; index: number; total: number; log_id: number; task_type: string; prompt_preview: string }
  | { type: "eval_result"; iteration: number; index: number; total: number; log_id: number; task_type: string; passed: boolean; reasoning: string; issues: string[]; api_calls: number; api_errors: number }
  | { type: "replay_error"; iteration: number; log_id: number; task_type: string; error: string }
  | { type: "iteration_summary"; iteration: number; passed: number; failed: number; total: number; total_passed_overall: number; total_remaining: number }
  | { type: "analyzing"; iteration: number; task_type: string; failed_count: number; message: string }
  | { type: "analyze_error"; iteration: number; task_type: string; error: string }
  | { type: "no_fixes"; iteration: number; task_type: string; message: string }
  | { type: "applying_fixes"; iteration: number; task_type: string; fix_count: number; fixes: AutoFixParsedFix[] }
  | { type: "fixes_applied"; iteration: number; task_type: string; results: AutoFixApplyResult[]; applied: number; total: number }
  | { type: "apply_error"; iteration: number; task_type: string; error: string }
  | { type: "iteration_fixes_done"; iteration: number; fixes_applied: number; message: string }
  | { type: "batch_done"; iterations: number; total_passed: number; total_failed: number; total_logs: number; message: string; failed_logs?: { log_id: number; task_type: string }[] }
  | { type: "error"; message: string }

// Batch task fix types (task picker auto-fix)
export interface LastEvalResult {
  task: string
  lang: string
  status: "PASS" | "FAIL" | "SKIP"
  classifier: {
    task: string
    predicted: string
    correct: boolean
  }
  prompt?: string
}

export type BatchTaskFixEvent =
  | { type: "batch_start"; total: number; task_names: string[] }
  | { type: "task_start"; index: number; total: number; task_name: string; tier: number }
  | { type: "attempt_start"; task_name: string; attempt: number; max_attempts: number }
  | { type: "eval_result"; task_name: string; attempt: number; correctness: number; score: number; max_possible: number; api_calls: number; api_errors: number; elapsed: number; checks: FieldCheck[]; passed: boolean; failed_count: number }
  | { type: "eval_error"; task_name: string; attempt: number; error: string }
  | { type: "analyzing"; task_name: string; attempt: number; message: string }
  | { type: "analyze_error"; task_name: string; attempt: number; error: string }
  | { type: "no_fixes"; task_name: string; attempt: number }
  | { type: "applying_fixes"; task_name: string; attempt: number; fix_count: number; fixes: AutoFixParsedFix[] }
  | { type: "fixes_applied"; task_name: string; attempt: number; results: AutoFixApplyResult[]; applied: number; total: number }
  | { type: "apply_error"; task_name: string; attempt: number; error: string }
  | { type: "task_done"; task_name: string; passed: boolean; correctness: number; score: number; max_possible: number; api_calls: number; api_errors: number; index: number; total: number }
  | { type: "batch_done"; total: number; passed: number; failed: number; results: { task_name: string; passed: boolean; correctness: number; score: number; max_possible: number; api_calls: number; api_errors: number }[] }
  | { type: "error"; message: string }

// Auto Test types
export interface AutoTestFieldCheck {
  field: string
  points: number
  max: number
  passed: boolean
  detail: string
}

export interface AutoTestResult {
  id: number
  solve_log_id: number
  task_type: string
  prompt: string
  expected_fields: string
  checks_json: string
  total_points: number
  max_points: number
  correctness: number
  intent_passed: number
  intent_reasoning: string
  issues: string
  api_calls: number
  api_errors: number
  created_at: string
}

export type AutoTestEvent =
  | { type: "batch_start"; total: number }
  | { type: "scoring"; index: number; total: number; log_id: number; task_type: string }
  | { type: "scored"; log_id: number; task_type: string; correctness: number; intent_passed: boolean; total_points: number; max_points: number; checks: AutoTestFieldCheck[]; intent_reasoning: string }
  | { type: "error"; log_id: number; error: string }
  | { type: "batch_done"; total: number; passed: number; failed: number; avg_score: number }

// Log evaluation types
export type LogEvalEvent =
  | { type: "phase"; phase: string; message: string }
  | { type: "eval_verdict"; iteration: number; passed: boolean; reasoning: string; issues: string[]; tool_calls: ToolCall[]; api_log_summary: { method: string; status: number; url: string }[]; agent_response: string }
  | { type: "fixes"; iteration: number; raw_text: string; parsed_fixes: AutoFixParsedFix[]; report: string }
  | { type: "applied"; iteration: number; results: AutoFixApplyResult[] }
  | { type: "rerun_result"; iteration: number; api_calls: number; api_errors: number; tool_count: number; agent_response: string }
  | { type: "error"; message: string }
