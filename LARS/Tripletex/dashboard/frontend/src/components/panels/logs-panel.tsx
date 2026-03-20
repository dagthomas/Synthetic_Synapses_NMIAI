import { useState, useCallback, useRef, useEffect, useMemo } from "react"
import { useLogs } from "@/hooks/use-api"
import { deleteAllLogs, streamLogEval, fetchLogJson, subscribeLiveEvents } from "@/lib/api"
import type { SolveLog, ToolCall, LogEvalEvent, AutoFixParsedFix, AutoFixApplyResult, LiveEvent } from "@/types/api"
import { PageHeader } from "@/components/layout/page-header"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"
import {
  CheckCircle,
  XCircle,
  Clock,
  Zap,
  ArrowRight,
  AlertTriangle,
  FileText,
  ChevronDown,
  ChevronRight,
  Tag,
  Wrench,
  Trash2,
  Copy,
  Check,
  MessageSquare,
  Globe,
  Play,
  RotateCcw,
  Brain,
  Loader2,
  Square,
  Download,
  Radio,
  ClipboardCopy,
  Bot,
  CircleDot,
} from "lucide-react"

function parseToolCalls(json?: string): ToolCall[] {
  if (!json) return []
  try {
    return JSON.parse(json)
  } catch {
    return []
  }
}

function parseFiles(json?: string): string[] {
  if (!json) return []
  try {
    return JSON.parse(json)
  } catch {
    return []
  }
}

function formatTime(iso?: string): string {
  if (!iso) return ""
  try {
    const d = new Date(iso)
    return d.toLocaleString("nb-NO", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    })
  } catch {
    return iso
  }
}

function formatJson(obj: unknown): string {
  try {
    if (typeof obj === "string") {
      const parsed = JSON.parse(obj)
      return JSON.stringify(parsed, null, 2)
    }
    return JSON.stringify(obj, null, 2)
  } catch {
    return String(obj)
  }
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      onClick={(e) => {
        e.stopPropagation()
        navigator.clipboard.writeText(text)
        setCopied(true)
        setTimeout(() => setCopied(false), 1500)
      }}
      className="p-1 rounded hover:bg-muted/60 transition-colors"
      title="Copy"
    >
      {copied ? (
        <Check className="h-3 w-3 text-emerald-500" />
      ) : (
        <Copy className="h-3 w-3 text-muted-foreground" />
      )}
    </button>
  )
}

// ── Live Activity types ────────────────────────────────────────────

interface LiveRequest {
  request_id: string
  prompt: string
  events: LiveEvent[]
  done: boolean
  startTs: string
}

function groupByRequest(events: LiveEvent[]): LiveRequest[] {
  const map = new Map<string, LiveRequest>()
  for (const ev of events) {
    if (ev.type === "error") continue
    const rid = ev.request_id
    if (!map.has(rid)) {
      map.set(rid, {
        request_id: rid,
        prompt: ev.type === "request_start" ? ev.prompt : "",
        events: [],
        done: false,
        startTs: ev.ts || "",
      })
    }
    const req = map.get(rid)!
    req.events.push(ev)
    if (ev.type === "request_start") req.prompt = ev.prompt
    if (ev.type === "request_done" || ev.type === "request_error") req.done = true
  }
  return Array.from(map.values()).reverse()
}

function buildErrorReport(requests: LiveRequest[]): string {
  const lines: string[] = ["## Agent Error Report", ""]
  let hasErrors = false

  for (const req of requests) {
    const errors: string[] = []
    const toolErrors: string[] = []
    const apiErrors: string[] = []
    let taskType = ""
    let classLevel = ""
    let elapsed = 0
    let apiCalls = 0
    let apiErrCount = 0
    let response = ""

    for (const ev of req.events) {
      if (ev.type === "classify") {
        taskType = ev.task_type || "unknown"
        classLevel = ev.classification_level
      }
      if (ev.type === "tool_call") {
        const result = req.events.find(
          (e) => e.type === "tool_result" && e.turn === ev.turn
        )
        if (result && result.type === "tool_result" && !result.ok) {
          toolErrors.push(
            `#${ev.turn} ${ev.tool}(${JSON.stringify(ev.args)}) -> ERROR: ${result.error || "unknown"}`
          )
        }
      }
      if (ev.type === "api_call" && !ev.ok) {
        const shortUrl = ev.url.replace(/^https?:\/\/[^/]+/, "")
        apiErrors.push(
          `${ev.method} ${shortUrl} -> ${ev.status} (${ev.elapsed}s) ${ev.error || ""}`
        )
      }
      if (ev.type === "request_done") {
        elapsed = ev.elapsed
        apiCalls = ev.api_calls
        apiErrCount = ev.api_errors
        response = ev.response
      }
      if (ev.type === "request_error") {
        errors.push(ev.error)
      }
    }

    if (toolErrors.length === 0 && apiErrors.length === 0 && errors.length === 0) continue
    hasErrors = true

    lines.push(`### ${taskType || "unknown"} (${classLevel})`)
    lines.push(`Prompt: "${req.prompt}"`)
    lines.push(`Time: ${elapsed}s | API: ${apiCalls} calls, ${apiErrCount} errors`)
    lines.push("")

    if (toolErrors.length > 0) {
      lines.push("**Tool Errors:**")
      toolErrors.forEach((e) => lines.push(`- ${e}`))
      lines.push("")
    }
    if (apiErrors.length > 0) {
      lines.push("**API Errors:**")
      apiErrors.forEach((e) => lines.push(`- ${e}`))
      lines.push("")
    }
    if (errors.length > 0) {
      lines.push("**Agent Errors:**")
      errors.forEach((e) => lines.push(`- ${e}`))
      lines.push("")
    }
    if (response) {
      lines.push(`**Agent Response:** "${response}"`)
      lines.push("")
    }
    lines.push("---")
    lines.push("")
  }

  if (!hasErrors) return ""
  return lines.join("\n")
}

function relativeTime(startTs: string, eventTs: string): string {
  try {
    const diff = (new Date(eventTs).getTime() - new Date(startTs).getTime()) / 1000
    return `+${diff.toFixed(1)}s`
  } catch {
    return ""
  }
}

// ── LiveActivitySection ────────────────────────────────────────────

function LiveActivitySection() {
  const [events, setEvents] = useState<LiveEvent[]>([])
  const [connected, setConnected] = useState(false)
  const [expandedReq, setExpandedReq] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    const sub = subscribeLiveEvents(
      (event) => {
        setEvents((prev) => {
          const next = [...prev, event]
          // Keep last 500 events
          return next.length > 500 ? next.slice(-400) : next
        })
        // Auto-expand new requests
        if (event.type === "request_start") {
          setExpandedReq(event.request_id)
        }
      },
      () => setConnected(true),
      () => setConnected(false),
    )
    return () => sub.close()
  }, [])

  const requests = useMemo(() => groupByRequest(events), [events])
  const errorReport = useMemo(() => buildErrorReport(requests), [requests])
  const hasErrors = errorReport.length > 0
  const activeCount = requests.filter((r) => !r.done).length

  const handleCopyErrors = useCallback(() => {
    navigator.clipboard.writeText(errorReport)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }, [errorReport])

  const handleClear = useCallback(() => {
    setEvents([])
    setExpandedReq(null)
  }, [])

  return (
    <Card className="shadow-premium mb-4 border-l-4 border-l-violet-500">
      <CardContent className="p-3">
        {/* Header */}
        <div className="flex items-center gap-2 mb-2">
          <div className="relative flex items-center">
            <Radio className={cn("h-4 w-4", connected ? "text-emerald-500" : "text-red-400")} />
            {connected && activeCount > 0 && (
              <span className="absolute h-4 w-4 rounded-full bg-emerald-400 animate-ping opacity-30" />
            )}
          </div>
          <span className="text-[13px] font-semibold">Live Activity</span>
          <Badge variant="secondary" className="text-[10px]">
            {connected ? (activeCount > 0 ? `${activeCount} active` : "listening") : "disconnected"}
          </Badge>
          <div className="flex-1" />

          {/* Copy Errors button */}
          <button
            onClick={handleCopyErrors}
            disabled={!hasErrors}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-semibold rounded-md transition-colors shrink-0",
              hasErrors
                ? "bg-red-500/10 text-red-600 hover:bg-red-500/20"
                : "bg-muted/40 text-muted-foreground cursor-not-allowed opacity-50"
            )}
          >
            {copied ? <Check className="h-3 w-3" /> : <ClipboardCopy className="h-3 w-3" />}
            {copied ? "Copied!" : "Copy Errors"}
          </button>

          {events.length > 0 && (
            <button
              onClick={handleClear}
              className="flex items-center gap-1 px-2 py-1 text-[10px] font-medium rounded bg-muted/40 text-muted-foreground hover:bg-muted/60 transition-colors shrink-0"
            >
              <Trash2 className="h-2.5 w-2.5" />
              Clear
            </button>
          )}
        </div>

        {/* Request list */}
        {requests.length === 0 ? (
          <div className="text-center py-4 text-muted-foreground text-[12px]">
            {connected
              ? "Waiting for /solve requests..."
              : "Connecting to agent..."}
          </div>
        ) : (
          <div className="space-y-1.5 max-h-[500px] overflow-y-auto">
            {requests.map((req) => (
              <LiveRequestCard
                key={req.request_id}
                req={req}
                expanded={expandedReq === req.request_id}
                onToggle={() =>
                  setExpandedReq(
                    expandedReq === req.request_id ? null : req.request_id
                  )
                }
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function LiveRequestCard({
  req,
  expanded,
  onToggle,
}: {
  req: LiveRequest
  expanded: boolean
  onToggle: () => void
}) {
  const doneEvent = req.events.find((e) => e.type === "request_done") as
    | Extract<LiveEvent, { type: "request_done" }>
    | undefined
  const classifyEvent = req.events.find((e) => e.type === "classify") as
    | Extract<LiveEvent, { type: "classify" }>
    | undefined
  const hasErrors =
    req.events.some(
      (e) =>
        (e.type === "tool_result" && !e.ok) ||
        (e.type === "api_call" && !e.ok) ||
        e.type === "request_error"
    )

  return (
    <div
      className={cn(
        "rounded-lg border text-[12px] overflow-hidden transition-colors",
        !req.done
          ? "border-violet-300 dark:border-violet-800 bg-violet-50/30 dark:bg-violet-950/10"
          : hasErrors
          ? "border-amber-200 dark:border-amber-900"
          : "border-border/50"
      )}
    >
      {/* Header */}
      <div
        className="flex items-center gap-2 px-3 py-1.5 cursor-pointer hover:bg-muted/30 transition-colors"
        onClick={onToggle}
      >
        {expanded ? (
          <ChevronDown className="h-3 w-3 text-muted-foreground shrink-0" />
        ) : (
          <ChevronRight className="h-3 w-3 text-muted-foreground shrink-0" />
        )}

        {!req.done ? (
          <Loader2 className="h-3.5 w-3.5 text-violet-500 animate-spin shrink-0" />
        ) : hasErrors ? (
          <AlertTriangle className="h-3.5 w-3.5 text-amber-500 shrink-0" />
        ) : (
          <CheckCircle className="h-3.5 w-3.5 text-emerald-500 shrink-0" />
        )}

        {classifyEvent?.task_type && (
          <Badge variant="outline" className="text-[9px] shrink-0">
            <Tag className="h-2 w-2 mr-0.5" />
            {classifyEvent.task_type}
          </Badge>
        )}

        <span className="truncate flex-1 text-[11px]">{req.prompt}</span>

        {doneEvent && (
          <>
            <Badge variant="secondary" className="text-[9px] tabular-nums shrink-0">
              <Clock className="h-2 w-2 mr-0.5" />
              {doneEvent.elapsed}s
            </Badge>
            <Badge variant="secondary" className="text-[9px] tabular-nums shrink-0">
              <Zap className="h-2 w-2 mr-0.5" />
              {doneEvent.api_calls}
            </Badge>
            {doneEvent.api_errors > 0 && (
              <Badge variant="destructive" className="text-[9px] tabular-nums shrink-0">
                {doneEvent.api_errors} err
              </Badge>
            )}
          </>
        )}
      </div>

      {/* Timeline */}
      {expanded && (
        <div className="border-t border-border/30 px-3 py-2">
          <div className="space-y-0.5">
            {req.events.map((ev, i) => (
              <LiveEventRow key={i} event={ev} startTs={req.startTs} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function LiveEventRow({ event: ev, startTs }: { event: LiveEvent; startTs: string }) {
  const rel = ev.ts ? relativeTime(startTs, ev.ts) : ""

  switch (ev.type) {
    case "request_start":
      return (
        <div className="flex items-center gap-2 text-[11px]">
          <CircleDot className="h-3 w-3 text-violet-500 shrink-0" />
          <span className="text-muted-foreground">Request started</span>
          {ev.files.length > 0 && (
            <Badge variant="outline" className="text-[9px]">
              {ev.files.length} file(s)
            </Badge>
          )}
          <span className="ml-auto text-[9px] text-muted-foreground/60 tabular-nums shrink-0">{rel}</span>
        </div>
      )

    case "classify":
      return (
        <div className="flex items-center gap-2 text-[11px]">
          <Tag className="h-3 w-3 text-blue-500 shrink-0" />
          <span>
            <span className="font-medium">{ev.task_type || "unknown"}</span>
            <span className="text-muted-foreground"> ({ev.classification_level}, {ev.tool_count}/{ev.total_tools} tools)</span>
          </span>
          <span className="ml-auto text-[9px] text-muted-foreground/60 tabular-nums shrink-0">{rel}</span>
        </div>
      )

    case "agent_start":
      return (
        <div className="flex items-center gap-2 text-[11px]">
          <Bot className="h-3 w-3 text-violet-500 shrink-0" />
          <span className="text-muted-foreground">Gemini agent started</span>
          <span className="ml-auto text-[9px] text-muted-foreground/60 tabular-nums shrink-0">{rel}</span>
        </div>
      )

    case "tool_call":
      return (
        <div className="flex items-center gap-2 text-[11px]">
          <ArrowRight className="h-3 w-3 text-blue-500 shrink-0" />
          <code className="font-semibold text-blue-600 dark:text-blue-400">{ev.tool}</code>
          <span className="text-muted-foreground truncate text-[10px]">
            ({Object.entries(ev.args || {}).map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join(", ").slice(0, 80)})
          </span>
          <span className="ml-auto text-[9px] text-muted-foreground/60 tabular-nums shrink-0">{rel}</span>
        </div>
      )

    case "tool_result":
      return (
        <div className="flex items-center gap-2 text-[11px]">
          {ev.ok ? (
            <CheckCircle className="h-3 w-3 text-emerald-500 shrink-0" />
          ) : (
            <XCircle className="h-3 w-3 text-red-500 shrink-0" />
          )}
          <code className={cn("font-medium", ev.ok ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400")}>
            {ev.tool}
          </code>
          {!ev.ok && ev.error && (
            <span className="text-red-500 truncate text-[10px]">{ev.error.slice(0, 100)}</span>
          )}
          <span className="ml-auto text-[9px] text-muted-foreground/60 tabular-nums shrink-0">{rel}</span>
        </div>
      )

    case "api_call":
      return (
        <div className="flex items-center gap-2 text-[11px] ml-4">
          <Globe className="h-2.5 w-2.5 text-muted-foreground shrink-0" />
          <span className={cn(
            "font-mono text-[10px] font-semibold w-10 shrink-0",
            ev.method === "POST" ? "text-blue-500" :
            ev.method === "PUT" ? "text-amber-500" :
            ev.method === "DELETE" ? "text-red-500" :
            "text-muted-foreground"
          )}>
            {ev.method}
          </span>
          <span className={cn(
            "text-[10px] w-6 shrink-0 tabular-nums text-center rounded px-0.5",
            ev.status >= 400 ? "bg-red-100 dark:bg-red-950 text-red-600" :
            ev.status >= 200 && ev.status < 300 ? "text-emerald-600" :
            "text-muted-foreground"
          )}>
            {ev.status}
          </span>
          <span className="text-muted-foreground truncate text-[10px]">
            {ev.url.replace(/^https?:\/\/[^/]+/, "")}
          </span>
          <span className="ml-auto text-[9px] text-muted-foreground/60 tabular-nums shrink-0">
            {ev.elapsed}s
          </span>
        </div>
      )

    case "text":
      return (
        <div className="flex items-center gap-2 text-[11px]">
          <MessageSquare className="h-3 w-3 text-muted-foreground shrink-0" />
          <span className="text-muted-foreground truncate">&ldquo;{ev.text.slice(0, 120)}&rdquo;</span>
          <span className="ml-auto text-[9px] text-muted-foreground/60 tabular-nums shrink-0">{rel}</span>
        </div>
      )

    case "request_done":
      return (
        <div className="flex items-center gap-2 text-[11px] font-medium">
          <CheckCircle className="h-3 w-3 text-emerald-500 shrink-0" />
          <span className="text-emerald-600 dark:text-emerald-400">
            Done {ev.elapsed}s ({ev.api_calls} API, {ev.api_errors} err, {ev.turns} turns)
          </span>
          <span className="ml-auto text-[9px] text-muted-foreground/60 tabular-nums shrink-0">{rel}</span>
        </div>
      )

    case "request_error":
      return (
        <div className="flex items-center gap-2 text-[11px]">
          <XCircle className="h-3 w-3 text-red-500 shrink-0" />
          <span className="text-red-500">{ev.error}</span>
          <span className="ml-auto text-[9px] text-muted-foreground/60 tabular-nums shrink-0">{rel}</span>
        </div>
      )

    default:
      return null
  }
}

// ── LogsPanel ──────────────────────────────────────────────────────

export function LogsPanel() {
  const { data: logs, isLoading, mutate } = useLogs()
  const [deleting, setDeleting] = useState(false)

  async function handleDeleteAll() {
    if (!confirm("Delete all solve logs and payload files?")) return
    setDeleting(true)
    try {
      await deleteAllLogs()
      mutate()
    } catch (e) {
      console.error("Failed to delete logs:", e)
    } finally {
      setDeleting(false)
    }
  }

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-48 w-full rounded-xl" />
      </div>
    )
  }

  const items = logs ?? []
  const totalCalls = items.reduce((s, l) => s + (l.api_calls || 0), 0)
  const totalErrors = items.reduce((s, l) => s + (l.api_errors || 0), 0)
  const totalTime = items.reduce((s, l) => s + (l.elapsed_seconds || 0), 0)

  return (
    <div>
      <div className="flex items-center justify-between">
        <PageHeader
          title="Live Solve Logs"
          description="All /solve requests from competition runs with full tool call and API traces"
        />
        {items.length > 0 && (
          <button
            onClick={handleDeleteAll}
            disabled={deleting}
            className="flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium rounded-md bg-red-500/10 text-red-600 hover:bg-red-500/20 disabled:opacity-50 transition-colors shrink-0"
          >
            <Trash2 className="h-3.5 w-3.5" />
            {deleting ? "Deleting..." : "Delete All Logs"}
          </button>
        )}
      </div>

      {/* Live activity stream */}
      <LiveActivitySection />

      {/* Summary stats */}
      {items.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 mb-4">
          <MetricCard label="Requests" value={items.length} />
          <MetricCard label="Total Time" value={`${totalTime.toFixed(0)}s`} icon={<Clock className="h-3 w-3" />} />
          <MetricCard label="Avg Time" value={`${(totalTime / items.length).toFixed(1)}s`} icon={<Clock className="h-3 w-3" />} />
          <MetricCard label="API Calls" value={totalCalls} icon={<Zap className="h-3 w-3" />} />
          <MetricCard label="API Errors" value={totalErrors} variant={totalErrors > 0 ? "danger" : "success"} />
        </div>
      )}

      {items.length === 0 ? (
        <Card className="shadow-premium">
          <CardContent className="py-16 text-center text-muted-foreground">
            <FileText className="h-10 w-10 mx-auto mb-3 opacity-30" />
            <p className="text-[13px]">No solve logs yet. Run a live submission to see logs here.</p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-2">
          {items.map((log, i) => (
            <LogCard key={log.id ?? log.request_id ?? i} log={log} index={i} />
          ))}
        </div>
      )}
    </div>
  )
}

function MetricCard({
  label,
  value,
  variant,
  icon,
}: {
  label: string
  value: string | number
  variant?: "success" | "danger"
  icon?: React.ReactNode
}) {
  const colorClass =
    variant === "success" ? "text-emerald-600" :
    variant === "danger" ? "text-red-600" :
    "text-foreground"

  return (
    <div className="text-center py-2.5 px-2 rounded-lg bg-muted/40 metric-card">
      <div className={cn("text-lg font-bold tabular-nums flex items-center justify-center gap-1", colorClass)}>
        {icon}
        {value}
      </div>
      <div className="text-[10px] text-muted-foreground font-medium mt-0.5">{label}</div>
    </div>
  )
}

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
  const [copyingJson, setCopyingJson] = useState(false)
  const [jsonCopied, setJsonCopied] = useState(false)
  const controllerRef = useRef<AbortController | null>(null)

  const handleCopyJson = useCallback(async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!log.id) return
    setCopyingJson(true)
    try {
      const data = await fetchLogJson(log.id)
      await navigator.clipboard.writeText(JSON.stringify(data, null, 2))
      setJsonCopied(true)
      setTimeout(() => setJsonCopied(false), 2000)
    } catch (err) {
      console.error("Failed to copy JSON:", err)
    } finally {
      setCopyingJson(false)
    }
  }, [log.id])

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

          {/* Copy Full JSON button */}
          {log.id && (
            <button
              onClick={handleCopyJson}
              disabled={copyingJson}
              className="flex items-center gap-1 px-2 py-1 text-[10px] font-medium rounded bg-blue-500/10 text-blue-600 hover:bg-blue-500/20 disabled:opacity-50 transition-colors shrink-0"
            >
              {jsonCopied ? <Check className="h-2.5 w-2.5" /> : copyingJson ? <Loader2 className="h-2.5 w-2.5 animate-spin" /> : <Download className="h-2.5 w-2.5" />}
              {jsonCopied ? "Copied!" : "Copy JSON"}
            </button>
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

function Section({
  icon,
  title,
  copyText,
  children,
}: {
  icon: React.ReactNode
  title: string
  copyText?: string
  children: React.ReactNode
}) {
  return (
    <div>
      <div className="flex items-center gap-1.5 mb-1">
        <span className="text-muted-foreground">{icon}</span>
        <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold">
          {title}
        </p>
        {copyText && <CopyButton text={copyText} />}
      </div>
      {children}
    </div>
  )
}

function ToolCallCard({ tc, index }: { tc: ToolCall; index: number }) {
  const [showResult, setShowResult] = useState(false)
  const ok = tc.result?.ok
  const hasData = tc.result?.data
  const argsStr = tc.args && Object.keys(tc.args).length > 0 ? formatJson(tc.args) : null

  return (
    <div
      className={cn(
        "rounded-lg border text-[12px] font-mono overflow-hidden",
        ok === false ? "border-red-200 dark:border-red-900" : "border-border/50"
      )}
    >
      {/* Tool header */}
      <div
        className={cn(
          "flex items-center gap-2 px-3 py-1.5",
          ok === false
            ? "bg-red-50 dark:bg-red-950/30"
            : "bg-muted/30"
        )}
      >
        <span className="text-[10px] text-muted-foreground w-4 text-right shrink-0">{index + 1}</span>
        <ArrowRight className="h-2.5 w-2.5 shrink-0 opacity-40" />
        {ok === true ? (
          <CheckCircle className="h-3 w-3 text-emerald-500 shrink-0" />
        ) : ok === false ? (
          <XCircle className="h-3 w-3 text-red-500 shrink-0" />
        ) : (
          <AlertTriangle className="h-3 w-3 text-amber-500 shrink-0" />
        )}
        <code className={cn(
          "font-semibold",
          ok === false ? "text-red-600 dark:text-red-400" : "text-emerald-700 dark:text-emerald-400"
        )}>
          {tc.tool}
        </code>
        <div className="flex-1" />
        {hasData && (
          <button
            onClick={() => setShowResult(!showResult)}
            className="text-[10px] text-muted-foreground hover:text-foreground transition-colors flex items-center gap-0.5"
          >
            {showResult ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            result
          </button>
        )}
      </div>

      {/* Args */}
      {argsStr && (
        <div className="px-3 py-1.5 border-t border-border/30">
          <div className="flex items-center gap-1 mb-0.5">
            <span className="text-[9px] uppercase tracking-wide text-muted-foreground font-semibold">args</span>
            <CopyButton text={argsStr} />
          </div>
          <pre className="text-[11px] text-muted-foreground whitespace-pre-wrap break-words max-h-[150px] overflow-y-auto">
            {argsStr}
          </pre>
        </div>
      )}

      {/* Error */}
      {ok === false && tc.result?.error && (
        <div className="px-3 py-1.5 border-t border-red-200 dark:border-red-900 bg-red-50/50 dark:bg-red-950/20">
          <span className="text-[9px] uppercase tracking-wide text-red-500 font-semibold">error</span>
          <pre className="text-[11px] text-red-500 whitespace-pre-wrap break-words mt-0.5">
            {tc.result.error}
          </pre>
        </div>
      )}

      {/* Result data */}
      {showResult && hasData && (
        <div className="px-3 py-1.5 border-t border-border/30 bg-muted/20">
          <div className="flex items-center gap-1 mb-0.5">
            <span className="text-[9px] uppercase tracking-wide text-muted-foreground font-semibold">response</span>
            <CopyButton text={String(tc.result!.data)} />
          </div>
          <pre className="text-[11px] text-muted-foreground whitespace-pre-wrap break-words max-h-[200px] overflow-y-auto">
            {formatJson(tc.result!.data)}
          </pre>
        </div>
      )}
    </div>
  )
}

function ApiLogSection({ json }: { json: string }) {
  const [expanded, setExpanded] = useState(false)

  let entries: Array<Record<string, unknown>> = []
  try {
    entries = JSON.parse(json)
  } catch {
    return null
  }

  if (entries.length === 0) return null

  const errorCount = entries.filter(e => e.ok === false).length

  return (
    <Section icon={<Globe className="h-3 w-3" />} title={`API Log (${entries.length} calls${errorCount > 0 ? `, ${errorCount} errors` : ""})`}>
      <button
        className="text-[10px] text-muted-foreground hover:text-foreground transition-colors flex items-center gap-0.5 mb-1"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        {expanded ? "Collapse" : "Expand"} details
      </button>

      <div className="bg-muted/30 rounded-lg overflow-hidden">
        {/* Compact summary row always visible */}
        <div className="px-3 py-1.5 text-[11px] font-mono space-y-0.5 max-h-[200px] overflow-y-auto">
          {entries.map((e, i) => {
            const isErr = e.ok === false
            const method = String(e.method || "")
            const status = Number(e.status || 0)
            const url = String(e.url || "")
            // Shorten URL: remove base URL prefix
            const shortUrl = url.replace(/^https?:\/\/[^/]+/, "")
            return (
              <div
                key={i}
                className={cn(
                  "flex gap-2 items-baseline",
                  isErr ? "text-red-500" : "text-muted-foreground"
                )}
              >
                <span className={cn(
                  "w-12 shrink-0 text-right font-semibold",
                  method === "POST" ? "text-blue-500" :
                  method === "PUT" ? "text-amber-500" :
                  method === "DELETE" ? "text-red-500" :
                  "text-muted-foreground"
                )}>
                  {method}
                </span>
                <span className={cn(
                  "w-8 shrink-0 tabular-nums text-center rounded px-1",
                  status >= 400 ? "bg-red-100 dark:bg-red-950 text-red-600" :
                  status >= 200 && status < 300 ? "text-emerald-600" :
                  "text-muted-foreground"
                )}>
                  {status}
                </span>
                <span className="truncate">{shortUrl}</span>
              </div>
            )
          })}
        </div>

        {/* Expanded: full URLs + request/response bodies */}
        {expanded && (
          <div className="border-t border-border/30 px-3 py-2 space-y-2 max-h-[600px] overflow-y-auto">
            {entries.map((e, i) => {
              const isErr = e.ok === false
              const hasReqBody = e.request_body != null
              const hasReqParams = e.request_params != null
              const hasRespBody = e.response_body != null
              return (
                <div key={i} className="text-[11px] font-mono border-b border-border/20 pb-2 last:border-0">
                  <div className={cn("flex gap-2", isErr ? "text-red-500" : "text-foreground")}>
                    <span className="font-semibold">{String(e.method)}</span>
                    <span className="tabular-nums">{String(e.status)}</span>
                    <span className="break-all flex-1">{String(e.url)}</span>
                    <span className="text-muted-foreground tabular-nums shrink-0">{String(e.elapsed || 0)}s</span>
                  </div>
                  {(e.error as string) && (
                    <div className="ml-4 text-red-400 mt-0.5">{String(e.error)}</div>
                  )}
                  {hasReqParams && (
                    <div className="ml-4 mt-0.5">
                      <span className="text-[9px] uppercase text-muted-foreground font-semibold">params</span>
                      <pre className="text-muted-foreground whitespace-pre-wrap break-words max-h-[80px] overflow-y-auto">
                        {formatJson(e.request_params)}
                      </pre>
                    </div>
                  )}
                  {hasReqBody && (
                    <div className="ml-4 mt-0.5">
                      <span className="text-[9px] uppercase text-muted-foreground font-semibold">request body</span>
                      <pre className="text-muted-foreground whitespace-pre-wrap break-words max-h-[150px] overflow-y-auto">
                        {formatJson(e.request_body)}
                      </pre>
                    </div>
                  )}
                  {hasRespBody && (
                    <div className="ml-4 mt-0.5">
                      <span className="text-[9px] uppercase text-muted-foreground font-semibold">response body</span>
                      <pre className={cn("whitespace-pre-wrap break-words max-h-[150px] overflow-y-auto", isErr ? "text-red-400" : "text-muted-foreground")}>
                        {formatJson(e.response_body)}
                      </pre>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </Section>
  )
}
