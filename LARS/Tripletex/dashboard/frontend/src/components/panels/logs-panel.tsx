import { useState } from "react"
import { useLogs } from "@/hooks/use-api"
import { deleteAllLogs } from "@/lib/api"
import type { SolveLog, ToolCall } from "@/types/api"
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

        {/* Expanded details */}
        {expanded && (
          <div className="mt-3 ml-4 space-y-3">
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

            {/* Tool calls — full detail */}
            {toolCalls.length > 0 && (
              <Section icon={<Wrench className="h-3 w-3" />} title={`Tool Calls (${toolCalls.length})`}>
                <div className="space-y-2">
                  {toolCalls.map((tc, j) => (
                    <ToolCallCard key={j} tc={tc} index={j} />
                  ))}
                </div>
              </Section>
            )}

            {/* API log — always visible */}
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

        {/* Expanded: full URLs + response bodies if available */}
        {expanded && (
          <div className="border-t border-border/30 px-3 py-2 space-y-2 max-h-[400px] overflow-y-auto">
            {entries.map((e, i) => {
              const isErr = e.ok === false
              return (
                <div key={i} className="text-[11px] font-mono">
                  <div className={cn("flex gap-2", isErr ? "text-red-500" : "text-foreground")}>
                    <span className="font-semibold">{String(e.method)}</span>
                    <span className="tabular-nums">{String(e.status)}</span>
                    <span className="break-all">{String(e.url)}</span>
                  </div>
                  {(e.error as string) && (
                    <div className="ml-4 text-red-400 mt-0.5">{String(e.error)}</div>
                  )}
                  {(e.body as string) && (
                    <pre className="ml-4 text-muted-foreground mt-0.5 whitespace-pre-wrap break-words max-h-[100px] overflow-y-auto">
                      {formatJson(e.body)}
                    </pre>
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
