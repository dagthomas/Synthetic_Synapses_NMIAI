import { useState } from "react"
import { useLogs } from "@/hooks/use-api"
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

export function LogsPanel() {
  const { data: logs, isLoading } = useLogs()

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
      <PageHeader
        title="Live Solve Logs"
        description="All /solve requests from competition runs with full tool call and API traces"
      />

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
  const [showApiLog, setShowApiLog] = useState(false)
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

        {/* Files */}
        {files.length > 0 && (
          <div className="mt-1 ml-12 flex gap-1 flex-wrap">
            {files.map((f, i) => (
              <Badge key={i} variant="outline" className="text-[10px]">
                <FileText className="h-2.5 w-2.5 mr-1" />
                {f}
              </Badge>
            ))}
          </div>
        )}

        {/* Expanded details */}
        {expanded && (
          <div className="mt-3 ml-8 space-y-3">
            {/* Agent response */}
            {log.agent_response && (
              <div>
                <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold mb-1">
                  Agent Response
                </p>
                <div className="bg-muted/30 rounded-lg p-3 text-[12px] whitespace-pre-wrap break-words max-h-[200px] overflow-y-auto">
                  {log.agent_response}
                </div>
              </div>
            )}

            {/* Tool calls timeline */}
            {toolCalls.length > 0 && (
              <div>
                <p className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold mb-1">
                  Tool Calls ({toolCalls.length})
                </p>
                <div className="space-y-1 border-l-2 border-border/50 pl-3">
                  {toolCalls.map((tc, j) => {
                    const ok = tc.result?.ok
                    return (
                      <div
                        key={j}
                        className={cn(
                          "text-[12px] font-mono py-0.5",
                          ok === false ? "text-red-600" : "text-emerald-700 dark:text-emerald-400"
                        )}
                      >
                        <div className="flex items-center gap-2">
                          <ArrowRight className="h-2.5 w-2.5 shrink-0 opacity-40" />
                          {ok === true ? (
                            <CheckCircle className="h-3 w-3 text-emerald-500 shrink-0" />
                          ) : ok === false ? (
                            <XCircle className="h-3 w-3 text-red-500 shrink-0" />
                          ) : (
                            <AlertTriangle className="h-3 w-3 text-amber-500 shrink-0" />
                          )}
                          <code className="truncate">{tc.tool}</code>
                        </div>
                        {tc.args && Object.keys(tc.args).length > 0 && (
                          <div className="ml-8 text-[11px] text-muted-foreground truncate">
                            {JSON.stringify(tc.args).slice(0, 200)}
                          </div>
                        )}
                        {ok === false && tc.result?.error && (
                          <div className="ml-8 text-red-400 text-[11px] truncate">
                            {tc.result.error}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* API log */}
            {log.api_log_json && (
              <div>
                <button
                  className="text-[10px] uppercase tracking-wide text-muted-foreground font-semibold mb-1 hover:text-foreground transition-colors flex items-center gap-1"
                  onClick={(e) => { e.stopPropagation(); setShowApiLog(!showApiLog) }}
                >
                  {showApiLog ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                  API Log
                </button>
                {showApiLog && (
                  <div className="bg-muted/30 rounded-lg p-2 text-[11px] font-mono max-h-[300px] overflow-y-auto space-y-0.5">
                    {(() => {
                      try {
                        const entries = JSON.parse(log.api_log_json)
                        return entries.map((e: Record<string, unknown>, i: number) => (
                          <div
                            key={i}
                            className={cn(
                              "flex gap-2",
                              e.ok === false ? "text-red-500" : "text-muted-foreground"
                            )}
                          >
                            <span className="w-10 shrink-0 text-right">{String(e.method)}</span>
                            <span className="w-8 shrink-0 tabular-nums">{String(e.status)}</span>
                            <span className="truncate">{String(e.url)}</span>
                          </div>
                        ))
                      } catch {
                        return <span className="text-muted-foreground">Could not parse API log</span>
                      }
                    })()}
                  </div>
                )}
              </div>
            )}

            <Separator />

            {/* Meta */}
            <div className="flex gap-4 text-[10px] text-muted-foreground">
              <span>ID: {log.request_id?.slice(0, 8)}</span>
              <span>Base: {log.base_url}</span>
              {log.source && <span>Source: {log.source}</span>}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
