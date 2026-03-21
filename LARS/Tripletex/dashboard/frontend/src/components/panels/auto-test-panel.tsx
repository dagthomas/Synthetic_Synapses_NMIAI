import { useState, useRef, useCallback, useMemo, useEffect } from "react"
import {
  fetchAutoTestResults,
  fetchAutoTestLogs,
  scoreAutoTestLog,
  streamAutoTestBatch,
  deleteAutoTestResults,
} from "@/lib/api"
import type {
  AutoTestResult,
  AutoTestEvent,
  AutoTestFieldCheck,
  SolveLog,
} from "@/types/api"
import { PageHeader } from "@/components/layout/page-header"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { toast } from "sonner"
import { cn } from "@/lib/utils"
import {
  Loader2,
  CheckCircle2,
  XCircle,
  ChevronDown,
  ChevronRight,
  Play,
  RotateCcw,
  Trash2,
  Filter,
} from "lucide-react"

export function AutoTestPanel() {
  const [results, setResults] = useState<AutoTestResult[]>([])
  const [logs, setLogs] = useState<SolveLog[]>([])
  const [loading, setLoading] = useState(true)
  const [scoring, setScoring] = useState(false)
  const [progress, setProgress] = useState({ current: 0, total: 0 })
  const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set())
  const [filterType, setFilterType] = useState<string>("")
  const [scoringLogId, setScoringLogId] = useState<number | null>(null)
  const controllerRef = useRef<AbortController | null>(null)

  // Load results and logs on mount
  useEffect(() => {
    loadData()
  }, [])

  const loadData = useCallback(async () => {
    setLoading(true)
    try {
      const [r, l] = await Promise.all([
        fetchAutoTestResults(200),
        fetchAutoTestLogs(200),
      ])
      setResults(r)
      setLogs(l)
    } catch (e) {
      toast.error("Failed to load data: " + String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  // Build a map of solve_log_id -> latest result
  const resultsByLogId = useMemo(() => {
    const map = new Map<number, AutoTestResult>()
    for (const r of results) {
      const existing = map.get(r.solve_log_id)
      if (!existing || r.id > existing.id) {
        map.set(r.solve_log_id, r)
      }
    }
    return map
  }, [results])

  // Get all unique task types
  const taskTypes = useMemo(() => {
    const types = new Set<string>()
    for (const l of logs) {
      if (l.task_type) types.add(l.task_type)
    }
    return Array.from(types).sort()
  }, [logs])

  // Filtered logs
  const filteredLogs = useMemo(() => {
    if (!filterType) return logs
    return logs.filter((l) => l.task_type === filterType)
  }, [logs, filterType])

  // Summary stats
  const summary = useMemo(() => {
    const total = filteredLogs.length
    let scored = 0
    let passed = 0
    let totalScore = 0

    for (const l of filteredLogs) {
      const r = resultsByLogId.get(l.id!)
      if (r) {
        scored++
        if (r.intent_passed) passed++
        totalScore += r.correctness
      }
    }

    return {
      total,
      scored,
      passed,
      passRate: scored > 0 ? Math.round((passed / scored) * 100) : 0,
      avgScore: scored > 0 ? Math.round((totalScore / scored) * 100) : 0,
    }
  }, [filteredLogs, resultsByLogId])

  // Toggle row expand
  const toggleRow = useCallback((logId: number) => {
    setExpandedRows((prev) => {
      const next = new Set(prev)
      if (next.has(logId)) next.delete(logId)
      else next.add(logId)
      return next
    })
  }, [])

  // Score single log
  const handleScoreSingle = useCallback(async (logId: number) => {
    setScoringLogId(logId)
    try {
      await scoreAutoTestLog(logId)
      toast.success(`Scored log #${logId}`)
      await loadData()
    } catch (e) {
      toast.error("Score failed: " + String(e))
    } finally {
      setScoringLogId(null)
    }
  }, [loadData])

  // Batch score
  const handleBatchScore = useCallback(() => {
    if (scoring) {
      controllerRef.current?.abort()
      setScoring(false)
      return
    }

    setScoring(true)
    setProgress({ current: 0, total: 0 })

    const logIds = filterType
      ? filteredLogs.map((l) => l.id!).filter(Boolean)
      : []

    controllerRef.current = streamAutoTestBatch(
      logIds,
      200,
      (event: AutoTestEvent) => {
        if (event.type === "batch_start") {
          setProgress({ current: 0, total: event.total })
        } else if (event.type === "scoring") {
          setProgress({ current: event.index + 1, total: event.total })
        } else if (event.type === "batch_done") {
          toast.success(
            `Batch complete: ${event.passed}/${event.total} passed (avg ${Math.round(event.avg_score * 100)}%)`
          )
        }
      },
      () => {
        setScoring(false)
        loadData()
      },
      (err) => {
        setScoring(false)
        toast.error("Batch error: " + err)
      },
    )
  }, [scoring, filterType, filteredLogs, loadData])

  // Clear results
  const handleClear = useCallback(async () => {
    try {
      await deleteAutoTestResults()
      setResults([])
      toast.success("Cleared all auto-test results")
    } catch (e) {
      toast.error("Clear failed: " + String(e))
    }
  }, [])

  // Parse checks from result
  const parseChecks = (r: AutoTestResult): AutoTestFieldCheck[] => {
    try {
      return JSON.parse(r.checks_json || "[]")
    } catch {
      return []
    }
  }

  const parseIssues = (r: AutoTestResult): string[] => {
    try {
      return JSON.parse(r.issues || "[]")
    } catch {
      return []
    }
  }

  return (
    <div>
      <PageHeader
        title="Auto Tester"
        description="Analyze real competition prompts and score agent execution quality."
      >
        <Button
          variant="outline"
          size="sm"
          onClick={handleClear}
          disabled={scoring || results.length === 0}
        >
          <Trash2 className="h-3.5 w-3.5 mr-1" />
          Clear
        </Button>
        <Button variant="outline" size="sm" onClick={loadData} disabled={scoring}>
          <RotateCcw className="h-3.5 w-3.5 mr-1" />
          Refresh
        </Button>
      </PageHeader>

      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-3 mb-4">
        <SummaryCard label="Total Prompts" value={summary.total} />
        <SummaryCard label="Scored" value={summary.scored} />
        <SummaryCard
          label="Pass Rate"
          value={`${summary.passRate}%`}
          color={summary.passRate >= 80 ? "text-emerald-600" : summary.passRate >= 50 ? "text-amber-600" : "text-red-600"}
        />
        <SummaryCard
          label="Avg Score"
          value={`${summary.avgScore}%`}
          color={summary.avgScore >= 80 ? "text-emerald-600" : summary.avgScore >= 50 ? "text-amber-600" : "text-red-600"}
        />
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3 mb-4">
        <Button
          size="sm"
          onClick={handleBatchScore}
          disabled={loading || filteredLogs.length === 0}
          className={scoring ? "bg-red-600 hover:bg-red-700" : ""}
        >
          {scoring ? (
            <>
              <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />
              Stop ({progress.current}/{progress.total})
            </>
          ) : (
            <>
              <Play className="h-3.5 w-3.5 mr-1" />
              Run Batch Score
            </>
          )}
        </Button>

        <div className="flex items-center gap-1.5">
          <Filter className="h-3.5 w-3.5 text-muted-foreground" />
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="h-8 px-2 rounded-md border border-input bg-background text-[13px]"
          >
            <option value="">All task types</option>
            {taskTypes.map((t) => (
              <option key={t} value={t}>
                {t} ({logs.filter((l) => l.task_type === t).length})
              </option>
            ))}
          </select>
        </div>

        {scoring && progress.total > 0 && (
          <div className="flex-1 max-w-xs">
            <Progress value={(progress.current / progress.total) * 100} className="h-2" />
          </div>
        )}
      </div>

      {/* Results Table */}
      {loading ? (
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground text-sm">
            Loading competition logs...
          </CardContent>
        </Card>
      ) : filteredLogs.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground text-sm">
            No competition logs found. Run the agent with live prompts first.
          </CardContent>
        </Card>
      ) : (
        <div className="border rounded-lg overflow-hidden">
          <table className="w-full text-[13px]">
            <thead>
              <tr className="bg-muted/50 border-b">
                <th className="text-left py-2 px-3 font-medium w-10">#</th>
                <th className="text-left py-2 px-3 font-medium w-32">Task</th>
                <th className="text-left py-2 px-3 font-medium">Prompt</th>
                <th className="text-center py-2 px-3 font-medium w-20">Score</th>
                <th className="text-center py-2 px-3 font-medium w-16">Intent</th>
                <th className="text-center py-2 px-3 font-medium w-20">API</th>
                <th className="text-center py-2 px-3 font-medium w-20"></th>
              </tr>
            </thead>
            <tbody>
              {filteredLogs.map((log) => {
                const result = resultsByLogId.get(log.id!)
                const expanded = expandedRows.has(log.id!)
                const checks = result ? parseChecks(result) : []
                const issues = result ? parseIssues(result) : []

                return (
                  <LogRow
                    key={log.id}
                    log={log}
                    result={result}
                    expanded={expanded}
                    checks={checks}
                    issues={issues}
                    onToggle={() => toggleRow(log.id!)}
                    onRescore={() => handleScoreSingle(log.id!)}
                    rescoring={scoringLogId === log.id}
                  />
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function SummaryCard({
  label,
  value,
  color,
}: {
  label: string
  value: string | number
  color?: string
}) {
  return (
    <Card>
      <CardContent className="py-3 px-4">
        <p className="text-[11px] uppercase tracking-wider text-muted-foreground font-medium">
          {label}
        </p>
        <p className={cn("text-2xl font-bold mt-0.5 tabular-nums", color)}>
          {value}
        </p>
      </CardContent>
    </Card>
  )
}

function LogRow({
  log,
  result,
  expanded,
  checks,
  issues,
  onToggle,
  onRescore,
  rescoring,
}: {
  log: SolveLog
  result: AutoTestResult | undefined
  expanded: boolean
  checks: AutoTestFieldCheck[]
  issues: string[]
  onToggle: () => void
  onRescore: () => void
  rescoring: boolean
}) {
  const scored = !!result

  return (
    <>
      <tr
        className={cn(
          "border-b hover:bg-muted/30 cursor-pointer transition-colors",
          expanded && "bg-muted/20"
        )}
        onClick={onToggle}
      >
        <td className="py-2 px-3 text-muted-foreground tabular-nums">
          <div className="flex items-center gap-1">
            {expanded ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )}
            {log.id}
          </div>
        </td>
        <td className="py-2 px-3">
          <Badge variant="outline" className="text-[11px] font-mono">
            {log.task_type || "unknown"}
          </Badge>
        </td>
        <td className="py-2 px-3">
          <span className="line-clamp-1 text-muted-foreground">
            {log.prompt?.slice(0, 120)}
          </span>
        </td>
        <td className="py-2 px-3 text-center">
          {scored ? (
            <ScoreBar
              total={result!.total_points}
              max={result!.max_points}
              correctness={result!.correctness}
            />
          ) : (
            <span className="text-muted-foreground/50">--</span>
          )}
        </td>
        <td className="py-2 px-3 text-center">
          {scored ? (
            result!.intent_passed ? (
              <CheckCircle2 className="h-4 w-4 text-emerald-500 mx-auto" />
            ) : (
              <XCircle className="h-4 w-4 text-red-500 mx-auto" />
            )
          ) : (
            <span className="text-muted-foreground/50">--</span>
          )}
        </td>
        <td className="py-2 px-3 text-center tabular-nums">
          <span className="text-muted-foreground">
            {log.api_calls}
            {log.api_errors > 0 && (
              <span className="text-red-500 ml-0.5">/{log.api_errors}e</span>
            )}
          </span>
        </td>
        <td className="py-2 px-3 text-center">
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-2 text-[11px]"
            onClick={(e) => {
              e.stopPropagation()
              onRescore()
            }}
            disabled={rescoring}
          >
            {rescoring ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <RotateCcw className="h-3 w-3" />
            )}
          </Button>
        </td>
      </tr>

      {/* Expanded detail row */}
      {expanded && (
        <tr className="bg-muted/10">
          <td colSpan={7} className="px-6 py-3">
            {scored ? (
              <div className="space-y-3">
                {/* Field checks */}
                <div>
                  <p className="text-[11px] uppercase tracking-wider text-muted-foreground font-medium mb-1.5">
                    Field Checks
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {checks.map((c, i) => (
                      <div
                        key={i}
                        className={cn(
                          "inline-flex items-center gap-1 px-2 py-0.5 rounded text-[12px] border",
                          c.passed
                            ? "bg-emerald-50 border-emerald-200 text-emerald-700 dark:bg-emerald-950/30 dark:border-emerald-800 dark:text-emerald-400"
                            : "bg-red-50 border-red-200 text-red-700 dark:bg-red-950/30 dark:border-red-800 dark:text-red-400"
                        )}
                      >
                        {c.passed ? (
                          <CheckCircle2 className="h-3 w-3" />
                        ) : (
                          <XCircle className="h-3 w-3" />
                        )}
                        <span className="font-mono">{c.field}</span>
                        <span className="text-[10px] opacity-60">
                          {c.points}/{c.max}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Intent reasoning */}
                <div>
                  <p className="text-[11px] uppercase tracking-wider text-muted-foreground font-medium mb-1">
                    Intent Verification
                  </p>
                  <div
                    className={cn(
                      "inline-flex items-start gap-1.5 px-2.5 py-1.5 rounded text-[12px] border",
                      result!.intent_passed
                        ? "bg-emerald-50 border-emerald-200 dark:bg-emerald-950/30 dark:border-emerald-800"
                        : "bg-red-50 border-red-200 dark:bg-red-950/30 dark:border-red-800"
                    )}
                  >
                    {result!.intent_passed ? (
                      <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500 mt-0.5 shrink-0" />
                    ) : (
                      <XCircle className="h-3.5 w-3.5 text-red-500 mt-0.5 shrink-0" />
                    )}
                    <span className="text-muted-foreground">
                      {result!.intent_reasoning || "No reasoning provided"}
                    </span>
                  </div>
                </div>

                {/* Issues */}
                {issues.length > 0 && (
                  <div>
                    <p className="text-[11px] uppercase tracking-wider text-muted-foreground font-medium mb-1">
                      Issues
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {issues.map((issue, i) => (
                        <Badge key={i} variant="destructive" className="text-[11px]">
                          {issue}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Prompt */}
                <div>
                  <p className="text-[11px] uppercase tracking-wider text-muted-foreground font-medium mb-1">
                    Full Prompt
                  </p>
                  <p className="text-[12px] text-muted-foreground bg-muted/50 rounded px-2.5 py-1.5 whitespace-pre-wrap">
                    {result!.prompt}
                  </p>
                </div>
              </div>
            ) : (
              <p className="text-[12px] text-muted-foreground italic">
                Not scored yet. Click the rescore button to analyze this prompt.
              </p>
            )}
          </td>
        </tr>
      )}
    </>
  )
}

function ScoreBar({
  total,
  max,
  correctness,
}: {
  total: number
  max: number
  correctness: number
}) {
  const pct = Math.round(correctness * 100)
  const color =
    pct >= 80
      ? "bg-emerald-500"
      : pct >= 50
        ? "bg-amber-500"
        : "bg-red-500"

  return (
    <div className="flex items-center gap-1.5">
      <span className="tabular-nums text-[12px] font-medium">
        {total}/{max}
      </span>
      <div className="w-8 h-1.5 rounded-full bg-muted overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all", color)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
