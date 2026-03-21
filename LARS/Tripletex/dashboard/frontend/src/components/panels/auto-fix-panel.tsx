import { useState, useRef, useCallback, useMemo, useEffect } from "react"
import { useTasks, useLanguages } from "@/hooks/use-api"
import { streamAutoFix, applyFixes, streamBatchAutoFix, fetchLastEvalResults, streamBatchTaskFix, streamLiveEval } from "@/lib/api"
import type {
  AutoFixEvent,
  AutoFixScore,
  AutoFixParsedFix,
  AutoFixApplyResult,
  BatchAutoFixEvent,
  BatchTaskFixEvent,
  LastEvalResult,
  FieldCheck,
  ToolCall,
  LiveEvalEvent,
  LiveLogExplanation,
} from "@/types/api"
import { PageHeader } from "@/components/layout/page-header"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Progress } from "@/components/ui/progress"
import { toast } from "sonner"
import { cn } from "@/lib/utils"
import {
  Wrench,
  Loader2,
  CheckCircle2,
  XCircle,
  FileCode,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Copy,
  Check,
  Rocket,
  ListChecks,
  ClipboardCopy,
  Play,
  Zap,
} from "lucide-react"

type Mode = "tasks" | "live" | "batch" | "single"

export function AutoFixPanel() {
  const [mode, setMode] = useState<Mode>("tasks")

  return (
    <div>
      <PageHeader
        title="Auto Fix"
        description="Pick tasks, run evals, and auto-fix code until everything passes."
      >
        <div className="flex items-center bg-muted/60 rounded-lg p-0.5">
          <button
            onClick={() => setMode("tasks")}
            className={cn(
              "h-7 px-3 rounded-md text-[12px] font-medium transition-all duration-150 flex items-center gap-1",
              mode === "tasks"
                ? "bg-blue-500/20 text-blue-700 shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <ListChecks className="h-3 w-3" />
            Tasks
          </button>
          <button
            onClick={() => setMode("live")}
            className={cn(
              "h-7 px-3 rounded-md text-[12px] font-medium transition-all duration-150 flex items-center gap-1",
              mode === "live"
                ? "bg-emerald-500/20 text-emerald-700 shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <Zap className="h-3 w-3" />
            Live
          </button>
          <button
            onClick={() => setMode("batch")}
            className={cn(
              "h-7 px-3 rounded-md text-[12px] font-medium transition-all duration-150 flex items-center gap-1",
              mode === "batch"
                ? "bg-orange-500/20 text-orange-700 shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <Rocket className="h-3 w-3" />
            Real Logs
          </button>
          <button
            onClick={() => setMode("single")}
            className={cn(
              "h-7 px-3 rounded-md text-[12px] font-medium transition-all duration-150 flex items-center gap-1",
              mode === "single"
                ? "bg-white text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <Wrench className="h-3 w-3" />
            Single Task
          </button>
        </div>
      </PageHeader>

      {mode === "tasks" ? <TaskPickerView /> : mode === "live" ? <LiveSubmissionsView /> : mode === "batch" ? <BatchAutoFixView /> : <SingleAutoFixView />}
    </div>
  )
}

// ── Live Submissions View ────────────────────────────────────────────

interface LiveLogResult {
  logId: number
  taskType: string
  passed: boolean
  reasoning: string
  issues: string[]
  apiCalls: number
  apiErrors: number
  runLogJson: string
  explanation?: LiveLogExplanation
}

function LiveSubmissionsView() {
  const [running, setRunning] = useState(false)
  const [autoFix, setAutoFix] = useState(false)
  const [logLimit, setLogLimit] = useState(50)
  const controllerRef = useRef<AbortController | null>(null)

  // State
  const [results, setResults] = useState<LiveLogResult[]>([])
  const [totalLogs, setTotalLogs] = useState(0)
  const [progress, setProgress] = useState(0)
  const [currentLog, setCurrentLog] = useState("")
  const [doneMessage, setDoneMessage] = useState("")
  const [errorMessage, setErrorMessage] = useState("")
  const [expandedLog, setExpandedLog] = useState<number | null>(null)
  const [copiedLogId, setCopiedLogId] = useState<number | null>(null)

  // Fix state
  const [fixHistory, setFixHistory] = useState<{ taskType: string; fixes: AutoFixApplyResult[] }[]>([])
  const [fixingType, setFixingType] = useState("")

  const passedCount = useMemo(() => results.filter(r => r.passed).length, [results])
  const failedCount = useMemo(() => results.filter(r => !r.passed).length, [results])

  const handleStart = useCallback(() => {
    setResults([])
    setDoneMessage("")
    setErrorMessage("")
    setFixHistory([])
    setFixingType("")
    setExpandedLog(null)
    setRunning(true)

    controllerRef.current = streamLiveEval(
      0, // since_id=0 → all recent
      logLimit,
      autoFix,
      (event: LiveEvalEvent) => {
        switch (event.type) {
          case "live_start":
            setTotalLogs(event.total)
            break

          case "evaluating":
            setCurrentLog(`#${event.log_id} ${event.task_type}`)
            setProgress(((event.index) / event.total) * 100)
            break

          case "log_result":
            setProgress(prev => prev + (100 / totalLogs || 1))
            setResults(prev => [...prev, {
              logId: event.log_id,
              taskType: event.task_type,
              passed: event.passed,
              reasoning: event.reasoning,
              issues: event.issues,
              apiCalls: event.api_calls,
              apiErrors: event.api_errors,
              runLogJson: JSON.stringify(event.run_log, null, 2),
              explanation: event.explanation,
            }])
            break

          case "eval_error":
            setResults(prev => [...prev, {
              logId: event.log_id,
              taskType: event.task_type,
              passed: false,
              reasoning: event.error,
              issues: [],
              apiCalls: 0,
              apiErrors: 0,
              runLogJson: "",
            }])
            break

          case "fixing":
            setFixingType(event.task_type)
            break

          case "fixes_applied":
            setFixHistory(prev => [...prev, { taskType: event.task_type, fixes: event.results }])
            setFixingType("")
            break

          case "fix_error":
            setFixingType("")
            toast.error(`Fix error (${event.task_type}): ${event.error}`)
            break

          case "live_done":
            setDoneMessage(event.message)
            setCurrentLog("")
            setProgress(100)
            break
        }
      },
      () => setRunning(false),
      (err) => {
        setErrorMessage(err)
        setRunning(false)
      }
    )
  }, [logLimit, autoFix, totalLogs])

  const handleStop = useCallback(() => {
    controllerRef.current?.abort()
    setRunning(false)
  }, [])

  const handleCopyLog = useCallback((logId: number, json: string) => {
    navigator.clipboard.writeText(json)
    setCopiedLogId(logId)
    toast.success("Run log copied to clipboard")
    setTimeout(() => setCopiedLogId(null), 2000)
  }, [])

  // Group results by task type
  const byTaskType = useMemo(() => {
    const map = new Map<string, LiveLogResult[]>()
    for (const r of results) {
      const arr = map.get(r.taskType) || []
      arr.push(r)
      map.set(r.taskType, arr)
    }
    return Array.from(map.entries()).sort(([, a], [, b]) => {
      const aFail = a.filter(r => !r.passed).length
      const bFail = b.filter(r => !r.passed).length
      return bFail - aFail
    })
  }, [results])

  return (
    <div className="space-y-4">
      {/* Controls */}
      <Card className="shadow-premium">
        <CardContent className="p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-9 w-9 rounded-lg bg-emerald-100 flex items-center justify-center shrink-0">
                <Zap className="h-4 w-4 text-emerald-700" />
              </div>
              <div>
                <p className="text-[13px] font-semibold">Live Submissions</p>
                <p className="text-[11px] text-muted-foreground">
                  Submit on app.ainm.no, then evaluate results here. Auto-fix failures.
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5">
                <label className="text-[11px] text-muted-foreground">Logs</label>
                <input
                  type="number" min={1} max={200} value={logLimit}
                  onChange={e => setLogLimit(Math.max(1, parseInt(e.target.value) || 50))}
                  disabled={running}
                  className="w-14 h-7 text-[12px] text-center tabular-nums border rounded-md bg-background"
                />
              </div>
              <label className="flex items-center gap-1.5 text-[12px]">
                <input
                  type="checkbox" checked={autoFix}
                  onChange={e => setAutoFix(e.target.checked)}
                  disabled={running}
                  className="rounded"
                />
                Auto-fix
              </label>
              {running ? (
                <Button variant="destructive" size="sm" onClick={handleStop}>Stop</Button>
              ) : (
                <Button
                  onClick={handleStart}
                  className="h-9 px-5 font-semibold bg-gradient-to-r from-emerald-500 to-teal-500 hover:shadow-lg hover:shadow-emerald-500/25 hover:scale-[1.02] active:scale-[0.98] transition-all"
                >
                  <Zap className="h-4 w-4 mr-2" />
                  Evaluate Submissions
                </Button>
              )}
            </div>
          </div>

          {/* Progress */}
          {(running || doneMessage) && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-[12px]">
                <div className="flex items-center gap-2">
                  {running && <Loader2 className="h-3.5 w-3.5 animate-spin text-emerald-500" />}
                  {doneMessage ? (
                    <span className="font-medium">{doneMessage}</span>
                  ) : fixingType ? (
                    <span className="text-muted-foreground">
                      Fixing <span className="font-medium text-foreground">{fixingType}</span>...
                    </span>
                  ) : currentLog ? (
                    <span className="text-muted-foreground">
                      Evaluating <span className="font-medium text-foreground">{currentLog}</span>
                    </span>
                  ) : (
                    <span className="text-muted-foreground">Starting...</span>
                  )}
                </div>
                <div className="flex items-center gap-3 tabular-nums">
                  {passedCount > 0 && <span className="text-emerald-600 font-semibold">{passedCount} pass</span>}
                  {failedCount > 0 && <span className="text-red-600 font-semibold">{failedCount} fail</span>}
                  {totalLogs > 0 && <span className="text-muted-foreground">{results.length}/{totalLogs}</span>}
                </div>
              </div>
              <Progress value={progress} className="h-2" />
            </div>
          )}
        </CardContent>
      </Card>

      {/* Error */}
      {errorMessage && (
        <Card className="border-red-200 bg-red-50/50">
          <CardContent className="p-4">
            <div className="flex items-start gap-2">
              <XCircle className="h-4 w-4 text-red-500 mt-0.5 shrink-0" />
              <p className="text-[13px] text-red-700">{errorMessage}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results by task type */}
      {byTaskType.length > 0 && (
        <Card className="shadow-premium">
          <CardContent className="p-0">
            <div className="px-4 py-2.5 border-b bg-muted/20 flex items-center justify-between">
              <span className="text-[12px] font-semibold text-muted-foreground">
                Submission Results
              </span>
              <div className="flex items-center gap-2">
                <Badge className="bg-emerald-500 text-[10px]">{passedCount} passed</Badge>
                {failedCount > 0 && <Badge variant="destructive" className="text-[10px]">{failedCount} failed</Badge>}
              </div>
            </div>
            <div className="divide-y">
              {byTaskType.map(([taskType, logs]) => {
                const tp = logs.filter(r => r.passed).length
                const tf = logs.filter(r => !r.passed).length
                const allPass = tf === 0
                return (
                  <div key={taskType}>
                    <button
                      onClick={() => setExpandedLog(expandedLog === logs[0].logId ? null : logs[0].logId)}
                      className={cn(
                        "w-full flex items-center gap-2 px-4 py-2.5 text-[12px] transition-colors",
                        allPass ? "hover:bg-emerald-50/60" : "hover:bg-red-50/60"
                      )}
                    >
                      {allPass ? (
                        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500 shrink-0" />
                      ) : (
                        <XCircle className="h-3.5 w-3.5 text-red-500 shrink-0" />
                      )}
                      <span className="font-medium flex-1 text-left">{taskType}</span>
                      <span className="tabular-nums text-muted-foreground">{tp}/{tp + tf}</span>
                      {expandedLog === logs[0].logId ? (
                        <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
                      ) : (
                        <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                      )}
                    </button>
                    {expandedLog === logs[0].logId && (
                      <div className="px-4 pb-3 space-y-2">
                        {logs.map(r => (
                          <LiveLogRow
                            key={r.logId}
                            result={r}
                            onCopyJson={() => handleCopyLog(r.logId, r.runLogJson)}
                            copied={copiedLogId === r.logId}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Fix History */}
      {fixHistory.length > 0 && (
        <Card className="shadow-premium border-orange-200">
          <CardContent className="p-4 space-y-3">
            <h3 className="text-[14px] font-semibold flex items-center gap-2">
              <FileCode className="h-4 w-4 text-orange-500" />
              Applied Fixes ({fixHistory.reduce((s, f) => s + f.fixes.filter(a => a.applied).length, 0)})
            </h3>
            <div className="space-y-2">
              {fixHistory.map((fh, i) => (
                <div key={i} className="rounded-lg border p-3 space-y-1.5">
                  <span className="text-[12px] font-medium">{fh.taskType}</span>
                  {fh.fixes.map((a, j) => (
                    <div key={j} className={cn("text-[11px] px-2 py-1 rounded", a.applied ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700")}>
                      {a.applied ? <><CheckCircle2 className="h-3 w-3 inline mr-1" />{a.file}: {a.reason}</> : <><AlertTriangle className="h-3 w-3 inline mr-1" />{a.file}: {a.error}</>}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function LiveLogRow({
  result,
  onCopyJson,
  copied,
}: {
  result: LiveLogResult
  onCopyJson: () => void
  copied: boolean
}) {
  const [showDetails, setShowDetails] = useState(!result.passed)

  return (
    <div className={cn(
      "rounded-lg border p-3 space-y-2",
      result.passed ? "border-emerald-200 bg-emerald-50/30" : "border-red-200 bg-red-50/30"
    )}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {result.passed ? (
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
          ) : (
            <XCircle className="h-3.5 w-3.5 text-red-500" />
          )}
          <span className="text-[12px] font-medium">{result.passed ? "PASS" : "FAIL"}</span>
          <span className="text-[11px] font-mono text-muted-foreground">#{result.logId}</span>
          <span className="text-[11px] tabular-nums text-muted-foreground">
            {result.apiCalls} calls
            {result.apiErrors > 0 && <span className="text-red-500"> ({result.apiErrors} err)</span>}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          {result.runLogJson && (
            <Button variant="outline" size="sm" onClick={onCopyJson} className="h-6 text-[10px] gap-1 px-2">
              {copied ? <Check className="h-3 w-3 text-emerald-500" /> : <ClipboardCopy className="h-3 w-3" />}
              Copy JSON
            </Button>
          )}
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="text-[10px] text-muted-foreground hover:text-foreground transition-colors"
          >
            {showDetails ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
          </button>
        </div>
      </div>

      {/* Reasoning */}
      <p className="text-[11px] text-muted-foreground">{result.reasoning}</p>

      {/* Expanded details */}
      {showDetails && result.explanation && (
        <div className="space-y-1.5 pt-1 border-t border-red-200/50">
          {/* Issues */}
          {result.issues.length > 0 && (
            <div className="space-y-0.5">
              {result.issues.map((issue, i) => (
                <p key={i} className="text-[11px] text-red-700 flex items-start gap-1.5">
                  <span className="text-red-400 shrink-0">·</span>
                  {issue}
                </p>
              ))}
            </div>
          )}

          {/* API errors */}
          {result.explanation.api_errors.length > 0 && (
            <div className="space-y-0.5">
              <p className="text-[10px] font-semibold text-red-600 uppercase tracking-wider">API Errors</p>
              {result.explanation.api_errors.map((e, i) => (
                <p key={i} className="text-[10px] text-red-600 font-mono">
                  {e.method} {e.url} → {e.status}
                </p>
              ))}
            </div>
          )}

          {/* Failed tools */}
          {result.explanation.failed_tools.length > 0 && (
            <div className="space-y-0.5">
              <p className="text-[10px] font-semibold text-red-600 uppercase tracking-wider">Failed Tools</p>
              {result.explanation.failed_tools.map((t, i) => (
                <p key={i} className="text-[10px] text-red-600 font-mono">{t.tool}: {t.error}</p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Task Picker View ─────────────────────────────────────────────────

interface TaskRunResult {
  task_name: string
  passed: boolean
  correctness: number
  score: number
  max_possible: number
  api_calls: number
  api_errors: number
  currentAttempt?: number
  status: "pending" | "running" | "done" | "error"
  error?: string
}

function TaskPickerView() {
  const { data: tasks, isLoading } = useTasks()
  const { data: languages } = useLanguages()
  const [lastResults, setLastResults] = useState<LastEvalResult[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [language, setLanguage] = useState("")
  const [maxRetries, setMaxRetries] = useState(2)
  const [autoApply, setAutoApply] = useState(true)
  const [running, setRunning] = useState(false)
  const controllerRef = useRef<AbortController | null>(null)

  // Run state
  const [runResults, setRunResults] = useState<Map<string, TaskRunResult>>(new Map())
  const [currentTask, setCurrentTask] = useState("")
  const [progress, setProgress] = useState(0)
  const [doneMessage, setDoneMessage] = useState("")
  const [fixHistory, setFixHistory] = useState<{ task: string; fixes: AutoFixApplyResult[] }[]>([])

  // Load last results on mount
  useEffect(() => {
    fetchLastEvalResults().then(setLastResults).catch(() => {})
  }, [])

  const lastResultsMap = useMemo(() => {
    const m = new Map<string, LastEvalResult>()
    for (const r of lastResults) m.set(r.task, r)
    return m
  }, [lastResults])

  // Counts
  const nPass = useMemo(() => [...lastResultsMap.values()].filter(r => r.status === "PASS").length, [lastResultsMap])
  const nFail = useMemo(() => [...lastResultsMap.values()].filter(r => r.status === "FAIL").length, [lastResultsMap])
  const nSkip = useMemo(() => [...lastResultsMap.values()].filter(r => r.status === "SKIP").length, [lastResultsMap])

  // Quick select helpers
  const selectAll = useCallback(() => {
    if (tasks) setSelected(new Set(tasks.map(t => t.name)))
  }, [tasks])
  const selectNone = useCallback(() => setSelected(new Set()), [])
  const selectFailed = useCallback(() => {
    setSelected(new Set([...lastResultsMap.entries()].filter(([, r]) => r.status === "FAIL").map(([k]) => k)))
  }, [lastResultsMap])
  const selectTier = useCallback((tier: number) => {
    if (tasks) setSelected(new Set(tasks.filter(t => t.tier === tier).map(t => t.name)))
  }, [tasks])
  const toggleTask = useCallback((name: string) => {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }, [])

  const handleStart = useCallback(() => {
    if (selected.size === 0) {
      toast.error("Select at least one task")
      return
    }
    const taskNames = [...selected]
    setRunning(true)
    setDoneMessage("")
    setFixHistory([])
    setRunResults(new Map(taskNames.map(n => [n, { task_name: n, passed: false, correctness: 0, score: 0, max_possible: 0, api_calls: 0, api_errors: 0, status: "pending" }])))

    controllerRef.current = streamBatchTaskFix(
      taskNames,
      language,
      maxRetries,
      autoApply,
      (event: BatchTaskFixEvent) => {
        switch (event.type) {
          case "task_start":
            setCurrentTask(event.task_name)
            setProgress(((event.index) / event.total) * 100)
            setRunResults(prev => {
              const next = new Map(prev)
              next.set(event.task_name, { ...next.get(event.task_name)!, status: "running" })
              return next
            })
            break

          case "eval_result":
            setRunResults(prev => {
              const next = new Map(prev)
              const existing = next.get(event.task_name)!
              next.set(event.task_name, {
                ...existing,
                correctness: event.correctness,
                score: event.score,
                max_possible: event.max_possible,
                api_calls: event.api_calls,
                api_errors: event.api_errors,
                passed: event.passed,
                currentAttempt: event.attempt,
              })
              return next
            })
            break

          case "fixes_applied":
            setFixHistory(prev => [...prev, { task: event.task_name, fixes: event.results }])
            break

          case "task_done":
            setProgress(((event.index + 1) / event.total) * 100)
            setRunResults(prev => {
              const next = new Map(prev)
              next.set(event.task_name, {
                task_name: event.task_name,
                passed: event.passed,
                correctness: event.correctness,
                score: event.score,
                max_possible: event.max_possible,
                api_calls: event.api_calls,
                api_errors: event.api_errors,
                status: "done",
              })
              return next
            })
            break

          case "batch_done":
            setDoneMessage(`Done: ${event.passed}/${event.total} passed`)
            setCurrentTask("")
            break

          case "eval_error":
            setRunResults(prev => {
              const next = new Map(prev)
              next.set(event.task_name, { ...next.get(event.task_name)!, status: "error", error: event.error })
              return next
            })
            break

          case "error":
            toast.error(event.message)
            break
        }
      },
      () => setRunning(false),
      (err) => {
        toast.error(err)
        setRunning(false)
      }
    )
  }, [selected, language, maxRetries, autoApply])

  const handleStop = useCallback(() => {
    controllerRef.current?.abort()
    setRunning(false)
  }, [])

  if (isLoading) {
    return <div className="space-y-4"><Skeleton className="h-8 w-48" /><Skeleton className="h-80 w-full rounded-xl" /></div>
  }

  const tierGroups = [1, 2, 3].map(tier => ({
    tier,
    tasks: (tasks ?? []).filter(t => t.tier === tier),
  }))

  const runResultsList = [...runResults.values()]
  const rPass = runResultsList.filter(r => r.passed).length
  const rFail = runResultsList.filter(r => r.status === "done" && !r.passed).length

  return (
    <div className="space-y-4">
      {/* Controls bar */}
      <Card className="shadow-premium">
        <CardContent className="p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-lg bg-blue-100 flex items-center justify-center shrink-0">
                <ListChecks className="h-4 w-4 text-blue-700" />
              </div>
              <div>
                <p className="text-[13px] font-semibold">Task Picker</p>
                <p className="text-[11px] text-muted-foreground">
                  Last eval: {nPass} pass, {nFail} fail, {nSkip} skip
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {/* Language */}
              <select
                value={language}
                onChange={e => setLanguage(e.target.value)}
                disabled={running}
                className="h-7 px-2 rounded-md border text-[11px] bg-background"
              >
                <option value="">Random lang</option>
                {languages && Object.entries(languages).map(([code]) => (
                  <option key={code} value={code}>{code}</option>
                ))}
              </select>
              {/* Retries */}
              <div className="flex items-center gap-1">
                <label className="text-[11px] text-muted-foreground">Retries</label>
                <input
                  type="number" min={1} max={5} value={maxRetries}
                  onChange={e => setMaxRetries(Math.max(1, parseInt(e.target.value) || 1))}
                  disabled={running}
                  className="w-10 h-7 text-[11px] text-center tabular-nums border rounded-md bg-background"
                />
              </div>
              {/* Auto-apply toggle */}
              <label className="flex items-center gap-1 text-[11px]">
                <input
                  type="checkbox" checked={autoApply}
                  onChange={e => setAutoApply(e.target.checked)}
                  disabled={running}
                  className="rounded"
                />
                Auto-fix
              </label>
              {/* Run button */}
              {running ? (
                <Button variant="destructive" size="sm" onClick={handleStop}>Stop</Button>
              ) : (
                <Button
                  onClick={handleStart}
                  disabled={selected.size === 0}
                  size="sm"
                  className="h-8 px-4 font-semibold bg-gradient-to-r from-blue-500 to-cyan-500 hover:shadow-lg hover:shadow-blue-500/25 hover:scale-[1.02] active:scale-[0.98] transition-all"
                >
                  <Wrench className="h-3.5 w-3.5 mr-1.5" />
                  Run {selected.size} task{selected.size !== 1 ? "s" : ""}
                </Button>
              )}
            </div>
          </div>

          {/* Quick select buttons */}
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-[11px] text-muted-foreground mr-1">Select:</span>
            <button onClick={selectAll} className="h-6 px-2 rounded text-[11px] bg-muted hover:bg-muted/80 transition-colors">All</button>
            <button onClick={selectNone} className="h-6 px-2 rounded text-[11px] bg-muted hover:bg-muted/80 transition-colors">None</button>
            <button onClick={selectFailed} disabled={nFail === 0} className="h-6 px-2 rounded text-[11px] bg-red-100 text-red-700 hover:bg-red-200 transition-colors disabled:opacity-40">Failed ({nFail})</button>
            <span className="text-muted-foreground">|</span>
            <button onClick={() => selectTier(1)} className="h-6 px-2 rounded text-[11px] bg-emerald-100 text-emerald-700 hover:bg-emerald-200 transition-colors">T1</button>
            <button onClick={() => selectTier(2)} className="h-6 px-2 rounded text-[11px] bg-amber-100 text-amber-700 hover:bg-amber-200 transition-colors">T2</button>
            <button onClick={() => selectTier(3)} className="h-6 px-2 rounded text-[11px] bg-red-100 text-red-700 hover:bg-red-200 transition-colors">T3</button>
          </div>

          {/* Progress bar when running */}
          {(running || doneMessage) && (
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-[12px]">
                <div className="flex items-center gap-2">
                  {running && <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-500" />}
                  {doneMessage ? (
                    <span className="font-medium">{doneMessage}</span>
                  ) : currentTask ? (
                    <span className="text-muted-foreground">Running <span className="font-medium text-foreground">{currentTask}</span>...</span>
                  ) : (
                    <span className="text-muted-foreground">Starting...</span>
                  )}
                </div>
                <div className="flex items-center gap-2 tabular-nums">
                  {rPass > 0 && <span className="text-emerald-600 font-semibold">{rPass} pass</span>}
                  {rFail > 0 && <span className="text-red-600 font-semibold">{rFail} fail</span>}
                </div>
              </div>
              <Progress value={progress} className="h-1.5" />
            </div>
          )}
        </CardContent>
      </Card>

      {/* Task grid by tier */}
      {tierGroups.map(({ tier, tasks: tierTasks }) => (
        <Card key={tier}>
          <CardContent className="p-0">
            <div className="px-4 py-2 border-b bg-muted/20 flex items-center justify-between">
              <span className="text-[12px] font-semibold">
                Tier {tier}
                <span className="text-muted-foreground font-normal ml-1.5">
                  ({tierTasks.length} tasks)
                </span>
              </span>
              <span className="text-[11px] text-muted-foreground">
                {tierTasks.filter(t => selected.has(t.name)).length} selected
              </span>
            </div>
            <div className="divide-y">
              {tierTasks.map(task => {
                const lr = lastResultsMap.get(task.name)
                const rr = runResults.get(task.name)
                const isSelected = selected.has(task.name)
                return (
                  <TaskRow
                    key={task.name}
                    name={task.name}
                    tier={tier}
                    description={task.description}
                    maxPoints={task.max_points}
                    lastStatus={lr?.status}
                    classifierOk={lr?.classifier?.correct}
                    runResult={rr}
                    isSelected={isSelected}
                    onToggle={() => toggleTask(task.name)}
                    disabled={running}
                  />
                )
              })}
            </div>
          </CardContent>
        </Card>
      ))}

      {/* Fix history */}
      {fixHistory.length > 0 && (
        <Card className="shadow-premium border-orange-200">
          <CardContent className="p-4 space-y-3">
            <h3 className="text-[14px] font-semibold flex items-center gap-2">
              <FileCode className="h-4 w-4 text-orange-500" />
              Applied Fixes ({fixHistory.reduce((s, f) => s + f.fixes.filter(a => a.applied).length, 0)})
            </h3>
            <div className="space-y-2">
              {fixHistory.map((fh, i) => (
                <div key={i} className="rounded-lg border p-3 space-y-1.5">
                  <span className="text-[12px] font-medium">{fh.task}</span>
                  {fh.fixes.map((a, j) => (
                    <div key={j} className={cn("text-[11px] px-2 py-1 rounded", a.applied ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700")}>
                      {a.applied ? <><CheckCircle2 className="h-3 w-3 inline mr-1" />{a.file}: {a.reason}</> : <><AlertTriangle className="h-3 w-3 inline mr-1" />{a.file}: {a.error}</>}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function TaskRow({
  name, tier, description, maxPoints,
  lastStatus, classifierOk, runResult, isSelected, onToggle, disabled,
}: {
  name: string
  tier: number
  description: string
  maxPoints: number
  lastStatus?: "PASS" | "FAIL" | "SKIP"
  classifierOk?: boolean
  runResult?: TaskRunResult
  isSelected: boolean
  onToggle: () => void
  disabled: boolean
}) {
  const tierColor = tier === 1 ? "emerald" : tier === 2 ? "amber" : "red"

  // Determine status badge from run result or last result
  let statusBadge: React.ReactNode = null
  if (runResult?.status === "running") {
    statusBadge = (
      <Badge variant="outline" className="text-[10px] border-blue-300 text-blue-600">
        <Loader2 className="h-2.5 w-2.5 mr-0.5 animate-spin" />
        {runResult.currentAttempt && runResult.currentAttempt > 1 ? `Retry ${runResult.currentAttempt}` : "Running"}
      </Badge>
    )
  } else if (runResult?.status === "done") {
    statusBadge = runResult.passed ? (
      <Badge className="text-[10px] bg-emerald-500">
        <CheckCircle2 className="h-2.5 w-2.5 mr-0.5" />
        {(runResult.correctness * 100).toFixed(0)}%
      </Badge>
    ) : (
      <Badge variant="destructive" className="text-[10px]">
        <XCircle className="h-2.5 w-2.5 mr-0.5" />
        {(runResult.correctness * 100).toFixed(0)}%
      </Badge>
    )
  } else if (runResult?.status === "error") {
    statusBadge = <Badge variant="destructive" className="text-[10px]">Error</Badge>
  } else if (lastStatus === "PASS") {
    statusBadge = <Badge variant="outline" className="text-[10px] border-emerald-300 text-emerald-600">PASS</Badge>
  } else if (lastStatus === "FAIL") {
    statusBadge = <Badge variant="outline" className="text-[10px] border-red-300 text-red-600">FAIL</Badge>
  } else if (lastStatus === "SKIP") {
    statusBadge = <Badge variant="outline" className="text-[10px] text-muted-foreground">SKIP</Badge>
  }

  return (
    <button
      onClick={onToggle}
      disabled={disabled}
      className={cn(
        "w-full flex items-center gap-3 px-4 py-2 text-left text-[12px] transition-colors",
        isSelected ? "bg-blue-50/80" : "hover:bg-muted/30",
        disabled && "opacity-60"
      )}
    >
      <input
        type="checkbox"
        checked={isSelected}
        readOnly
        className="rounded border-gray-300 pointer-events-none"
      />
      <div className={cn("w-1.5 h-1.5 rounded-full shrink-0", `bg-${tierColor}-500`)} />
      <span className="font-medium w-[260px] truncate">{name.replace(/_/g, " ")}</span>
      <span className="text-muted-foreground flex-1 truncate">{description}</span>
      <span className="tabular-nums text-muted-foreground w-[50px] text-right shrink-0">{maxPoints}p</span>
      {classifierOk === false && (
        <Badge variant="outline" className="text-[9px] border-amber-300 text-amber-600 shrink-0">cls</Badge>
      )}
      <div className="w-[70px] flex justify-end shrink-0">{statusBadge}</div>
      {runResult?.status === "done" && (
        <span className="tabular-nums text-[11px] text-muted-foreground w-[80px] text-right shrink-0">
          {runResult.score}/{runResult.max_possible} | {runResult.api_calls}c
        </span>
      )}
    </button>
  )
}

// ── Batch Auto-Fix View (Real Logs) ─────────────────────────────────

interface LogResult {
  logId: number
  taskType: string
  promptPreview: string
  passed: boolean
  reasoning: string
  issues: string[]
  apiCalls: number
  apiErrors: number
  error?: string
}

interface FixResult {
  iteration: number
  taskType: string
  fixes: AutoFixParsedFix[]
  applied: AutoFixApplyResult[]
}

function BatchAutoFixView() {
  const [running, setRunning] = useState(false)
  const [maxIterations, setMaxIterations] = useState(5)
  const [logLimit, setLogLimit] = useState(50)
  const controllerRef = useRef<AbortController | null>(null)

  // State from events
  const [iteration, setIteration] = useState(0)
  const [totalLogs, setTotalLogs] = useState(0)
  const [currentLog, setCurrentLog] = useState("")
  const [progress, setProgress] = useState(0)
  const [results, setResults] = useState<LogResult[]>([])
  const [fixHistory, setFixHistory] = useState<FixResult[]>([])
  const [doneMessage, setDoneMessage] = useState("")
  const [errorMessage, setErrorMessage] = useState("")
  const [analyzing, setAnalyzing] = useState("")
  const [iterationLogs, setIterationLogs] = useState<string[]>([])

  const passedCount = useMemo(() => results.filter((r) => r.passed).length, [results])
  const failedCount = useMemo(() => results.filter((r) => !r.passed).length, [results])

  // Group results by task type
  const taskTypeSummary = useMemo(() => {
    const map = new Map<string, { passed: number; failed: number; logs: LogResult[] }>()
    for (const r of results) {
      const entry = map.get(r.taskType) || { passed: 0, failed: 0, logs: [] }
      if (r.passed) entry.passed++
      else entry.failed++
      entry.logs.push(r)
      map.set(r.taskType, entry)
    }
    return Array.from(map.entries()).sort(([, a], [, b]) => b.failed - a.failed)
  }, [results])

  const reset = useCallback(() => {
    setIteration(0)
    setTotalLogs(0)
    setCurrentLog("")
    setProgress(0)
    setResults([])
    setFixHistory([])
    setDoneMessage("")
    setErrorMessage("")
    setAnalyzing("")
    setIterationLogs([])
  }, [])

  const addLog = useCallback((msg: string) => {
    setIterationLogs((prev) => [...prev.slice(-200), msg])
  }, [])

  const handleStart = useCallback(() => {
    reset()
    setRunning(true)

    controllerRef.current = streamBatchAutoFix(
      [],  // empty = all recent logs
      logLimit,
      maxIterations,
      (event: BatchAutoFixEvent) => {
        switch (event.type) {
          case "batch_start":
            setTotalLogs(event.total_logs)
            addLog(`Starting batch: ${event.total_logs} real solve logs, max ${event.max_iterations} iterations`)
            break

          case "iteration_start":
            setIteration(event.iteration)
            setProgress(0)
            setAnalyzing("")
            addLog(`\n--- Iteration ${event.iteration}: ${event.logs_remaining} logs remaining ---`)
            break

          case "replaying":
            setCurrentLog(`#${event.log_id} ${event.task_type}`)
            setProgress(((event.index - 1) / event.total) * 100)
            break

          case "eval_result":
            setProgress((event.index / event.total) * 100)
            setResults((prev) => {
              const filtered = prev.filter((r) => r.logId !== event.log_id)
              return [
                ...filtered,
                {
                  logId: event.log_id,
                  taskType: event.task_type,
                  promptPreview: "",
                  passed: event.passed,
                  reasoning: event.reasoning,
                  issues: event.issues,
                  apiCalls: event.api_calls,
                  apiErrors: event.api_errors,
                },
              ]
            })
            if (!event.passed) {
              addLog(`  FAIL: log#${event.log_id} (${event.task_type}): ${event.reasoning}`)
            }
            break

          case "replay_error":
            setResults((prev) => [
              ...prev.filter((r) => r.logId !== event.log_id),
              {
                logId: event.log_id,
                taskType: event.task_type,
                promptPreview: "",
                passed: false,
                reasoning: "",
                issues: [],
                apiCalls: 0,
                apiErrors: 0,
                error: event.error,
              },
            ])
            addLog(`  ERROR: log#${event.log_id} (${event.task_type}): ${event.error}`)
            break

          case "iteration_summary":
            addLog(`  Summary: ${event.passed} passed, ${event.failed} failed (${event.total_passed_overall}/${event.total_passed_overall + event.total_remaining} total)`)
            break

          case "analyzing":
            setAnalyzing(event.task_type)
            addLog(`  Analyzing: ${event.task_type} (${event.failed_count} log(s) failed)`)
            break

          case "no_fixes":
            addLog(`  No fixes: ${event.task_type}`)
            break

          case "analyze_error":
            addLog(`  Analyze error: ${event.task_type}: ${event.error}`)
            break

          case "applying_fixes":
            addLog(`  Applying ${event.fix_count} fix(es) for ${event.task_type}`)
            break

          case "fixes_applied":
            setFixHistory((prev) => [
              ...prev,
              {
                iteration: event.iteration,
                taskType: event.task_type,
                fixes: [],
                applied: event.results,
              },
            ])
            addLog(`  Applied: ${event.applied}/${event.total} for ${event.task_type}`)
            for (const r of event.results) {
              if (r.applied) addLog(`    OK: ${r.file} — ${r.reason || ""}`)
              else addLog(`    SKIP: ${r.file} — ${r.error || ""}`)
            }
            break

          case "apply_error":
            addLog(`  Apply error: ${event.task_type}: ${event.error}`)
            break

          case "iteration_fixes_done":
            setAnalyzing("")
            addLog(`  ${event.message}`)
            break

          case "batch_done":
            setDoneMessage(event.message)
            setAnalyzing("")
            setCurrentLog("")
            addLog(`\n${event.message}`)
            break

          case "error":
            setErrorMessage(event.message)
            addLog(`ERROR: ${event.message}`)
            break
        }
      },
      () => setRunning(false),
      (err) => {
        setErrorMessage(err)
        setRunning(false)
      }
    )
  }, [logLimit, maxIterations, reset, addLog])

  const handleStop = useCallback(() => {
    controllerRef.current?.abort()
    setRunning(false)
  }, [])

  return (
    <div className="space-y-4">
      {/* Controls */}
      <Card className="shadow-premium">
        <CardContent className="p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-9 w-9 rounded-lg bg-orange-100 flex items-center justify-center shrink-0">
                <Rocket className="h-4 w-4 text-orange-700" />
              </div>
              <div>
                <p className="text-[13px] font-semibold">
                  Real Logs Auto-Fix
                </p>
                <p className="text-[11px] text-muted-foreground">
                  Replay real solve logs, LLM-evaluate, auto-fix failures, loop until clean
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5">
                <label className="text-[11px] text-muted-foreground">Logs</label>
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={logLimit}
                  onChange={(e) => setLogLimit(Math.max(1, parseInt(e.target.value) || 50))}
                  disabled={running}
                  className="w-14 h-7 text-[12px] text-center tabular-nums border border-border rounded-md bg-background"
                />
              </div>
              <div className="flex items-center gap-1.5">
                <label className="text-[11px] text-muted-foreground">Max iter.</label>
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={maxIterations}
                  onChange={(e) => setMaxIterations(Math.max(1, parseInt(e.target.value) || 1))}
                  disabled={running}
                  className="w-14 h-7 text-[12px] text-center tabular-nums border border-border rounded-md bg-background"
                />
              </div>
              {running ? (
                <Button variant="destructive" size="sm" onClick={handleStop}>
                  Stop
                </Button>
              ) : (
                <Button
                  onClick={handleStart}
                  className="h-9 px-5 font-semibold bg-gradient-to-r from-orange-500 to-amber-500 hover:shadow-lg hover:shadow-orange-500/25 hover:scale-[1.02] active:scale-[0.98] transition-all"
                >
                  <Rocket className="h-4 w-4 mr-2" />
                  Start Auto-Fix
                </Button>
              )}
            </div>
          </div>

          {/* Progress */}
          {(running || doneMessage) && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-[12px]">
                <div className="flex items-center gap-2">
                  {running && <Loader2 className="h-3.5 w-3.5 animate-spin text-orange-500" />}
                  {doneMessage ? (
                    <span className="font-medium">{doneMessage}</span>
                  ) : analyzing ? (
                    <span className="text-muted-foreground">
                      Iter {iteration}: Analyzing <span className="font-medium text-foreground">{analyzing}</span>...
                    </span>
                  ) : currentLog ? (
                    <span className="text-muted-foreground">
                      Iter {iteration}: Replaying <span className="font-medium text-foreground">{currentLog}</span>
                    </span>
                  ) : (
                    <span className="text-muted-foreground">Preparing...</span>
                  )}
                </div>
                <div className="flex items-center gap-3 tabular-nums">
                  <span className="text-emerald-600 font-semibold">{passedCount} passed</span>
                  {failedCount > 0 && (
                    <span className="text-red-600 font-semibold">{failedCount} failed</span>
                  )}
                  {totalLogs > 0 && (
                    <span className="text-muted-foreground">{passedCount + failedCount}/{totalLogs}</span>
                  )}
                </div>
              </div>
              <Progress value={progress} className="h-2" />
            </div>
          )}
        </CardContent>
      </Card>

      {/* Error message */}
      {errorMessage && (
        <Card className="border-red-200 bg-red-50/50">
          <CardContent className="p-4">
            <div className="flex items-start gap-2">
              <XCircle className="h-4 w-4 text-red-500 mt-0.5 shrink-0" />
              <p className="text-[13px] text-red-700">{errorMessage}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results by task type */}
      {taskTypeSummary.length > 0 && (
        <Card className="shadow-premium">
          <CardContent className="p-0">
            <div className="px-4 py-2.5 border-b bg-muted/20 flex items-center justify-between">
              <span className="text-[12px] font-semibold text-muted-foreground">
                Log Results — Iteration {iteration}
              </span>
              <div className="flex items-center gap-2">
                <Badge className="bg-emerald-500 text-[10px]">{passedCount} passed</Badge>
                {failedCount > 0 && (
                  <Badge variant="destructive" className="text-[10px]">{failedCount} failed</Badge>
                )}
              </div>
            </div>
            <div className="p-4">
              <div className="grid grid-cols-1 gap-1.5">
                {taskTypeSummary.map(([taskType, info]) => (
                  <LogTaskTypeRow
                    key={taskType}
                    taskType={taskType}
                    passedCount={info.passed}
                    total={info.passed + info.failed}
                    allPassed={info.failed === 0}
                    logs={info.logs}
                  />
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Fix History */}
      {fixHistory.length > 0 && (
        <Card className="shadow-premium border-orange-200">
          <CardContent className="p-4 space-y-3">
            <h3 className="text-[14px] font-semibold flex items-center gap-2">
              <FileCode className="h-4 w-4 text-orange-500" />
              Applied Fixes ({fixHistory.reduce((s, f) => s + f.applied.filter((a) => a.applied).length, 0)})
            </h3>
            <div className="space-y-2">
              {fixHistory.map((fh, i) => (
                <div key={i} className="rounded-lg border p-3 space-y-1.5">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-[10px]">Iter {fh.iteration}</Badge>
                    <span className="text-[12px] font-medium">{fh.taskType}</span>
                  </div>
                  {fh.applied.map((a, j) => (
                    <div
                      key={j}
                      className={cn(
                        "text-[11px] px-2 py-1 rounded",
                        a.applied ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"
                      )}
                    >
                      {a.applied ? (
                        <><CheckCircle2 className="h-3 w-3 inline mr-1" />{a.file}: {a.reason}</>
                      ) : (
                        <><AlertTriangle className="h-3 w-3 inline mr-1" />{a.file}: {a.error}</>
                      )}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Live Log */}
      {iterationLogs.length > 0 && (
        <Card>
          <CardContent className="p-4">
            <ExpandableSection title="Live Log" defaultOpen={false}>
              <ScrollArea className="h-[300px] rounded-lg border bg-slate-950 p-4">
                <pre className="text-[11px] text-slate-300 font-mono whitespace-pre-wrap">
                  {iterationLogs.join("\n")}
                </pre>
              </ScrollArea>
            </ExpandableSection>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function LogTaskTypeRow({
  taskType,
  passedCount,
  total,
  allPassed,
  logs,
}: {
  taskType: string
  passedCount: number
  total: number
  allPassed: boolean
  logs: LogResult[]
}) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div>
      <button
        onClick={() => setExpanded(!expanded)}
        className={cn(
          "w-full flex items-center gap-2 px-3 py-2 rounded-md text-[12px] transition-colors",
          allPassed
            ? "bg-emerald-50 hover:bg-emerald-100/60 border border-emerald-100"
            : "bg-red-50 hover:bg-red-100/60 border border-red-100"
        )}
      >
        {allPassed ? (
          <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500 shrink-0" />
        ) : (
          <XCircle className="h-3.5 w-3.5 text-red-500 shrink-0" />
        )}
        <span className="font-medium flex-1 text-left">{taskType}</span>
        <span className="tabular-nums text-muted-foreground">
          {passedCount}/{total}
        </span>
        {expanded ? (
          <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
        )}
      </button>
      {expanded && (
        <div className="ml-6 mt-1 space-y-0.5">
          {logs.map((r) => (
            <div
              key={r.logId}
              className={cn(
                "flex items-center gap-2 px-2 py-1 rounded text-[11px]",
                r.passed ? "text-emerald-700" : "text-red-700"
              )}
            >
              {r.passed ? (
                <CheckCircle2 className="h-3 w-3 text-emerald-500" />
              ) : (
                <XCircle className="h-3 w-3 text-red-500" />
              )}
              <span className="font-mono text-muted-foreground">#{r.logId}</span>
              <span className="tabular-nums">{r.apiCalls} calls</span>
              {r.apiErrors > 0 && (
                <span className="text-red-500 tabular-nums">{r.apiErrors} err</span>
              )}
              {!r.passed && r.reasoning && (
                <span className="text-muted-foreground truncate flex-1">{r.reasoning}</span>
              )}
              {r.error && (
                <span className="text-red-500 truncate flex-1">{r.error}</span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function ExpandableSection({
  title,
  defaultOpen = false,
  children,
}: {
  title: string
  defaultOpen?: boolean
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors"
      >
        {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
        {title}
      </button>
      {open && <div className="mt-2">{children}</div>}
    </div>
  )
}

// ── Single Auto-Fix View (original) ──────────────────────────────────

function SingleAutoFixView() {
  const { data: tasks, isLoading: tasksLoading } = useTasks()
  const { data: languages } = useLanguages()

  const [selectedTask, setSelectedTask] = useState("")
  const [selectedLang, setSelectedLang] = useState("")
  const [autoApply, setAutoApply] = useState(false)
  const [running, setRunning] = useState(false)
  const [phase, setPhase] = useState("")
  const [phaseMessage, setPhaseMessage] = useState("")

  // Results
  const [evalResult, setEvalResult] = useState<{
    score: AutoFixScore
    prompt: string
    expected: Record<string, unknown>
    language: string
    api_calls: number
    api_errors: number
    elapsed: number
    checks: FieldCheck[]
    tool_calls: ToolCall[]
    agent_response: string
  } | null>(null)
  const [runLogJson, setRunLogJson] = useState<string>("")
  const [explanation, setExplanation] = useState<{
    summary: string
    issues: string[]
    api_errors: { method: string; url: string; status: number; response: string }[]
    failed_tools: { tool: string; error: string }[]
  } | null>(null)
  const [fixes, setFixes] = useState<AutoFixParsedFix[]>([])
  const [fixRawText, setFixRawText] = useState("")
  const [errorReport, setErrorReport] = useState("")
  const [applyResults, setApplyResults] = useState<AutoFixApplyResult[]>([])
  const [errorMessage, setErrorMessage] = useState("")

  // UI state
  const [showReport, setShowReport] = useState(false)
  const [showToolCalls, setShowToolCalls] = useState(false)
  const [applying, setApplying] = useState(false)
  const [copiedJson, setCopiedJson] = useState(false)
  const [fixing, setFixing] = useState(false)

  const controllerRef = useRef<AbortController | null>(null)

  const reset = useCallback(() => {
    setEvalResult(null)
    setRunLogJson("")
    setExplanation(null)
    setFixes([])
    setFixRawText("")
    setErrorReport("")
    setApplyResults([])
    setErrorMessage("")
    setPhase("")
    setPhaseMessage("")
    setShowReport(false)
    setShowToolCalls(false)
    setCopiedJson(false)
    setFixing(false)
  }, [])

  const handleRun = useCallback(() => {
    if (!selectedTask) {
      toast.error("Select a task first")
      return
    }

    reset()
    setRunning(true)
    setPhase("running")
    setPhaseMessage("Running submission...")

    controllerRef.current = streamAutoFix(
      selectedTask,
      selectedLang,
      autoApply,
      (event: AutoFixEvent) => {
        switch (event.type) {
          case "phase":
            setPhase(event.phase)
            setPhaseMessage(event.message)
            break
          case "eval_result":
            setEvalResult({
              score: event.score,
              prompt: event.prompt,
              expected: event.expected,
              language: event.language,
              api_calls: event.api_calls,
              api_errors: event.api_errors,
              elapsed: event.elapsed,
              checks: event.checks,
              tool_calls: event.tool_calls,
              agent_response: event.agent_response,
            })
            break
          case "run_log":
            setRunLogJson(JSON.stringify(event.log, null, 2))
            break
          case "explanation":
            setExplanation({
              summary: event.summary,
              issues: event.issues,
              api_errors: event.api_errors,
              failed_tools: event.failed_tools,
            })
            break
          case "fixes":
            setFixes(event.parsed_fixes)
            setFixRawText(event.raw_text)
            setErrorReport(event.report)
            break
          case "applied":
            setApplyResults(event.results)
            break
          case "error":
            setErrorMessage(event.message)
            break
        }
      },
      () => setRunning(false),
      (err) => {
        setErrorMessage(err)
        setRunning(false)
      }
    )
  }, [selectedTask, selectedLang, autoApply, reset])

  // Manual "Auto Fix Task" trigger — runs with auto_apply=true
  const handleAutoFix = useCallback(() => {
    if (!selectedTask) return
    setFixing(true)
    setFixes([])
    setFixRawText("")
    setErrorReport("")
    setApplyResults([])

    controllerRef.current = streamAutoFix(
      selectedTask,
      selectedLang,
      true, // auto_apply = true
      (event: AutoFixEvent) => {
        switch (event.type) {
          case "phase":
            setPhase(event.phase)
            setPhaseMessage(event.message)
            break
          case "fixes":
            setFixes(event.parsed_fixes)
            setFixRawText(event.raw_text)
            setErrorReport(event.report)
            break
          case "applied":
            setApplyResults(event.results)
            break
          case "error":
            setErrorMessage(event.message)
            break
        }
      },
      () => setFixing(false),
      (err) => {
        setErrorMessage(err)
        setFixing(false)
      }
    )
  }, [selectedTask, selectedLang])

  const handleApply = useCallback(async () => {
    if (fixes.length === 0) return
    setApplying(true)
    try {
      const result = await applyFixes(fixes)
      setApplyResults(result.results)
      const applied = result.results.filter((r) => r.applied).length
      toast.success(`Applied ${applied}/${fixes.length} fix(es)`)
    } catch (err) {
      toast.error("Failed to apply: " + (err as Error).message)
    } finally {
      setApplying(false)
    }
  }, [fixes])

  const handleCopyJson = useCallback(() => {
    if (!runLogJson) return
    navigator.clipboard.writeText(runLogJson)
    setCopiedJson(true)
    toast.success("Run log copied to clipboard")
    setTimeout(() => setCopiedJson(false), 2000)
  }, [runLogJson])

  const handleStop = useCallback(() => {
    controllerRef.current?.abort()
    setRunning(false)
    setFixing(false)
  }, [])

  if (tasksLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-40 w-full rounded-xl" />
      </div>
    )
  }

  const passedChecks = evalResult?.checks.filter((c) => c.passed) ?? []
  const isPerfect = evalResult?.score.correctness === 1
  const isRunningOrFixing = running || fixing

  return (
    <div>
      {/* Controls */}
      <Card className="shadow-premium mb-4">
        <CardContent className="p-5 space-y-4">
          <div className="flex flex-wrap gap-4 items-end">
            {/* Task selector */}
            <div className="flex-1 min-w-[200px]">
              <label className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5 block">
                Task
              </label>
              <select
                value={selectedTask}
                onChange={(e) => setSelectedTask(e.target.value)}
                disabled={isRunningOrFixing}
                className="w-full h-9 px-3 rounded-lg border border-border bg-background text-[13px] focus:outline-none focus:ring-2 focus:ring-primary/20"
              >
                <option value="">Select task...</option>
                {[1, 2, 3].map((tier) => (
                  <optgroup key={tier} label={`Tier ${tier}`}>
                    {tasks
                      ?.filter((t) => t.tier === tier)
                      .map((t) => (
                        <option key={t.name} value={t.name}>
                          {t.name.replace(/_/g, " ")} ({t.max_points}p)
                        </option>
                      ))}
                  </optgroup>
                ))}
              </select>
            </div>

            {/* Language selector */}
            <div className="w-[140px]">
              <label className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5 block">
                Language
              </label>
              <select
                value={selectedLang}
                onChange={(e) => setSelectedLang(e.target.value)}
                disabled={isRunningOrFixing}
                className="w-full h-9 px-3 rounded-lg border border-border bg-background text-[13px] focus:outline-none focus:ring-2 focus:ring-primary/20"
              >
                <option value="">Random</option>
                {languages &&
                  Object.entries(languages).map(([code, name]) => (
                    <option key={code} value={code}>
                      {name}
                    </option>
                  ))}
              </select>
            </div>

            {/* Auto-fix toggle */}
            <label className="flex items-center gap-1.5 text-[12px] pb-1.5">
              <input
                type="checkbox"
                checked={autoApply}
                onChange={(e) => setAutoApply(e.target.checked)}
                disabled={isRunningOrFixing}
                className="rounded"
              />
              Auto-fix on fail
            </label>

            {/* Buttons */}
            <div className="flex gap-2">
              {isRunningOrFixing ? (
                <Button variant="destructive" size="sm" onClick={handleStop}>
                  Stop
                </Button>
              ) : (
                <Button
                  onClick={handleRun}
                  disabled={!selectedTask}
                  className="h-9 px-5 font-semibold bg-gradient-to-r from-blue-500 to-cyan-500 hover:shadow-lg hover:shadow-blue-500/25 hover:scale-[1.02] active:scale-[0.98] transition-all"
                >
                  <Play className="h-4 w-4 mr-2" />
                  Run Submission
                </Button>
              )}
            </div>
          </div>

          {/* Progress indicator */}
          {isRunningOrFixing && (
            <div className="flex items-center gap-2 text-[13px] text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
              <span>{phaseMessage}</span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Error message */}
      {errorMessage && (
        <Card className="mb-4 border-red-200 bg-red-50/50">
          <CardContent className="p-4">
            <div className="flex items-start gap-2">
              <XCircle className="h-4 w-4 text-red-500 mt-0.5 shrink-0" />
              <p className="text-[13px] text-red-700">{errorMessage}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* LLM Verdict Card */}
      {evalResult && (
        <Card className={cn("mb-4 shadow-premium", isPerfect ? "border-emerald-200" : "border-red-200")}>
          <CardContent className="p-5 space-y-4">
            {/* Verdict header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={cn(
                  "h-10 w-10 rounded-xl flex items-center justify-center",
                  isPerfect ? "bg-emerald-100" : "bg-red-100"
                )}>
                  {isPerfect ? (
                    <CheckCircle2 className="h-5 w-5 text-emerald-600" />
                  ) : (
                    <XCircle className="h-5 w-5 text-red-600" />
                  )}
                </div>
                <div>
                  <h3 className="text-[15px] font-bold">
                    {isPerfect ? "PASS" : "FAIL"} — LLM Eval
                  </h3>
                  <p className="text-[12px] text-muted-foreground">
                    {evalResult.elapsed}s · {evalResult.api_calls} API calls
                    {evalResult.api_errors > 0 && (
                      <span className="text-red-500"> · {evalResult.api_errors} errors</span>
                    )}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Badge
                  variant={isPerfect ? "default" : "destructive"}
                  className={cn("text-[13px] font-bold px-3 py-1", isPerfect && "bg-emerald-500")}
                >
                  {(evalResult.score.correctness * 100).toFixed(0)}%
                </Badge>
                {runLogJson && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleCopyJson}
                    className="h-8 gap-1.5"
                  >
                    {copiedJson ? (
                      <Check className="h-3.5 w-3.5 text-emerald-500" />
                    ) : (
                      <ClipboardCopy className="h-3.5 w-3.5" />
                    )}
                    Copy JSON
                  </Button>
                )}
              </div>
            </div>

            {/* Prompt */}
            <div className="bg-muted/50 rounded-lg p-3">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                Prompt ({evalResult.language})
              </p>
              <p className="text-[13px]">{evalResult.prompt}</p>
            </div>

            {/* Explanation — only on failure */}
            {explanation && (
              <div className="rounded-lg border border-red-200 bg-red-50/50 p-4 space-y-3">
                <div className="flex items-start gap-2">
                  <AlertTriangle className="h-4 w-4 text-red-500 mt-0.5 shrink-0" />
                  <div className="space-y-2 flex-1">
                    <p className="text-[13px] font-medium text-red-800">{explanation.summary}</p>

                    {/* Issues */}
                    {explanation.issues.length > 0 && (
                      <div className="space-y-0.5">
                        {explanation.issues.map((issue, i) => (
                          <p key={i} className="text-[12px] text-red-700 flex items-start gap-1.5">
                            <span className="text-red-400 shrink-0">·</span>
                            {issue}
                          </p>
                        ))}
                      </div>
                    )}

                    {/* API errors */}
                    {explanation.api_errors.length > 0 && (
                      <div className="space-y-0.5 mt-1">
                        <p className="text-[11px] font-semibold text-red-600 uppercase tracking-wider">API Errors</p>
                        {explanation.api_errors.map((e, i) => (
                          <p key={i} className="text-[11px] text-red-600 font-mono">
                            {e.method} {e.url} → {e.status}
                            {e.response && <span className="text-red-400 ml-1">({e.response.slice(0, 80)})</span>}
                          </p>
                        ))}
                      </div>
                    )}

                    {/* Failed tools */}
                    {explanation.failed_tools.length > 0 && (
                      <div className="space-y-0.5 mt-1">
                        <p className="text-[11px] font-semibold text-red-600 uppercase tracking-wider">Failed Tool Calls</p>
                        {explanation.failed_tools.map((t, i) => (
                          <p key={i} className="text-[11px] text-red-600 font-mono">
                            {t.tool}: {t.error}
                          </p>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* Auto Fix button — only when not auto-applying and no fixes generated yet */}
                {!autoApply && fixes.length === 0 && !fixing && (
                  <div className="flex items-center gap-3 pt-2 border-t border-red-200">
                    <Button
                      onClick={handleAutoFix}
                      className="h-9 px-5 font-semibold bg-gradient-to-r from-orange-500 to-amber-500 hover:shadow-lg hover:shadow-orange-500/25 hover:scale-[1.02] active:scale-[0.98] transition-all"
                    >
                      <Zap className="h-4 w-4 mr-2" />
                      Auto Fix Task
                    </Button>
                    <span className="text-[11px] text-muted-foreground">
                      Analyze errors and generate code fixes
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* Field Checks */}
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                Checks ({passedChecks.length}/{evalResult.checks.length} passed)
              </p>
              <div className="space-y-1">
                {evalResult.checks.map((c, i) => (
                  <div
                    key={i}
                    className={cn(
                      "flex items-center gap-2 px-3 py-1.5 rounded-md text-[12px]",
                      c.passed
                        ? "bg-emerald-50 text-emerald-800 border border-emerald-100"
                        : "bg-red-50 text-red-800 border border-red-100"
                    )}
                  >
                    {c.passed ? (
                      <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500 shrink-0" />
                    ) : (
                      <XCircle className="h-3.5 w-3.5 text-red-500 shrink-0" />
                    )}
                    <span className="font-medium w-[160px] shrink-0">{c.field}</span>
                    <span className="flex-1 truncate text-muted-foreground">{c.detail}</span>
                    <span className="tabular-nums font-medium shrink-0">{c.points}/{c.max}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Tool Calls (collapsible) */}
            <div>
              <button
                onClick={() => setShowToolCalls(!showToolCalls)}
                className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors"
              >
                {showToolCalls ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
                Tool Calls ({evalResult.tool_calls.length})
              </button>
              {showToolCalls && (
                <div className="mt-2 space-y-1">
                  {evalResult.tool_calls.map((tc, i) => {
                    const isError = tc.result && tc.result.ok === false
                    return (
                      <div
                        key={i}
                        className={cn(
                          "px-3 py-1.5 rounded-md text-[12px] font-mono border",
                          isError ? "bg-red-50 border-red-100" : "bg-muted/30 border-border/30"
                        )}
                      >
                        <span className="font-semibold">{tc.tool}</span>
                        <span className="text-muted-foreground ml-1">
                          ({Object.entries(tc.args || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ").slice(0, 150)})
                        </span>
                        {isError && tc.result?.error && (
                          <div className="text-red-600 mt-0.5 text-[11px]">{tc.result.error.slice(0, 200)}</div>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </div>

            {/* Agent Response */}
            {evalResult.agent_response && (
              <div className="bg-muted/50 rounded-lg p-3">
                <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">Agent Response</p>
                <p className="text-[12px] text-muted-foreground">{evalResult.agent_response}</p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Fix Suggestions */}
      {fixes.length > 0 && (
        <Card className="mb-4 shadow-premium border-orange-200">
          <CardContent className="p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-[14px] font-semibold flex items-center gap-2">
                <FileCode className="h-4 w-4 text-orange-500" />
                Suggested Fixes ({fixes.length})
              </h3>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={() => setShowReport(!showReport)}>
                  {showReport ? "Hide" : "Show"} Error Report
                </Button>
                {applyResults.length === 0 && (
                  <Button
                    size="sm"
                    onClick={handleApply}
                    disabled={applying}
                    className="bg-gradient-to-r from-orange-500 to-amber-500 hover:shadow-lg"
                  >
                    {applying ? <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" /> : <Wrench className="h-3.5 w-3.5 mr-1.5" />}
                    Apply All Fixes
                  </Button>
                )}
              </div>
            </div>

            {showReport && (
              <ScrollArea className="h-[300px] rounded-lg border bg-slate-950 p-4">
                <pre className="text-[11px] text-slate-300 font-mono whitespace-pre-wrap">{errorReport}</pre>
              </ScrollArea>
            )}

            <div className="space-y-3">
              {fixes.map((fix, i) => (
                <FixCard key={i} fix={fix} index={i} applyResult={applyResults[i]} />
              ))}
            </div>

            {applyResults.length > 0 && (
              <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted/50">
                {applyResults.every((r) => r.applied) ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                ) : (
                  <AlertTriangle className="h-4 w-4 text-amber-500" />
                )}
                <span className="text-[13px]">
                  {applyResults.filter((r) => r.applied).length}/{applyResults.length} fixes applied
                </span>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Done state with no fixes needed */}
      {phase === "done" && isPerfect && (
        <Card>
          <CardContent className="p-8 text-center">
            <CheckCircle2 className="h-8 w-8 text-emerald-500 mx-auto mb-2" />
            <p className="text-[14px] font-medium">All checks passed! No fixes needed.</p>
          </CardContent>
        </Card>
      )}

      {/* Gemini raw output (collapsible) */}
      {fixRawText && (
        <Card className="mt-4">
          <CardContent className="p-4">
            <ExpandableSection title="Raw LLM Output">
              <ScrollArea className="h-[300px] rounded-lg border bg-slate-950 p-4">
                <pre className="text-[11px] text-slate-300 font-mono whitespace-pre-wrap">{fixRawText}</pre>
              </ScrollArea>
            </ExpandableSection>
          </CardContent>
        </Card>
      )}

      {/* Run log JSON (collapsible) */}
      {runLogJson && (
        <Card className="mt-4">
          <CardContent className="p-4">
            <ExpandableSection title="Full Run Log (JSON)">
              <div className="relative">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleCopyJson}
                  className="absolute top-2 right-2 z-10 h-7 gap-1"
                >
                  {copiedJson ? <Check className="h-3 w-3 text-emerald-500" /> : <Copy className="h-3 w-3" />}
                  Copy
                </Button>
                <ScrollArea className="h-[400px] rounded-lg border bg-slate-950 p-4">
                  <pre className="text-[11px] text-slate-300 font-mono whitespace-pre-wrap">{runLogJson}</pre>
                </ScrollArea>
              </div>
            </ExpandableSection>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function FixCard({
  fix,
  index,
  applyResult,
}: {
  fix: AutoFixParsedFix
  index: number
  applyResult?: AutoFixApplyResult
}) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(fix.new)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div
      className={cn(
        "rounded-lg border p-4 space-y-3",
        applyResult?.applied
          ? "border-emerald-200 bg-emerald-50/30"
          : applyResult && !applyResult.applied
          ? "border-amber-200 bg-amber-50/30"
          : "border-border"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="font-mono text-[11px]">
            {fix.file}
          </Badge>
          <span className="text-[12px] text-muted-foreground">
            Fix #{index + 1}
          </span>
          {applyResult && (
            <Badge
              variant={applyResult.applied ? "default" : "secondary"}
              className={cn(
                "text-[10px]",
                applyResult.applied && "bg-emerald-500"
              )}
            >
              {applyResult.applied ? "Applied" : "Skipped"}
            </Badge>
          )}
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleCopy}
          className="h-7 px-2"
        >
          {copied ? (
            <Check className="h-3.5 w-3.5 text-emerald-500" />
          ) : (
            <Copy className="h-3.5 w-3.5" />
          )}
        </Button>
      </div>

      {/* Reason */}
      <p className="text-[12px] text-foreground">{fix.reason}</p>

      {/* Diff view */}
      <div className="grid grid-cols-2 gap-2">
        <div className="rounded-md bg-red-950/80 border border-red-900/30 p-3 overflow-auto">
          <p className="text-[10px] font-semibold text-red-400 mb-1.5 uppercase tracking-wider">
            Old
          </p>
          <pre className="text-[11px] text-red-300 font-mono whitespace-pre-wrap">
            {fix.old}
          </pre>
        </div>
        <div className="rounded-md bg-emerald-950/80 border border-emerald-900/30 p-3 overflow-auto">
          <p className="text-[10px] font-semibold text-emerald-400 mb-1.5 uppercase tracking-wider">
            New
          </p>
          <pre className="text-[11px] text-emerald-300 font-mono whitespace-pre-wrap">
            {fix.new}
          </pre>
        </div>
      </div>

      {/* Error message if skipped */}
      {applyResult && !applyResult.applied && applyResult.error && (
        <p className="text-[11px] text-amber-600">{applyResult.error}</p>
      )}
    </div>
  )
}
