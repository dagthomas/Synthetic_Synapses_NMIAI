import { useState, useRef, useCallback } from "react"
import { useTasks, useLanguages } from "@/hooks/use-api"
import { streamAutoFix, applyFixes } from "@/lib/api"
import type {
  AutoFixEvent,
  AutoFixScore,
  AutoFixParsedFix,
  AutoFixApplyResult,
  FieldCheck,
  ToolCall,
} from "@/types/api"
import { PageHeader } from "@/components/layout/page-header"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
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
} from "lucide-react"

export function AutoFixPanel() {
  const { data: tasks, isLoading: tasksLoading } = useTasks()
  const { data: languages } = useLanguages()

  const [selectedTask, setSelectedTask] = useState("")
  const [selectedLang, setSelectedLang] = useState("")
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
  const [fixes, setFixes] = useState<AutoFixParsedFix[]>([])
  const [fixRawText, setFixRawText] = useState("")
  const [errorReport, setErrorReport] = useState("")
  const [applyResults, setApplyResults] = useState<AutoFixApplyResult[]>([])
  const [errorMessage, setErrorMessage] = useState("")

  // UI state
  const [showReport, setShowReport] = useState(false)
  const [showToolCalls, setShowToolCalls] = useState(false)
  const [applying, setApplying] = useState(false)

  const controllerRef = useRef<AbortController | null>(null)

  const reset = useCallback(() => {
    setEvalResult(null)
    setFixes([])
    setFixRawText("")
    setErrorReport("")
    setApplyResults([])
    setErrorMessage("")
    setPhase("")
    setPhaseMessage("")
    setShowReport(false)
    setShowToolCalls(false)
  }, [])

  const handleRun = useCallback(() => {
    if (!selectedTask) {
      toast.error("Select a task first")
      return
    }

    reset()
    setRunning(true)
    setPhase("generating")
    setPhaseMessage("Starting...")

    controllerRef.current = streamAutoFix(
      selectedTask,
      selectedLang,
      false,
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
  }, [selectedTask, selectedLang, reset])

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

  const handleStop = useCallback(() => {
    controllerRef.current?.abort()
    setRunning(false)
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

  return (
    <div>
      <PageHeader
        title="Auto Fix"
        description="Run a task, analyze failures, and get AI-suggested code fixes."
      />

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
                disabled={running}
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
                disabled={running}
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

            {/* Buttons */}
            <div className="flex gap-2">
              {running ? (
                <Button variant="destructive" size="sm" onClick={handleStop}>
                  Stop
                </Button>
              ) : (
                <Button
                  onClick={handleRun}
                  disabled={!selectedTask}
                  className="h-9 px-5 font-semibold bg-gradient-to-r from-orange-500 to-amber-500 hover:shadow-lg hover:shadow-orange-500/25 hover:scale-[1.02] active:scale-[0.98] transition-all"
                >
                  <Wrench className="h-4 w-4 mr-2" />
                  Run & Analyze
                </Button>
              )}
            </div>
          </div>

          {/* Progress indicator */}
          {running && (
            <div className="flex items-center gap-2 text-[13px] text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin text-orange-500" />
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

      {/* Eval Result */}
      {evalResult && (
        <Card className="mb-4 shadow-premium">
          <CardContent className="p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-[14px] font-semibold flex items-center gap-2">
                {isPerfect ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                ) : (
                  <AlertTriangle className="h-4 w-4 text-amber-500" />
                )}
                Eval Result
              </h3>
              <div className="flex items-center gap-3">
                <Badge
                  variant={isPerfect ? "default" : "destructive"}
                  className={cn(
                    "text-[12px] font-bold",
                    isPerfect && "bg-emerald-500"
                  )}
                >
                  {(evalResult.score.correctness * 100).toFixed(0)}%
                </Badge>
                <span className="text-[12px] text-muted-foreground tabular-nums">
                  {evalResult.score.final_score}/{evalResult.score.max_possible}
                </span>
                <span className="text-[12px] text-muted-foreground tabular-nums">
                  {evalResult.elapsed}s
                </span>
                <span className="text-[12px] text-muted-foreground tabular-nums">
                  {evalResult.api_calls} calls
                  {evalResult.api_errors > 0 && (
                    <span className="text-red-500 ml-1">
                      ({evalResult.api_errors} errors)
                    </span>
                  )}
                </span>
              </div>
            </div>

            {/* Prompt */}
            <div className="bg-muted/50 rounded-lg p-3">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                Prompt ({evalResult.language})
              </p>
              <p className="text-[13px]">{evalResult.prompt}</p>
            </div>

            {/* Field Checks */}
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                Field Checks ({passedChecks.length}/{evalResult.checks.length} passed)
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
                    <span className="flex-1 truncate text-muted-foreground">
                      {c.detail}
                    </span>
                    <span className="tabular-nums font-medium shrink-0">
                      {c.points}/{c.max}
                    </span>
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
                {showToolCalls ? (
                  <ChevronDown className="h-3.5 w-3.5" />
                ) : (
                  <ChevronRight className="h-3.5 w-3.5" />
                )}
                Tool Calls ({evalResult.tool_calls.length})
              </button>
              {showToolCalls && (
                <div className="mt-2 space-y-1">
                  {evalResult.tool_calls.map((tc, i) => {
                    const isError =
                      tc.result && tc.result.ok === false
                    return (
                      <div
                        key={i}
                        className={cn(
                          "px-3 py-1.5 rounded-md text-[12px] font-mono border",
                          isError
                            ? "bg-red-50 border-red-100"
                            : "bg-muted/30 border-border/30"
                        )}
                      >
                        <span className="font-semibold">{tc.tool}</span>
                        <span className="text-muted-foreground ml-1">
                          ({Object.entries(tc.args || {})
                            .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                            .join(", ")
                            .slice(0, 150)}
                          )
                        </span>
                        {isError && tc.result?.error && (
                          <div className="text-red-600 mt-0.5 text-[11px]">
                            {tc.result.error.slice(0, 200)}
                          </div>
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
                <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                  Agent Response
                </p>
                <p className="text-[12px] text-muted-foreground">
                  {evalResult.agent_response}
                </p>
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
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowReport(!showReport)}
                >
                  {showReport ? "Hide" : "Show"} Error Report
                </Button>
                {applyResults.length === 0 && (
                  <Button
                    size="sm"
                    onClick={handleApply}
                    disabled={applying}
                    className="bg-gradient-to-r from-orange-500 to-amber-500 hover:shadow-lg"
                  >
                    {applying ? (
                      <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                    ) : (
                      <Wrench className="h-3.5 w-3.5 mr-1.5" />
                    )}
                    Apply All Fixes
                  </Button>
                )}
              </div>
            </div>

            {/* Error Report (collapsible) */}
            {showReport && (
              <ScrollArea className="h-[300px] rounded-lg border bg-slate-950 p-4">
                <pre className="text-[11px] text-slate-300 font-mono whitespace-pre-wrap">
                  {errorReport}
                </pre>
              </ScrollArea>
            )}

            {/* Individual fixes */}
            <div className="space-y-3">
              {fixes.map((fix, i) => (
                <FixCard
                  key={i}
                  fix={fix}
                  index={i}
                  applyResult={applyResults[i]}
                />
              ))}
            </div>

            {/* Apply results summary */}
            {applyResults.length > 0 && (
              <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted/50">
                {applyResults.every((r) => r.applied) ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                ) : (
                  <AlertTriangle className="h-4 w-4 text-amber-500" />
                )}
                <span className="text-[13px]">
                  {applyResults.filter((r) => r.applied).length}/{applyResults.length}{" "}
                  fixes applied
                </span>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Done state with no fixes needed */}
      {phase === "done" && !evalResult && !errorMessage && (
        <Card>
          <CardContent className="p-8 text-center">
            <CheckCircle2 className="h-8 w-8 text-emerald-500 mx-auto mb-2" />
            <p className="text-[14px] font-medium">{phaseMessage}</p>
          </CardContent>
        </Card>
      )}

      {/* Gemini raw output (collapsible) */}
      {fixRawText && (
        <Card className="mt-4">
          <CardContent className="p-4">
            <button
              onClick={() => {
                const el = document.getElementById("raw-fix-text")
                if (el) el.classList.toggle("hidden")
              }}
              className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1.5"
            >
              <ChevronRight className="h-3.5 w-3.5" />
              Raw LLM Output
            </button>
            <div id="raw-fix-text" className="hidden mt-2">
              <ScrollArea className="h-[300px] rounded-lg border bg-slate-950 p-4">
                <pre className="text-[11px] text-slate-300 font-mono whitespace-pre-wrap">
                  {fixRawText}
                </pre>
              </ScrollArea>
            </div>
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
