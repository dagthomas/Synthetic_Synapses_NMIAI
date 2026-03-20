import { useState } from "react"
import { useRuns } from "@/hooks/use-api"
import { deleteRun, deleteRuns } from "@/lib/api"
import type { EvalRun, FieldCheck } from "@/types/api"
import { PageHeader } from "@/components/layout/page-header"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { toast } from "sonner"
import { cn } from "@/lib/utils"
import {
  Copy,
  Trash2,
  AlertTriangle,
  XCircle,
  CheckCircle,
  Check,
  Bug,
  AlertOctagon,
  ChevronDown,
  ChevronRight,
} from "lucide-react"

export function ErrorsPanel({ source = "all" }: { source?: "simulator" | "competition" | "all" }) {
  const { data: runs, isLoading, mutate } = useRuns("all", source, 5000)
  const [copied, setCopied] = useState(false)

  const problems = (runs ?? []).filter(
    (r) =>
      r.status === "failed" ||
      (r.status === "completed" && r.correctness != null && r.correctness < 1)
  )

  const byTask: Record<string, EvalRun[]> = {}
  problems.forEach((r) => {
    if (!byTask[r.task_name]) byTask[r.task_name] = []
    byTask[r.task_name].push(r)
  })

  const crashed = problems.filter((r) => r.status === "failed").length
  const failedChecks = problems.filter((r) => r.status === "completed").length
  const taskCount = Object.keys(byTask).length

  async function handleDelete(id: number) {
    try {
      await deleteRun(id)
      toast.success("Deleted")
      mutate()
    } catch (err) {
      toast.error((err as Error).message)
    }
  }

  async function handleDeleteAll() {
    if (!problems.length) return
    try {
      await deleteRuns(problems.map((r) => r.id))
      toast.success(`Deleted ${problems.length} error(s)`)
      mutate()
    } catch (err) {
      toast.error((err as Error).message)
    }
  }

  function handleCopyErrors() {
    let text = "# Tripletex Error Report\n\n"
    for (const [taskName, taskRuns] of Object.entries(byTask).sort()) {
      text += `## ${taskName}\n`
      for (const r of taskRuns) {
        text += `  [${r.language}] #${r.id}: `
        if (r.status === "failed") {
          text += `CRASHED — ${r.error_message || "unknown"}\n`
        } else {
          text += `${((r.correctness ?? 0) * 100).toFixed(0)}% correct\n`
          try {
            const checks: FieldCheck[] = JSON.parse(r.checks_json || "[]")
            checks
              .filter((c) => !c.passed)
              .forEach((c) => {
                text += `    FAIL ${c.field}: ${c.detail} (${c.points}/${c.max} pts)\n`
              })
          } catch { /* ignore */ }
          text += `    Prompt: ${r.prompt}\n`
        }
      }
      text += "\n"
    }
    navigator.clipboard.writeText(text)
    setCopied(true)
    toast.success("Errors copied")
    setTimeout(() => setCopied(false), 2000)
  }

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-32 w-full rounded-xl" />
      </div>
    )
  }

  if (problems.length === 0) {
    return (
      <div>
        <PageHeader title="Errors" />
        <div className="text-center py-20 text-muted-foreground">
          <CheckCircle className="h-12 w-12 mx-auto mb-4 opacity-20 text-emerald-500" />
          <p className="text-[14px] font-medium mb-1 text-emerald-600">All clear</p>
          <p className="text-[12px] text-muted-foreground/70">No errors found. All runs passed.</p>
        </div>
      </div>
    )
  }

  return (
    <div>
      <PageHeader title="Errors & Failed Checks">
        <Button
          variant={copied ? "default" : "outline"}
          size="sm"
          onClick={handleCopyErrors}
          className={cn(
            "transition-all duration-200",
            copied && "bg-emerald-600 hover:bg-emerald-600 text-white"
          )}
        >
          {copied ? (
            <Check className="h-3.5 w-3.5 mr-1.5" />
          ) : (
            <Copy className="h-3.5 w-3.5 mr-1.5" />
          )}
          {copied ? "Copied!" : "Copy Errors"}
        </Button>
        <Button
          variant="destructive"
          size="sm"
          onClick={handleDeleteAll}
        >
          <Trash2 className="h-3.5 w-3.5 mr-1.5" />
          Delete All
        </Button>
      </PageHeader>

      {/* Summary */}
      <div className="grid grid-cols-4 gap-3 mb-4">
        <SummaryCard
          icon={<AlertOctagon className="h-4 w-4" />}
          label="Total Errors"
          value={problems.length}
          iconBg="bg-red-50 text-red-600"
          valueClass="text-red-600"
        />
        <SummaryCard
          icon={<Bug className="h-4 w-4" />}
          label="Crashed"
          value={crashed}
          iconBg="bg-red-50 text-red-600"
          valueClass="text-red-600"
        />
        <SummaryCard
          icon={<AlertTriangle className="h-4 w-4" />}
          label="Failed Checks"
          value={failedChecks}
          iconBg="bg-amber-50 text-amber-600"
          valueClass="text-amber-600"
        />
        <SummaryCard
          icon={<XCircle className="h-4 w-4" />}
          label="Tasks Affected"
          value={taskCount}
          iconBg="bg-slate-100 text-slate-600"
        />
      </div>

      {/* Error cards */}
      <div className="space-y-2">
        {Object.keys(byTask)
          .sort()
          .flatMap((taskName) =>
            byTask[taskName].map((r, i) => (
              <ErrorCard key={r.id} run={r} onDelete={handleDelete} index={i} />
            ))
          )}
      </div>
    </div>
  )
}

function SummaryCard({
  icon,
  label,
  value,
  iconBg,
  valueClass,
}: {
  icon: React.ReactNode
  label: string
  value: string | number
  iconBg: string
  valueClass?: string
}) {
  return (
    <Card className="shadow-sm metric-card">
      <CardContent className="p-4">
        <div className={cn("h-8 w-8 rounded-lg flex items-center justify-center mb-2", iconBg)}>
          {icon}
        </div>
        <div className={cn("text-xl font-bold tabular-nums tracking-tight", valueClass)}>
          {value}
        </div>
        <div className="text-[10px] text-muted-foreground font-medium mt-0.5">{label}</div>
      </CardContent>
    </Card>
  )
}

function ErrorCard({ run: r, onDelete, index }: { run: EvalRun; onDelete: (id: number) => void; index: number }) {
  const isCrash = r.status === "failed"
  const [expanded, setExpanded] = useState(false)

  let checks: FieldCheck[] = []
  try {
    checks = JSON.parse(r.checks_json || "[]")
  } catch { /* ignore */ }

  const failedChecks = checks.filter((c) => !c.passed)
  const passedChecks = checks.filter((c) => c.passed)

  return (
    <Card
      className={cn(
        "shadow-sm transition-all duration-200 animate-fade-in-up overflow-hidden",
        isCrash
          ? "border-l-4 border-l-red-500"
          : "border-l-4 border-l-amber-500"
      )}
      style={{ animationDelay: `${index * 30}ms` }}
    >
      <CardContent className="p-4">
        <div className="flex items-center gap-2 mb-2">
          {isCrash ? (
            <div className="h-7 w-7 rounded-lg bg-red-50 flex items-center justify-center shrink-0">
              <XCircle className="h-4 w-4 text-red-500" />
            </div>
          ) : (
            <div className="h-7 w-7 rounded-lg bg-amber-50 flex items-center justify-center shrink-0">
              <AlertTriangle className="h-4 w-4 text-amber-500" />
            </div>
          )}
          <span className="font-semibold text-[13px]">{r.task_name}</span>
          <Badge variant="secondary" className="text-[10px] font-mono">{r.language}</Badge>
          <Badge variant="outline" className="text-[10px] font-mono text-muted-foreground">#{r.id}</Badge>
          <Button
            variant="ghost"
            size="sm"
            className="ml-auto h-7 w-7 p-0 text-muted-foreground hover:text-red-600 hover:bg-red-50"
            onClick={(e) => { e.stopPropagation(); onDelete(r.id) }}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>

        {isCrash ? (
          <pre className="text-[11px] font-mono bg-red-50/50 rounded-lg p-3 whitespace-pre-wrap text-red-600 border border-red-100">
            {r.error_message || "Unknown error"}
          </pre>
        ) : (
          <div>
            {failedChecks.length > 0 && (
              <div className="space-y-1 mb-2">
                {failedChecks.map((c, i) => (
                  <div key={i} className="flex items-start gap-2 text-[12px] text-red-600 bg-red-50/50 rounded-md px-2.5 py-1.5">
                    <XCircle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                    <div>
                      <strong className="font-semibold">{c.field}</strong>
                      <span className="text-red-500 ml-1">{c.detail}</span>
                      <span className="text-muted-foreground ml-1.5 text-[10px]">
                        ({c.points}/{c.max} pts)
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
            {passedChecks.length > 0 && (
              <button
                className="flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
                onClick={() => setExpanded(!expanded)}
              >
                {expanded ? (
                  <ChevronDown className="h-3 w-3" />
                ) : (
                  <ChevronRight className="h-3 w-3" />
                )}
                {passedChecks.length} passed checks
              </button>
            )}
            {expanded && (
              <div className="space-y-0.5 mt-1.5 ml-1">
                {passedChecks.map((c, i) => (
                  <div key={i} className="flex items-center gap-1.5 text-[12px] text-emerald-700">
                    <CheckCircle className="h-3 w-3 shrink-0" />
                    <span>{c.field}: {c.detail}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <Separator className="my-3" />
        <div className="flex items-center gap-4 text-[11px] text-muted-foreground">
          <span>
            Score: <strong className="text-foreground">{r.final_score ?? "-"}/{r.max_possible ?? "-"}</strong>
          </span>
          <span>
            Correctness:{" "}
            <strong className={cn(
              r.correctness != null && r.correctness >= 0.8 ? "text-emerald-600" :
              r.correctness != null && r.correctness >= 0.5 ? "text-amber-600" :
              "text-red-600"
            )}>
              {r.correctness != null ? `${(r.correctness * 100).toFixed(0)}%` : "-"}
            </strong>
          </span>
          <span>
            Time: <strong className="text-foreground tabular-nums">{r.elapsed_seconds ? `${r.elapsed_seconds.toFixed(1)}s` : "-"}</strong>
          </span>
        </div>
      </CardContent>
    </Card>
  )
}
