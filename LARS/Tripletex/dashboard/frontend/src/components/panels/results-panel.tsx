import { useState, useMemo } from "react"
import { useRuns } from "@/hooks/use-api"
import { cleanupStale as apiCleanupStale } from "@/lib/api"
import type { EvalRun, FieldCheck } from "@/types/api"
import { PageHeader } from "@/components/layout/page-header"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { toast } from "sonner"
import { cn } from "@/lib/utils"
import {
  Trash2,
  Table2,
  CheckCircle2,
  XCircle,
  Loader2,
  ChevronDown,
  ChevronRight,
  Radio,
  FlaskConical,
} from "lucide-react"

type StatusFilter = "all" | "completed" | "failed" | "running"
type SourceFilter = "all" | "simulator" | "competition"

function formatDateTime(iso: string | null): string {
  if (!iso) return "-"
  try {
    const d = new Date(iso)
    if (isNaN(d.getTime())) return "-"
    return d.toLocaleString("nb-NO", {
      day: "2-digit",
      month: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    })
  } catch {
    return "-"
  }
}

export function ResultsPanel({ defaultSource = "all" }: { defaultSource?: SourceFilter }) {
  const [filter, setFilter] = useState<StatusFilter>("all")
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>(defaultSource)
  const { data: runs, isLoading, mutate } = useRuns(filter, sourceFilter, 5000)
  const [expandedId, setExpandedId] = useState<number | null>(null)

  const sortedRuns = useMemo(() => runs ?? [], [runs])

  const failedN = sortedRuns.filter((r) => r.status === "failed").length

  async function handleCleanup() {
    try {
      const result = await apiCleanupStale()
      if (result.cleaned > 0) {
        toast.success(`Cleaned up ${result.cleaned} stale run(s)`)
        mutate()
      } else {
        toast.info("No stale runs found")
      }
    } catch (err) {
      toast.error((err as Error).message)
    }
  }

  const filters: { label: string; value: StatusFilter }[] = [
    { label: "All", value: "all" },
    { label: "Completed", value: "completed" },
    { label: "Failed", value: "failed" },
    { label: "Running", value: "running" },
  ]

  const sourceFilters: { label: string; value: SourceFilter }[] = [
    { label: "All", value: "all" },
    { label: "Live", value: "competition" },
    { label: "Simulator", value: "simulator" },
  ]

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-64 w-full rounded-xl" />
      </div>
    )
  }

  return (
    <div>
      <PageHeader title="Results">
        <div className="flex items-center gap-1.5">
          {/* Source filter */}
          <div className="flex items-center bg-muted/60 rounded-lg p-0.5">
            {sourceFilters.map((f) => (
              <button
                key={f.value}
                onClick={() => setSourceFilter(f.value)}
                className={cn(
                  "h-7 px-3 rounded-md text-[12px] font-medium transition-all duration-150",
                  sourceFilter === f.value
                    ? "bg-white text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {f.label}
              </button>
            ))}
          </div>
          {/* Status filter */}
          <div className="flex items-center bg-muted/60 rounded-lg p-0.5">
            {filters.map((f) => (
              <button
                key={f.value}
                onClick={() => setFilter(f.value)}
                className={cn(
                  "h-7 px-3 rounded-md text-[12px] font-medium transition-all duration-150",
                  filter === f.value
                    ? "bg-white text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {f.label}
              </button>
            ))}
          </div>
          <Button
            variant="outline"
            size="sm"
            className="h-7 text-[12px] px-2.5 text-amber-600 border-amber-200 hover:bg-amber-50"
            onClick={handleCleanup}
          >
            <Trash2 className="h-3 w-3 mr-1" />
            Clean Stale
          </Button>
        </div>
      </PageHeader>

      {sortedRuns.length === 0 ? (
        <div className="text-center py-20 text-muted-foreground">
          <Table2 className="h-12 w-12 mx-auto mb-4 opacity-20" />
          <p className="text-[14px] font-medium mb-1">No eval runs yet</p>
          <p className="text-[12px] text-muted-foreground/70">Select tasks and click Run to start evaluating.</p>
        </div>
      ) : (
        <Card className="shadow-premium overflow-hidden">
          <CardContent className="p-0">
            <div className="px-4 py-2.5 border-b bg-muted/20 flex items-center justify-between">
              <span className="text-[12px] text-muted-foreground font-medium">
                {sortedRuns.length} shown
                {failedN ? ` / ${failedN} failed` : ""}
              </span>
            </div>
            <Table>
              <TableHeader>
                <TableRow className="text-[11px] bg-muted/10">
                  <TableHead className="w-10 font-semibold">#</TableHead>
                  <TableHead className="font-semibold">Task</TableHead>
                  <TableHead className="w-16 font-semibold">Source</TableHead>
                  <TableHead className="w-12 font-semibold">Lang</TableHead>
                  <TableHead className="w-24 font-semibold">Correct</TableHead>
                  <TableHead className="w-20 font-semibold">Score</TableHead>
                  <TableHead className="w-16 font-semibold">API</TableHead>
                  <TableHead className="w-16 font-semibold">Time</TableHead>
                  <TableHead className="w-28 font-semibold">Date</TableHead>
                  <TableHead className="w-24 font-semibold">Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sortedRuns.map((r) => (
                  <RunRow
                    key={r.id}
                    run={r}
                    expanded={expandedId === r.id}
                    onToggle={() =>
                      setExpandedId(expandedId === r.id ? null : r.id)
                    }
                  />
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function RunRow({
  run: r,
  expanded,
  onToggle,
}: {
  run: EvalRun
  expanded: boolean
  onToggle: () => void
}) {
  const corr =
    r.correctness != null ? `${(r.correctness * 100).toFixed(0)}%` : "-"
  const corrClass =
    r.correctness != null
      ? r.correctness >= 1
        ? "text-emerald-600"
        : r.correctness >= 0.5
        ? "text-amber-600"
        : "text-red-600"
      : ""
  const score =
    r.final_score != null
      ? `${r.final_score.toFixed(2)}${r.max_possible != null ? `/${r.max_possible}` : ""}`
      : "-"

  // Correctness as mini bar
  const corrPct = r.correctness != null ? r.correctness * 100 : 0
  const corrBarColor =
    r.correctness != null
      ? r.correctness >= 1
        ? "bg-emerald-500"
        : r.correctness >= 0.5
        ? "bg-amber-500"
        : "bg-red-500"
      : "bg-muted"

  return (
    <>
      <TableRow
        className={cn(
          "cursor-pointer text-[12px] transition-colors duration-100 group",
          expanded ? "bg-muted/30" : "hover:bg-muted/20"
        )}
        onClick={onToggle}
      >
        <TableCell className="font-mono text-muted-foreground py-2.5 pl-4">
          <div className="flex items-center gap-1">
            {expanded ? (
              <ChevronDown className="h-3 w-3 text-muted-foreground/50" />
            ) : (
              <ChevronRight className="h-3 w-3 text-muted-foreground/30 group-hover:text-muted-foreground/60 transition-colors" />
            )}
            {r.id}
          </div>
        </TableCell>
        <TableCell className="py-2.5 font-medium">{r.task_name}</TableCell>
        <TableCell className="py-2.5">
          <SourceBadge source={r.source} />
        </TableCell>
        <TableCell className="py-2.5">
          <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
            {r.language || "-"}
          </span>
        </TableCell>
        <TableCell className={cn("py-2.5", corrClass)}>
          <div className="flex items-center gap-2">
            <span className="font-semibold tabular-nums w-8">{corr}</span>
            {r.correctness != null && (
              <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden max-w-[60px]">
                <div
                  className={cn("h-full rounded-full transition-all duration-500", corrBarColor)}
                  style={{ width: `${corrPct}%` }}
                />
              </div>
            )}
          </div>
        </TableCell>
        <TableCell className={cn("py-2.5 font-semibold tabular-nums", corrClass)}>
          {score}
        </TableCell>
        <TableCell className="py-2.5 tabular-nums text-muted-foreground">{r.api_calls || "-"}</TableCell>
        <TableCell className="py-2.5 tabular-nums text-muted-foreground">
          {r.elapsed_seconds ? `${r.elapsed_seconds.toFixed(1)}s` : "-"}
        </TableCell>
        <TableCell className="py-2.5 tabular-nums text-muted-foreground text-[11px]">
          {formatDateTime(r.created_at)}
        </TableCell>
        <TableCell className="py-2.5">
          <StatusBadge status={r.status} />
        </TableCell>
      </TableRow>
      {expanded && (
        <TableRow>
          <TableCell colSpan={10} className="bg-muted/20 p-0">
            <div className="p-4 animate-fade-in-up">
              <RunDetail run={r} />
            </div>
          </TableCell>
        </TableRow>
      )}
    </>
  )
}

function SourceBadge({ source }: { source: string | null }) {
  const s = source || "simulator"
  if (s === "competition") {
    return (
      <Badge variant="outline" className="text-[10px] font-semibold bg-violet-50 text-violet-700 border-violet-200">
        <Radio className="h-2.5 w-2.5 mr-1" />
        live
      </Badge>
    )
  }
  return (
    <Badge variant="outline" className="text-[10px] font-semibold bg-slate-50 text-slate-600 border-slate-200">
      <FlaskConical className="h-2.5 w-2.5 mr-1" />
      sim
    </Badge>
  )
}

function StatusBadge({ status }: { status: string }) {
  if (status === "completed") {
    return (
      <Badge variant="outline" className="text-[10px] font-semibold bg-emerald-50 text-emerald-700 border-emerald-200">
        <CheckCircle2 className="h-2.5 w-2.5 mr-1" />
        {status}
      </Badge>
    )
  }
  if (status === "running") {
    return (
      <Badge variant="outline" className="text-[10px] font-semibold bg-blue-50 text-blue-700 border-blue-200">
        <Loader2 className="h-2.5 w-2.5 mr-1 animate-spin" />
        {status}
      </Badge>
    )
  }
  return (
    <Badge variant="outline" className="text-[10px] font-semibold bg-red-50 text-red-700 border-red-200">
      <XCircle className="h-2.5 w-2.5 mr-1" />
      {status}
    </Badge>
  )
}

function RunDetail({ run: r }: { run: EvalRun }) {
  let checks: FieldCheck[] = []
  try {
    checks = JSON.parse(r.checks_json || "[]")
  } catch { /* ignore */ }

  return (
    <Card className="terminal-bg border-white/[0.06] overflow-hidden">
      <div className="flex items-center gap-1.5 px-4 py-2 border-b border-white/[0.06]">
        <span className="h-2 w-2 rounded-full bg-red-500/80" />
        <span className="h-2 w-2 rounded-full bg-amber-500/80" />
        <span className="h-2 w-2 rounded-full bg-green-500/80" />
        <span className="ml-3 text-[10px] text-white/25 font-mono">
          run #{r.id} | {r.source || "simulator"} | {r.created_at ? new Date(r.created_at).toLocaleString("nb-NO") : ""}
        </span>
      </div>
      <pre className="p-4 text-[11px] font-mono whitespace-pre-wrap leading-relaxed max-h-[300px] overflow-y-auto text-slate-300">
        {r.error_message && (
          <span className="text-red-400 font-semibold">
            ERROR: {r.error_message}
            {"\n\n"}
          </span>
        )}
        {checks.length > 0 && (
          <>
            <span className="text-slate-500">Field Checks:</span>{"\n"}
            {checks.map((c, i) => (
              <span key={i}>
                {"  "}
                <span className={c.passed ? "text-emerald-400" : "text-red-400"}>
                  {c.passed ? "PASS" : "FAIL"}
                </span>
                {" "}
                <span className="text-blue-300">{c.field.padEnd(25)}</span>
                {" "}
                <span className="text-slate-400">{c.points}/{c.max}</span>
                {"  "}
                <span className="text-slate-500">{c.detail}</span>
                {"\n"}
              </span>
            ))}
            {"\n"}
          </>
        )}
        <span className="text-slate-500">Prompt:</span>{"\n"}
        <span className="text-slate-300">{r.prompt || "(none)"}</span>{"\n"}
        {r.expected_json && (
          <>
            {"\n"}
            <span className="text-slate-500">Expected:</span>{"\n"}
            <span className="text-slate-400">{r.expected_json}</span>
          </>
        )}
      </pre>
    </Card>
  )
}
