import { useState, useMemo, useCallback } from "react"
import { useTasksLiveSummary, useLanguages } from "@/hooks/use-api"
import { startBatch } from "@/lib/api"
import type { TaskLiveSummary } from "@/types/api"
import { PageHeader } from "@/components/layout/page-header"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { cn } from "@/lib/utils"
import {
  Activity,
  Clock,
  Zap,
  Rocket,
  Loader2,
  CheckCircle2,
  FlaskConical,
  Globe,
} from "lucide-react"
import { toast } from "sonner"

type TypeFilter = "all" | "real" | "synthetic"
type TierFilter = "all" | 1 | 2 | 3

const TIER_COLORS: Record<number, string> = {
  1: "bg-emerald-50 text-emerald-700 border-emerald-200",
  2: "bg-blue-50 text-blue-700 border-blue-200",
  3: "bg-purple-50 text-purple-700 border-purple-200",
}

export function TasksPanel() {
  const { data: tasks, isLoading } = useTasksLiveSummary()
  const { data: languages } = useLanguages()
  const [typeFilter, setTypeFilter] = useState<TypeFilter>("all")
  const [tierFilter, setTierFilter] = useState<TierFilter>("all")
  const [runningReal, setRunningReal] = useState(false)
  const [realCount, setRealCount] = useState(1)

  const { realTasks, syntheticTasks } = useMemo(() => {
    if (!tasks) return { realTasks: [], syntheticTasks: [] }
    const real = tasks.filter((t) => t.live_runs > 0)
    const synthetic = tasks.filter((t) => t.live_runs === 0)
    return { realTasks: real, syntheticTasks: synthetic }
  }, [tasks])

  const filtered = useMemo(() => {
    if (!tasks) return []
    let list = tasks
    if (typeFilter === "real") list = realTasks
    else if (typeFilter === "synthetic") list = syntheticTasks
    if (tierFilter !== "all") list = list.filter((t) => t.tier === tierFilter)
    return list.sort((a, b) => {
      // Real tasks first, then by tier, then by name
      const aReal = a.live_runs > 0 ? 0 : 1
      const bReal = b.live_runs > 0 ? 0 : 1
      if (aReal !== bReal) return aReal - bReal
      return a.tier - b.tier || a.name.localeCompare(b.name)
    })
  }, [tasks, realTasks, syntheticTasks, typeFilter, tierFilter])

  const summary = useMemo(() => {
    if (!tasks) return { total: 0, real: 0, synthetic: 0, totalRuns: 0 }
    return {
      total: tasks.length,
      real: realTasks.length,
      synthetic: syntheticTasks.length,
      totalRuns: tasks.reduce((s, t) => s + t.live_runs, 0),
    }
  }, [tasks, realTasks, syntheticTasks])

  const handleRunAllReal = useCallback(async () => {
    if (realTasks.length === 0) {
      toast.error("No real tasks discovered yet")
      return
    }
    const langCodes = languages ? Object.keys(languages) : ["no"]
    const taskNames = realTasks.map((t) => t.name)
    const total = taskNames.length * langCodes.length * realCount

    setRunningReal(true)
    try {
      await startBatch(taskNames, langCodes, realCount)
      toast.success(
        `Launched ${total} evals (${taskNames.length} real tasks x ${langCodes.length} langs x ${realCount})`
      )
    } catch (err) {
      toast.error("Failed: " + (err as Error).message)
    } finally {
      setRunningReal(false)
    }
  }, [realTasks, languages, realCount])

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <div className="grid grid-cols-3 gap-3">
          <Skeleton className="h-20 rounded-xl" />
          <Skeleton className="h-20 rounded-xl" />
          <Skeleton className="h-20 rounded-xl" />
        </div>
        <Skeleton className="h-96 w-full rounded-xl" />
      </div>
    )
  }

  const typeFilters: { label: string; value: TypeFilter; icon: React.ReactNode }[] = [
    { label: "All", value: "all", icon: null },
    {
      label: `Real (${summary.real})`,
      value: "real",
      icon: <CheckCircle2 className="h-3 w-3" />,
    },
    {
      label: `Synthetic (${summary.synthetic})`,
      value: "synthetic",
      icon: <FlaskConical className="h-3 w-3" />,
    },
  ]

  const tierFilters: { label: string; value: TierFilter }[] = [
    { label: "All", value: "all" },
    { label: "T1", value: 1 },
    { label: "T2", value: 2 },
    { label: "T3", value: 3 },
  ]

  // Show a separator row between real and synthetic when showing "all"
  const showSeparator = typeFilter === "all" && realTasks.length > 0 && syntheticTasks.length > 0

  return (
    <div>
      <PageHeader
        title="Tasks"
        description={`${summary.real} real task types discovered from competition runs`}
      >
        <div className="flex items-center gap-2">
          {/* Type filter */}
          <div className="flex items-center bg-muted/60 rounded-lg p-0.5">
            {typeFilters.map((f) => (
              <button
                key={f.value}
                onClick={() => setTypeFilter(f.value)}
                className={cn(
                  "h-7 px-3 rounded-md text-[12px] font-medium transition-all duration-150 flex items-center gap-1",
                  typeFilter === f.value
                    ? f.value === "real"
                      ? "bg-emerald-600 text-white shadow-sm"
                      : "bg-white text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {f.icon}
                {f.label}
              </button>
            ))}
          </div>
          {/* Tier filter */}
          <div className="flex items-center bg-muted/60 rounded-lg p-0.5">
            {tierFilters.map((f) => (
              <button
                key={String(f.value)}
                onClick={() => setTierFilter(f.value)}
                className={cn(
                  "h-7 px-2.5 rounded-md text-[12px] font-medium transition-all duration-150",
                  tierFilter === f.value
                    ? "bg-white text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>
      </PageHeader>

      {/* Summary cards */}
      <div className="grid grid-cols-4 gap-3 mb-5">
        <SummaryCard
          icon={CheckCircle2}
          label="Real Tasks"
          value={String(summary.real)}
          sub="from competition"
          accent="emerald"
        />
        <SummaryCard
          icon={FlaskConical}
          label="Synthetic"
          value={String(summary.synthetic)}
          sub="not yet observed"
        />
        <SummaryCard
          icon={Activity}
          label="Live Runs"
          value={String(summary.totalRuns)}
          sub="competition requests"
        />
        <SummaryCard
          icon={Zap}
          label="Coverage"
          value={
            summary.total > 0
              ? `${((summary.real / summary.total) * 100).toFixed(0)}%`
              : "0%"
          }
          sub="tasks discovered"
        />
      </div>

      {/* Run all real tasks action bar */}
      {summary.real > 0 && (
        <Card className="shadow-premium mb-5 border-emerald-200/60 bg-gradient-to-r from-emerald-50/50 to-transparent">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="h-9 w-9 rounded-lg bg-emerald-100 flex items-center justify-center shrink-0">
                  <Rocket className="h-4 w-4 text-emerald-700" />
                </div>
                <div>
                  <p className="text-[13px] font-semibold text-emerald-900">
                    Run All Real Tasks
                  </p>
                  <p className="text-[11px] text-emerald-700/70">
                    {summary.real} tasks x{" "}
                    {languages ? Object.keys(languages).length : 1} languages x{" "}
                    {realCount} = {summary.real * (languages ? Object.keys(languages).length : 1) * realCount} evals
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5">
                  <Globe className="h-3.5 w-3.5 text-emerald-600/60" />
                  <span className="text-[11px] text-emerald-700/70">
                    {languages ? Object.keys(languages).length : 1} langs
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <label className="text-[11px] text-emerald-700/70">Variations</label>
                  <Input
                    type="number"
                    min={1}
                    max={10}
                    value={realCount}
                    onChange={(e) => setRealCount(Math.max(1, parseInt(e.target.value) || 1))}
                    className="w-14 h-7 text-[12px] text-center tabular-nums border-emerald-200"
                  />
                </div>
                <Button
                  onClick={handleRunAllReal}
                  disabled={runningReal}
                  size="sm"
                  className="h-8 px-4 font-semibold bg-emerald-600 hover:bg-emerald-700 text-white"
                >
                  {runningReal ? (
                    <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                  ) : (
                    <Rocket className="h-3.5 w-3.5 mr-1.5" />
                  )}
                  {runningReal ? "Launching..." : "Run"}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Task table */}
      <Card className="shadow-premium overflow-hidden">
        <CardContent className="p-0">
          <div className="px-4 py-2.5 border-b bg-muted/20 flex items-center justify-between">
            <span className="text-[12px] text-muted-foreground font-medium">
              {filtered.length} tasks shown
            </span>
          </div>
          <Table>
            <TableHeader>
              <TableRow className="text-[11px] bg-muted/10">
                <TableHead className="font-semibold">Task</TableHead>
                <TableHead className="w-20 font-semibold">Type</TableHead>
                <TableHead className="w-16 font-semibold">Tier</TableHead>
                <TableHead className="w-16 font-semibold text-center">Runs</TableHead>
                <TableHead className="w-20 font-semibold text-right">Avg API</TableHead>
                <TableHead className="w-20 font-semibold text-right">Baseline</TableHead>
                <TableHead className="w-20 font-semibold text-right">Errors</TableHead>
                <TableHead className="w-20 font-semibold text-right">Avg Time</TableHead>
                <TableHead className="w-24 font-semibold text-right">Last Run</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((t, i) => {
                const isReal = t.live_runs > 0
                const prevIsReal = i > 0 && filtered[i - 1].live_runs > 0
                const showSep = showSeparator && !isReal && prevIsReal
                return (
                  <>
                    {showSep && (
                      <TableRow key="separator" className="border-0">
                        <TableCell colSpan={9} className="py-3 px-4">
                          <div className="flex items-center gap-2">
                            <div className="h-px flex-1 bg-border" />
                            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1">
                              <FlaskConical className="h-3 w-3" />
                              Synthetic Tasks — not yet observed in competition
                            </span>
                            <div className="h-px flex-1 bg-border" />
                          </div>
                        </TableCell>
                      </TableRow>
                    )}
                    <TaskRow key={t.name} task={t} />
                  </>
                )
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}

function TaskRow({ task: t }: { task: TaskLiveSummary }) {
  const isReal = t.live_runs > 0
  const avgCalls = t.avg_api_calls != null ? t.avg_api_calls.toFixed(1) : "-"
  const efficiency =
    isReal && t.avg_api_calls != null
      ? t.avg_api_calls <= t.baseline_calls
        ? "text-emerald-600"
        : t.avg_api_calls <= t.baseline_calls * 1.5
        ? "text-amber-600"
        : "text-red-600"
      : ""

  return (
    <TableRow
      className={cn(
        "text-[12px] transition-colors",
        isReal
          ? "bg-emerald-50/30 hover:bg-emerald-50/60"
          : "opacity-50 hover:opacity-75"
      )}
    >
      <TableCell className="py-2.5">
        <div>
          <span className="font-medium">{t.name}</span>
          <p className="text-[10px] text-muted-foreground mt-0.5 max-w-[280px] truncate">
            {t.description}
          </p>
        </div>
      </TableCell>
      <TableCell className="py-2.5">
        {isReal ? (
          <Badge className="text-[10px] font-semibold bg-emerald-100 text-emerald-700 border-emerald-200 hover:bg-emerald-100">
            <CheckCircle2 className="h-3 w-3 mr-0.5" />
            REAL
          </Badge>
        ) : (
          <Badge
            variant="outline"
            className="text-[10px] font-medium text-muted-foreground border-dashed"
          >
            <FlaskConical className="h-3 w-3 mr-0.5" />
            Synth
          </Badge>
        )}
      </TableCell>
      <TableCell className="py-2.5">
        <Badge
          variant="outline"
          className={cn("text-[10px] font-semibold", TIER_COLORS[t.tier])}
        >
          T{t.tier}
        </Badge>
      </TableCell>
      <TableCell className="py-2.5 text-center">
        {isReal ? (
          <span className="font-semibold tabular-nums">{t.live_runs}</span>
        ) : (
          <span className="text-muted-foreground">-</span>
        )}
      </TableCell>
      <TableCell
        className={cn(
          "py-2.5 text-right tabular-nums font-semibold",
          efficiency
        )}
      >
        {avgCalls}
      </TableCell>
      <TableCell className="py-2.5 text-right tabular-nums text-muted-foreground">
        {t.baseline_calls}
      </TableCell>
      <TableCell className="py-2.5 text-right tabular-nums">
        {isReal && t.avg_api_errors != null ? (
          <span
            className={
              t.avg_api_errors > 0
                ? "text-red-600 font-semibold"
                : "text-muted-foreground"
            }
          >
            {t.avg_api_errors.toFixed(1)}
          </span>
        ) : (
          <span className="text-muted-foreground">-</span>
        )}
      </TableCell>
      <TableCell className="py-2.5 text-right tabular-nums text-muted-foreground">
        {isReal && t.avg_elapsed != null ? (
          <span className="flex items-center justify-end gap-1">
            <Clock className="h-3 w-3 opacity-40" />
            {t.avg_elapsed.toFixed(1)}s
          </span>
        ) : (
          "-"
        )}
      </TableCell>
      <TableCell className="py-2.5 text-right text-[10px] text-muted-foreground">
        {t.last_run ? formatRelative(t.last_run) : "-"}
      </TableCell>
    </TableRow>
  )
}

function SummaryCard({
  icon: Icon,
  label,
  value,
  sub,
  accent,
}: {
  icon: React.ComponentType<{ className?: string }>
  label: string
  value: string
  sub: string
  accent?: string
}) {
  const isAccent = accent === "emerald"
  return (
    <Card
      className={cn(
        "shadow-premium",
        isAccent && "border-emerald-200/60 bg-emerald-50/30"
      )}
    >
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          <div
            className={cn(
              "h-9 w-9 rounded-lg flex items-center justify-center shrink-0",
              isAccent ? "bg-emerald-100" : "bg-muted/60"
            )}
          >
            <Icon
              className={cn(
                "h-4 w-4",
                isAccent ? "text-emerald-700" : "text-muted-foreground"
              )}
            />
          </div>
          <div>
            <p
              className={cn(
                "text-[10px] font-medium uppercase tracking-wide",
                isAccent ? "text-emerald-700" : "text-muted-foreground"
              )}
            >
              {label}
            </p>
            <p
              className={cn(
                "text-lg font-bold tabular-nums leading-tight",
                isAccent && "text-emerald-800"
              )}
            >
              {value}
            </p>
            <p
              className={cn(
                "text-[10px]",
                isAccent ? "text-emerald-600/70" : "text-muted-foreground"
              )}
            >
              {sub}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function formatRelative(iso: string): string {
  try {
    const d = new Date(iso)
    const now = new Date()
    const diff = Math.floor((now.getTime() - d.getTime()) / 1000)
    if (diff < 60) return "just now"
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
    return `${Math.floor(diff / 86400)}d ago`
  } catch {
    return iso
  }
}
