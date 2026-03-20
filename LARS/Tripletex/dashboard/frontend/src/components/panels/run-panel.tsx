import { useState, useCallback, useRef, useEffect } from "react"
import { useTasks, useLanguages, useTasksLiveSummary } from "@/hooks/use-api"
import { startBatch, fetchRuns, fetchSandboxHealth, seedSandbox } from "@/lib/api"
import type { EvalRun } from "@/types/api"
import { PageHeader } from "@/components/layout/page-header"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { toast } from "sonner"
import { cn } from "@/lib/utils"
import { Play, Rocket, Layers, Globe, AlertCircle, Loader2, CheckCircle2 } from "lucide-react"

interface RunPanelProps {
  onBatchDone: (runs: EvalRun[]) => void
  onNavigate: (panel: string) => void
}

export function RunPanel({ onBatchDone, onNavigate }: RunPanelProps) {
  const { data: tasks, isLoading: tasksLoading } = useTasks()
  const { data: languages } = useLanguages()
  const { data: liveSummary } = useTasksLiveSummary()

  const realTaskNames = (liveSummary ?? [])
    .filter((t) => t.live_runs > 0)
    .map((t) => t.name)

  const [selectedTier, setSelectedTier] = useState<number | null>(null)
  const [selectedTasks, setSelectedTasks] = useState<Set<string>>(new Set())
  const [selectedLangs, setSelectedLangs] = useState<Set<string>>(
    new Set(["no"])
  )
  const [count, setCount] = useState(1)
  const [running, setRunning] = useState(false)
  const [progressPct, setProgressPct] = useState(0)
  const [progressText, setProgressText] = useState("")
  const [showPreflight, setShowPreflight] = useState(false)
  const [preflightMissing, setPreflightMissing] = useState<string[]>([])
  const preflightResolveRef = useRef<((action: string) => void) | null>(null)

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const batchStartRef = useRef(0)
  const batchTotalRef = useRef(0)

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [])

  const ENTITY_LABELS: Record<string, string> = {
    department: "Departments",
    employee: "Employees",
    customer: "Customers",
    product: "Products",
    contact: "Contacts",
    project: "Projects",
    invoice: "Invoices",
    travelExpense: "Travel Expenses",
  }

  const tiers = [1, 2, 3]
  const tierCounts = tiers.map(
    (t) => tasks?.filter((task) => task.tier === t).length ?? 0
  )

  function selectRealTasks() {
    setSelectedTier(null)
    setSelectedTasks(new Set(realTaskNames))
    // Also select all languages
    if (languages) {
      setSelectedLangs(new Set(Object.keys(languages)))
    }
  }

  function toggleTier(tier: number) {
    const newTier = selectedTier === tier ? null : tier
    setSelectedTier(newTier)
    if (newTier && tasks) {
      setSelectedTasks(
        new Set(tasks.filter((t) => t.tier === newTier).map((t) => t.name))
      )
    } else {
      setSelectedTasks(new Set())
    }
  }

  function toggleTask(name: string) {
    setSelectedTasks((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
    setSelectedTier(null)
  }

  function toggleLang(code: string) {
    setSelectedLangs((prev) => {
      const next = new Set(prev)
      if (next.has(code)) next.delete(code)
      else next.add(code)
      return next
    })
  }

  const checkPreflight = useCallback(async () => {
    try {
      const health = await fetchSandboxHealth()
      if (!health.connected) return { ready: false, missing: ["Cannot connect to sandbox"] }
      if (health.ready) return { ready: true, missing: [] }
      const missing: string[] = []
      for (const [type, info] of Object.entries(health.entities || {})) {
        if (!info.ok) missing.push(ENTITY_LABELS[type] || type)
      }
      if (!health.bank_account_1920) missing.push("Bank Account 1920")
      return { ready: false, missing }
    } catch {
      return { ready: false, missing: ["Cannot connect to sandbox"] }
    }
  }, [])

  const showPreflightDialog = useCallback(
    (missing: string[]) => {
      setPreflightMissing(missing)
      setShowPreflight(true)
      return new Promise<string>((resolve) => {
        preflightResolveRef.current = resolve
      })
    },
    []
  )

  async function handleRun() {
    if (selectedTasks.size === 0) {
      toast.error("Select at least one task")
      return
    }
    if (selectedLangs.size === 0) {
      toast.error("Select at least one language")
      return
    }

    setRunning(true)
    setProgressPct(0)
    setProgressText("Checking sandbox...")

    const pf = await checkPreflight()
    if (!pf.ready) {
      const action = await showPreflightDialog(pf.missing)
      if (action === "cancel") {
        setRunning(false)
        setProgressText("")
        return
      }
      if (action === "seed") {
        setProgressText("Seeding sandbox...")
        try {
          await seedSandbox(["all"])
          toast.success("Sandbox seeded")
        } catch (err) {
          toast.error("Failed to seed: " + (err as Error).message)
          setRunning(false)
          setProgressText("")
          return
        }
      }
    }

    const taskNames = [...selectedTasks]
    const langs = [...selectedLangs]
    const total = taskNames.length * langs.length * count
    batchTotalRef.current = total
    batchStartRef.current = Date.now()

    setProgressText(
      `Starting ${total} evals (${taskNames.length} tasks x ${langs.length} langs x ${count})...`
    )

    try {
      const result = await startBatch(taskNames, langs, count)
      if ((result as unknown as { error?: string }).error) {
        throw new Error((result as unknown as { error: string }).error)
      }
      setProgressText(`Launched ${total} evals...`)
      if (pollRef.current) clearInterval(pollRef.current)
      pollRef.current = setInterval(() => pollProgress(), 3000)
    } catch (err) {
      toast.error("Error: " + (err as Error).message)
      setRunning(false)
      setProgressText("")
    }
  }

  async function pollProgress() {
    try {
      const runs = await fetchRuns({ limit: 500 })
      const cutoff = new Date(batchStartRef.current - 5000).toISOString()
      const batch = runs.filter((r) => r.created_at > cutoff)
      const done = batch.filter(
        (r) => r.status === "completed" || r.status === "failed"
      ).length
      const runningN = batch.filter((r) => r.status === "running").length
      const pct =
        batchTotalRef.current > 0
          ? Math.min(100, (done / batchTotalRef.current) * 100)
          : 0

      setProgressPct(pct)
      setProgressText(`${done}/${batchTotalRef.current} done, ${runningN} running`)

      if (done >= batchTotalRef.current || (done > 0 && runningN === 0)) {
        if (pollRef.current) clearInterval(pollRef.current)
        pollRef.current = null
        setRunning(false)
        setProgressPct(100)
        const elapsed = ((Date.now() - batchStartRef.current) / 1000).toFixed(0)
        const failed = batch.filter((r) => r.status === "failed").length
        setProgressText(
          `Done: ${done} evals in ${elapsed}s` +
            (failed ? ` (${failed} failed)` : "")
        )
        onBatchDone(batch)
        toast.success(`Batch complete: ${done} evals`)
        onNavigate("report")
      }
    } catch {
      // ignore poll errors
    }
  }

  if (tasksLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-40 w-full rounded-xl" />
      </div>
    )
  }

  const langFlags: Record<string, string> = { no: "NO", en: "EN", sv: "SV", da: "DA", fi: "FI", de: "DE" }

  return (
    <div>
      <PageHeader
        title="Run Evaluations"
        description="Select tasks, languages, and run count to evaluate your Tripletex agent."
      />

      <Card className="shadow-premium">
        <CardContent className="p-5 space-y-5">
          {/* Quick select + Tier selection */}
          <div>
            <label className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2 flex items-center gap-1.5">
              <Layers className="h-3.5 w-3.5" />
              Quick Select
            </label>
            <div className="flex gap-2">
              {realTaskNames.length > 0 && (
                <button
                  onClick={selectRealTasks}
                  className={cn(
                    "relative px-4 py-2 rounded-lg text-[13px] font-semibold transition-all duration-150 border",
                    selectedTasks.size > 0 &&
                      realTaskNames.length === selectedTasks.size &&
                      realTaskNames.every((n) => selectedTasks.has(n))
                      ? "bg-emerald-600 text-white border-emerald-600 shadow-sm"
                      : "bg-card text-foreground border-emerald-300 hover:border-emerald-500 hover:bg-emerald-50/50"
                  )}
                >
                  <CheckCircle2 className="h-3.5 w-3.5 inline mr-1.5 -mt-0.5" />
                  Real Tasks
                  <span className="ml-1.5 text-[11px] opacity-70">
                    ({realTaskNames.length})
                  </span>
                </button>
              )}
              {tiers.map((tier, i) => {
                const isActive = selectedTier === tier
                return (
                  <button
                    key={tier}
                    onClick={() => toggleTier(tier)}
                    className={cn(
                      "relative px-4 py-2 rounded-lg text-[13px] font-semibold transition-all duration-150 border",
                      isActive
                        ? "bg-primary text-primary-foreground border-primary shadow-sm"
                        : "bg-card text-foreground border-border hover:border-primary/30 hover:bg-primary/[0.03]"
                    )}
                  >
                    Tier {tier}
                    <span className={cn(
                      "ml-1.5 text-[11px]",
                      isActive ? "text-primary-foreground/70" : "text-muted-foreground"
                    )}>
                      ({tierCounts[i]})
                    </span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Task cards */}
          <div>
            <label className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2 block">
              Tasks
              {selectedTasks.size > 0 && (
                <span className="ml-1.5 text-primary font-bold">
                  {selectedTasks.size} selected
                </span>
              )}
            </label>
            <div className="flex flex-wrap gap-1.5">
              {tasks?.map((t) => {
                const isSelected = selectedTasks.has(t.name)
                return (
                  <button
                    key={t.name}
                    onClick={() => toggleTask(t.name)}
                    className={cn(
                      "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[12px] font-medium transition-all duration-100 border",
                      isSelected
                        ? "bg-primary/10 text-primary border-primary/25"
                        : "bg-card text-muted-foreground border-border/60 hover:border-primary/20 hover:text-foreground"
                    )}
                  >
                    <span className={cn(
                      "inline-block h-1.5 w-1.5 rounded-full shrink-0",
                      t.tier === 1 ? "bg-emerald-400" : t.tier === 2 ? "bg-amber-400" : "bg-red-400"
                    )} />
                    {t.name.replace(/_/g, " ")}
                    <span className={cn(
                      "text-[10px] tabular-nums",
                      isSelected ? "text-primary/60" : "text-muted-foreground/50"
                    )}>
                      {t.max_points}p
                    </span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Language selection */}
          <div>
            <label className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2 flex items-center gap-1.5">
              <Globe className="h-3.5 w-3.5" />
              Languages
            </label>
            <div className="flex flex-wrap gap-1.5">
              {languages &&
                Object.entries(languages).map(([code, name]) => {
                  const isSelected = selectedLangs.has(code)
                  return (
                    <button
                      key={code}
                      onClick={() => toggleLang(code)}
                      className={cn(
                        "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[12px] font-medium transition-all duration-100 border",
                        isSelected
                          ? "bg-primary/10 text-primary border-primary/25"
                          : "bg-card text-muted-foreground border-border/60 hover:border-primary/20 hover:text-foreground"
                      )}
                    >
                      <span className="font-bold text-[10px] uppercase tracking-wide opacity-60">
                        {langFlags[code] || code.toUpperCase()}
                      </span>
                      {name}
                    </button>
                  )
                })}
            </div>
          </div>

          {/* Run controls */}
          <div className="flex items-center gap-4 pt-1">
            <div className="flex items-center gap-2">
              <label className="text-[12px] text-muted-foreground font-medium">
                Variations
              </label>
              <Input
                type="number"
                min={1}
                max={20}
                value={count}
                onChange={(e) => setCount(parseInt(e.target.value) || 1)}
                className="w-16 h-8 text-sm text-center tabular-nums"
              />
            </div>
            <Button
              onClick={handleRun}
              disabled={running}
              className={cn(
                "h-9 px-5 font-semibold transition-all duration-200",
                !running && "bg-gradient-to-r from-primary to-blue-600 hover:shadow-lg hover:shadow-primary/25 hover:scale-[1.02] active:scale-[0.98]"
              )}
            >
              {running ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Rocket className="h-4 w-4 mr-2" />
              )}
              {running ? "Running..." : "Run Evaluation"}
            </Button>
            {selectedTasks.size > 0 && !running && (
              <span className="text-[12px] text-muted-foreground tabular-nums">
                {selectedTasks.size * selectedLangs.size * count} total evals
              </span>
            )}
          </div>

          {/* Progress */}
          {(running || progressText) && (
            <div className="space-y-2 pt-1">
              <div className="relative">
                <Progress value={progressPct} className="h-2" />
                {running && (
                  <div className="absolute inset-0 rounded-full overflow-hidden">
                    <div className="h-full shimmer" />
                  </div>
                )}
              </div>
              <div className="flex items-center justify-between">
                <p className="text-[12px] text-muted-foreground">
                  {progressText}
                </p>
                {progressPct > 0 && (
                  <span className="text-[12px] font-semibold tabular-nums text-primary">
                    {Math.round(progressPct)}%
                  </span>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Preflight Dialog */}
      <Dialog open={showPreflight} onOpenChange={(open) => {
        if (!open) {
          setShowPreflight(false)
          preflightResolveRef.current?.("cancel")
        }
      }}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertCircle className="h-5 w-5 text-amber-500" />
              Sandbox Not Ready
            </DialogTitle>
            <DialogDescription>
              The following entities are missing. Evaluations may fail without them.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-1.5 my-3">
            {preflightMissing.map((m) => (
              <div
                key={m}
                className="flex items-center gap-2 text-[13px] px-3 py-2 rounded-lg bg-amber-50 border border-amber-200/60 text-amber-800"
              >
                <AlertCircle className="h-3.5 w-3.5 shrink-0 text-amber-500" />
                {m}
              </div>
            ))}
          </div>
          <DialogFooter className="gap-2 sm:gap-2">
            <Button
              variant="ghost"
              onClick={() => {
                setShowPreflight(false)
                preflightResolveRef.current?.("cancel")
              }}
            >
              Cancel
            </Button>
            <Button
              variant="outline"
              onClick={() => {
                setShowPreflight(false)
                preflightResolveRef.current?.("skip")
              }}
            >
              Run Anyway
            </Button>
            <Button
              onClick={() => {
                setShowPreflight(false)
                preflightResolveRef.current?.("seed")
              }}
              className="bg-gradient-to-r from-primary to-blue-600"
            >
              <Play className="h-3.5 w-3.5 mr-1.5" />
              Seed & Run
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
