import { useState, useMemo } from "react"
import type { EvalRun, Languages } from "@/types/api"
import { useLanguages } from "@/hooks/use-api"
import { PageHeader } from "@/components/layout/page-header"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { ScrollArea } from "@/components/ui/scroll-area"
import { toast } from "sonner"
import { cn } from "@/lib/utils"
import {
  Copy,
  FileText,
  Trophy,
  AlertTriangle,
  Clock,
  BarChart3,
  Check,
} from "lucide-react"

interface ReportPanelProps {
  batchRuns: EvalRun[]
}

function generateReportText(runs: EvalRun[], languages: Languages): string {
  const completed = runs.filter((r) => r.status === "completed")
  const crashed = runs.filter((r) => r.status === "failed")
  const tiers = [...new Set(runs.map((r) => r.tier))].sort()
  const tierLabel = tiers.map((t) => `Tier ${t}`).join(", ")
  const langs = [...new Set(runs.map((r) => r.language))].sort()
  const langLabel = langs.map((l) => languages[l] || l).join(", ")

  let out = `# Tripletex Eval Report\n${tierLabel} | ${langLabel} | ${runs.length} runs\n\n`
  const perfect = completed.filter((r) => (r.correctness ?? 0) >= 1).length
  const withFailures = completed.filter((r) => (r.correctness ?? 0) < 1)
  out += `## Summary\n- ${completed.length} completed, ${crashed.length} crashed\n- ${perfect}/${completed.length} scored 100% correctness\n`
  if (withFailures.length) out += `- ${withFailures.length} tasks had failed field checks\n`
  out += `\n`

  const byTask: Record<string, EvalRun[]> = {}
  runs.forEach((r) => {
    if (!byTask[r.task_name]) byTask[r.task_name] = []
    byTask[r.task_name].push(r)
  })

  const taskNames = Object.keys(byTask).sort((a, b) => {
    const aF = byTask[a].some((r) => r.status === "completed" && (r.correctness ?? 0) < 1)
    const bF = byTask[b].some((r) => r.status === "completed" && (r.correctness ?? 0) < 1)
    if (aF !== bF) return aF ? -1 : 1
    return a.localeCompare(b)
  })

  for (const taskName of taskNames) {
    const taskRuns = byTask[taskName]
    const taskCompleted = taskRuns.filter((r) => r.status === "completed")
    const taskCrashed = taskRuns.filter((r) => r.status === "failed")
    const taskPerfect = taskCompleted.filter((r) => (r.correctness ?? 0) >= 1)
    const tier = taskRuns[0]?.tier || "?"
    const allPerfect = taskPerfect.length === taskCompleted.length && !taskCrashed.length
    out += `## [${allPerfect ? "PASS" : "FAIL"}] ${taskName} (tier ${tier}) — ${taskPerfect.length}/${taskRuns.length} perfect\n`
    const multiLang = langs.length > 1
    taskRuns.forEach((r, idx) => {
      const langTag = multiLang ? ` [${r.language}]` : ""
      if (r.status === "failed") {
        out += `  Run ${idx + 1}${langTag}: CRASHED\n    Error: ${r.error_message || "unknown"}\n`
        return
      }
      const corr = r.correctness ?? 0
      out += `  Run ${idx + 1}${langTag}: ${(corr * 100).toFixed(0)}% correct, score ${r.final_score}/${r.max_possible}\n`
      if (corr < 1) {
        out += `    Prompt: ${r.prompt}\n`
        try {
          const checks = JSON.parse(r.checks_json || "[]")
          const failed = checks.filter((c: { passed: boolean }) => !c.passed)
          if (failed.length) {
            out += `    Failed checks:\n`
            failed.forEach((c: { field: string; detail: string; points: number; max: number }) => {
              out += `      FAIL ${c.field}: ${c.detail} (${c.points}/${c.max} pts)\n`
            })
          }
        } catch { /* ignore */ }
      }
    })
    out += `\n`
  }

  return out
}

export function ReportPanel({ batchRuns }: ReportPanelProps) {
  const { data: languages } = useLanguages()
  const [copied, setCopied] = useState(false)

  const reportText = useMemo(
    () => (batchRuns.length ? generateReportText(batchRuns, languages ?? {}) : ""),
    [batchRuns, languages]
  )

  const completed = batchRuns.filter((r) => r.status === "completed")
  const crashed = batchRuns.filter((r) => r.status === "failed")
  const perfect = completed.filter((r) => (r.correctness ?? 0) >= 1)
  const avgScore = completed.length
    ? completed.reduce((s, r) => s + (r.final_score ?? 0), 0) / completed.length
    : 0
  const avgCorr = completed.length
    ? completed.reduce((s, r) => s + (r.correctness ?? 0), 0) / completed.length
    : 0
  const avgTime = completed.length
    ? completed.reduce((s, r) => s + (r.elapsed_seconds ?? 0), 0) / completed.length
    : 0

  function handleCopy() {
    navigator.clipboard.writeText(reportText)
    setCopied(true)
    toast.success("Report copied")
    setTimeout(() => setCopied(false), 2000)
  }

  if (!batchRuns.length) {
    return (
      <div>
        <PageHeader title="Report" />
        <div className="text-center py-20 text-muted-foreground">
          <FileText className="h-12 w-12 mx-auto mb-4 opacity-20" />
          <p className="text-[14px] font-medium mb-1">No report yet</p>
          <p className="text-[12px] text-muted-foreground/70">Run a batch evaluation to generate a report.</p>
        </div>
      </div>
    )
  }

  return (
    <div>
      <PageHeader title="Eval Report">
        <Button
          variant={copied ? "default" : "outline"}
          size="sm"
          onClick={handleCopy}
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
          {copied ? "Copied!" : "Copy Report"}
        </Button>
      </PageHeader>

      {/* Hero metrics */}
      <div className="grid grid-cols-5 gap-3 mb-4">
        <MetricCard
          icon={<BarChart3 className="h-4 w-4" />}
          label="Total Runs"
          value={batchRuns.length}
          iconBg="bg-blue-50 text-blue-600"
        />
        <MetricCard
          icon={<BarChart3 className="h-4 w-4" />}
          label="Avg Score"
          value={avgScore.toFixed(2)}
          iconBg={avgCorr >= 0.8 ? "bg-emerald-50 text-emerald-600" : avgCorr >= 0.5 ? "bg-amber-50 text-amber-600" : "bg-red-50 text-red-600"}
          valueClass={avgCorr >= 0.8 ? "text-emerald-600" : avgCorr >= 0.5 ? "text-amber-600" : "text-red-600"}
        />
        <MetricCard
          icon={<Trophy className="h-4 w-4" />}
          label="Perfect"
          value={`${perfect.length}/${completed.length}`}
          iconBg={perfect.length === completed.length ? "bg-emerald-50 text-emerald-600" : "bg-amber-50 text-amber-600"}
          valueClass={perfect.length === completed.length ? "text-emerald-600" : "text-amber-600"}
        />
        <MetricCard
          icon={<AlertTriangle className="h-4 w-4" />}
          label="Crashed"
          value={crashed.length}
          iconBg={crashed.length ? "bg-red-50 text-red-600" : "bg-emerald-50 text-emerald-600"}
          valueClass={crashed.length ? "text-red-600" : "text-emerald-600"}
        />
        <MetricCard
          icon={<Clock className="h-4 w-4" />}
          label="Avg Time"
          value={`${avgTime.toFixed(1)}s`}
          iconBg="bg-slate-100 text-slate-600"
        />
      </div>

      {/* Report output */}
      <Card className="shadow-premium">
        <CardHeader className="p-4 pb-0">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <FileText className="h-4 w-4 text-muted-foreground" />
            Full Report
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          <Separator className="mb-4" />
          <Card className="terminal-bg border-white/[0.06] overflow-hidden">
            <div className="flex items-center gap-1.5 px-4 py-2.5 border-b border-white/[0.06]">
              <span className="h-2.5 w-2.5 rounded-full bg-red-500/80" />
              <span className="h-2.5 w-2.5 rounded-full bg-amber-500/80" />
              <span className="h-2.5 w-2.5 rounded-full bg-green-500/80" />
              <span className="ml-3 text-[11px] text-white/30 font-mono">eval-report.md</span>
            </div>
            <ScrollArea className="h-[350px]">
              <pre className="p-4 font-mono text-[12px] leading-relaxed text-slate-300 whitespace-pre-wrap">
                {reportText}
              </pre>
            </ScrollArea>
          </Card>
        </CardContent>
      </Card>
    </div>
  )
}

function MetricCard({
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
