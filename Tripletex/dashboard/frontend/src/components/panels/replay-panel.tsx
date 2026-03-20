import { useState } from "react"
import { usePayloads } from "@/hooks/use-api"
import { replayPayloads } from "@/lib/api"
import type { ReplayResult, ToolCall } from "@/types/api"
import { PageHeader } from "@/components/layout/page-header"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { Separator } from "@/components/ui/separator"
import { toast } from "sonner"
import { cn } from "@/lib/utils"
import {
  Play,
  Copy,
  CheckCircle,
  XCircle,
  Loader2,
  FileJson,
  Clock,
  ArrowRight,
  Zap,
  AlertTriangle,
} from "lucide-react"

function generateReplayText(results: ReplayResult[]): string {
  const ok = results.filter((r) => r.status === "OK").length
  let out = `# Replay Report\n\n**${ok}/${results.length} passed**\n\n`
  out += `| # | Payload | Status | Time | Tools | API Calls | API Errors |\n`
  out += `|---|---------|--------|------|-------|-----------|------------|\n`
  results.forEach((r, i) => {
    out += `| ${i + 1} | ${r.filename} | ${r.status} | ${(r.elapsed || 0).toFixed(0)}s | ${(r.tool_calls || []).length} | ${r.api_calls || 0} | ${r.api_errors || 0} |\n`
  })
  return out
}

export function ReplayPanel() {
  const { data: payloads, isLoading } = usePayloads()
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [running, setRunning] = useState(false)
  const [results, setResults] = useState<ReplayResult[] | null>(null)

  // Deduplicate by prompt
  const unique = (() => {
    if (!payloads) return []
    const seen = new Set<string>()
    return payloads.filter((p) => {
      if (seen.has(p.prompt)) return false
      seen.add(p.prompt)
      return true
    })
  })()

  function toggleAll(checked: boolean) {
    setSelected(checked ? new Set(unique.map((p) => p.filename)) : new Set())
  }

  function toggle(filename: string) {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(filename)) next.delete(filename)
      else next.add(filename)
      return next
    })
  }

  async function handleReplay() {
    if (selected.size === 0) {
      toast.error("Select at least one payload")
      return
    }
    setRunning(true)
    try {
      const res = await replayPayloads([...selected])
      setResults(res)
      const ok = res.filter((r) => r.status === "OK").length
      toast.success(`Replay done: ${ok}/${res.length} passed`)
    } catch (err) {
      toast.error((err as Error).message)
    } finally {
      setRunning(false)
    }
  }

  function copyReport() {
    if (!results) return
    navigator.clipboard.writeText(generateReplayText(results))
    toast.success("Report copied")
  }

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-48 w-full rounded-xl" />
      </div>
    )
  }

  const okCount = results?.filter((r) => r.status === "OK").length ?? 0
  const failCount = results?.filter((r) => r.status === "FAIL").length ?? 0
  const totalTime = results?.reduce((s, r) => s + (r.elapsed || 0), 0) ?? 0
  const totalApi = results?.reduce((s, r) => s + (r.api_calls || 0), 0) ?? 0
  const totalApiErr = results?.reduce((s, r) => s + (r.api_errors || 0), 0) ?? 0

  return (
    <div>
      <PageHeader
        title="Replay Saved Payloads"
        description="Replay captured payloads through /solve-debug with live sandbox credentials"
      />

      <Card className="mb-4 shadow-premium">
        <CardContent className="p-5 space-y-4">
          <div className="flex gap-2 items-center">
            <Button size="sm" variant="outline" onClick={() => toggleAll(true)}>
              Select All
            </Button>
            <Button size="sm" variant="ghost" onClick={() => toggleAll(false)}>
              Deselect
            </Button>
            <div className="flex-1" />
            <Button
              size="sm"
              onClick={handleReplay}
              disabled={running || selected.size === 0}
              className={cn(
                "font-semibold",
                !running && selected.size > 0 && "bg-gradient-to-r from-primary to-blue-600 hover:shadow-lg hover:shadow-primary/20"
              )}
            >
              {running ? (
                <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
              ) : (
                <Play className="h-3.5 w-3.5 mr-1.5" />
              )}
              {running ? "Replaying..." : `Replay ${selected.size > 0 ? selected.size : ""} Selected`}
            </Button>
          </div>

          {unique.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <FileJson className="h-10 w-10 mx-auto mb-3 opacity-30" />
              <p className="text-[13px]">No saved payloads found in payloads/ directory.</p>
            </div>
          ) : (
            <ScrollArea className="h-[260px] border rounded-lg">
              <div className="divide-y divide-border/50">
                {unique.map((p) => {
                  const isChecked = selected.has(p.filename)
                  return (
                    <label
                      key={p.filename}
                      className={cn(
                        "flex items-center gap-3 px-3 py-2.5 cursor-pointer transition-colors duration-100",
                        isChecked ? "bg-primary/[0.03]" : "hover:bg-muted/40"
                      )}
                    >
                      <Checkbox
                        checked={isChecked}
                        onCheckedChange={() => toggle(p.filename)}
                      />
                      <span className={cn(
                        "text-[11px] font-mono min-w-[180px] shrink-0",
                        isChecked ? "text-primary" : "text-muted-foreground"
                      )}>
                        {p.filename}
                      </span>
                      <span className="text-[12px] truncate text-muted-foreground">{p.prompt}</span>
                    </label>
                  )
                })}
              </div>
            </ScrollArea>
          )}
        </CardContent>
      </Card>

      {/* Results */}
      {results && (
        <Card className="shadow-premium">
          <CardHeader className="p-4 pb-0 flex flex-row items-center justify-between">
            <CardTitle className="text-sm font-semibold">Replay Results</CardTitle>
            <Button variant="outline" size="sm" onClick={copyReport}>
              <Copy className="h-3.5 w-3.5 mr-1.5" />
              Copy Report
            </Button>
          </CardHeader>
          <CardContent className="p-4 space-y-4">
            {/* Summary stats */}
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
              <MetricCard label="Total" value={results.length} />
              <MetricCard label="Passed" value={okCount} variant="success" />
              <MetricCard label="Failed" value={failCount} variant={failCount ? "danger" : "success"} />
              <MetricCard label="Time" value={`${totalTime.toFixed(0)}s`} icon={<Clock className="h-3 w-3" />} />
              <MetricCard label="API Calls" value={totalApi} icon={<Zap className="h-3 w-3" />} />
              <MetricCard label="API Errors" value={totalApiErr} variant={totalApiErr ? "danger" : "success"} />
            </div>

            <Separator />

            {/* Per-payload results */}
            <div className="space-y-2">
              {results.map((r, i) => (
                <ReplayResultCard key={r.filename} result={r} index={i} />
              ))}
            </div>
          </CardContent>
        </Card>
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
  variant?: "success" | "danger" | "warning"
  icon?: React.ReactNode
}) {
  const colorClass =
    variant === "success" ? "text-emerald-600" :
    variant === "danger" ? "text-red-600" :
    variant === "warning" ? "text-amber-600" :
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

function ReplayResultCard({ result: r, index }: { result: ReplayResult; index: number }) {
  const [expanded, setExpanded] = useState(false)
  const isFail = r.status === "FAIL"

  return (
    <Card
      className={cn(
        "shadow-sm transition-all duration-200 cursor-pointer hover:shadow-md overflow-hidden",
        isFail ? "border-l-4 border-l-red-500" : "border-l-4 border-l-emerald-500"
      )}
      style={{ animationDelay: `${index * 40}ms` }}
      onClick={() => setExpanded(!expanded)}
    >
      <CardContent className="p-3">
        <div className="flex items-center gap-2 mb-1.5">
          {isFail ? (
            <div className="h-6 w-6 rounded-full bg-red-50 flex items-center justify-center shrink-0">
              <XCircle className="h-3.5 w-3.5 text-red-500" />
            </div>
          ) : (
            <div className="h-6 w-6 rounded-full bg-emerald-50 flex items-center justify-center shrink-0">
              <CheckCircle className="h-3.5 w-3.5 text-emerald-500" />
            </div>
          )}
          <span className="font-medium text-[13px]">{r.filename}</span>
          <Badge variant="secondary" className="text-[10px] ml-auto tabular-nums">
            <Clock className="h-2.5 w-2.5 mr-1" />
            {r.elapsed?.toFixed(1) || "?"}s
          </Badge>
          <Badge variant="secondary" className="text-[10px] tabular-nums">
            <Zap className="h-2.5 w-2.5 mr-1" />
            {r.api_calls || 0} API
          </Badge>
        </div>
        <p className="text-[12px] text-muted-foreground truncate pl-8">{r.prompt}</p>

        {/* Tool calls timeline */}
        {expanded && r.tool_calls?.length > 0 && (
          <div className="mt-3 ml-8 space-y-1 border-l-2 border-border/50 pl-3">
            {r.tool_calls.map((tc: ToolCall, j: number) => {
              const ok = tc.result?.ok
              return (
                <div
                  key={j}
                  className={cn(
                    "flex items-center gap-2 text-[12px] font-mono py-0.5",
                    ok === false ? "text-red-600" : "text-emerald-700"
                  )}
                >
                  <ArrowRight className="h-2.5 w-2.5 shrink-0 opacity-40" />
                  {ok === true ? (
                    <CheckCircle className="h-3 w-3 text-emerald-500 shrink-0" />
                  ) : ok === false ? (
                    <XCircle className="h-3 w-3 text-red-500 shrink-0" />
                  ) : (
                    <AlertTriangle className="h-3 w-3 text-amber-500 shrink-0" />
                  )}
                  <code className="truncate">{tc.tool}</code>
                  {ok === false && tc.result?.error && (
                    <span className="text-red-400 text-[11px] truncate ml-1">
                      {tc.result.error}
                    </span>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
