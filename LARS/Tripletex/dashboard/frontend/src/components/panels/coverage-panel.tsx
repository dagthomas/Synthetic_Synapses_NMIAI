import { useState, useMemo } from "react"
import { useCoverage } from "@/hooks/use-api"
import type { CoverageCategory } from "@/types/api"
import { PageHeader } from "@/components/layout/page-header"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { toast } from "sonner"
import { cn } from "@/lib/utils"
import {
  Copy,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Layers,
  Wrench,
  Shield,
  ChevronDown,
  ChevronRight,
  BarChart3,
} from "lucide-react"

export function CoveragePanel() {
  const { data, isLoading } = useCoverage()

  const stats = useMemo(() => {
    if (!data) return null
    const totalEndpoints = data.reduce((s, c) => s + c.endpoint_count, 0)
    const totalCovered = data.reduce((s, c) => s + c.covered.length, 0)
    const totalUncovered = data.reduce((s, c) => s + c.uncovered.length, 0)
    const totalTools = data.reduce((s, c) => s + c.tools.length, 0)
    const coveredCats = data.filter((c) => c.tools.length > 0).length
    const uncoveredCats = data.filter((c) => c.tools.length === 0).length
    const pct = totalEndpoints > 0 ? (totalCovered / totalEndpoints) * 100 : 0
    return { totalEndpoints, totalCovered, totalUncovered, totalTools, coveredCats, uncoveredCats, pct }
  }, [data])

  const sorted = useMemo(() => {
    if (!data) return []
    return [...data].sort((a, b) => {
      const aHas = a.tools.length > 0 ? 1 : 0
      const bHas = b.tools.length > 0 ? 1 : 0
      if (aHas !== bHas) return bHas - aHas
      if (aHas) {
        const aR = a.covered.length / Math.max(1, a.endpoint_count)
        const bR = b.covered.length / Math.max(1, b.endpoint_count)
        return bR - aR
      }
      return b.endpoint_count - a.endpoint_count
    })
  }, [data])

  function copyReport() {
    if (!data || !stats) return
    let out = `# API Coverage Report\n\n`
    out += `**${stats.totalCovered}/${stats.totalEndpoints}** endpoints covered (${stats.pct.toFixed(0)}%) | ${stats.totalTools} tools | ${stats.coveredCats}/${stats.coveredCats + stats.uncoveredCats} categories\n\n`
    for (const cat of sorted) {
      const pct = cat.endpoint_count > 0 ? ((cat.covered.length / cat.endpoint_count) * 100).toFixed(0) : "0"
      out += `## ${cat.category} — ${pct}% (${cat.covered.length}/${cat.endpoint_count})\n`
      if (cat.tools.length) out += `Tools: ${cat.tools.join(", ")}\n`
      if (cat.uncovered.length) {
        out += `Uncovered:\n`
        cat.uncovered.forEach((e) => (out += `  - ${e}\n`))
      }
      out += `\n`
    }
    navigator.clipboard.writeText(out)
    toast.success("Coverage report copied")
  }

  return (
    <div>
      <PageHeader
        title="API Coverage"
        description="Tripletex API endpoints vs implemented agent tools"
      >
        {data && (
          <Button variant="outline" size="sm" onClick={copyReport}>
            <Copy className="h-3.5 w-3.5 mr-1.5" />
            Copy Report
          </Button>
        )}
      </PageHeader>

      {isLoading && (
        <div className="space-y-3">
          <div className="grid grid-cols-4 gap-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-20" />
            ))}
          </div>
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
      )}

      {stats && (
        <>
          {/* Hero coverage percentage */}
          <Card className="mb-4 overflow-hidden">
            <CardContent className="p-4">
              <div className="flex items-center gap-4 mb-3">
                <div
                  className="h-10 w-10 rounded-xl flex items-center justify-center shrink-0"
                  style={{
                    background:
                      stats.pct > 50
                        ? "linear-gradient(135deg, hsl(142 76% 36%), hsl(142 76% 46%))"
                        : stats.pct > 25
                          ? "linear-gradient(135deg, hsl(38 92% 50%), hsl(45 93% 47%))"
                          : "linear-gradient(135deg, hsl(0 84% 60%), hsl(0 84% 70%))",
                    boxShadow:
                      stats.pct > 50
                        ? "0 2px 8px hsl(142 76% 36% / 0.3)"
                        : "0 2px 8px hsl(38 92% 50% / 0.3)",
                  }}
                >
                  <BarChart3 className="h-5 w-5 text-white" />
                </div>
                <div>
                  <div className="text-2xl font-bold tabular-nums">{stats.pct.toFixed(0)}%</div>
                  <div className="text-xs text-muted-foreground">
                    {stats.totalCovered} of {stats.totalEndpoints} endpoints covered
                  </div>
                </div>
              </div>
              <Progress
                value={stats.pct}
                className="h-2"
              />
            </CardContent>
          </Card>

          {/* Stats grid */}
          <div className="grid grid-cols-5 gap-2.5 mb-4">
            <StatCard icon={Layers} label="Endpoints" value={stats.totalEndpoints} />
            <StatCard icon={CheckCircle} label="Covered" value={stats.totalCovered} className="text-green-600" />
            <StatCard icon={XCircle} label="Uncovered" value={stats.totalUncovered} className={stats.totalUncovered ? "text-red-600" : "text-green-600"} />
            <StatCard icon={Wrench} label="Tools" value={stats.totalTools} />
            <StatCard icon={Shield} label="Categories" value={`${stats.coveredCats}/${stats.coveredCats + stats.uncoveredCats}`} />
          </div>
        </>
      )}

      {/* Category cards */}
      {sorted.length > 0 && (
        <div className="space-y-2">
          {sorted.map((cat) => (
            <CategoryCard key={cat.category} category={cat} />
          ))}
        </div>
      )}
    </div>
  )
}

function StatCard({
  icon: Icon,
  label,
  value,
  className,
}: {
  icon: React.ComponentType<{ className?: string }>
  label: string
  value: string | number
  className?: string
}) {
  return (
    <div className="border rounded-lg p-3 text-center metric-card">
      <Icon className={cn("h-4 w-4 mx-auto mb-1 text-muted-foreground", className)} />
      <div className={cn("text-lg font-bold tabular-nums", className)}>{value}</div>
      <div className="text-[10px] text-muted-foreground">{label}</div>
    </div>
  )
}

function CategoryCard({ category: cat }: { category: CoverageCategory }) {
  const [showCovered, setShowCovered] = useState(false)
  const [showUncovered, setShowUncovered] = useState(false)
  const hasTool = cat.tools.length > 0
  const pct = cat.endpoint_count > 0 ? (cat.covered.length / cat.endpoint_count) * 100 : hasTool ? 100 : 0

  return (
    <Card
      className={cn(
        "border-l-4 overflow-hidden",
        hasTool
          ? pct >= 50
            ? "border-l-green-500"
            : "border-l-amber-500"
          : "border-l-red-500"
      )}
    >
      <CardContent className="p-3">
        {/* Header */}
        <div className="flex items-center gap-2 mb-2">
          {hasTool ? (
            pct >= 50 ? (
              <CheckCircle className="h-4 w-4 text-green-500 shrink-0" />
            ) : (
              <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0" />
            )
          ) : (
            <XCircle className="h-4 w-4 text-red-500 shrink-0" />
          )}
          <span className="text-[13px] font-semibold">{cat.category}</span>
          <Badge variant="secondary" className="text-[10px]">
            {cat.endpoint_count} endpoints
          </Badge>
          <Badge
            variant="secondary"
            className={cn(
              "text-[10px]",
              hasTool ? "bg-green-50 text-green-700" : "bg-red-50 text-red-700"
            )}
          >
            {cat.tools.length} tools
          </Badge>
          {cat.endpoint_count > 0 && (
            <span className="ml-auto text-[12px] font-bold tabular-nums text-muted-foreground">
              {pct.toFixed(0)}%
            </span>
          )}
        </div>

        {/* Progress bar */}
        {cat.endpoint_count > 0 && (
          <Progress value={pct} className="h-1.5 mb-2" />
        )}

        {/* Tools */}
        {cat.tools.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-2">
            {cat.tools.map((t) => (
              <span
                key={t}
                className="inline-block bg-primary/10 text-primary px-2 py-0.5 rounded text-[11px] font-mono"
              >
                {t}
              </span>
            ))}
          </div>
        )}

        {/* Covered endpoints */}
        {cat.covered.length > 0 && (
          <div>
            <button
              onClick={() => setShowCovered(!showCovered)}
              className="flex items-center gap-1 text-[11px] font-semibold text-green-600 hover:text-green-700 transition-colors"
            >
              {showCovered ? (
                <ChevronDown className="h-3 w-3" />
              ) : (
                <ChevronRight className="h-3 w-3" />
              )}
              {cat.covered.length} covered endpoints
            </button>
            {showCovered && (
              <div className="mt-1 ml-4 space-y-0.5">
                {cat.covered.map((e) => (
                  <div key={e} className="flex items-center gap-1.5 text-[11px] font-mono text-green-700">
                    <CheckCircle className="h-3 w-3 shrink-0" />
                    {e}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Uncovered endpoints */}
        {cat.uncovered.length > 0 && (
          <div className="mt-1">
            <button
              onClick={() => setShowUncovered(!showUncovered)}
              className="flex items-center gap-1 text-[11px] font-semibold text-red-600 hover:text-red-700 transition-colors"
            >
              {showUncovered ? (
                <ChevronDown className="h-3 w-3" />
              ) : (
                <ChevronRight className="h-3 w-3" />
              )}
              {cat.uncovered.length} uncovered endpoints
            </button>
            {showUncovered && (
              <div className="mt-1 ml-4 space-y-0.5">
                {cat.uncovered.map((e) => (
                  <div key={e} className="flex items-center gap-1.5 text-[11px] font-mono text-red-600">
                    <XCircle className="h-3 w-3 shrink-0" />
                    {e}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
