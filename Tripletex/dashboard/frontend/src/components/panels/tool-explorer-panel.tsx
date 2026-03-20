import { useState, useRef, useEffect, useMemo } from "react"
import useSWR from "swr"
import type { ToolInfo, ToolTestResult } from "@/types/api"
import { fetchToolCatalog } from "@/lib/api"
import { PageHeader } from "@/components/layout/page-header"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import { toast } from "sonner"
import { cn } from "@/lib/utils"
import {
  Play,
  Copy,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Square,
  Clock,
  Activity,
  ArrowRight,
  Search,
  ChevronDown,
  ChevronRight,
  Package,
  Code2,
  Braces,
  FlaskConical,
} from "lucide-react"

// ── Tool Catalog Section ────────────────────────────────────────

function ToolCatalog({
  tools,
  testResults,
}: {
  tools: ToolInfo[]
  testResults: Map<string, ToolTestResult>
}) {
  const [search, setSearch] = useState("")
  const [expandedModule, setExpandedModule] = useState<string | null>(null)
  const [expandedTool, setExpandedTool] = useState<string | null>(null)

  const filtered = useMemo(() => {
    if (!search) return tools
    const q = search.toLowerCase()
    return tools.filter(
      (t) =>
        t.name.toLowerCase().includes(q) ||
        t.module.toLowerCase().includes(q) ||
        t.summary.toLowerCase().includes(q)
    )
  }, [tools, search])

  // Group by module
  const grouped = useMemo(() => {
    const map = new Map<string, ToolInfo[]>()
    for (const t of filtered) {
      if (!map.has(t.module)) map.set(t.module, [])
      map.get(t.module)!.push(t)
    }
    return map
  }, [filtered])

  return (
    <div>
      {/* Search bar */}
      <div className="relative mb-4">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search tools by name, module, or description..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="pl-10 h-9"
        />
        <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[11px] text-muted-foreground tabular-nums">
          {filtered.length} tool{filtered.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Module groups */}
      <div className="space-y-1.5">
        {Array.from(grouped.entries()).map(([module, moduleTools]) => {
          const isExpanded = expandedModule === module || search.length > 0
          const passCount = moduleTools.filter(
            (t) => testResults.get(t.name)?.status === "OK"
          ).length
          const failCount = moduleTools.filter(
            (t) =>
              testResults.get(t.name)?.status === "FAIL" ||
              testResults.get(t.name)?.status === "EXCEPTION"
          ).length
          const testedCount = passCount + failCount

          return (
            <Card key={module} className="shadow-sm">
              <button
                className="w-full p-3 flex items-center gap-2.5 hover:bg-muted/30 transition-colors rounded-t-lg"
                onClick={() =>
                  setExpandedModule(isExpanded && !search ? null : module)
                }
              >
                {isExpanded ? (
                  <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0" />
                )}
                <Package className="h-4 w-4 text-blue-500 shrink-0" />
                <span className="text-[13px] font-semibold">{module}</span>
                <Badge variant="secondary" className="text-[10px] ml-1">
                  {moduleTools.length} tool{moduleTools.length !== 1 ? "s" : ""}
                </Badge>
                {testedCount > 0 && (
                  <div className="ml-auto flex items-center gap-1.5">
                    {passCount > 0 && (
                      <Badge className="text-[10px] bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-50">
                        <CheckCircle className="h-2.5 w-2.5 mr-0.5" />
                        {passCount}
                      </Badge>
                    )}
                    {failCount > 0 && (
                      <Badge className="text-[10px] bg-red-50 text-red-700 border-red-200 hover:bg-red-50">
                        <XCircle className="h-2.5 w-2.5 mr-0.5" />
                        {failCount}
                      </Badge>
                    )}
                  </div>
                )}
              </button>
              {isExpanded && (
                <CardContent className="pt-0 px-3 pb-3">
                  <div className="space-y-1">
                    {moduleTools.map((tool) => {
                      const testResult = testResults.get(tool.name)
                      const isToolExpanded = expandedTool === tool.name
                      return (
                        <div
                          key={tool.name}
                          className={cn(
                            "border rounded-lg transition-all duration-150",
                            testResult?.status === "OK"
                              ? "border-l-4 border-l-emerald-400"
                              : testResult?.status === "FAIL"
                              ? "border-l-4 border-l-red-400"
                              : testResult?.status === "EXCEPTION"
                              ? "border-l-4 border-l-amber-400"
                              : "border-l-4 border-l-transparent"
                          )}
                        >
                          <button
                            className="w-full p-2.5 flex items-center gap-2 hover:bg-muted/20 transition-colors text-left rounded-lg"
                            onClick={() =>
                              setExpandedTool(isToolExpanded ? null : tool.name)
                            }
                          >
                            <Code2 className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                            <span className="text-[13px] font-medium font-mono">
                              {tool.name}
                            </span>
                            <span className="text-[11px] text-muted-foreground truncate ml-1 flex-1">
                              {tool.summary}
                            </span>
                            <div className="flex items-center gap-1.5 shrink-0">
                              {tool.params.length > 0 && (
                                <Badge
                                  variant="outline"
                                  className="text-[10px] tabular-nums"
                                >
                                  <Braces className="h-2.5 w-2.5 mr-0.5" />
                                  {tool.params.filter((p) => p.required).length}
                                  /{tool.params.length}
                                </Badge>
                              )}
                              {testResult && (
                                <Badge
                                  className={cn(
                                    "text-[10px] tabular-nums",
                                    testResult.status === "OK"
                                      ? "bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-50"
                                      : testResult.status === "FAIL"
                                      ? "bg-red-50 text-red-700 border-red-200 hover:bg-red-50"
                                      : "bg-amber-50 text-amber-700 border-amber-200 hover:bg-amber-50"
                                  )}
                                >
                                  {testResult.status}
                                  {testResult.elapsed && (
                                    <span className="ml-1">
                                      {testResult.elapsed}s
                                    </span>
                                  )}
                                </Badge>
                              )}
                            </div>
                          </button>
                          {isToolExpanded && (
                            <div className="px-3 pb-3 space-y-2.5">
                              <Separator />
                              {/* Docstring */}
                              {tool.docstring && (
                                <pre className="text-[11px] text-muted-foreground whitespace-pre-wrap font-mono bg-muted/30 rounded p-2.5 leading-relaxed">
                                  {tool.docstring}
                                </pre>
                              )}
                              {/* Parameters */}
                              {tool.params.length > 0 && (
                                <div>
                                  <p className="text-[11px] font-semibold text-muted-foreground mb-1.5 uppercase tracking-wider">
                                    Parameters
                                  </p>
                                  <div className="rounded-lg border overflow-hidden">
                                    <table className="w-full text-[12px]">
                                      <thead>
                                        <tr className="bg-muted/50 text-[10px] uppercase tracking-wider text-muted-foreground">
                                          <th className="text-left p-2 font-semibold">
                                            Name
                                          </th>
                                          <th className="text-left p-2 font-semibold">
                                            Type
                                          </th>
                                          <th className="text-left p-2 font-semibold">
                                            Required
                                          </th>
                                          <th className="text-left p-2 font-semibold">
                                            Default
                                          </th>
                                        </tr>
                                      </thead>
                                      <tbody>
                                        {tool.params.map((p) => (
                                          <tr
                                            key={p.name}
                                            className="border-t hover:bg-muted/20"
                                          >
                                            <td className="p-2 font-mono font-medium">
                                              {p.name}
                                            </td>
                                            <td className="p-2 text-muted-foreground font-mono">
                                              {p.type || "any"}
                                            </td>
                                            <td className="p-2">
                                              {p.required ? (
                                                <Badge className="text-[9px] bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-50">
                                                  required
                                                </Badge>
                                              ) : (
                                                <span className="text-muted-foreground/50">
                                                  optional
                                                </span>
                                              )}
                                            </td>
                                            <td className="p-2 text-muted-foreground font-mono">
                                              {p.default ?? "—"}
                                            </td>
                                          </tr>
                                        ))}
                                      </tbody>
                                    </table>
                                  </div>
                                </div>
                              )}
                              {/* Test error details */}
                              {testResult?.error && (
                                <pre className="text-[11px] text-red-600 whitespace-pre-wrap font-mono bg-red-50/50 rounded p-2">
                                  {testResult.error}
                                </pre>
                              )}
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </CardContent>
              )}
            </Card>
          )
        })}
      </div>
    </div>
  )
}

// ── Test Runner Section (adapted from tools-panel) ──────────────

const ESTIMATED_TOTAL = 43

function generateToolTestReport(results: ToolTestResult[]): string {
  const ok = results.filter((r) => r.status === "OK").length
  const total = results.length
  const totalTime = results.reduce((s: number, r) => s + r.elapsed, 0).toFixed(1)
  let out = `# Tool Test Report\n\n`
  out += `**${ok}/${total} passed** | ${totalTime}s\n\n`
  out += `| # | Tool | Status | Time | Error |\n`
  out += `|---|------|--------|------|-------|\n`
  results.forEach((r, i) => {
    const err = r.error ? r.error.substring(0, 100) : ""
    out += `| ${i + 1} | ${r.tool} | ${r.status} | ${r.elapsed}s | ${err} |\n`
  })
  return out
}

function TestRunner({
  onResults,
}: {
  onResults: (results: ToolTestResult[]) => void
}) {
  const [running, setRunning] = useState(false)
  const [results, setResults] = useState<ToolTestResult[]>([])
  const [done, setDone] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom as results arrive
  useEffect(() => {
    if (running && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" })
    }
  }, [results.length, running])

  // Propagate results up
  useEffect(() => {
    if (done && results.length > 0) {
      onResults(results)
    }
  }, [done]) // eslint-disable-line react-hooks/exhaustive-deps

  async function handleRun() {
    setRunning(true)
    setResults([])
    setDone(false)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const resp = await fetch("/api/test-tools/stream", {
        signal: controller.signal,
      })
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({ error: resp.statusText }))
        throw new Error(body.error || resp.statusText)
      }

      const reader = resp.body?.getReader()
      if (!reader) throw new Error("No readable stream")

      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { done: streamDone, value } = await reader.read()
        if (streamDone) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue
          const payload = line.slice(6).trim()
          if (payload === "[DONE]") {
            setDone(true)
            setRunning(false)
            return
          }
          try {
            const result: ToolTestResult = JSON.parse(payload)
            setResults((prev) => [...prev, result])
          } catch {
            // skip malformed
          }
        }
      }

      setDone(true)
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        toast.error((err as Error).message)
      }
    } finally {
      setRunning(false)
      abortRef.current = null
    }
  }

  function handleStop() {
    abortRef.current?.abort()
    setRunning(false)
    setDone(true)
  }

  function copyReport() {
    if (!results.length) return
    navigator.clipboard.writeText(generateToolTestReport(results))
    toast.success("Report copied")
  }

  const ok = results.filter((r) => r.status === "OK").length
  const fail = results.filter((r) => r.status === "FAIL").length
  const exception = results.filter((r) => r.status === "EXCEPTION").length
  const totalTime = results.reduce((s: number, r) => s + r.elapsed, 0)
  const progress = Math.min((results.length / ESTIMATED_TOTAL) * 100, 100)

  // Notify when complete
  useEffect(() => {
    if (done && results.length > 0) {
      toast.success(`Tool tests done: ${ok}/${results.length} passed`)
    }
  }, [done]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div>
      <div className="flex items-center gap-3 mb-4">
        {!running ? (
          <Button
            onClick={handleRun}
            size="sm"
            className="font-semibold bg-gradient-to-r from-primary to-blue-600 hover:shadow-lg hover:shadow-primary/20"
          >
            <Play className="h-3.5 w-3.5 mr-1.5" />
            Run All Tool Tests
          </Button>
        ) : (
          <Button onClick={handleStop} size="sm" variant="destructive">
            <Square className="h-3.5 w-3.5 mr-1.5" />
            Stop
          </Button>
        )}
        {running && (
          <div className="flex items-center gap-2">
            <Activity className="h-3.5 w-3.5 text-primary animate-pulse" />
            <span className="text-xs text-muted-foreground tabular-nums font-medium">
              {results.length}/{ESTIMATED_TOTAL} tests...
            </span>
          </div>
        )}
        {results.length > 0 && !running && (
          <Button variant="outline" size="sm" onClick={copyReport}>
            <Copy className="h-3.5 w-3.5 mr-1.5" />
            Copy Report
          </Button>
        )}
      </div>

      {/* Progress bar while running */}
      {running && (
        <div className="relative mb-4">
          <Progress value={progress} className="h-2" />
          <div className="absolute inset-0 rounded-full overflow-hidden">
            <div className="h-full shimmer" />
          </div>
        </div>
      )}

      {/* Summary stats */}
      {results.length > 0 && (
        <div className="grid grid-cols-5 gap-2 mb-4">
          <StatCard
            label="Total"
            value={results.length}
            icon={<Activity className="h-3 w-3" />}
          />
          <StatCard
            label="Passed"
            value={ok}
            variant="success"
            icon={<CheckCircle className="h-3 w-3" />}
          />
          <StatCard
            label="Failed"
            value={fail}
            variant={fail ? "danger" : "success"}
            icon={<XCircle className="h-3 w-3" />}
          />
          <StatCard
            label="Exception"
            value={exception}
            variant={exception ? "danger" : "success"}
            icon={<AlertTriangle className="h-3 w-3" />}
          />
          <StatCard
            label="Time"
            value={`${totalTime.toFixed(1)}s`}
            icon={<Clock className="h-3 w-3" />}
          />
        </div>
      )}

      {/* Live results log */}
      {results.length > 0 && (
        <Card className="shadow-premium">
          <CardHeader className="p-4 pb-0">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              Test Results
              {running && (
                <span className="flex items-center gap-1.5 text-primary">
                  <span className="inline-block h-2 w-2 rounded-full bg-primary animate-pulse" />
                  <span className="text-[11px] font-normal text-muted-foreground">
                    streaming...
                  </span>
                </span>
              )}
              {done && !running && (
                <Badge
                  variant="secondary"
                  className={cn(
                    "text-[10px] font-semibold",
                    fail === 0 && exception === 0
                      ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                      : "bg-red-50 text-red-700 border-red-200"
                  )}
                >
                  {fail === 0 && exception === 0
                    ? "All Passed"
                    : `${fail + exception} Issues`}
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4">
            <div className="space-y-1.5">
              {results.map((r, i) => (
                <ToolResultCard key={`${r.tool}-${i}`} result={r} index={i + 1} />
              ))}
            </div>
            <div ref={bottomRef} />
          </CardContent>
        </Card>
      )}
    </div>
  )
}

// ── Shared Components ────────────────────────────────────────────

function StatCard({
  label,
  value,
  variant,
  icon,
}: {
  label: string
  value: string | number
  variant?: "success" | "danger"
  icon?: React.ReactNode
}) {
  const colorClass =
    variant === "success"
      ? "text-emerald-600"
      : variant === "danger"
      ? "text-red-600"
      : "text-foreground"

  return (
    <div className="text-center border rounded-lg py-2.5 bg-card metric-card shadow-sm">
      <div
        className={cn(
          "text-lg font-bold tabular-nums flex items-center justify-center gap-1",
          colorClass
        )}
      >
        {icon}
        {value}
      </div>
      <div className="text-[10px] text-muted-foreground font-medium mt-0.5">
        {label}
      </div>
    </div>
  )
}

function ToolResultCard({
  result: r,
  index,
}: {
  result: ToolTestResult
  index: number
}) {
  const isOk = r.status === "OK"
  const [expanded, setExpanded] = useState(false)

  return (
    <div
      className={cn(
        "border rounded-lg p-3 transition-all duration-200 cursor-pointer hover:shadow-sm animate-fade-in-up",
        isOk
          ? "border-l-4 border-l-emerald-500 hover:bg-emerald-50/30"
          : r.status === "FAIL"
          ? "border-l-4 border-l-red-500 hover:bg-red-50/30"
          : "border-l-4 border-l-amber-500 hover:bg-amber-50/30"
      )}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center gap-2">
        <span className="text-[10px] text-muted-foreground tabular-nums w-5 text-right font-mono">
          {index}
        </span>
        <ArrowRight className="h-3 w-3 text-muted-foreground/30 shrink-0" />
        {isOk ? (
          <CheckCircle className="h-3.5 w-3.5 text-emerald-500 shrink-0" />
        ) : r.status === "FAIL" ? (
          <XCircle className="h-3.5 w-3.5 text-red-500 shrink-0" />
        ) : (
          <AlertTriangle className="h-3.5 w-3.5 text-amber-500 shrink-0" />
        )}
        <span className="text-[13px] font-medium">{r.tool}</span>
        <div className="ml-auto flex items-center gap-1.5">
          <Badge variant="secondary" className="text-[10px] tabular-nums font-mono">
            <Clock className="h-2.5 w-2.5 mr-0.5" />
            {r.elapsed}s
          </Badge>
          {r.status_code && (
            <Badge
              variant="secondary"
              className={cn(
                "text-[10px] tabular-nums",
                r.status_code >= 400
                  ? "bg-red-50 text-red-600"
                  : "bg-emerald-50 text-emerald-600"
              )}
            >
              {r.status_code}
            </Badge>
          )}
        </div>
      </div>
      {r.error && (
        <pre className="text-[11px] text-red-600 mt-2 whitespace-pre-wrap font-mono bg-red-50/50 rounded p-2 ml-[52px]">
          {r.error}
        </pre>
      )}
      {expanded && r.result_preview && (
        <pre className="text-[11px] text-muted-foreground mt-2 whitespace-pre-wrap font-mono max-h-[150px] overflow-y-auto bg-muted/30 rounded p-2 ml-[52px]">
          {r.result_preview}
        </pre>
      )}
    </div>
  )
}

// ── Main Panel ──────────────────────────────────────────────────

type TabId = "catalog" | "test"

export function ToolExplorerPanel() {
  const [activeTab, setActiveTab] = useState<TabId>("catalog")
  const [testResults, setTestResults] = useState<Map<string, ToolTestResult>>(
    new Map()
  )
  const { data: tools } = useSWR("tool-catalog", fetchToolCatalog)

  const handleTestResults = (results: ToolTestResult[]) => {
    const map = new Map<string, ToolTestResult>()
    for (const r of results) {
      // Strip suffix like " (2)" or " (credit)" for matching to catalog
      const baseName = r.tool.replace(/\s*\(.*\)$/, "")
      // Keep the first (or best) result per base tool name
      if (!map.has(baseName) || r.status !== "OK") {
        map.set(baseName, r)
      }
    }
    setTestResults(map)
  }

  const toolCount = tools?.length ?? 0
  const moduleCount = new Set(tools?.map((t) => t.module) ?? []).size

  return (
    <div>
      <PageHeader
        title="Tool Explorer"
        description={`Browse ${toolCount} tools across ${moduleCount} modules, inspect parameters, and run integration tests`}
      />

      {/* Sub-tabs */}
      <div className="flex items-center gap-1 mb-5 border-b">
        <button
          onClick={() => setActiveTab("catalog")}
          className={cn(
            "px-4 py-2 text-[13px] font-medium border-b-2 -mb-px transition-colors",
            activeTab === "catalog"
              ? "border-primary text-primary"
              : "border-transparent text-muted-foreground hover:text-foreground"
          )}
        >
          <Package className="h-3.5 w-3.5 inline-block mr-1.5 -mt-0.5" />
          Catalog
          <Badge variant="secondary" className="ml-2 text-[10px]">
            {toolCount}
          </Badge>
        </button>
        <button
          onClick={() => setActiveTab("test")}
          className={cn(
            "px-4 py-2 text-[13px] font-medium border-b-2 -mb-px transition-colors",
            activeTab === "test"
              ? "border-primary text-primary"
              : "border-transparent text-muted-foreground hover:text-foreground"
          )}
        >
          <FlaskConical className="h-3.5 w-3.5 inline-block mr-1.5 -mt-0.5" />
          Test Runner
          {testResults.size > 0 && (
            <Badge
              className={cn(
                "ml-2 text-[10px]",
                Array.from(testResults.values()).every((r) => r.status === "OK")
                  ? "bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-50"
                  : "bg-red-50 text-red-700 border-red-200 hover:bg-red-50"
              )}
            >
              {
                Array.from(testResults.values()).filter(
                  (r) => r.status === "OK"
                ).length
              }
              /{testResults.size}
            </Badge>
          )}
        </button>
      </div>

      {/* Tab content */}
      <div className="animate-fade-in-up">
        {activeTab === "catalog" && (
          <ToolCatalog tools={tools ?? []} testResults={testResults} />
        )}
        {activeTab === "test" && (
          <TestRunner onResults={handleTestResults} />
        )}
      </div>
    </div>
  )
}
