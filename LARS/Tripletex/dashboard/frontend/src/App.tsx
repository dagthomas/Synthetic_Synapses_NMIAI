import { useState, useCallback, useMemo, useEffect } from "react"
import { SWRConfig } from "swr"
import { Toaster } from "@/components/ui/sonner"
import { TooltipProvider } from "@/components/ui/tooltip"
import { Sidebar } from "@/components/layout/sidebar"
import type { PanelId, TabId } from "@/components/layout/sidebar"
import { RunPanel } from "@/components/panels/run-panel"
import { SandboxPanel } from "@/components/panels/sandbox-panel"
import { ReplayPanel } from "@/components/panels/replay-panel"
import { ToolExplorerPanel } from "@/components/panels/tool-explorer-panel"
import { ReportPanel } from "@/components/panels/report-panel"
import { ErrorsPanel } from "@/components/panels/errors-panel"
import { ResultsPanel } from "@/components/panels/results-panel"
import { CoveragePanel } from "@/components/panels/coverage-panel"
import { LogsPanel } from "@/components/panels/logs-panel"
import { TasksPanel } from "@/components/panels/tasks-panel"
import { AutoFixPanel } from "@/components/panels/auto-fix-panel"
import { AutoTestPanel } from "@/components/panels/auto-test-panel"
import { ScoreOverviewPanel } from "@/components/panels/score-overview-panel"
import { useRuns } from "@/hooks/use-api"
import type { EvalRun } from "@/types/api"
import { ScrollArea } from "@/components/ui/scroll-area"

function Dashboard() {
  const [activePanel, setActivePanel] = useState<PanelId>("run")
  const [activeTab, setActiveTab] = useState<TabId>("eval")
  const [batchRuns, setBatchRuns] = useState<EvalRun[]>([])
  const [connected, setConnected] = useState(false)
  const [preSelectedTasks, setPreSelectedTasks] = useState<string[]>([])


  // Separate error counts per source
  const { data: evalRuns } = useRuns("all", "simulator", 60_000)
  const { data: liveRuns } = useRuns("all", "competition", 60_000)

  const evalErrorCount = useMemo(() => {
    if (!evalRuns) return 0
    return evalRuns.filter(
      (r) =>
        r.status === "failed" ||
        (r.status === "completed" && r.correctness != null && r.correctness < 1)
    ).length
  }, [evalRuns])

  const liveErrorCount = useMemo(() => {
    if (!liveRuns) return 0
    return liveRuns.filter(
      (r) =>
        r.status === "failed" ||
        (r.status === "completed" && r.correctness != null && r.correctness < 1)
    ).length
  }, [liveRuns])

  const errorCount = activeTab === "live" ? liveErrorCount : evalErrorCount

  // Check agent connectivity
  useEffect(() => {
    const check = () => {
      fetch("http://localhost:8005/docs")
        .then((r) => setConnected(r.ok))
        .catch(() => setConnected(false))
    }
    check()
    const id = setInterval(check, 10000)
    return () => clearInterval(id)
  }, [])

  const handleBatchDone = useCallback((runs: EvalRun[]) => {
    setBatchRuns(runs)
  }, [])

  const handleNavigate = useCallback((panel: PanelId | string) => {
    setActivePanel(panel as PanelId)
  }, [])

  const handleTabChange = useCallback((tab: TabId) => {
    setActiveTab(tab)
    // Switch to first panel of the new tab
    setActivePanel(tab === "live" ? "logs" : "run")
  }, [])

  const handleRunEvalFromTasks = useCallback((taskNames: string[]) => {
    setPreSelectedTasks(taskNames)
    setActiveTab("eval")
    setActivePanel("run")
  }, [])

  // Determine source filter for errors panel based on active tab
  const errorsSource = activeTab === "live" ? "competition" : "simulator"

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar
        activePanel={activePanel}
        onNavigate={setActivePanel}
        errorCount={errorCount}
        connected={connected}
        activeTab={activeTab}
        onTabChange={handleTabChange}
      />
      <ScrollArea className="flex-1 h-screen">
        <main className="p-6 max-w-[980px]">
          <div className="animate-fade-in-up">
            {activePanel === "run" && (
              <RunPanel
                onBatchDone={handleBatchDone}
                onNavigate={handleNavigate}
                preSelectedTasks={preSelectedTasks}
                onPreSelectedConsumed={() => setPreSelectedTasks([])}
              />
            )}
            {activePanel === "sandbox" && <SandboxPanel />}
            {activePanel === "replay" && <ReplayPanel />}
            {activePanel === "explorer" && <ToolExplorerPanel />}
            {activePanel === "autofix" && <AutoFixPanel />}
            {activePanel === "autotest" && <AutoTestPanel />}
            {activePanel === "tasks" && <TasksPanel onRunEval={handleRunEvalFromTasks} />}
            {activePanel === "logs" && <LogsPanel />}
            {activePanel === "report" && <ReportPanel batchRuns={batchRuns} />}
            {activePanel === "errors" && <ErrorsPanel source={errorsSource} />}
            {activePanel === "scores" && <ScoreOverviewPanel />}
            {activePanel === "coverage" && <CoveragePanel />}
            {activePanel === "results" && (
              <ResultsPanel defaultSource={activeTab === "eval" ? "simulator" : "all"} />
            )}
          </div>
        </main>
      </ScrollArea>
    </div>
  )
}

function App() {
  return (
    <SWRConfig value={{ revalidateOnFocus: false }}>
      <TooltipProvider>
        <Dashboard />
        <Toaster position="bottom-right" richColors closeButton />
      </TooltipProvider>
    </SWRConfig>
  )
}

export default App
