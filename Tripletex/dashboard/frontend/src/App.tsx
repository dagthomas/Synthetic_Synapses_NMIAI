import { useState, useCallback, useMemo, useEffect } from "react"
import { SWRConfig } from "swr"
import { Toaster } from "@/components/ui/sonner"
import { TooltipProvider } from "@/components/ui/tooltip"
import { Sidebar } from "@/components/layout/sidebar"
import type { PanelId } from "@/components/layout/sidebar"
import { RunPanel } from "@/components/panels/run-panel"
import { SandboxPanel } from "@/components/panels/sandbox-panel"
import { ReplayPanel } from "@/components/panels/replay-panel"
import { ToolsPanel } from "@/components/panels/tools-panel"
import { ReportPanel } from "@/components/panels/report-panel"
import { ErrorsPanel } from "@/components/panels/errors-panel"
import { ResultsPanel } from "@/components/panels/results-panel"
import { CoveragePanel } from "@/components/panels/coverage-panel"
import { useRuns } from "@/hooks/use-api"
import type { EvalRun } from "@/types/api"
import { ScrollArea } from "@/components/ui/scroll-area"

function Dashboard() {
  const [activePanel, setActivePanel] = useState<PanelId>("run")
  const [batchRuns, setBatchRuns] = useState<EvalRun[]>([])
  const [connected, setConnected] = useState(false)
  const { data: allRuns } = useRuns("all", 60_000)

  const errorCount = useMemo(() => {
    if (!allRuns) return 0
    return allRuns.filter(
      (r) =>
        r.status === "failed" ||
        (r.status === "completed" && r.correctness != null && r.correctness < 1)
    ).length
  }, [allRuns])

  // Check agent connectivity
  useEffect(() => {
    const check = () => {
      fetch("http://localhost:8000/docs")
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

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar
        activePanel={activePanel}
        onNavigate={setActivePanel}
        errorCount={errorCount}
        connected={connected}
      />
      <ScrollArea className="flex-1 h-screen">
        <main className="p-6 max-w-[980px]">
          <div className="animate-fade-in-up">
            {activePanel === "run" && (
              <RunPanel onBatchDone={handleBatchDone} onNavigate={handleNavigate} />
            )}
            {activePanel === "sandbox" && <SandboxPanel />}
            {activePanel === "replay" && <ReplayPanel />}
            {activePanel === "tools" && <ToolsPanel />}
            {activePanel === "report" && <ReportPanel batchRuns={batchRuns} />}
            {activePanel === "errors" && <ErrorsPanel />}
            {activePanel === "coverage" && <CoveragePanel />}
            {activePanel === "results" && <ResultsPanel />}
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
