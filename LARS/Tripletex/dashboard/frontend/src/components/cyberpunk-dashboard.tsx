import { useState, useEffect, useCallback, useMemo, useRef } from "react"
import { useScoreAutoPolling } from "@/hooks/use-api"
import { subscribeLiveEvents } from "@/lib/api"
import { mapTaskNumber, setScoreAuth, fetchScoreAuthStatus } from "@/lib/api"
import type { LiveEvent, ScoreTask } from "@/types/api"
import { cn } from "@/lib/utils"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { X, Save, Trash2, MessageSquare, Plus, Copy, Check, ArrowUp, ChevronDown, ChevronUp } from "lucide-react"

// ── localStorage for task metadata ─────────────────────────────

interface TaskPrompt { text: string; addedAt: string }
interface TaskMeta {
  name?: string
  testsPassed?: number
  totalTests?: number
  prompts?: TaskPrompt[]
}

const STORAGE_KEY = "tripletex_score_overview"
function loadMeta(): Record<number, TaskMeta> {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}") } catch { return {} }
}
function saveMeta(data: Record<number, TaskMeta>) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
}

// ── Helpers ─────────────────────────────────────────────────────

function pad(n: number) { return String(n).padStart(2, "0") }
function ts() { return new Date().toLocaleTimeString("en-GB", { hour12: false }) }

type FeedEntry = {
  id: number
  time: string
  type: "REQ" | "TASK" | "TOOL" | "OK" | "FAIL" | "DONE" | "INFO"
  text: string
}

const TYPE_COLORS: Record<string, string> = {
  REQ: "neon-text-cyan",
  TASK: "neon-text-amber",
  TOOL: "neon-text-magenta",
  OK: "neon-text-green",
  FAIL: "neon-text-red",
  DONE: "neon-text-green",
  INFO: "neon-text-amber",
}

// ── Main Component ──────────────────────────────────────────────

export function CyberpunkDashboard() {
  // State
  const [meta, setMeta] = useState<Record<number, TaskMeta>>(loadMeta)
  const [sseConnected, setSseConnected] = useState(false)
  const [agentConnected, setAgentConnected] = useState(false)
  const [feed, setFeed] = useState<FeedEntry[]>([])
  const [editingTask, setEditingTask] = useState<number | null>(null)
  const [modalView, setModalView] = useState<"edit" | "prompts">("edit")
  const [editName, setEditName] = useState("")
  const [editPassed, setEditPassed] = useState("")
  const [editTotal, setEditTotal] = useState("")
  const [newPrompt, setNewPrompt] = useState("")
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null)
  const [flashTasks, setFlashTasks] = useState<Set<number>>(new Set())
  const [pollEnabled, setPollEnabled] = useState(true)
  const [showAuthInput, setShowAuthInput] = useState(false)
  const [authInput, setAuthInput] = useState("")
  const [hasCookie, setHasCookie] = useState(false)
  const [feedCollapsed, setFeedCollapsed] = useState(false)
  const [changesCollapsed, setChangesCollapsed] = useState(false)
  const [priorityCollapsed, setPriorityCollapsed] = useState(false)
  const feedRef = useRef<HTMLDivElement>(null)
  const feedIdRef = useRef(0)

  // Score polling
  const { data: pollData, error: pollError, isLoading: pollLoading } = useScoreAutoPolling(pollEnabled)

  // Check auth on mount
  useEffect(() => {
    fetchScoreAuthStatus().then(r => setHasCookie(r.has_cookie)).catch(() => {})
  }, [])

  // Save meta to localStorage
  useEffect(() => { saveMeta(meta) }, [meta])

  // Flash changed tasks
  useEffect(() => {
    if (!pollData?.changes?.length) return
    const nums = new Set(pollData.changes.map(c => c.task_number))
    setFlashTasks(nums)
    const timer = setTimeout(() => setFlashTasks(new Set()), 2000)
    return () => clearTimeout(timer)
  }, [pollData?.changes])

  // Agent connectivity check (via backend proxy to avoid CORS)
  useEffect(() => {
    const check = () => {
      fetch("/api/agent/health")
        .then(r => r.json())
        .then(d => setAgentConnected(d.ok === true))
        .catch(() => setAgentConnected(false))
    }
    check()
    const id = setInterval(check, 10000)
    return () => clearInterval(id)
  }, [])

  // SSE live events
  useEffect(() => {
    const sub = subscribeLiveEvents(
      (ev: LiveEvent) => {
        const entry = liveEventToFeed(ev)
        if (!entry) return
        setFeed(prev => [entry, ...prev].slice(0, 50))
      },
      () => setSseConnected(true),
      () => setSseConnected(false),
    )
    return () => sub.close()
  }, [])

  const liveEventToFeed = useCallback((ev: LiveEvent): FeedEntry | null => {
    const id = ++feedIdRef.current
    const time = ts()
    switch (ev.type) {
      case "request_start":
        return { id, time, type: "REQ", text: `${ev.prompt.slice(0, 100)}` }
      case "classify":
        return { id, time, type: "TASK", text: `▸ ${ev.task_type}  ${ev.tool_count}c/${ev.total_tools || 0}t` }
      case "tool_call":
        return { id, time, type: "TOOL", text: `${ev.tool}(${Object.keys(ev.args).join(", ")})` }
      case "tool_result":
        return { id, time, type: ev.ok ? "OK" : "FAIL", text: `${ev.tool} → ${ev.ok ? "OK" : ev.error?.slice(0, 60)}` }
      case "request_done":
        return { id, time, type: "DONE", text: `${ev.task_type} done in ${ev.elapsed.toFixed(1)}s (${ev.api_calls} calls)` }
      case "request_error":
        return { id, time, type: "FAIL", text: ev.error.slice(0, 80) }
      default:
        return null
    }
  }, [])

  // Build task data (merge poll data with meta)
  const allTasks = useMemo(() => {
    const taskMap = new Map<number, ScoreTask>()
    if (pollData?.tasks) {
      for (const t of pollData.tasks) taskMap.set(t.task_number, t)
    }
    const tasks: Array<{
      num: number
      score: number
      checks_passed: number
      checks_total: number
      mapped: string | null
      meta: TaskMeta
    }> = []
    for (let i = 1; i <= 30; i++) {
      const st = taskMap.get(i)
      tasks.push({
        num: i,
        score: st?.score ?? 0,
        checks_passed: st?.checks_passed ?? 0,
        checks_total: st?.checks_total ?? 0,
        mapped: st?.mapped_task_type || pollData?.mappings?.[String(i)]?.task_type || null,
        meta: meta[i] || {},
      })
    }
    return tasks
  }, [pollData, meta])

  const totalScore = pollData?.total_score ?? allTasks.reduce((s, t) => s + t.score, 0)
  const rank = pollData?.rank
  const scored = allTasks.filter(t => t.score > 0).length

  // Priority sorting
  const priorities = useMemo(() => {
    return [...allTasks].sort((a, b) => {
      const pa = getPriority(a)
      const pb = getPriority(b)
      if (pa !== pb) return pa - pb
      return a.num - b.num
    })
  }, [allTasks])

  function getPriority(t: typeof allTasks[0]) {
    const m = t.meta
    const hasFailed = t.checks_total > 0 && t.checks_passed < t.checks_total
    if (t.score === 0 && t.checks_total > 0) return 1 // broken: scored 0 but has tries
    if (t.score === 0 && t.checks_total === 0) return 2 // never attempted
    if (hasFailed || (m.testsPassed !== undefined && m.totalTests !== undefined && m.testsPassed < m.totalTests)) return 3 // partial
    return 4 // all good
  }

  function priorityLabel(p: number) {
    switch (p) {
      case 1: return { label: "BROKEN", color: "neon-text-red" }
      case 2: return { label: "UNTRIED", color: "neon-text-cyan" }
      case 3: return { label: "PARTIAL", color: "neon-text-amber" }
      case 4: return { label: "PASSING", color: "neon-text-green" }
      default: return { label: "?", color: "" }
    }
  }

  // Card glow class
  function cardGlow(t: typeof allTasks[0]) {
    if (t.checks_total > 0 && t.checks_passed === t.checks_total && t.score > 0) return "cyber-glow-green border-[#00ff8833]"
    if (t.score > 0 && t.checks_total > 0 && t.checks_passed < t.checks_total) return "cyber-glow-amber border-[#ffaa0033]"
    if (t.score === 0 && t.checks_total > 0) return "cyber-glow-red border-[#ff224433]"
    if (t.score > 0) return "cyber-glow-green border-[#00ff8833]"
    return "border-[#2a2a3e]"
  }

  // Edit modal
  const openEdit = useCallback((num: number) => {
    const m = meta[num] || {}
    setEditName(m.name || "")
    setEditPassed(m.testsPassed !== undefined ? String(m.testsPassed) : "")
    setEditTotal(m.totalTests !== undefined ? String(m.totalTests) : "")
    setEditingTask(num)
    setModalView("edit")
    setNewPrompt("")
    setCopiedIdx(null)
  }, [meta])

  const saveEdit = useCallback(() => {
    if (editingTask === null) return
    const d: TaskMeta = {}
    if (editName.trim()) d.name = editName.trim()
    if (editPassed.trim()) d.testsPassed = parseInt(editPassed) || 0
    if (editTotal.trim()) d.totalTests = parseInt(editTotal) || 0
    setMeta(prev => {
      const next = { ...prev }
      const existing = prev[editingTask] || {}
      next[editingTask] = { ...existing, ...d }
      return next
    })
    setEditingTask(null)
  }, [editingTask, editName, editPassed, editTotal])

  const clearEdit = useCallback(() => {
    if (editingTask === null) return
    setMeta(prev => { const next = { ...prev }; delete next[editingTask]; return next })
    setEditingTask(null)
  }, [editingTask])

  const addPrompt = useCallback(() => {
    if (editingTask === null || !newPrompt.trim()) return
    setMeta(prev => {
      const existing = prev[editingTask] || {}
      const prompts = [...(existing.prompts || []), { text: newPrompt.trim(), addedAt: new Date().toISOString() }]
      return { ...prev, [editingTask]: { ...existing, prompts } }
    })
    setNewPrompt("")
  }, [editingTask, newPrompt])

  const deletePrompt = useCallback((idx: number) => {
    if (editingTask === null) return
    setMeta(prev => {
      const existing = prev[editingTask] || {}
      const prompts = (existing.prompts || []).filter((_, i) => i !== idx)
      return { ...prev, [editingTask]: { ...existing, prompts: prompts.length > 0 ? prompts : undefined } }
    })
  }, [editingTask])

  const copyPrompt = useCallback((text: string, idx: number) => {
    navigator.clipboard.writeText(text)
    setCopiedIdx(idx)
    setTimeout(() => setCopiedIdx(null), 1500)
  }, [])

  const saveAuth = useCallback(async () => {
    if (!authInput.trim()) return
    await setScoreAuth(authInput.trim())
    setHasCookie(true)
    setShowAuthInput(false)
    setAuthInput("")
  }, [authInput])

  // Map a task number to task type inline
  const [mappingTask, setMappingTask] = useState<number | null>(null)
  const [mappingType, setMappingType] = useState("")
  const saveMapping = useCallback(async () => {
    if (mappingTask === null || !mappingType.trim()) return
    await mapTaskNumber(mappingTask, mappingType.trim())
    setMappingTask(null)
    setMappingType("")
  }, [mappingTask, mappingType])

  return (
    <div className="min-h-screen bg-background cyber-grid-bg">
      {/* ─── HEADER BAR ─────────────────────────────────────────── */}
      <header className="sticky top-0 z-40 border-b border-[#2a2a3e] bg-[#0a0a0fdd] backdrop-blur-md">
        <div className="max-w-[1400px] mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-bold neon-text-cyan tracking-wider">
              SYNTHETIC SYNAPSES
            </h1>
          </div>

          <div className="flex items-center gap-6">
            {/* Score */}
            <div className="text-right">
              <div className="text-2xl font-bold neon-text-cyan tabular-nums animate-neon-pulse">
                {totalScore.toFixed(2)} pts
              </div>
              {rank && (
                <div className="text-xs text-muted-foreground">
                  Rank #{rank}
                </div>
              )}
            </div>

            {/* Status dots */}
            <div className="flex flex-col gap-1.5 text-xs">
              <div className="flex items-center gap-2">
                <span className={cn(
                  "w-2 h-2 rounded-full",
                  agentConnected ? "bg-[#00ff88]" : "bg-[#ff2244]"
                )} />
                <span className={agentConnected ? "neon-text-green" : "neon-text-red"}>
                  Agent
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className={cn(
                  "w-2 h-2 rounded-full",
                  sseConnected ? "bg-[#00ff88]" : "bg-[#ff2244]"
                )} />
                <span className={sseConnected ? "neon-text-green" : "neon-text-red"}>
                  SSE
                </span>
              </div>
            </div>

            {/* Stats */}
            <div className="text-xs text-muted-foreground">
              <div>{scored}/30 scored</div>
              <div className="flex items-center gap-1">
                {pollLoading && <span className="neon-text-cyan">polling...</span>}
                {pollError && <span className="neon-text-red">poll err</span>}
                {!pollLoading && !pollError && pollData && <span className="neon-text-green">live</span>}
              </div>
            </div>

            {/* Auth / Poll toggle */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPollEnabled(!pollEnabled)}
                className={cn(
                  "text-[10px] px-2 py-1 rounded border",
                  pollEnabled
                    ? "border-[#00ff8844] neon-text-green"
                    : "border-[#ff224444] neon-text-red"
                )}
              >
                {pollEnabled ? "POLL ON" : "POLL OFF"}
              </button>
              <button
                onClick={() => setShowAuthInput(!showAuthInput)}
                className={cn(
                  "text-[10px] px-2 py-1 rounded border",
                  hasCookie
                    ? "border-[#00ff8844] neon-text-green"
                    : "border-[#ffaa0044] neon-text-amber"
                )}
              >
                {hasCookie ? "AUTH OK" : "SET AUTH"}
              </button>
            </div>
          </div>
        </div>

        {/* Auth input row */}
        {showAuthInput && (
          <div className="max-w-[1400px] mx-auto px-4 pb-3 flex gap-2">
            <Input
              value={authInput}
              onChange={e => setAuthInput(e.target.value)}
              placeholder="Paste JWT or session cookie..."
              className="flex-1 bg-[#12121a] border-[#2a2a3e] text-xs"
              onKeyDown={e => { if (e.key === "Enter") saveAuth() }}
            />
            <Button size="sm" onClick={saveAuth} className="bg-[#00f0ff] text-[#0a0a0f] hover:bg-[#00d0dd] text-xs">
              Save
            </Button>
          </div>
        )}
      </header>

      <div className="max-w-[1400px] mx-auto px-4 py-6 space-y-6">

        {/* ─── SCORE GRID ────────────────────────────────────────── */}
        <section>
          <h2 className="text-sm font-bold neon-text-cyan mb-3 tracking-wider uppercase">
            Task Grid
          </h2>
          <div className="grid grid-cols-6 gap-2">
            {allTasks.map(t => {
              const promptCount = t.meta.prompts?.length || 0
              const hasTests = t.meta.testsPassed !== undefined && t.meta.totalTests !== undefined && (t.meta.totalTests ?? 0) > 0

              return (
                <div
                  key={t.num}
                  className={cn(
                    "cyber-card rounded cursor-pointer p-2.5 text-center relative",
                    cardGlow(t),
                    flashTasks.has(t.num) && "animate-magenta-flash"
                  )}
                  onClick={() => openEdit(t.num)}
                >
                  <div className="text-[10px] text-muted-foreground flex items-center justify-center gap-1">
                    <span>T{pad(t.num)}</span>
                    {promptCount > 0 && (
                      <span className="text-[8px] px-1 rounded bg-[#ff00ff22] neon-text-magenta font-bold">
                        {promptCount}
                      </span>
                    )}
                  </div>
                  {(t.meta.name || t.mapped) && (
                    <div className="text-[9px] neon-text-cyan truncate" title={t.meta.name || t.mapped || ""}>
                      {t.meta.name || t.mapped}
                    </div>
                  )}
                  <div className={cn(
                    "text-lg font-bold tabular-nums",
                    t.score > 0 ? "neon-text-green" : "text-muted-foreground"
                  )}>
                    {t.score > 0 ? t.score.toFixed(2) : "\u2014"}
                  </div>
                  {t.checks_total > 0 && (
                    <div className={cn(
                      "text-[10px] font-semibold",
                      t.checks_passed === t.checks_total ? "neon-text-green" :
                      t.checks_passed === 0 ? "neon-text-red" : "neon-text-amber"
                    )}>
                      {t.checks_passed}/{t.checks_total} checks
                    </div>
                  )}
                  {hasTests && (
                    <div className={cn(
                      "text-[10px] font-semibold",
                      t.meta.testsPassed === t.meta.totalTests ? "neon-text-green" :
                      t.meta.testsPassed === 0 ? "neon-text-red" : "neon-text-amber"
                    )}>
                      {t.meta.testsPassed}/{t.meta.totalTests} tests
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </section>

        {/* ─── TWO COLUMNS: FEED + CHANGES/PRIORITY ──────────────── */}
        <div className="grid grid-cols-2 gap-4">

          {/* LEFT: Live Activity Feed */}
          <section className="cyber-card rounded p-3">
            <button
              className="flex items-center justify-between w-full text-sm font-bold neon-text-magenta mb-2 tracking-wider uppercase"
              onClick={() => setFeedCollapsed(!feedCollapsed)}
            >
              <span>Live Activity</span>
              <div className="flex items-center gap-2">
                <span className={cn(
                  "w-2 h-2 rounded-full",
                  sseConnected ? "bg-[#00ff88]" : "bg-[#ff2244]"
                )} />
                {feedCollapsed ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronUp className="h-3.5 w-3.5" />}
              </div>
            </button>
            {!feedCollapsed && (
              <div
                ref={feedRef}
                className="h-[280px] overflow-y-auto text-[11px] space-y-0.5 font-mono"
              >
                {feed.length === 0 && (
                  <div className="text-muted-foreground text-center py-8">
                    Waiting for events...
                  </div>
                )}
                {feed.map(f => (
                  <div key={f.id} className="flex gap-2 py-0.5 border-b border-[#1a1a2e]">
                    <span className="text-muted-foreground shrink-0">[{f.time}]</span>
                    <span className={cn("shrink-0 w-10 font-bold", TYPE_COLORS[f.type])}>
                      {f.type}
                    </span>
                    <span className="text-foreground/80 truncate">{f.text}</span>
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* RIGHT: Recent Changes + Priority */}
          <div className="space-y-4">
            {/* Recent Changes */}
            <section className="cyber-card rounded p-3">
              <button
                className="flex items-center justify-between w-full text-sm font-bold neon-text-green mb-2 tracking-wider uppercase"
                onClick={() => setChangesCollapsed(!changesCollapsed)}
              >
                <span>Recent Changes</span>
                {changesCollapsed ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronUp className="h-3.5 w-3.5" />}
              </button>
              {!changesCollapsed && (
                <div className="space-y-1 text-[11px] max-h-[120px] overflow-y-auto">
                  {(!pollData?.changes || pollData.changes.length === 0) ? (
                    <div className="text-muted-foreground text-center py-4">
                      No changes detected yet
                    </div>
                  ) : (
                    pollData.changes.map((c, i) => (
                      <div key={i} className="flex items-center gap-2 py-0.5">
                        {c.type === "new" ? (
                          <>
                            <span className="px-1.5 py-0.5 rounded bg-[#00f0ff22] neon-text-cyan font-bold text-[9px]">NEW</span>
                            <span>T{pad(c.task_number)}</span>
                            <span className="neon-text-green">{c.score.toFixed(2)}</span>
                          </>
                        ) : (
                          <>
                            <ArrowUp className="h-3 w-3 neon-text-green" />
                            <span>T{pad(c.task_number)}</span>
                            <span className="text-muted-foreground">{(c.prev_score ?? 0).toFixed(2)}</span>
                            <span className="text-muted-foreground">&rarr;</span>
                            <span className="neon-text-green">{c.score.toFixed(2)}</span>
                            {c.checks_passed !== c.prev_checks_passed && (
                              <span className="text-[9px] neon-text-amber">
                                ({c.prev_checks_passed}&rarr;{c.checks_passed} checks)
                              </span>
                            )}
                          </>
                        )}
                      </div>
                    ))
                  )}
                  {pollData?.polled_at && (
                    <div className="text-[9px] text-muted-foreground mt-1">
                      Last poll: {new Date(pollData.polled_at).toLocaleTimeString()}
                    </div>
                  )}
                </div>
              )}
            </section>

            {/* Priority Targets */}
            <section className="cyber-card rounded p-3">
              <button
                className="flex items-center justify-between w-full text-sm font-bold neon-text-amber mb-2 tracking-wider uppercase"
                onClick={() => setPriorityCollapsed(!priorityCollapsed)}
              >
                <span>Priority Targets</span>
                {priorityCollapsed ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronUp className="h-3.5 w-3.5" />}
              </button>
              {!priorityCollapsed && (
                <div className="space-y-0.5 text-[11px] max-h-[200px] overflow-y-auto">
                  {priorities.filter(t => getPriority(t) < 4).map(t => {
                    const p = getPriority(t)
                    const pl = priorityLabel(p)
                    return (
                      <div
                        key={t.num}
                        className="flex items-center gap-2 py-1 border-b border-[#1a1a2e] cursor-pointer hover:bg-[#1a1a2e55]"
                        onClick={() => openEdit(t.num)}
                      >
                        <span className={cn("w-14 text-[9px] font-bold text-center px-1 rounded", pl.color,
                          p === 1 ? "bg-[#ff224418]" :
                          p === 2 ? "bg-[#00f0ff18]" : "bg-[#ffaa0018]"
                        )}>
                          {pl.label}
                        </span>
                        <span className="w-8">T{pad(t.num)}</span>
                        <span className="flex-1 truncate text-muted-foreground">
                          {t.meta.name || t.mapped || "—"}
                        </span>
                        <span className={t.score > 0 ? "neon-text-green" : "text-muted-foreground"}>
                          {t.score > 0 ? t.score.toFixed(2) : "0"}
                        </span>
                        {t.checks_total > 0 && (
                          <span className={cn("text-[9px]",
                            t.checks_passed === t.checks_total ? "neon-text-green" : "neon-text-amber"
                          )}>
                            {t.checks_passed}/{t.checks_total}
                          </span>
                        )}
                      </div>
                    )
                  })}
                  {priorities.filter(t => getPriority(t) < 4).length === 0 && (
                    <div className="text-center text-muted-foreground py-4">
                      All tasks passing!
                    </div>
                  )}
                </div>
              )}
            </section>
          </div>
        </div>
      </div>

      {/* ─── EDIT MODAL ──────────────────────────────────────────── */}
      {editingTask !== null && (
        <div
          className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center"
          onClick={e => { if (e.target === e.currentTarget) setEditingTask(null) }}
        >
          <div className="bg-[#12121a] border border-[#2a2a3e] rounded p-5 w-[480px] shadow-2xl max-h-[80vh] flex flex-col cyber-glow-cyan">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <h3 className="text-base font-bold neon-text-cyan">
                  Task {pad(editingTask)}
                </h3>
                {meta[editingTask]?.name && (
                  <span className="text-xs neon-text-magenta">{meta[editingTask].name}</span>
                )}
                {/* Inline mapping */}
                {allTasks.find(t => t.num === editingTask) && (
                  <span className="text-[10px] text-muted-foreground">
                    {allTasks.find(t => t.num === editingTask)?.mapped || "unmapped"}
                  </span>
                )}
              </div>
              <button onClick={() => setEditingTask(null)} className="text-muted-foreground hover:text-foreground">
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Score info from poll */}
            {pollData?.tasks && (() => {
              const st = pollData.tasks.find(t => t.task_number === editingTask)
              if (!st) return null
              return (
                <div className="flex gap-4 mb-4 text-xs">
                  <div>
                    <span className="text-muted-foreground">Score: </span>
                    <span className="neon-text-green font-bold">{st.score.toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Checks: </span>
                    <span className={st.checks_passed === st.checks_total ? "neon-text-green" : "neon-text-amber"}>
                      {st.checks_passed}/{st.checks_total}
                    </span>
                  </div>
                </div>
              )
            })()}

            {/* Tab buttons */}
            <div className="flex gap-1 mb-4 bg-[#0a0a0f] rounded p-1">
              <button
                className={cn(
                  "flex-1 text-xs font-medium py-1.5 rounded transition-colors",
                  modalView === "edit"
                    ? "bg-[#1a1a2e] neon-text-cyan"
                    : "text-muted-foreground hover:text-foreground"
                )}
                onClick={() => setModalView("edit")}
              >
                Edit
              </button>
              <button
                className={cn(
                  "flex-1 text-xs font-medium py-1.5 rounded transition-colors flex items-center justify-center gap-1.5",
                  modalView === "prompts"
                    ? "bg-[#1a1a2e] neon-text-magenta"
                    : "text-muted-foreground hover:text-foreground"
                )}
                onClick={() => setModalView("prompts")}
              >
                <MessageSquare className="h-3 w-3" />
                Prompts
                {(meta[editingTask]?.prompts?.length || 0) > 0 && (
                  <span className="text-[9px] bg-[#ff00ff22] neon-text-magenta px-1.5 rounded-full font-bold">
                    {meta[editingTask]?.prompts?.length}
                  </span>
                )}
              </button>
            </div>

            {/* Edit view */}
            {modalView === "edit" && (
              <>
                <div className="space-y-3">
                  <div>
                    <label className="text-[10px] font-semibold text-muted-foreground mb-1 block uppercase tracking-wider">Name</label>
                    <Input
                      value={editName}
                      onChange={e => setEditName(e.target.value)}
                      placeholder="e.g. create_employee"
                      autoFocus
                      className="bg-[#0a0a0f] border-[#2a2a3e] text-xs"
                      onKeyDown={e => { if (e.key === "Enter") saveEdit(); if (e.key === "Escape") setEditingTask(null) }}
                    />
                  </div>
                  <div className="flex gap-3">
                    <div className="flex-1">
                      <label className="text-[10px] font-semibold text-muted-foreground mb-1 block uppercase tracking-wider">Tests Passed</label>
                      <Input
                        type="number" value={editPassed} onChange={e => setEditPassed(e.target.value)}
                        placeholder="0" min={0}
                        className="bg-[#0a0a0f] border-[#2a2a3e] text-xs"
                        onKeyDown={e => { if (e.key === "Enter") saveEdit(); if (e.key === "Escape") setEditingTask(null) }}
                      />
                    </div>
                    <div className="flex-1">
                      <label className="text-[10px] font-semibold text-muted-foreground mb-1 block uppercase tracking-wider">Total Tests</label>
                      <Input
                        type="number" value={editTotal} onChange={e => setEditTotal(e.target.value)}
                        placeholder="0" min={0}
                        className="bg-[#0a0a0f] border-[#2a2a3e] text-xs"
                        onKeyDown={e => { if (e.key === "Enter") saveEdit(); if (e.key === "Escape") setEditingTask(null) }}
                      />
                    </div>
                  </div>
                  {/* Map task type */}
                  <div>
                    <label className="text-[10px] font-semibold text-muted-foreground mb-1 block uppercase tracking-wider">Map to Task Type</label>
                    <div className="flex gap-2">
                      <Input
                        value={mappingTask === editingTask ? mappingType : (allTasks.find(t => t.num === editingTask)?.mapped || "")}
                        onChange={e => { setMappingTask(editingTask); setMappingType(e.target.value) }}
                        placeholder="e.g. create_employee"
                        className="bg-[#0a0a0f] border-[#2a2a3e] text-xs flex-1"
                        onKeyDown={e => { if (e.key === "Enter") saveMapping() }}
                      />
                      <Button size="sm" onClick={saveMapping} className="bg-[#00f0ff] text-[#0a0a0f] hover:bg-[#00d0dd] text-xs" disabled={mappingTask !== editingTask || !mappingType.trim()}>
                        Map
                      </Button>
                    </div>
                  </div>
                </div>

                <div className="flex justify-end gap-2 mt-5">
                  <Button variant="destructive" size="sm" onClick={clearEdit} className="text-xs">
                    <Trash2 className="h-3 w-3 mr-1" /> Clear
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => setEditingTask(null)} className="text-xs border-[#2a2a3e]">
                    Cancel
                  </Button>
                  <Button size="sm" onClick={saveEdit} className="bg-[#00f0ff] text-[#0a0a0f] hover:bg-[#00d0dd] text-xs">
                    <Save className="h-3 w-3 mr-1" /> Save
                  </Button>
                </div>
              </>
            )}

            {/* Prompts view */}
            {modalView === "prompts" && (
              <>
                <div className="flex gap-2 mb-3">
                  <textarea
                    value={newPrompt}
                    onChange={e => setNewPrompt(e.target.value)}
                    placeholder="Paste or type a prompt..."
                    className="flex-1 min-h-[60px] rounded border border-[#2a2a3e] bg-[#0a0a0f] px-3 py-2 text-xs resize-y focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#00f0ff]"
                    onKeyDown={e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); addPrompt() } }}
                  />
                  <Button size="sm" onClick={addPrompt} disabled={!newPrompt.trim()} className="self-end bg-[#ff00ff] text-white hover:bg-[#dd00dd] text-xs">
                    <Plus className="h-3 w-3 mr-1" /> Add
                  </Button>
                </div>
                <div className="text-[9px] text-muted-foreground mb-3">Ctrl+Enter to add</div>

                <div className="flex-1 overflow-y-auto space-y-2 min-h-0">
                  {(!meta[editingTask]?.prompts || meta[editingTask]!.prompts!.length === 0) ? (
                    <div className="text-center text-muted-foreground text-xs py-8">No prompts added yet</div>
                  ) : (
                    meta[editingTask]!.prompts!.map((p, idx) => (
                      <div key={idx} className="group border border-[#2a2a3e] rounded p-3 bg-[#0a0a0f] hover:border-[#00f0ff33] transition-colors">
                        <div className="flex items-start justify-between gap-2">
                          <pre className="text-[11px] whitespace-pre-wrap break-words flex-1 font-mono leading-relaxed text-foreground/80">{p.text}</pre>
                          <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                            <button onClick={() => copyPrompt(p.text, idx)} className="p-1 rounded hover:bg-[#1a1a2e] text-muted-foreground hover:text-foreground" title="Copy">
                              {copiedIdx === idx ? <Check className="h-3 w-3 neon-text-green" /> : <Copy className="h-3 w-3" />}
                            </button>
                            <button onClick={() => deletePrompt(idx)} className="p-1 rounded hover:bg-[#ff224418] text-muted-foreground hover:neon-text-red" title="Delete">
                              <Trash2 className="h-3 w-3" />
                            </button>
                          </div>
                        </div>
                        <div className="text-[9px] text-muted-foreground mt-1.5">{new Date(p.addedAt).toLocaleString()}</div>
                      </div>
                    ))
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
