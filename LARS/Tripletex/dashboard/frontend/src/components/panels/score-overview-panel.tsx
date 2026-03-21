import { useState, useEffect, useCallback } from "react"
import { PageHeader } from "@/components/layout/page-header"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import { X, Save, Trash2, MessageSquare, Plus, Copy, Check } from "lucide-react"

interface TaskInfo {
  num: number
  score: number | null
  tries: number
}

interface TaskPrompt {
  text: string
  addedAt: string
}

interface TaskMeta {
  name?: string
  testsPassed?: number
  totalTests?: number
  score?: number | null
  tries?: number
  prompts?: TaskPrompt[]
}

const TASKS: TaskInfo[] = [
  { num: 1, score: 1.5, tries: 6 },
  { num: 2, score: 2.0, tries: 5 },
  { num: 3, score: 2.0, tries: 9 },
  { num: 4, score: 2.0, tries: 8 },
  { num: 5, score: 1.33, tries: 8 },
  { num: 6, score: 1.2, tries: 7 },
  { num: 7, score: 0.29, tries: 9 },
  { num: 8, score: null, tries: 7 },
  { num: 9, score: null, tries: 7 },
  { num: 10, score: null, tries: 7 },
  { num: 11, score: null, tries: 8 },
  { num: 12, score: null, tries: 6 },
  { num: 13, score: 1.13, tries: 9 },
  { num: 14, score: 0.25, tries: 7 },
  { num: 15, score: 0.5, tries: 8 },
  { num: 16, score: null, tries: 8 },
  { num: 17, score: 3.5, tries: 7 },
  { num: 18, score: 2.8, tries: 6 },
  { num: 19, score: 2.05, tries: 3 },
  { num: 20, score: 0.6, tries: 4 },
  { num: 21, score: 1.71, tries: 1 },
  { num: 22, score: null, tries: 1 },
  { num: 23, score: null, tries: 1 },
  { num: 24, score: 2.25, tries: 4 },
  { num: 25, score: null, tries: 3 },
  { num: 26, score: 2.4, tries: 3 },
  { num: 27, score: 1.5, tries: 2 },
  { num: 28, score: 1.5, tries: 2 },
  { num: 29, score: 0.55, tries: 1 },
  { num: 30, score: null, tries: 4 },
]

const STORAGE_KEY = "tripletex_score_overview"

function loadMeta(): Record<number, TaskMeta> {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}")
  } catch {
    return {}
  }
}

function saveMeta(data: Record<number, TaskMeta>) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
}

function pad(n: number) {
  return String(n).padStart(2, "0")
}

export function ScoreOverviewPanel() {
  const [meta, setMeta] = useState<Record<number, TaskMeta>>(loadMeta)
  const [editingTask, setEditingTask] = useState<number | null>(null)
  const [editName, setEditName] = useState("")
  const [editPassed, setEditPassed] = useState("")
  const [editTotal, setEditTotal] = useState("")
  const [editScore, setEditScore] = useState("")
  const [editTries, setEditTries] = useState("")
  const [modalView, setModalView] = useState<"edit" | "prompts">("edit")
  const [newPrompt, setNewPrompt] = useState("")
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null)

  useEffect(() => {
    saveMeta(meta)
  }, [meta])

  const openEdit = useCallback((num: number) => {
    const m = meta[num] || {}
    const t = TASKS.find((x) => x.num === num)!
    setEditName(m.name || "")
    setEditPassed(m.testsPassed !== undefined ? String(m.testsPassed) : "")
    setEditTotal(m.totalTests !== undefined ? String(m.totalTests) : "")
    const effectiveScore = m.score !== undefined ? m.score : t.score
    setEditScore(effectiveScore !== null ? String(effectiveScore) : "")
    setEditTries(String(m.tries !== undefined ? m.tries : t.tries))
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
    const t = TASKS.find((x) => x.num === editingTask)!
    if (editScore.trim() === "") {
      d.score = null
    } else {
      const parsed = parseFloat(editScore)
      if (!isNaN(parsed) && parsed !== t.score) d.score = parsed
    }
    if (editTries.trim()) {
      const parsed = parseInt(editTries)
      if (!isNaN(parsed) && parsed !== t.tries) d.tries = parsed
    }
    setMeta((prev) => {
      const next = { ...prev }
      if (Object.keys(d).length > 0) next[editingTask] = { ...(prev[editingTask] || {}), ...d }
      else delete next[editingTask]
      return next
    })
    setEditingTask(null)
  }, [editingTask, editName, editPassed, editTotal, editScore, editTries])

  const clearEdit = useCallback(() => {
    if (editingTask === null) return
    setMeta((prev) => {
      const next = { ...prev }
      delete next[editingTask]
      return next
    })
    setEditingTask(null)
  }, [editingTask])

  const addPrompt = useCallback(() => {
    if (editingTask === null || !newPrompt.trim()) return
    setMeta((prev) => {
      const existing = prev[editingTask] || {}
      const prompts = [...(existing.prompts || []), { text: newPrompt.trim(), addedAt: new Date().toISOString() }]
      return { ...prev, [editingTask]: { ...existing, prompts } }
    })
    setNewPrompt("")
  }, [editingTask, newPrompt])

  const deletePrompt = useCallback((idx: number) => {
    if (editingTask === null) return
    setMeta((prev) => {
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

  const getScore = (t: TaskInfo) => {
    const m = meta[t.num]
    if (m?.score !== undefined) return m.score
    return t.score
  }
  const getTries = (t: TaskInfo) => {
    const m = meta[t.num]
    if (m?.tries !== undefined) return m.tries
    return t.tries
  }
  const totalScore = TASKS.reduce((s, t) => s + (getScore(t) || 0), 0)
  const scored = TASKS.filter((t) => getScore(t) !== null).length

  return (
    <div>
      <PageHeader
        title="Score Overview"
        description="Competition task scores with test tracking. Click any card to tag it."
      >
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="text-sm font-mono">
            {scored}/30 scored
          </Badge>
          <span className="text-2xl font-bold tabular-nums">{totalScore.toFixed(2)}</span>
        </div>
      </PageHeader>

      {/* Task grid */}
      <div className="grid grid-cols-6 gap-2.5">
        {TASKS.map((t) => {
          const m = meta[t.num] || {}
          const score = getScore(t)
          const tries = getTries(t)
          const hasScore = score !== null
          const hasTests = m.testsPassed !== undefined && m.totalTests !== undefined && (m.totalTests ?? 0) > 0
          const promptCount = m.prompts?.length || 0

          return (
            <Card
              key={t.num}
              className={cn(
                "cursor-pointer transition-all hover:shadow-md hover:-translate-y-0.5 relative",
                hasScore
                  ? "bg-emerald-50 border-emerald-200"
                  : "bg-white border-border"
              )}
              onClick={() => openEdit(t.num)}
            >
              <CardContent className="p-3 text-center">
                <div className="text-[11px] text-muted-foreground">
                  Task {pad(t.num)}
                  {promptCount > 0 && (
                    <span className="inline-flex items-center ml-1 px-1 py-0 text-[9px] font-bold bg-blue-100 text-blue-700 rounded">
                      <MessageSquare className="h-2.5 w-2.5 mr-0.5" />
                      {promptCount}
                    </span>
                  )}
                </div>
                {m.name && (
                  <div
                    className="text-[9px] font-semibold text-blue-600 truncate"
                    title={m.name}
                  >
                    {m.name}
                  </div>
                )}
                <div
                  className={cn(
                    "text-xl font-bold tabular-nums",
                    hasScore ? "text-emerald-700" : "text-muted-foreground"
                  )}
                >
                  {hasScore ? score!.toFixed(2) : "\u2014"}
                </div>
                <div className="text-[10px] text-muted-foreground">
                  {tries} {tries === 1 ? "try" : "tries"}
                </div>
                {hasTests && (
                  <div
                    className={cn(
                      "text-[10px] font-semibold mt-0.5",
                      m.testsPassed === m.totalTests
                        ? "text-emerald-600"
                        : m.testsPassed === 0
                        ? "text-red-500"
                        : "text-amber-600"
                    )}
                  >
                    {m.testsPassed}/{m.totalTests} tests
                  </div>
                )}
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Edit / Prompts modal */}
      {editingTask !== null && (
        <div
          className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center"
          onClick={(e) => {
            if (e.target === e.currentTarget) setEditingTask(null)
          }}
        >
          <div className="bg-white rounded-xl p-6 w-[480px] shadow-2xl max-h-[80vh] flex flex-col">
            {/* Header with tab toggle */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <h3 className="text-lg font-bold">
                  Task {pad(editingTask)}
                  {meta[editingTask]?.name && (
                    <span className="text-sm font-normal text-blue-600 ml-2">
                      {meta[editingTask].name}
                    </span>
                  )}
                </h3>
              </div>
              <button
                onClick={() => setEditingTask(null)}
                className="text-muted-foreground hover:text-foreground"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Tab buttons */}
            <div className="flex gap-1 mb-4 bg-gray-100 rounded-lg p-1">
              <button
                className={cn(
                  "flex-1 text-sm font-medium py-1.5 rounded-md transition-colors",
                  modalView === "edit"
                    ? "bg-white shadow-sm text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
                onClick={() => setModalView("edit")}
              >
                Edit
              </button>
              <button
                className={cn(
                  "flex-1 text-sm font-medium py-1.5 rounded-md transition-colors flex items-center justify-center gap-1.5",
                  modalView === "prompts"
                    ? "bg-white shadow-sm text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
                onClick={() => setModalView("prompts")}
              >
                <MessageSquare className="h-3.5 w-3.5" />
                Prompts
                {(meta[editingTask]?.prompts?.length || 0) > 0 && (
                  <span className="text-[10px] bg-blue-100 text-blue-700 px-1.5 rounded-full font-bold">
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
                    <label className="text-xs font-semibold text-muted-foreground mb-1 block">
                      Name
                    </label>
                    <Input
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      placeholder="e.g. create_employee"
                      autoFocus
                      onKeyDown={(e) => {
                        if (e.key === "Enter") saveEdit()
                        if (e.key === "Escape") setEditingTask(null)
                      }}
                    />
                  </div>
                  <div className="flex gap-3">
                    <div className="flex-1">
                      <label className="text-xs font-semibold text-muted-foreground mb-1 block">
                        Score
                      </label>
                      <Input
                        type="number"
                        step="0.01"
                        value={editScore}
                        onChange={(e) => setEditScore(e.target.value)}
                        placeholder="—"
                        min={0}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") saveEdit()
                          if (e.key === "Escape") setEditingTask(null)
                        }}
                      />
                    </div>
                    <div className="flex-1">
                      <label className="text-xs font-semibold text-muted-foreground mb-1 block">
                        Tries
                      </label>
                      <Input
                        type="number"
                        value={editTries}
                        onChange={(e) => setEditTries(e.target.value)}
                        placeholder="0"
                        min={0}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") saveEdit()
                          if (e.key === "Escape") setEditingTask(null)
                        }}
                      />
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <div className="flex-1">
                      <label className="text-xs font-semibold text-muted-foreground mb-1 block">
                        Tests Passed
                      </label>
                      <Input
                        type="number"
                        value={editPassed}
                        onChange={(e) => setEditPassed(e.target.value)}
                        placeholder="0"
                        min={0}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") saveEdit()
                          if (e.key === "Escape") setEditingTask(null)
                        }}
                      />
                    </div>
                    <div className="flex-1">
                      <label className="text-xs font-semibold text-muted-foreground mb-1 block">
                        Total Tests
                      </label>
                      <Input
                        type="number"
                        value={editTotal}
                        onChange={(e) => setEditTotal(e.target.value)}
                        placeholder="0"
                        min={0}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") saveEdit()
                          if (e.key === "Escape") setEditingTask(null)
                        }}
                      />
                    </div>
                  </div>
                </div>

                <div className="flex justify-end gap-2 mt-5">
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={clearEdit}
                  >
                    <Trash2 className="h-3.5 w-3.5 mr-1" />
                    Clear
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setEditingTask(null)}
                  >
                    Cancel
                  </Button>
                  <Button size="sm" onClick={saveEdit}>
                    <Save className="h-3.5 w-3.5 mr-1" />
                    Save
                  </Button>
                </div>
              </>
            )}

            {/* Prompts view */}
            {modalView === "prompts" && (
              <>
                {/* Add new prompt */}
                <div className="flex gap-2 mb-3">
                  <textarea
                    value={newPrompt}
                    onChange={(e) => setNewPrompt(e.target.value)}
                    placeholder="Paste or type a prompt..."
                    className="flex-1 min-h-[60px] rounded-md border border-input bg-background px-3 py-2 text-sm resize-y focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                        e.preventDefault()
                        addPrompt()
                      }
                    }}
                  />
                  <Button
                    size="sm"
                    onClick={addPrompt}
                    disabled={!newPrompt.trim()}
                    className="self-end"
                  >
                    <Plus className="h-3.5 w-3.5 mr-1" />
                    Add
                  </Button>
                </div>
                <div className="text-[10px] text-muted-foreground mb-3">
                  Ctrl+Enter to add
                </div>

                {/* Prompt list */}
                <div className="flex-1 overflow-y-auto space-y-2 min-h-0">
                  {(!meta[editingTask]?.prompts || meta[editingTask]!.prompts!.length === 0) ? (
                    <div className="text-center text-muted-foreground text-sm py-8">
                      No prompts added yet
                    </div>
                  ) : (
                    meta[editingTask]!.prompts!.map((p, idx) => (
                      <div
                        key={idx}
                        className="group border rounded-lg p-3 bg-gray-50 hover:bg-gray-100 transition-colors"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <pre className="text-xs whitespace-pre-wrap break-words flex-1 font-mono leading-relaxed">
                            {p.text}
                          </pre>
                          <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                            <button
                              onClick={() => copyPrompt(p.text, idx)}
                              className="p-1 rounded hover:bg-gray-200 text-muted-foreground hover:text-foreground"
                              title="Copy"
                            >
                              {copiedIdx === idx ? (
                                <Check className="h-3.5 w-3.5 text-emerald-600" />
                              ) : (
                                <Copy className="h-3.5 w-3.5" />
                              )}
                            </button>
                            <button
                              onClick={() => deletePrompt(idx)}
                              className="p-1 rounded hover:bg-red-100 text-muted-foreground hover:text-red-600"
                              title="Delete"
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </button>
                          </div>
                        </div>
                        <div className="text-[10px] text-muted-foreground mt-1.5">
                          {new Date(p.addedAt).toLocaleString()}
                        </div>
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
