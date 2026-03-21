import { useState, useEffect, useCallback } from "react"
import { PageHeader } from "@/components/layout/page-header"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import { X, Save, Trash2 } from "lucide-react"

interface TaskInfo {
  num: number
  score: number | null
  tries: number
}

interface TaskMeta {
  name?: string
  testsPassed?: number
  totalTests?: number
  score?: number | null
  tries?: number
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

          return (
            <Card
              key={t.num}
              className={cn(
                "cursor-pointer transition-all hover:shadow-md hover:-translate-y-0.5",
                hasScore
                  ? "bg-emerald-50 border-emerald-200"
                  : "bg-white border-border"
              )}
              onClick={() => openEdit(t.num)}
            >
              <CardContent className="p-3 text-center">
                <div className="text-[11px] text-muted-foreground">
                  Task {pad(t.num)}
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

      {/* Edit modal */}
      {editingTask !== null && (
        <div
          className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center"
          onClick={(e) => {
            if (e.target === e.currentTarget) setEditingTask(null)
          }}
        >
          <div className="bg-white rounded-xl p-6 w-[380px] shadow-2xl">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold">
                Tag Task {pad(editingTask)}
              </h3>
              <button
                onClick={() => setEditingTask(null)}
                className="text-muted-foreground hover:text-foreground"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

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
          </div>
        </div>
      )}
    </div>
  )
}
