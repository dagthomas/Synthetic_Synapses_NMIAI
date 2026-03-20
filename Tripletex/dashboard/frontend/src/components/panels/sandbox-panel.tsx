import { useState, useCallback } from "react"
import { useSandboxHealth } from "@/hooks/use-api"
import { seedSandbox, cleanSandbox as apiClean } from "@/lib/api"
import { PageHeader } from "@/components/layout/page-header"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { ScrollArea } from "@/components/ui/scroll-area"
import { toast } from "sonner"
import { cn } from "@/lib/utils"
import {
  RefreshCw,
  Sprout,
  Trash2,
  Building2,
  Users,
  UserCheck,
  Package,
  BookUser,
  FolderKanban,
  Receipt,
  Plane,
  Landmark,
  Loader2,
  Wifi,
  WifiOff,
  CheckCircle2,
  AlertCircle,
} from "lucide-react"

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

const ENTITY_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  department: Building2,
  employee: Users,
  customer: UserCheck,
  product: Package,
  contact: BookUser,
  project: FolderKanban,
  invoice: Receipt,
  travelExpense: Plane,
}

const ENTITY_ORDER = [
  "department",
  "employee",
  "customer",
  "product",
  "contact",
  "project",
  "invoice",
  "travelExpense",
]

export function SandboxPanel() {
  const { data: health, isLoading, mutate } = useSandboxHealth()
  const [logs, setLogs] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [seedingType, setSeedingType] = useState<string | null>(null)

  const addLog = useCallback((msg: string) => {
    setLogs((prev) => [...prev, msg])
  }, [])

  async function handleSeed(types: string[], clean = false) {
    setBusy(true)
    setSeedingType(types[0] || null)
    setLogs([])
    const action = clean ? "Clean & Reseed" : types[0] === "all" ? "Seed All" : `Seed ${ENTITY_LABELS[types[0]] || types[0]}`
    addLog(`Starting ${action}...`)

    try {
      const result = await seedSandbox(types, clean)
      for (const [type, r] of Object.entries(result.results || {})) {
        const label = ENTITY_LABELS[type] || type
        addLog(
          `  ${label}: ${r.created} created` +
            (r.errors?.length ? `, ${r.errors.length} errors` : "")
        )
        r.errors?.forEach((e) => addLog(`    ERROR: ${e}`))
      }
      if (result.bank_account) {
        const ba = result.bank_account
        addLog(
          `  Bank account: ${
            ba.ok
              ? ba.already_set
                ? "already set"
                : "configured"
              : "FAILED: " + (ba.error || "")
          }`
        )
      }
      addLog(`\nDone: ${result.total_created} created, ${result.total_errors} errors`)
      toast.success(`${action}: ${result.total_created} created`)
    } catch (err) {
      addLog(`ERROR: ${(err as Error).message}`)
      toast.error((err as Error).message)
    } finally {
      setBusy(false)
      setSeedingType(null)
      mutate()
    }
  }

  async function handleClean() {
    setBusy(true)
    setSeedingType("clean")
    setLogs([])
    addLog("Cleaning all sandbox entities...")

    try {
      const result = await apiClean()
      for (const [type, r] of Object.entries(result.results || {})) {
        const label = ENTITY_LABELS[type] || type
        if (r.deleted > 0) addLog(`  ${label}: ${r.deleted} deleted`)
      }
      addLog(`\nDone: ${result.total_deleted} deleted`)
      toast.success(`Cleaned: ${result.total_deleted} deleted`)
    } catch (err) {
      addLog(`ERROR: ${(err as Error).message}`)
      toast.error((err as Error).message)
    } finally {
      setBusy(false)
      setSeedingType(null)
      mutate()
    }
  }

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-20 w-full rounded-xl" />
        <div className="grid grid-cols-3 gap-3">
          {Array.from({ length: 9 }).map((_, i) => (
            <Skeleton key={i} className="h-28 rounded-xl" />
          ))}
        </div>
      </div>
    )
  }

  const connected = health?.connected ?? false
  const ready = health?.ready ?? false

  return (
    <div>
      <PageHeader title="Sandbox" description="Manage Tripletex sandbox entities and data seeding">
        <Button variant="outline" size="sm" onClick={() => mutate()} disabled={busy}>
          <RefreshCw className={cn("h-3.5 w-3.5 mr-1.5", busy && "animate-spin")} />
          Refresh
        </Button>
      </PageHeader>

      {/* Status bar */}
      <Card className={cn(
        "mb-4 shadow-premium transition-all duration-300",
        !connected ? "glow-destructive" : ready ? "glow-success" : "glow-warning"
      )}>
        <CardContent className="p-4 flex items-center gap-3">
          <div className="relative flex items-center justify-center">
            {!connected ? (
              <WifiOff className="h-5 w-5 text-red-500" />
            ) : ready ? (
              <Wifi className="h-5 w-5 text-emerald-500" />
            ) : (
              <AlertCircle className="h-5 w-5 text-amber-500" />
            )}
          </div>
          <div>
            <span className="font-semibold text-[13px]">
              {!connected
                ? "Cannot Connect"
                : ready
                ? "Sandbox Ready"
                : "Setup Required"}
            </span>
            {health?.base_url && (
              <p className="text-[11px] text-muted-foreground font-mono mt-0.5">
                {health.base_url}
              </p>
            )}
          </div>
          <Badge
            variant="outline"
            className={cn(
              "ml-auto text-[10px] font-semibold",
              !connected
                ? "border-red-200 text-red-600 bg-red-50"
                : ready
                ? "border-emerald-200 text-emerald-600 bg-emerald-50"
                : "border-amber-200 text-amber-600 bg-amber-50"
            )}
          >
            {!connected ? "Offline" : ready ? "Healthy" : "Incomplete"}
          </Badge>
        </CardContent>
      </Card>

      {/* Entity grid */}
      {connected && (
        <div className="grid grid-cols-3 gap-3 mb-4">
          {ENTITY_ORDER.map((type, idx) => {
            const info = health?.entities[type] || { count: 0, ok: false }
            const Icon = ENTITY_ICONS[type] || Building2
            const isSeeding = seedingType === type
            return (
              <Card
                key={type}
                className={cn(
                  "shadow-premium metric-card transition-all duration-200",
                  !info.ok && "border-amber-200/50"
                )}
                style={{ animationDelay: `${idx * 50}ms` }}
              >
                <CardContent className="p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className={cn(
                      "h-9 w-9 rounded-lg flex items-center justify-center",
                      info.ok
                        ? "bg-emerald-50 text-emerald-600"
                        : "bg-amber-50 text-amber-600"
                    )}>
                      <Icon className="h-4.5 w-4.5" />
                    </div>
                    {info.ok ? (
                      <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                    ) : (
                      <AlertCircle className="h-4 w-4 text-amber-500" />
                    )}
                  </div>
                  <div
                    className={cn(
                      "text-2xl font-bold tabular-nums tracking-tight",
                      info.ok ? "text-foreground" : "text-amber-600"
                    )}
                  >
                    {info.count}
                  </div>
                  <div className="text-[11px] text-muted-foreground mt-0.5 font-medium">
                    {ENTITY_LABELS[type]}
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className={cn(
                      "mt-2 h-7 text-[11px] w-full",
                      !info.ok && "text-amber-600 hover:text-amber-700 hover:bg-amber-50"
                    )}
                    onClick={() => handleSeed([type])}
                    disabled={busy}
                  >
                    {isSeeding ? (
                      <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                    ) : (
                      <Sprout className="h-3 w-3 mr-1" />
                    )}
                    {isSeeding ? "Seeding..." : "Seed"}
                  </Button>
                </CardContent>
              </Card>
            )
          })}

          {/* Bank account card */}
          <Card className={cn(
            "shadow-premium metric-card",
            !health?.bank_account_1920 && "border-amber-200/50"
          )}>
            <CardContent className="p-4">
              <div className="flex items-start justify-between mb-3">
                <div className={cn(
                  "h-9 w-9 rounded-lg flex items-center justify-center",
                  health?.bank_account_1920
                    ? "bg-emerald-50 text-emerald-600"
                    : "bg-amber-50 text-amber-600"
                )}>
                  <Landmark className="h-4.5 w-4.5" />
                </div>
                {health?.bank_account_1920 ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                ) : (
                  <AlertCircle className="h-4 w-4 text-amber-500" />
                )}
              </div>
              <div className={cn(
                "text-lg font-bold",
                health?.bank_account_1920 ? "text-foreground" : "text-amber-600"
              )}>
                {health?.bank_account_1920 ? "Configured" : "Missing"}
              </div>
              <div className="text-[11px] text-muted-foreground mt-0.5 font-medium">
                Bank Account 1920
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2 mb-4">
        <Button
          size="sm"
          onClick={() => handleSeed(["all"])}
          disabled={busy}
          className="bg-gradient-to-r from-primary to-blue-600 hover:shadow-lg hover:shadow-primary/20"
        >
          {seedingType === "all" ? (
            <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
          ) : (
            <Sprout className="h-3.5 w-3.5 mr-1.5" />
          )}
          Seed All
        </Button>
        <Button
          size="sm"
          variant="outline"
          className="border-amber-300 text-amber-700 hover:bg-amber-50"
          onClick={() => handleSeed(["all"], true)}
          disabled={busy}
        >
          <RefreshCw className={cn("h-3.5 w-3.5 mr-1.5", seedingType === "all" && "animate-spin")} />
          Clean & Reseed
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={handleClean}
          disabled={busy}
        >
          {seedingType === "clean" ? (
            <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
          ) : (
            <Trash2 className="h-3.5 w-3.5 mr-1.5" />
          )}
          Clean Only
        </Button>
      </div>

      {/* Terminal log */}
      {logs.length > 0 && (
        <Card className="terminal-bg border-white/[0.06] shadow-premium-lg overflow-hidden">
          <div className="flex items-center gap-1.5 px-4 py-2.5 border-b border-white/[0.06]">
            <span className="h-2.5 w-2.5 rounded-full bg-red-500/80" />
            <span className="h-2.5 w-2.5 rounded-full bg-amber-500/80" />
            <span className="h-2.5 w-2.5 rounded-full bg-green-500/80" />
            <span className="ml-3 text-[11px] text-white/30 font-mono">output</span>
          </div>
          <ScrollArea className="h-[200px]">
            <CardContent className="p-4">
              <pre className="text-[12px] font-mono text-slate-300 whitespace-pre-wrap leading-relaxed">
                {logs.map((line, i) => (
                  <span key={i}>
                    <span className="text-slate-600 select-none mr-3 text-[10px]">
                      {String(i + 1).padStart(3, " ")}
                    </span>
                    <span className={cn(
                      line.includes("ERROR") ? "text-red-400" :
                      line.includes("Done") ? "text-emerald-400" :
                      "text-slate-300"
                    )}>
                      {line}
                    </span>
                    {"\n"}
                  </span>
                ))}
              </pre>
            </CardContent>
          </ScrollArea>
        </Card>
      )}
    </div>
  )
}
