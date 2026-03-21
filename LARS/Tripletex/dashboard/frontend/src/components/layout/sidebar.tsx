import { cn } from "@/lib/utils"
import {
  Play,
  Database,
  RotateCcw,
  Compass,
  FileText,
  AlertTriangle,
  Table2,
  Zap,
  Shield,
  ScrollText,
  ListChecks,
  Wrench,
  Radio,
  FlaskConical,
  ClipboardCheck,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"

export type PanelId =
  | "run"
  | "sandbox"
  | "replay"
  | "explorer"
  | "autofix"
  | "autotest"
  | "tasks"
  | "logs"
  | "report"
  | "errors"
  | "coverage"
  | "results"

export type TabId = "live" | "eval"

interface NavItem {
  id: PanelId
  label: string
  icon: React.ComponentType<{ className?: string }>
  shortcut?: string
}

const LIVE_NAV: NavItem[] = [
  { id: "logs", label: "Solve Logs", icon: ScrollText, shortcut: "1" },
  { id: "tasks", label: "Tasks", icon: ListChecks, shortcut: "2" },
  { id: "autotest", label: "Auto Tester", icon: ClipboardCheck, shortcut: "3" },
  { id: "coverage", label: "API Coverage", icon: Shield, shortcut: "4" },
  { id: "errors", label: "Errors", icon: AlertTriangle, shortcut: "5" },
]

const EVAL_NAV: NavItem[] = [
  { id: "run", label: "Run Evals", icon: Play, shortcut: "1" },
  { id: "results", label: "Results", icon: Table2, shortcut: "2" },
  { id: "report", label: "Report", icon: FileText, shortcut: "3" },
  { id: "errors", label: "Errors", icon: AlertTriangle, shortcut: "4" },
  { id: "autofix", label: "Auto Fix", icon: Wrench, shortcut: "5" },
  { id: "replay", label: "Replay", icon: RotateCcw, shortcut: "6" },
]

const SHARED_NAV: NavItem[] = [
  { id: "sandbox", label: "Sandbox", icon: Database, shortcut: "8" },
  { id: "explorer", label: "Tool Explorer", icon: Compass, shortcut: "9" },
]

interface SidebarProps {
  activePanel: PanelId
  onNavigate: (id: PanelId) => void
  errorCount: number
  connected: boolean
  activeTab: TabId
  onTabChange: (tab: TabId) => void
}

export function Sidebar({
  activePanel,
  onNavigate,
  errorCount,
  connected,
  activeTab,
  onTabChange,
}: SidebarProps) {
  const tabNav = activeTab === "live" ? LIVE_NAV : EVAL_NAV

  return (
    <aside
      className="w-[232px] shrink-0 flex flex-col h-screen sticky top-0"
      style={{
        background: "linear-gradient(180deg, hsl(228 25% 12%) 0%, hsl(228 28% 8%) 100%)",
      }}
    >
      {/* Brand */}
      <div className="px-5 pt-5 pb-4">
        <div className="flex items-center gap-2.5">
          <div
            className="h-8 w-8 rounded-lg flex items-center justify-center shrink-0"
            style={{
              background: "linear-gradient(135deg, hsl(221 83% 53%), hsl(262 83% 58%))",
              boxShadow: "0 2px 8px hsl(221 83% 53% / 0.35)",
            }}
          >
            <Zap className="h-4 w-4 text-white" />
          </div>
          <div>
            <h1 className="text-[14px] font-semibold text-white tracking-tight leading-tight">
              Tripletex Eval
            </h1>
            <p className="text-[10px] text-white/40 font-medium tracking-wide uppercase">
              Dashboard
            </p>
          </div>
        </div>
      </div>

      {/* Tab toggle */}
      <div className="px-3 pb-3">
        <div className="flex rounded-lg bg-white/[0.06] p-0.5">
          <button
            onClick={() => onTabChange("live")}
            className={cn(
              "flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md text-[12px] font-semibold transition-all duration-150",
              activeTab === "live"
                ? "bg-emerald-500/20 text-emerald-400 shadow-sm"
                : "text-white/40 hover:text-white/60"
            )}
          >
            <Radio className="h-3.5 w-3.5" />
            Live
          </button>
          <button
            onClick={() => onTabChange("eval")}
            className={cn(
              "flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md text-[12px] font-semibold transition-all duration-150",
              activeTab === "eval"
                ? "bg-blue-500/20 text-blue-400 shadow-sm"
                : "text-white/40 hover:text-white/60"
            )}
          >
            <FlaskConical className="h-3.5 w-3.5" />
            Eval
          </button>
        </div>
      </div>

      {/* Divider */}
      <div className="mx-4 h-px bg-white/[0.06]" />

      {/* Navigation */}
      <nav className="flex-1 px-3 pt-4 space-y-5 overflow-y-auto">
        {/* Tab-specific section */}
        <div>
          <p className="px-2 mb-1.5 text-[10px] font-semibold uppercase tracking-widest text-white/25">
            {activeTab === "live" ? "Competition" : "Testing"}
          </p>
          <div className="space-y-0.5">
            {tabNav.map((item) => (
              <NavButton
                key={item.id}
                item={item}
                active={activePanel === item.id}
                onClick={() => onNavigate(item.id)}
                errorCount={item.id === "errors" ? errorCount : 0}
                accentColor={activeTab === "live" ? "emerald" : "blue"}
              />
            ))}
          </div>
        </div>

        {/* Shared section */}
        <div>
          <p className="px-2 mb-1.5 text-[10px] font-semibold uppercase tracking-widest text-white/25">
            Tools
          </p>
          <div className="space-y-0.5">
            {SHARED_NAV.map((item) => (
              <NavButton
                key={item.id}
                item={item}
                active={activePanel === item.id}
                onClick={() => onNavigate(item.id)}
                errorCount={0}
                accentColor={activeTab === "live" ? "emerald" : "blue"}
              />
            ))}
          </div>
        </div>
      </nav>

      {/* Bottom divider */}
      <div className="mx-4 h-px bg-white/[0.06]" />

      {/* Connection status */}
      <div className="px-4 py-3.5">
        <Tooltip>
          <TooltipTrigger>
            <div className="flex items-center gap-2.5 cursor-default">
              <div className="relative flex items-center justify-center">
                <span
                  className={cn(
                    "h-2 w-2 rounded-full shrink-0",
                    connected ? "bg-emerald-400" : "bg-red-400"
                  )}
                />
                {connected && (
                  <span className="absolute h-2 w-2 rounded-full bg-emerald-400 animate-ping opacity-50" />
                )}
              </div>
              <span className="text-[11px] font-medium text-white/45">
                {connected ? "Agent connected" : "Agent disconnected"}
              </span>
            </div>
          </TooltipTrigger>
          <TooltipContent side="right" className="text-xs">
            {connected
              ? "FastAPI agent at localhost:8000 is reachable"
              : "Cannot reach agent at localhost:8000"}
          </TooltipContent>
        </Tooltip>
      </div>
    </aside>
  )
}

function NavButton({
  item,
  active,
  onClick,
  errorCount,
  accentColor = "blue",
}: {
  item: NavItem
  active: boolean
  onClick: () => void
  errorCount: number
  accentColor?: "blue" | "emerald"
}) {
  const iconActiveClass = accentColor === "emerald" ? "text-emerald-400" : "text-blue-400"

  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full flex items-center gap-2.5 px-2.5 py-[7px] rounded-lg text-[13px] font-medium transition-all duration-150 group",
        active
          ? "sidebar-active-glow text-white"
          : "text-white/55 hover:text-white/85 hover:bg-white/[0.04]"
      )}
    >
      <item.icon
        className={cn(
          "h-4 w-4 shrink-0 transition-colors duration-150",
          active ? iconActiveClass : "text-white/40 group-hover:text-white/60"
        )}
      />
      <span className="truncate">{item.label}</span>

      {errorCount > 0 && (
        <Badge
          variant="destructive"
          className="ml-auto h-[18px] min-w-[18px] px-1 text-[10px] font-bold bg-red-500/90 text-white border-0"
        >
          {errorCount}
        </Badge>
      )}

      {item.shortcut && errorCount === 0 && (
        <span className="ml-auto text-[10px] font-mono text-white/15 tabular-nums group-hover:text-white/25 transition-colors">
          {item.shortcut}
        </span>
      )}
    </button>
  )
}
