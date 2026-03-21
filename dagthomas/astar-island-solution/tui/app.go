package main

import (
	"fmt"
	"path/filepath"
	"strings"

	"astar-tui/api"
	"astar-tui/components"
	"astar-tui/internal"
	"astar-tui/tabs"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// Tab interface
type Tab interface {
	Title() string
	ShortKey() string
	Rune() string
}

// AppModel is the root model
type AppModel struct {
	activeTab   int
	layout      *LayoutNode
	focused     *LayoutNode
	width       int
	height      int
	tabBar      components.TabBar
	statusBar   components.StatusBar
	apiClient   *api.Client
	procManager *internal.ProcessManager
	envConfig   *internal.EnvConfig
	dataDir     string
	showHelp    bool
	connStatus  string

	// Tab models
	dashboard tabs.DashboardModel
	rounds    tabs.RoundsModel
	submit    tabs.SubmitModel
	explorer  tabs.ExplorerModel
	autoloop  tabs.AutoloopModel
	research  tabs.ResearchModel
	backtest  tabs.BacktestModel
	logs      tabs.LogsModel
	settings  tabs.SettingsModel
	metrics   tabs.MetricsModel

	// Hacker scramble effect
	scramble components.ScrambleModel
}

func NewApp(client *api.Client, procMgr *internal.ProcessManager, env *internal.EnvConfig, dataDir string) AppModel {
	tabInfos := []components.TabInfo{
		{Key: "1", Title: "Dashboard", Rune: "ᛟ"},
		{Key: "2", Title: "Rounds", Rune: "ᚱ"},
		{Key: "3", Title: "Submit", Rune: "ᛏ"},
		{Key: "4", Title: "Explorer", Rune: "ᚨ"},
		{Key: "5", Title: "Autoiterate", Rune: "ᚹ"},
		{Key: "6", Title: "Research", Rune: "ᚷ"},
		{Key: "7", Title: "Backtest", Rune: "ᛞ"},
		{Key: "8", Title: "Logs", Rune: "ᛚ"},
		{Key: "9", Title: "Settings", Rune: "ᛁ"},
		{Key: "0", Title: "Metrics", Rune: "ᛗ"},
	}

	autoloopLog := dataDir + "/autoloop_log.jsonl"

	// Load persisted layout or create default
	layout, focused, _ := LoadLayout(dataDir)
	if layout == nil {
		layout = NewLayout()
		focused = layout
	}

	tb := components.NewTabBar(tabInfos)
	tb.ActiveIdx = focused.TabIdx

	app := AppModel{
		apiClient:   client,
		procManager: procMgr,
		envConfig:   env,
		dataDir:     dataDir,
		tabBar:      tb,
		activeTab:   focused.TabIdx,
		connStatus:  "connected",
		layout:      layout,
		focused:     focused,
		scramble:    components.NewScramble(),

		dashboard: tabs.NewDashboard(client, dataDir),
		rounds:    tabs.NewRounds(client),
		submit:    tabs.NewSubmit(client, procMgr),
		explorer:  tabs.NewExplorer(client, dataDir),
		autoloop:  tabs.NewAutoloop(client, procMgr, autoloopLog),
		research:  tabs.NewResearch(client, procMgr, dataDir),
		backtest:  tabs.NewBacktest(client, procMgr),
		logs:      tabs.NewLogs(dataDir),
		settings:  tabs.NewSettings(env, filepath.Dir(dataDir)),
		metrics:   tabs.NewMetrics(client, dataDir),
	}

	// Sync pane tabs from loaded layout
	leaves := layout.Leaves()
	paneTabs := make([]int, 0, len(leaves))
	for _, leaf := range leaves {
		if leaf != focused {
			paneTabs = append(paneTabs, leaf.TabIdx)
		}
	}
	app.tabBar.PaneTabs = paneTabs

	return app
}

func (m AppModel) Init() tea.Cmd {
	return tea.Batch(
		m.dashboard.Init(),
		m.rounds.Init(),
		m.explorer.Init(),
		m.autoloop.Init(),
		m.research.Init(),
		m.logs.Init(),
		m.metrics.Init(),
		components.ScrambleTickCmd(), // Start scramble animation
	)
}

func (m AppModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case components.ScrambleTickMsg:
		m.scramble = m.scramble.Tick()
		if m.scramble.IsActive() {
			return m, components.ScrambleTickCmd()
		}
		return m, nil

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.tabBar.Width = msg.Width
		m.statusBar.Width = msg.Width

		contentH := msg.Height - 4 // tab bar + status bar
		m.dashboard = m.dashboard.SetSize(msg.Width, contentH)
		m.rounds = m.rounds.SetSize(msg.Width, contentH)
		m.submit = m.submit.SetSize(msg.Width, contentH)
		m.explorer = m.explorer.SetSize(msg.Width, contentH)
		m.autoloop = m.autoloop.SetSize(msg.Width, contentH)
		m.research = m.research.SetSize(msg.Width, contentH)
		m.backtest = m.backtest.SetSize(msg.Width, contentH)
		m.logs = m.logs.SetSize(msg.Width, contentH)
		m.settings = m.settings.SetSize(msg.Width, contentH)
		m.metrics = m.metrics.SetSize(msg.Width, contentH)
		// Trigger initial scramble effect on first render
		if !m.scramble.IsDone() && !m.scramble.IsActive() {
			m.scramble = m.scramble.SetTarget(m.renderTab(m.focused.TabIdx))
			return m, components.ScrambleTickCmd()
		}
		return m, nil

	case tea.KeyMsg:
		// Global keys
		switch msg.String() {
		case "ctrl+c", "q":
			if !m.showHelp {
				return m, tea.Quit
			}
			m.showHelp = false
			return m, nil
		case "?":
			m.showHelp = !m.showHelp
			return m, nil

		// Layout: split / close
		case "alt+v":
			if m.focused.IsLeaf() {
				m.focused = m.focused.Split(LayoutVSplit)
				m.syncTabBar()
			}
			return m, nil
		case "alt+s":
			if m.focused.IsLeaf() {
				m.focused = m.focused.Split(LayoutHSplit)
				m.syncTabBar()
			}
			return m, nil
		case "alt+w":
			if m.layout.LeafCount() > 1 {
				m.focused = m.focused.Close()
				m.syncTabBar()
			}
			return m, nil

		// Layout: navigate between panes
		case "alt+left", "alt+h":
			if next := m.focused.Navigate(DirLeft); next != nil {
				m.focused = next
				m.syncTabBar()
			}
			return m, nil
		case "alt+right", "alt+l":
			if next := m.focused.Navigate(DirRight); next != nil {
				m.focused = next
				m.syncTabBar()
			}
			return m, nil
		case "alt+up", "alt+k":
			if next := m.focused.Navigate(DirUp); next != nil {
				m.focused = next
				m.syncTabBar()
			}
			return m, nil
		case "alt+down", "alt+j":
			if next := m.focused.Navigate(DirDown); next != nil {
				m.focused = next
				m.syncTabBar()
			}
			return m, nil

		// Layout: resize split ratio
		case "alt+=":
			m.focused.Resize(0.05)
			return m, nil
		case "alt+-":
			m.focused.Resize(-0.05)
			return m, nil

		// Tab cycling in focused pane
		case "tab":
			m.focused.TabIdx = (m.focused.TabIdx + 1) % 10
			m.syncTabBar()
			m.scramble = m.scramble.SetTarget(m.renderTab(m.focused.TabIdx))
			return m, components.ScrambleTickCmd()
		case "shift+tab":
			m.focused.TabIdx = (m.focused.TabIdx + 9) % 10
			m.syncTabBar()
			m.scramble = m.scramble.SetTarget(m.renderTab(m.focused.TabIdx))
			return m, components.ScrambleTickCmd()
		}

		// Tab switching via number keys (only when not in a text input mode)
		if !m.isInputMode() {
			if idx := numKeyToTabIdx(msg.String()); idx >= 0 && m.focused.TabIdx != idx {
				m.focused.TabIdx = idx
				m.syncTabBar()
				m.scramble = m.scramble.SetTarget(m.renderTab(m.focused.TabIdx))
				return m, components.ScrambleTickCmd()
			}
		}
	}

	var cmd tea.Cmd

	// KeyMsg → focused pane's tab only
	if _, isKey := msg.(tea.KeyMsg); isKey {
		switch m.focused.TabIdx {
		case 0:
			m.dashboard, cmd = m.dashboard.Update(msg)
		case 1:
			m.rounds, cmd = m.rounds.Update(msg)
		case 2:
			m.submit, cmd = m.submit.Update(msg)
		case 3:
			m.explorer, cmd = m.explorer.Update(msg)
		case 4:
			m.autoloop, cmd = m.autoloop.Update(msg)
		case 5:
			m.research, cmd = m.research.Update(msg)
		case 6:
			m.backtest, cmd = m.backtest.Update(msg)
		case 7:
			m.logs, cmd = m.logs.Update(msg)
		case 8:
			m.settings, cmd = m.settings.Update(msg)
		case 9:
			m.metrics, cmd = m.metrics.Update(msg)
		}

		m.statusBar.RunningProcs = m.procManager.Running()
		m.statusBar.ConnStatus = m.connStatus
		m.statusBar.PaneCount = m.layout.LeafCount()
		return m, cmd
	}

	// Data messages → route to ALL tabs so each receives its Init() response
	var cmds []tea.Cmd
	m.dashboard, cmd = m.dashboard.Update(msg)
	if cmd != nil {
		cmds = append(cmds, cmd)
	}
	m.rounds, cmd = m.rounds.Update(msg)
	if cmd != nil {
		cmds = append(cmds, cmd)
	}
	m.submit, cmd = m.submit.Update(msg)
	if cmd != nil {
		cmds = append(cmds, cmd)
	}
	m.explorer, cmd = m.explorer.Update(msg)
	if cmd != nil {
		cmds = append(cmds, cmd)
	}
	m.autoloop, cmd = m.autoloop.Update(msg)
	if cmd != nil {
		cmds = append(cmds, cmd)
	}
	m.research, cmd = m.research.Update(msg)
	if cmd != nil {
		cmds = append(cmds, cmd)
	}
	m.backtest, cmd = m.backtest.Update(msg)
	if cmd != nil {
		cmds = append(cmds, cmd)
	}
	m.logs, cmd = m.logs.Update(msg)
	if cmd != nil {
		cmds = append(cmds, cmd)
	}
	m.settings, cmd = m.settings.Update(msg)
	if cmd != nil {
		cmds = append(cmds, cmd)
	}
	m.metrics, cmd = m.metrics.Update(msg)
	if cmd != nil {
		cmds = append(cmds, cmd)
	}

	// Update status bar
	m.statusBar.RunningProcs = m.procManager.Running()
	m.statusBar.ConnStatus = m.connStatus
	m.statusBar.PaneCount = m.layout.LeafCount()

	return m, tea.Batch(cmds...)
}

func (m AppModel) isInputMode() bool {
	// Explorer uses 1-5 for seed switching, Settings has text inputs
	return m.focused.TabIdx == 3 || m.focused.TabIdx == 8
}

// persistLayout saves the current layout state to disk.
func (m *AppModel) persistLayout() {
	_ = SaveLayout(m.dataDir, m.layout, m.focused)
}

// syncTabBar keeps activeTab and tabBar in sync with the focused pane,
// and persists the layout to disk. Also triggers scramble effect.
func (m *AppModel) syncTabBar() {
	m.activeTab = m.focused.TabIdx
	m.tabBar.ActiveIdx = m.activeTab
	leaves := m.layout.Leaves()
	paneTabs := make([]int, 0, len(leaves))
	for _, leaf := range leaves {
		if leaf != m.focused {
			paneTabs = append(paneTabs, leaf.TabIdx)
		}
	}
	m.tabBar.PaneTabs = paneTabs
	m.persistLayout()
}

// numKeyToTabIdx converts "1"-"9" to 0-8 and "0" to 9, or returns -1.
func numKeyToTabIdx(key string) int {
	if len(key) == 1 && key[0] >= '1' && key[0] <= '9' {
		return int(key[0] - '1')
	}
	if key == "0" {
		return 9 // Metrics tab
	}
	return -1
}

func (m AppModel) View() string {
	if m.width == 0 || m.height == 0 {
		return ""
	}
	if m.showHelp {
		return m.renderHelp()
	}

	var sections []string
	sections = append(sections, m.tabBar.View())

	contentH := m.height - 3

	if m.layout.LeafCount() == 1 {
		// Single pane — unchanged rendering path (no tinting)
		content := m.renderTab(m.focused.TabIdx)
		// Apply hacker scramble effect if active
		if m.scramble.IsActive() {
			content = m.scramble.View(content)
		}
		contentLines := strings.Count(content, "\n") + 1
		if contentLines < contentH {
			content += strings.Repeat("\n", contentH-contentLines)
		}
		sections = append(sections, content)
	} else {
		// Multi-pane layout — build leaf index map for tinting
		leafIdx := 0
		content := m.renderLayoutNodeTinted(m.layout, m.width, contentH, &leafIdx)
		sections = append(sections, content)
	}

	sections = append(sections, m.statusBar.View())
	return lipgloss.JoinVertical(lipgloss.Left, sections...)
}

func (m AppModel) renderTab(tabIdx int) string {
	switch tabIdx {
	case 0:
		return m.dashboard.View()
	case 1:
		return m.rounds.View()
	case 2:
		return m.submit.View()
	case 3:
		return m.explorer.View()
	case 4:
		return m.autoloop.View()
	case 5:
		return m.research.View()
	case 6:
		return m.backtest.View()
	case 7:
		return m.logs.View()
	case 8:
		return m.settings.View()
	case 9:
		return m.metrics.View()
	default:
		return ""
	}
}

// renderLayoutNode recursively renders the layout tree (no tinting).
func (m AppModel) renderLayoutNode(node *LayoutNode, w, h int) string {
	if node.IsLeaf() {
		tabH := h - 1
		if tabH < 1 {
			tabH = 1
		}
		m.setTabSize(node.TabIdx, w, tabH)
		content := m.renderTab(node.TabIdx)
		return padToSize(content, w, h)
	}

	switch node.Kind {
	case LayoutVSplit:
		leftW, rightW := splitDim(w, node.Ratio)
		left := m.renderLayoutNode(node.Left, leftW, h)
		divFocused := node.Left == m.focused || node.Right == m.focused
		divider := renderVDivider(h, divFocused)
		right := m.renderLayoutNode(node.Right, rightW, h)
		return lipgloss.JoinHorizontal(lipgloss.Top, left, divider, right)

	case LayoutHSplit:
		topH, botH := splitDim(h, node.Ratio)
		top := m.renderLayoutNode(node.Left, w, topH)
		divFocused := node.Left == m.focused || node.Right == m.focused
		divider := renderHDivider(w, divFocused)
		bottom := m.renderLayoutNode(node.Right, w, botH)
		return lipgloss.JoinVertical(lipgloss.Left, top, divider, bottom)
	}

	return ""
}

// renderLayoutNodeTinted renders with per-pane accent colors.
// leafIdx is incremented for each leaf encountered (left-to-right order).
func (m AppModel) renderLayoutNodeTinted(node *LayoutNode, w, h int, leafIdx *int) string {
	if node.IsLeaf() {
		idx := *leafIdx
		*leafIdx++
		isFocused := node == m.focused
		accent := PaneAccent(idx)

		// Reserve 1 line for the pane header bar
		tabH := h - 2
		if tabH < 1 {
			tabH = 1
		}
		m.setTabSize(node.TabIdx, w, tabH)
		content := m.renderTab(node.TabIdx)
		content = padToSize(content, w, h-1)

		// Build header bar with pane's accent color
		tabInfo := m.tabBar.Tabs[node.TabIdx]
		label := fmt.Sprintf(" %s %s ", tabInfo.Rune, tabInfo.Title)
		labelW := lipgloss.Width(label)
		lineRest := w - labelW
		if lineRest < 0 {
			lineRest = 0
		}

		if isFocused {
			// Focused: inverted accent label + glitch-style accent line
			headerStyle := lipgloss.NewStyle().
				Foreground(lipgloss.Color("#0a0a12")).
				Background(accent).
				Bold(true)
			lineStyle := lipgloss.NewStyle().Foreground(accent)
			glitchEdge := lipgloss.NewStyle().Foreground(accent).Render("▓▒░")
			lineLen := lineRest - 3
			if lineLen < 0 {
				lineLen = 0
			}
			header := headerStyle.Render(label) + glitchEdge + lineStyle.Render(strings.Repeat("═", lineLen))
			content = header + "\n" + content
		} else {
			// Unfocused: dimmed label + dark line
			headerStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#2a2a44"))
			lineStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#1a1a2e"))
			header := headerStyle.Render(label) + lineStyle.Render(strings.Repeat("─", lineRest))
			content = header + "\n" + content
		}

		return content
	}

	switch node.Kind {
	case LayoutVSplit:
		leftW, rightW := splitDim(w, node.Ratio)
		left := m.renderLayoutNodeTinted(node.Left, leftW, h, leafIdx)
		// Use the focused pane's accent for divider, else muted
		if node.Left == m.focused || node.Right == m.focused {
			focusedAccent := m.focusedPaneAccent()
			divider := renderVDividerAccent(h, focusedAccent)
			right := m.renderLayoutNodeTinted(node.Right, rightW, h, leafIdx)
			return lipgloss.JoinHorizontal(lipgloss.Top, left, divider, right)
		}
		divider := renderVDivider(h, false)
		right := m.renderLayoutNodeTinted(node.Right, rightW, h, leafIdx)
		return lipgloss.JoinHorizontal(lipgloss.Top, left, divider, right)

	case LayoutHSplit:
		topH, botH := splitDim(h, node.Ratio)
		top := m.renderLayoutNodeTinted(node.Left, w, topH, leafIdx)
		if node.Left == m.focused || node.Right == m.focused {
			focusedAccent := m.focusedPaneAccent()
			divider := renderHDividerAccent(w, focusedAccent)
			bottom := m.renderLayoutNodeTinted(node.Right, w, botH, leafIdx)
			return lipgloss.JoinVertical(lipgloss.Left, top, divider, bottom)
		}
		divider := renderHDivider(w, false)
		bottom := m.renderLayoutNodeTinted(node.Right, w, botH, leafIdx)
		return lipgloss.JoinVertical(lipgloss.Left, top, divider, bottom)
	}

	return ""
}

// focusedPaneAccent returns the accent color of the currently focused pane.
func (m AppModel) focusedPaneAccent() lipgloss.Color {
	leaves := m.layout.Leaves()
	for i, leaf := range leaves {
		if leaf == m.focused {
			return PaneAccent(i)
		}
	}
	return PaneAccents[0]
}

// setTabSize updates a tab's dimensions on the local model copy.
func (m *AppModel) setTabSize(tabIdx, w, h int) {
	switch tabIdx {
	case 0:
		m.dashboard = m.dashboard.SetSize(w, h)
	case 1:
		m.rounds = m.rounds.SetSize(w, h)
	case 2:
		m.submit = m.submit.SetSize(w, h)
	case 3:
		m.explorer = m.explorer.SetSize(w, h)
	case 4:
		m.autoloop = m.autoloop.SetSize(w, h)
	case 5:
		m.research = m.research.SetSize(w, h)
	case 6:
		m.backtest = m.backtest.SetSize(w, h)
	case 7:
		m.logs = m.logs.SetSize(w, h)
	case 8:
		m.settings = m.settings.SetSize(w, h)
	case 9:
		m.metrics = m.metrics.SetSize(w, h)
	}
}

func (m AppModel) renderHelp() string {
	helpStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(ColorNeonCyan).
		Padding(1, 2).
		Width(60)

	title := lipgloss.NewStyle().
		Foreground(ColorNeonCyan).
		Bold(true)

	key := lipgloss.NewStyle().
		Foreground(ColorNeonPink).
		Bold(true)

	desc := lipgloss.NewStyle().
		Foreground(ColorFg)

	header := title.Render(`
   ░▒▓ ASTAR ISLAND //EXPLORER ▓▒░
   ᚨ ᛋ ᛏ ᚨ ᚱ ᛫ ᛁ ᛋ ᛚ ᚨ ᚾ ᛞ
`)

	bindings := []struct{ k, d string }{
		{"0-9", "Switch tab in focused pane"},
		{"Tab/Shift+Tab", "Cycle tabs in pane"},
		{"q / Ctrl+C", "Quit"},
		{"?", "Toggle help"},
		{"", ""},
		{"Layout (splits)", ""},
		{"Alt+V", "Split vertical (side-by-side)"},
		{"Alt+S", "Split horizontal (stacked)"},
		{"Alt+W", "Close focused pane"},
		{"Alt+←→↑↓", "Navigate between panes"},
		{"Alt+h/j/k/l", "Navigate (vim style)"},
		{"Alt+=", "Grow focused pane"},
		{"Alt+-", "Shrink focused pane"},
		{"", ""},
		{"Dashboard (1)", ""},
		{"r", "Refresh all data"},
		{"", ""},
		{"Rounds (2)", ""},
		{"↑↓", "Navigate rounds"},
		{"Enter", "Drill into round seeds"},
		{"Esc", "Back to table"},
		{"", ""},
		{"Submit (3)", ""},
		{"↑↓", "Select variant"},
		{"Enter", "Confirm & run"},
		{"x", "Cancel running"},
		{"", ""},
		{"Explorer (4)", ""},
		{"1-5", "Switch seed"},
		{"←→↑↓", "Move viewport"},
		{"v", "Toggle viewport overlay"},
		{"o", "Load observations"},
		{"p", "Load predictions (compare)"},
		{"", ""},
		{"Autoiterate (5)", ""},
		{"s", "Start autoiterate"},
		{"x", "Stop autoiterate"},
		{"", ""},
		{"Research (6)", ""},
		{"g/a/m", "Switch agent"},
		{"s", "Start agent"},
		{"x", "Stop agent"},
		{"", ""},
		{"Logs (8)", ""},
		{"f", "Switch log file"},
		{"t", "Toggle tail/browse"},
		{"/", "Search filter"},
		{"G", "Jump to tail"},
		{"", ""},
		{"Metrics (0)", ""},
		{"r", "Refresh all data"},
	}

	var rows []string
	for _, b := range bindings {
		if b.k == "" {
			rows = append(rows, "")
		} else if b.d == "" {
			rows = append(rows, title.Render(b.k))
		} else {
			rows = append(rows, fmt.Sprintf("  %s  %s", key.Render(fmt.Sprintf("%-16s", b.k)), desc.Render(b.d)))
		}
	}

	return helpStyle.Render(header + strings.Join(rows, "\n") + "\n\n" +
		lipgloss.NewStyle().Foreground(ColorMuted).Render("  Press ? or q to close"))
}
