package tabs

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"astar-tui/api"
	"astar-tui/components"
	"astar-tui/internal"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
)

type researchTickMsg time.Time

type researchLogMsg struct {
	agent   string
	entries []api.ResearchEntry
	err     error
}

// ResearchAgent defines a research agent config
type ResearchAgent struct {
	Name    string
	Key     string
	LogPath string
	Command string
	Args    []string
	Rune    string
	Desc    string
	HowTo   string
}

// ResearchModel manages research agents
type ResearchModel struct {
	client      *api.Client
	procMgr     *internal.ProcessManager
	width       int
	height      int
	agents      []ResearchAgent
	activeAgent int
	readers     map[string]*internal.JSONLReader
	entries     map[string][]api.ResearchEntry
	scores      map[string][]float64
	bestScore   map[string]float64
	err         error
	// Detail view
	showDetail  bool
	detailIdx   int
	detailScroll int
	// Per-round score breakdown
	roundScores map[string]map[string][]float64 // agent -> round -> scores
}

func NewResearch(client *api.Client, procMgr *internal.ProcessManager, dataDir string) ResearchModel {
	agents := []ResearchAgent{
		{
			Name:    "Gemini Researcher",
			Key:     "g",
			LogPath: dataDir + "/gemini_research_log.jsonl",
			Command: "python",
			Args:    []string{"gemini_researcher.py"},
			Rune:    "ᚷ",
			Desc:    "Uses Google Gemini API to propose STRUCTURAL algorithmic changes",
			HowTo: `### Gemini Researcher
Generates prediction function code using Gemini AI. Each iteration:
1. **Analyzes** past experiments and knowledge base
2. **Proposes** a structural change (not parameter tweaks)
3. **Generates** complete prediction function code
4. **Backtests** against R2-R5 ground truth
5. **Logs** results and updates knowledge if improved

Set **GOOGLE_API_KEY** in .env to enable.`,
		},
		{
			Name:    "ADK Agent",
			Key:     "a",
			LogPath: dataDir + "/adk_research_log.jsonl",
			Command: "python",
			Args:    []string{"-m", "research_agent.run"},
			Rune:    "ᚨ",
			Desc:    "Google ADK framework agent for systematic algorithm exploration",
			HowTo: `### ADK Research Agent
A Google Agent Development Kit (ADK) agent that systematically explores:
- **Calibration models** and feature engineering
- **Blending strategies** for prior/empirical data
- **Structural zero** handling (ocean, mountains, inland ports)
- **Global multiplier** tuning strategies
- **Smoothing** and floor policies

Uses tool-calling patterns for structured experimentation.`,
		},
		{
			Name:    "Multi Researcher",
			Key:     "m",
			LogPath: dataDir + "/multi_research_log.jsonl",
			Command: "python",
			Args:    []string{"multi_researcher.py"},
			Rune:    "ᛖ",
			Desc:    "Gemini 3.1 Pro + Flash researcher",
			HowTo: `### Multi-Model Researcher
Orchestrates Gemini models with different strengths:
1. **Gemini 3 Flash** analyzes experiment log and picks direction (fast, 2-5s)
2. **Gemini 3.1 Pro** writes prediction function code (10-30s)
3. **Gemini 3 Flash** extracts clean code from response
4. **Backtest** evaluates against ground truth (3s)
5. Repeat

Requires GOOGLE_API_KEY in .env.`,
		},
	}

	readers := make(map[string]*internal.JSONLReader)
	for _, a := range agents {
		readers[a.Key] = internal.NewJSONLReader(a.LogPath)
	}

	return ResearchModel{
		client:      client,
		procMgr:     procMgr,
		agents:      agents,
		readers:     readers,
		entries:     make(map[string][]api.ResearchEntry),
		scores:      make(map[string][]float64),
		bestScore:   make(map[string]float64),
		roundScores: make(map[string]map[string][]float64),
	}
}

func (ResearchModel) Title() string    { return "Research" }
func (ResearchModel) ShortKey() string { return "6" }
func (ResearchModel) Rune() string     { return "ᚷ" }

func (m ResearchModel) Init() tea.Cmd {
	var cmds []tea.Cmd
	for _, a := range m.agents {
		cmds = append(cmds, m.loadAgentLog(a.Key))
	}
	cmds = append(cmds, m.tick())
	return tea.Batch(cmds...)
}

func (m ResearchModel) tick() tea.Cmd {
	return tea.Tick(2*time.Second, func(t time.Time) tea.Msg {
		return researchTickMsg(t)
	})
}

func (m ResearchModel) loadAgentLog(key string) tea.Cmd {
	reader := m.readers[key]
	return func() tea.Msg {
		raw, err := reader.ReadLast(100)
		if err != nil {
			return researchLogMsg{agent: key, err: err}
		}
		var entries []api.ResearchEntry
		for _, r := range raw {
			var e api.ResearchEntry
			if err := json.Unmarshal(r, &e); err == nil {
				e.Normalize()
				entries = append(entries, e)
			}
		}
		return researchLogMsg{agent: key, entries: entries}
	}
}

func (m ResearchModel) Update(msg tea.Msg) (ResearchModel, tea.Cmd) {
	switch msg := msg.(type) {
	case researchLogMsg:
		if msg.err == nil && msg.entries != nil {
			m.entries[msg.agent] = msg.entries
			var sc []float64
			var best float64
			rs := make(map[string][]float64)
			for _, e := range msg.entries {
				avg, ok := e.Scores["avg"]
				if ok {
					sc = append(sc, avg)
					if avg > best {
						best = avg
					}
				}
				// Collect per-round scores
				for k, v := range e.Scores {
					if k != "avg" {
						rs[k] = append(rs[k], v)
					}
				}
			}
			m.scores[msg.agent] = sc
			m.bestScore[msg.agent] = best
			m.roundScores[msg.agent] = rs
		}
		if msg.err != nil {
			m.err = msg.err
		}

	case researchTickMsg:
		a := m.agents[m.activeAgent]
		return m, tea.Batch(m.loadAgentLog(a.Key), m.tick())

	case tea.KeyMsg:
		if m.showDetail {
			switch msg.String() {
			case "esc":
				m.showDetail = false
			case "up", "k":
				if m.detailScroll > 0 {
					m.detailScroll--
				}
			case "down", "j":
				m.detailScroll++
			}
			return m, nil
		}

		switch msg.String() {
		case "g":
			m.activeAgent = 0
		case "a":
			m.activeAgent = 1
		case "m":
			if len(m.agents) > 2 {
				m.activeAgent = 2
			}
		case "s":
			a := m.agents[m.activeAgent]
			proc := m.procMgr.Get(a.Key)
			if proc == nil || proc.State != internal.ProcessRunning {
				err := m.procMgr.Start(a.Key, a.Command, a.Args, nil)
				if err != nil {
					m.err = err
				}
			}
		case "x":
			a := m.agents[m.activeAgent]
			m.procMgr.Stop(a.Key)
		case "r":
			a := m.agents[m.activeAgent]
			return m, m.loadAgentLog(a.Key)
		case "enter":
			// Show detail of last experiment
			entries := m.entries[m.agents[m.activeAgent].Key]
			if len(entries) > 0 {
				m.showDetail = true
				m.detailIdx = len(entries) - 1
				m.detailScroll = 0
			}
		case "up", "k":
			// Select previous entry for detail
			entries := m.entries[m.agents[m.activeAgent].Key]
			if len(entries) > 0 && m.detailIdx > 0 {
				m.detailIdx--
			}
		case "down", "j":
			entries := m.entries[m.agents[m.activeAgent].Key]
			if len(entries) > 0 && m.detailIdx < len(entries)-1 {
				m.detailIdx++
			}
		case "i":
			// Show agent info/how-to
			m.showDetail = true
			m.detailIdx = -1 // special: show info
			m.detailScroll = 0
		}
	}
	return m, nil
}

func (m ResearchModel) View() string {
	if m.showDetail {
		if m.detailIdx == -1 {
			return m.viewAgentInfo()
		}
		return m.viewDetail()
	}
	return m.viewMain()
}

func (m ResearchModel) viewMain() string {
	var sb strings.Builder

	title := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render("  ᚷ Research Agents — Autonomous AI Experimenters")
	sb.WriteString(title + "\n\n")

	// Agent selector tabs
	sb.WriteString("  ")
	for i, a := range m.agents {
		label := fmt.Sprintf("%s %s:%s", a.Rune, a.Key, a.Name)
		if i == m.activeAgent {
			sb.WriteString(lipgloss.NewStyle().
				Foreground(lipgloss.Color("#0a0a12")).
				Background(lipgloss.Color("#00e5ff")).
				Bold(true).
				Padding(0, 1).
				Render(label))
		} else {
			sb.WriteString(lipgloss.NewStyle().
				Foreground(lipgloss.Color("#c0c8e0")).
				Background(lipgloss.Color("#0d0d1a")).
				Padding(0, 1).
				Render(label))
		}
		sb.WriteString(" ")
	}
	sb.WriteString("\n\n")

	a := m.agents[m.activeAgent]

	// Agent description
	sb.WriteString(lipgloss.NewStyle().
		Foreground(lipgloss.Color("#3a3a5a")).
		Italic(true).
		Render("  "+a.Desc) + "\n\n")

	// Process status & controls
	proc := m.procMgr.Get(a.Key)
	if proc != nil && proc.State == internal.ProcessRunning {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00ff41")).
			Bold(true).
			Render(fmt.Sprintf("  ⚙ RUNNING (%s)", proc.Uptime().Round(time.Second))))
		sb.WriteString("  " + lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a")).
			Render("[x] stop"))
	} else {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a")).
			Render("  ● Stopped"))
		sb.WriteString("  " + lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a")).
			Render("[s] start"))
	}
	sb.WriteString("\n")

	// Stats panel
	entries := m.entries[a.Key]
	successCount := 0
	errorCount := 0
	improvedCount := 0
	for _, e := range entries {
		if e.Status == "success" {
			successCount++
			if e.Improvement > 0 {
				improvedCount++
			}
		} else {
			errorCount++
		}
	}

	statsPanel := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#00e5ff")).
		Padding(0, 1).
		Width(60)

	statsContent := fmt.Sprintf(
		"%s %s  %s %s  %s %s  %s %s",
		lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a")).Render("Total:"),
		lipgloss.NewStyle().Foreground(lipgloss.Color("#c0c8e0")).Bold(true).Render(fmt.Sprintf("%d", len(entries))),
		lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff41")).Render("✓:"),
		lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff41")).Bold(true).Render(fmt.Sprintf("%d", successCount)),
		lipgloss.NewStyle().Foreground(lipgloss.Color("#ff0033")).Render("✕:"),
		lipgloss.NewStyle().Foreground(lipgloss.Color("#ff0033")).Bold(true).Render(fmt.Sprintf("%d", errorCount)),
		lipgloss.NewStyle().Foreground(lipgloss.Color("#00e5ff")).Render("↑:"),
		lipgloss.NewStyle().Foreground(lipgloss.Color("#00e5ff")).Bold(true).Render(fmt.Sprintf("%d", improvedCount)),
	)

	// Best score
	if best, ok := m.bestScore[a.Key]; ok && best > 0 {
		statsContent += "\n" + lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00e5ff")).Bold(true).
			Render("ᛏ Best: ") + scoreColor(best).Render(fmt.Sprintf("%.3f", best))
	}

	sb.WriteString("\n" + statsPanel.Render(statsContent) + "\n")

	// Per-round score sparklines
	if rs, ok := m.roundScores[a.Key]; ok && len(rs) > 0 {
		sb.WriteString("\n")
		roundOrder := []string{"round2", "round3", "round4", "round5"}
		roundColors := map[string]lipgloss.Color{
			"round2": lipgloss.Color("#bf00ff"),
			"round3": lipgloss.Color("#00ff41"),
			"round4": lipgloss.Color("#00e5ff"),
			"round5": lipgloss.Color("#ffaa00"),
		}

		for _, rname := range roundOrder {
			vals, ok := rs[rname]
			if !ok || len(vals) < 2 {
				continue
			}
			// Mini sparkline
			last := vals[len(vals)-1]
			best := vals[0]
			for _, v := range vals {
				if v > best {
					best = v
				}
			}

			barColor := roundColors[rname]
			label := lipgloss.NewStyle().Foreground(barColor).Bold(true).
				Render(fmt.Sprintf("  %-8s", rname))
			// Mini inline sparkline (last 30 values)
			miniVals := vals
			if len(miniVals) > 30 {
				miniVals = miniVals[len(miniVals)-30:]
			}
			mini := miniSparkline(miniVals, barColor)
			lastStr := scoreColor(last).Render(fmt.Sprintf("%.1f", last))
			bestStr := lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a")).Render(fmt.Sprintf("best:%.1f", best))
			sb.WriteString(fmt.Sprintf("%s %s %s %s\n", label, mini, lastStr, bestStr))
		}
	}

	// Average score sparkline
	if sc, ok := m.scores[a.Key]; ok && len(sc) > 2 {
		spark := components.Sparkline{
			Title:  fmt.Sprintf("  %s Average Score", a.Rune),
			Values: sc,
			Width:  m.width - 14,
		}
		sb.WriteString("\n" + spark.View() + "\n")
	}

	// Experiments table
	if len(entries) > 0 {
		sb.WriteString("\n")
		hdr := lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00e5ff")).
			Bold(true).
			Underline(true)
		sb.WriteString(hdr.Render(fmt.Sprintf("  %-4s %-30s %6s %6s %8s %8s %6s", "ID", "Name", "Model", "Status", "Avg", "Δ", "Time")) + "\n")

		maxShow := 12
		if m.height > 0 {
			maxShow = m.height - 30
			if maxShow < 5 {
				maxShow = 5
			}
		}
		start := 0
		if len(entries) > maxShow {
			start = len(entries) - maxShow
		}

		for idx, e := range entries[start:] {
			globalIdx := start + idx
			name := e.Name
			if name == "" {
				name = fmt.Sprintf("Exp #%d", e.ID)
				if avg, ok := e.Scores["avg"]; ok {
					name += fmt.Sprintf(" (%.1f)", avg)
				}
			}
			if len(name) > 30 {
				name = name[:29] + "…"
			}

			statusStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff41"))
			if e.Status != "success" {
				statusStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#ff0033"))
			}

			avg := e.Scores["avg"]
			delta := e.Improvement
			deltaStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a"))
			if delta > 0 {
				deltaStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff41")).Bold(true)
			} else if delta < 0 {
				deltaStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#ff0033"))
			}

			cursor := "  "
			if globalIdx == m.detailIdx {
				cursor = "▸ "
			}

			modelStr := e.Model
			if modelStr == "" {
				modelStr = "—"
			}
			if len(modelStr) > 6 {
				modelStr = modelStr[:6]
			}

			sb.WriteString(fmt.Sprintf("%s%-4d %-30s %6s %s %s %s %6.1fs\n",
				cursor,
				e.ID,
				name,
				modelStr,
				statusStyle.Render(fmt.Sprintf("%6s", e.Status)),
				scoreColor(avg).Render(fmt.Sprintf("%8.3f", avg)),
				deltaStyle.Render(fmt.Sprintf("%+8.3f", delta)),
				e.Elapsed,
			))
		}
	}

	if m.err != nil {
		sb.WriteString("\n  " + lipgloss.NewStyle().
			Foreground(lipgloss.Color("#ff0033")).
			Render(fmt.Sprintf("Error: %v", m.err)))
	}

	sb.WriteString("\n" + lipgloss.NewStyle().
		Foreground(lipgloss.Color("#3a3a5a")).
		Render("  g/a/m agent  s start  x stop  ↑↓ select  Enter detail  i info  r reload"))

	return sb.String()
}

func (m ResearchModel) viewDetail() string {
	a := m.agents[m.activeAgent]
	entries := m.entries[a.Key]
	if m.detailIdx < 0 || m.detailIdx >= len(entries) {
		return "  No entry selected"
	}

	e := entries[m.detailIdx]
	var sb strings.Builder

	title := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render(fmt.Sprintf("  ᚷ Experiment #%d — %s", e.ID, a.Name))
	sb.WriteString(title + "\n\n")

	// Metadata panel
	metaStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#00e5ff")).
		Padding(0, 1)

	label := lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a"))
	value := lipgloss.NewStyle().Foreground(lipgloss.Color("#c0c8e0")).Bold(true)

	timeStr := fmt.Sprintf("%.1fs", e.Elapsed)
	if e.Timings != nil {
		timeStr = fmt.Sprintf("%.1fs (analysis:%.1fs code:%.1fs backtest:%.1fs)",
			e.Timings.Total, e.Timings.Analysis, e.Timings.Code, e.Timings.Backtest)
	}

	modelStr := e.Model
	if modelStr == "" {
		modelStr = "—"
	}

	meta := fmt.Sprintf(
		"%s %s\n%s %s\n%s %s\n%s %s\n%s %s",
		label.Render("Name:"), value.Render(e.Name),
		label.Render("Model:"), value.Render(modelStr),
		label.Render("Status:"), statusColor(e.Status).Render(e.Status),
		label.Render("Time:"), value.Render(timeStr),
		label.Render("Timestamp:"), value.Render(e.Timestamp),
	)
	sb.WriteString(metaStyle.Render(meta) + "\n\n")

	// Scores with bar chart
	if len(e.Scores) > 0 {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00e5ff")).
			Bold(true).
			Render("  ᚠ Scores") + "\n")

		rounds := []string{"round2", "round3", "round4", "round5", "avg"}
		for _, rname := range rounds {
			if v, ok := e.Scores[rname]; ok {
				barW := 25
				barLen := int(v / 100 * float64(barW))
				if barLen < 0 {
					barLen = 0
				}
				bar := lipgloss.NewStyle().Foreground(lipgloss.Color("#00e5ff")).Render(strings.Repeat("█", barLen))
				rest := lipgloss.NewStyle().Foreground(lipgloss.Color("#0d0d1a")).Render(strings.Repeat("░", barW-barLen))
				nameStr := lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a")).Width(10).Render(rname)
				sb.WriteString(fmt.Sprintf("  %s %s%s %s\n", nameStr, bar, rest, scoreColor(v).Render(fmt.Sprintf("%.2f", v))))
			}
		}

		if e.Improvement != 0 {
			deltaStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a"))
			if e.Improvement > 0 {
				deltaStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff41")).Bold(true)
			} else {
				deltaStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#ff0033"))
			}
			sb.WriteString("\n  " + deltaStyle.Render(fmt.Sprintf("Improvement: %+.3f", e.Improvement)) + "\n")
		}
	}

	// Hypothesis
	if e.Hypothesis != "" {
		sb.WriteString("\n" + lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00e5ff")).
			Bold(true).
			Render("  ᚨ Hypothesis") + "\n")
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#c0c8e0")).
			Width(m.width - 6).
			Render("  " + e.Hypothesis) + "\n")
	}

	// Error
	if e.Error != "" {
		sb.WriteString("\n" + lipgloss.NewStyle().
			Foreground(lipgloss.Color("#ff0033")).
			Bold(true).
			Render("  ✕ Error") + "\n")
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#ff0033")).
			Render("  "+e.Error) + "\n")
	}

	// Code (with syntax highlighting via glamour)
	if e.Code != "" {
		sb.WriteString("\n" + lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00e5ff")).
			Bold(true).
			Render("  ᚲ Generated Code") + "\n\n")

		// Render code as markdown code block via glamour
		md := "```python\n" + e.Code + "\n```"
		rendered, err := glamour.Render(md, "dark")
		if err != nil {
			// Fallback: plain code
			lines := strings.Split(e.Code, "\n")
			scroll := m.detailScroll
			maxLines := m.height - 30
			if maxLines < 10 {
				maxLines = 20
			}
			if scroll > len(lines)-maxLines {
				scroll = len(lines) - maxLines
			}
			if scroll < 0 {
				scroll = 0
			}
			end := scroll + maxLines
			if end > len(lines) {
				end = len(lines)
			}
			for _, line := range lines[scroll:end] {
				sb.WriteString(lipgloss.NewStyle().
					Foreground(lipgloss.Color("#c0c8e0")).
					Background(lipgloss.Color("#06060e")).
					Render("  "+line) + "\n")
			}
		} else {
			// Apply scroll
			lines := strings.Split(rendered, "\n")
			scroll := m.detailScroll
			maxLines := m.height - 30
			if maxLines < 10 {
				maxLines = 20
			}
			if scroll > len(lines)-maxLines {
				scroll = len(lines) - maxLines
			}
			if scroll < 0 {
				scroll = 0
			}
			end := scroll + maxLines
			if end > len(lines) {
				end = len(lines)
			}
			for _, line := range lines[scroll:end] {
				sb.WriteString(line + "\n")
			}
		}
	}

	sb.WriteString("\n" + lipgloss.NewStyle().
		Foreground(lipgloss.Color("#3a3a5a")).
		Render("  Esc back  ↑↓ scroll code"))

	return sb.String()
}

func (m ResearchModel) viewAgentInfo() string {
	a := m.agents[m.activeAgent]

	var sb strings.Builder
	title := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render(fmt.Sprintf("  %s %s — How It Works", a.Rune, a.Name))
	sb.WriteString(title + "\n\n")

	// Render the HowTo markdown
	rendered, err := glamour.Render(a.HowTo, "dark")
	if err != nil {
		sb.WriteString("  " + a.HowTo)
	} else {
		sb.WriteString(rendered)
	}

	// Process output if running
	proc := m.procMgr.Get(a.Key)
	if proc != nil {
		sb.WriteString("\n" + lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00e5ff")).
			Bold(true).
			Render("  ⚙ Process Output") + "\n")

		lines := proc.GetOutput()
		maxLines := 15
		start := 0
		if len(lines) > maxLines {
			start = len(lines) - maxLines
		}
		for _, line := range lines[start:] {
			sb.WriteString("  " + lipgloss.NewStyle().
				Foreground(lipgloss.Color("#c0c8e0")).
				Render(line) + "\n")
		}
	}

	sb.WriteString("\n" + lipgloss.NewStyle().
		Foreground(lipgloss.Color("#3a3a5a")).
		Render("  Esc back"))

	return sb.String()
}

func (m ResearchModel) SetSize(w, h int) ResearchModel {
	m.width = w
	m.height = h
	return m
}

func statusColor(status string) lipgloss.Style {
	if status == "success" {
		return lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff41")).Bold(true)
	}
	return lipgloss.NewStyle().Foreground(lipgloss.Color("#ff0033"))
}

// miniSparkline renders a compact inline sparkline
func miniSparkline(vals []float64, color lipgloss.Color) string {
	if len(vals) == 0 {
		return ""
	}
	blocks := []string{"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"}

	minVal, maxVal := vals[0], vals[0]
	for _, v := range vals {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	rng := maxVal - minVal
	if rng < 0.1 {
		rng = 1.0
		minVal -= 0.5
	}

	style := lipgloss.NewStyle().Foreground(color)
	var sb strings.Builder
	for _, v := range vals {
		norm := (v - minVal) / rng
		idx := int(norm * float64(len(blocks)-1))
		if idx < 0 {
			idx = 0
		}
		if idx >= len(blocks) {
			idx = len(blocks) - 1
		}
		sb.WriteString(style.Render(blocks[idx]))
	}
	return sb.String()
}
