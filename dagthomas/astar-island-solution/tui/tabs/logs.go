package tabs

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"astar-tui/internal"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type logsTickMsg time.Time

type logsDataMsg struct {
	lines []string
	err   error
}

// LogSource describes a log file
type LogSource struct {
	Name string
	Path string
	Key  string
}

// LogsModel is the log viewer tab
type LogsModel struct {
	width       int
	height      int
	sources     []LogSource
	activeSource int
	readers     map[string]*internal.JSONLReader
	lines       []string
	tailMode    bool
	offset      int
	filter      string
	filtering   bool
	filterBuf   string
	err         error
}

func NewLogs(dataDir string) LogsModel {
	sources := []LogSource{
		{Name: "Autoloop", Path: dataDir + "/autoloop_log.jsonl", Key: "autoloop"},
		{Name: "ADK Research", Path: dataDir + "/adk_research_log.jsonl", Key: "adk"},
		{Name: "Gemini Research", Path: dataDir + "/gemini_research_log.jsonl", Key: "gemini"},
		{Name: "ADK History", Path: dataDir + "/adk_agent_history.jsonl", Key: "history"},
		{Name: "Multi Research", Path: dataDir + "/multi_research_log.jsonl", Key: "multi"},
	}

	readers := make(map[string]*internal.JSONLReader)
	for _, s := range sources {
		readers[s.Key] = internal.NewJSONLReader(s.Path)
	}

	return LogsModel{
		sources:  sources,
		readers:  readers,
		tailMode: true,
	}
}

func (LogsModel) Title() string    { return "Logs" }
func (LogsModel) ShortKey() string { return "8" }
func (LogsModel) Rune() string     { return "ᛚ" }

func (m LogsModel) Init() tea.Cmd {
	return tea.Batch(m.loadLog(), m.tick())
}

func (m LogsModel) tick() tea.Cmd {
	return tea.Tick(time.Second, func(t time.Time) tea.Msg {
		return logsTickMsg(t)
	})
}

func (m LogsModel) loadLog() tea.Cmd {
	source := m.sources[m.activeSource]
	reader := m.readers[source.Key]

	return func() tea.Msg {
		raw, err := reader.ReadLast(200)
		if err != nil {
			return logsDataMsg{err: err}
		}

		var lines []string
		for _, r := range raw {
			lines = append(lines, formatJSONLine(r))
		}
		return logsDataMsg{lines: lines}
	}
}

func (m LogsModel) loadNew() tea.Cmd {
	source := m.sources[m.activeSource]
	reader := m.readers[source.Key]

	return func() tea.Msg {
		raw, err := reader.ReadNew()
		if err != nil {
			return logsDataMsg{err: err}
		}
		if len(raw) == 0 {
			return logsDataMsg{}
		}

		var lines []string
		for _, r := range raw {
			lines = append(lines, formatJSONLine(r))
		}
		return logsDataMsg{lines: lines}
	}
}

func (m LogsModel) Update(msg tea.Msg) (LogsModel, tea.Cmd) {
	switch msg := msg.(type) {
	case logsDataMsg:
		if msg.err != nil {
			m.err = msg.err
		} else if msg.lines != nil {
			if len(m.lines) == 0 {
				m.lines = msg.lines
			} else {
				m.lines = append(m.lines, msg.lines...)
			}
			if len(m.lines) > 500 {
				m.lines = m.lines[len(m.lines)-500:]
			}
			if m.tailMode {
				m.offset = max(0, len(m.lines)-m.viewableLines())
			}
		}

	case logsTickMsg:
		if m.tailMode {
			return m, tea.Batch(m.loadNew(), m.tick())
		}
		return m, m.tick()

	case tea.KeyMsg:
		if m.filtering {
			switch msg.String() {
			case "enter":
				m.filter = m.filterBuf
				m.filtering = false
			case "esc":
				m.filtering = false
				m.filterBuf = ""
			case "backspace":
				if len(m.filterBuf) > 0 {
					m.filterBuf = m.filterBuf[:len(m.filterBuf)-1]
				}
			default:
				if len(msg.String()) == 1 {
					m.filterBuf += msg.String()
				}
			}
			return m, nil
		}

		switch msg.String() {
		case "f":
			// Cycle through sources
			m.activeSource = (m.activeSource + 1) % len(m.sources)
			m.lines = nil
			m.offset = 0
			return m, m.loadLog()
		case "t":
			m.tailMode = !m.tailMode
		case "/":
			m.filtering = true
			m.filterBuf = m.filter
		case "up", "k":
			if m.offset > 0 {
				m.offset--
				m.tailMode = false
			}
		case "down", "j":
			if m.offset < len(m.lines)-1 {
				m.offset++
			}
		case "pgup":
			m.offset -= m.viewableLines()
			if m.offset < 0 {
				m.offset = 0
			}
			m.tailMode = false
		case "pgdown":
			m.offset += m.viewableLines()
			maxOffset := len(m.lines) - m.viewableLines()
			if maxOffset < 0 {
				maxOffset = 0
			}
			if m.offset > maxOffset {
				m.offset = maxOffset
			}
		case "G":
			m.tailMode = true
			m.offset = max(0, len(m.lines)-m.viewableLines())
		case "c":
			m.filter = ""
		}
	}
	return m, nil
}

func (m LogsModel) viewableLines() int {
	h := m.height - 8
	if h < 10 {
		h = 30
	}
	return h
}

func (m LogsModel) View() string {
	var sb strings.Builder

	title := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render("  ᛚ Log Viewer")
	sb.WriteString(title + "\n")

	// Source selector
	sb.WriteString("  ")
	for i, s := range m.sources {
		label := s.Name
		if i == m.activeSource {
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
	sb.WriteString("\n")

	// Mode indicator
	modeStr := "browse"
	modeColor := lipgloss.Color("#3a3a5a")
	if m.tailMode {
		modeStr = "tail ◉"
		modeColor = lipgloss.Color("#00ff41")
	}
	sb.WriteString("  " + lipgloss.NewStyle().Foreground(modeColor).Render(modeStr))

	if m.filter != "" {
		sb.WriteString("  " + lipgloss.NewStyle().
			Foreground(lipgloss.Color("#bf00ff")).
			Render(fmt.Sprintf("filter: %q", m.filter)))
	}
	if m.filtering {
		sb.WriteString("  " + lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00e5ff")).
			Render(fmt.Sprintf("/ %s█", m.filterBuf)))
	}

	sb.WriteString(fmt.Sprintf("  [%d lines]", len(m.lines)))
	sb.WriteString("\n\n")

	if m.err != nil {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#ff0033")).
			Render(fmt.Sprintf("  Error: %v\n", m.err)))
	}

	// Filtered lines
	var filtered []string
	for _, line := range m.lines {
		if m.filter == "" || strings.Contains(strings.ToLower(line), strings.ToLower(m.filter)) {
			filtered = append(filtered, line)
		}
	}

	// Display viewport
	viewH := m.viewableLines()
	start := m.offset
	if start > len(filtered) {
		start = len(filtered)
	}
	end := start + viewH
	if end > len(filtered) {
		end = len(filtered)
	}

	for _, line := range filtered[start:end] {
		// Colorize based on content
		style := lipgloss.NewStyle().Foreground(lipgloss.Color("#c0c8e0"))
		if strings.Contains(line, "error") || strings.Contains(line, "Error") || strings.Contains(line, "failed") || strings.Contains(line, "ERR:") {
			style = lipgloss.NewStyle().Foreground(lipgloss.Color("#ff0033"))
		} else if strings.Contains(line, "accepted") || strings.Contains(line, "success") || strings.Contains(line, "✓") || strings.Contains(line, "NEW BEST") {
			style = lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff41"))
		} else if strings.Contains(line, "baseline") {
			style = lipgloss.NewStyle().Foreground(lipgloss.Color("#00e5ff"))
		}
		sb.WriteString("  " + style.Render(truncate(line, m.width-4)) + "\n")
	}

	sb.WriteString("\n" + lipgloss.NewStyle().
		Foreground(lipgloss.Color("#3a3a5a")).
		Render("  f switch file  t tail/browse  / filter  c clear filter  ↑↓ scroll  G tail"))

	return sb.String()
}

func (m LogsModel) SetSize(w, h int) LogsModel {
	m.width = w
	m.height = h
	return m
}

// formatJSONLine formats a JSONL entry for display
func formatJSONLine(raw json.RawMessage) string {
	var generic map[string]any
	if err := json.Unmarshal(raw, &generic); err != nil {
		return string(raw)
	}

	// Extract key fields
	var parts []string

	if id, ok := generic["id"]; ok {
		parts = append(parts, fmt.Sprintf("#%v", id))
	}
	if ts, ok := generic["timestamp"].(string); ok {
		if len(ts) > 19 {
			ts = ts[11:19] // just time
		}
		parts = append(parts, ts)
	}
	if name, ok := generic["name"].(string); ok && name != "" {
		if len(name) > 40 {
			name = name[:39] + "…"
		}
		parts = append(parts, name)
	}
	if status, ok := generic["status"].(string); ok {
		parts = append(parts, "["+status+"]")
	}
	if accepted, ok := generic["accepted"].(bool); ok {
		if accepted {
			parts = append(parts, "✓")
		}
	}
	if scores, ok := generic["scores_quick"].(map[string]any); ok {
		if avg, ok := scores["avg"]; ok {
			parts = append(parts, fmt.Sprintf("avg=%.3f", avg))
		}
	}
	if scores, ok := generic["scores"].(map[string]any); ok {
		if avg, ok := scores["avg"]; ok {
			parts = append(parts, fmt.Sprintf("avg=%.3f", avg))
		}
	}
	if model, ok := generic["model"].(string); ok && model != "" {
		parts = append(parts, "["+model+"]")
	}
	if imp, ok := generic["improvement"].(float64); ok {
		parts = append(parts, fmt.Sprintf("Δ=%+.3f", imp))
	}
	if timings, ok := generic["timings"].(map[string]any); ok {
		if total, ok := timings["total"]; ok {
			parts = append(parts, fmt.Sprintf("%.0fs", total))
		}
	}
	if errMsg, ok := generic["error"].(string); ok && errMsg != "" {
		if len(errMsg) > 50 {
			errMsg = errMsg[:49] + "…"
		}
		parts = append(parts, "ERR: "+errMsg)
	}

	if len(parts) == 0 {
		return string(raw)
	}
	return strings.Join(parts, "  ")
}

func truncate(s string, maxLen int) string {
	if maxLen <= 0 {
		return s
	}
	// Count visible chars (rough approximation)
	if len(s) > maxLen*2 { // rough cut for very long strings
		return s[:maxLen*2]
	}
	return s
}
