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
	"github.com/charmbracelet/lipgloss"
)

type autoloopTickMsg time.Time

type autoloopLogMsg struct {
	entries []api.AutoloopEntry
	total   int
	err     error
}

// AutoloopModel monitors the autoloop optimizer
type AutoloopModel struct {
	client     *api.Client
	procMgr    *internal.ProcessManager
	logReader  *internal.JSONLReader
	width      int
	height     int
	entries    []api.AutoloopEntry
	total      int
	bestScore  float64
	bestName   string
	scores     []float64 // for sparkline
	running    bool
	loading    bool
	err        error
	startTime  time.Time
}

func NewAutoloop(client *api.Client, procMgr *internal.ProcessManager, logPath string) AutoloopModel {
	return AutoloopModel{
		client:    client,
		procMgr:   procMgr,
		logReader: internal.NewJSONLReader(logPath),
	}
}

func (AutoloopModel) Title() string    { return "Autoiterate" }
func (AutoloopModel) ShortKey() string { return "5" }
func (AutoloopModel) Rune() string     { return "ᚹ" }

func (m AutoloopModel) Init() tea.Cmd {
	return tea.Batch(m.loadLog(), m.tick())
}

func (m AutoloopModel) tick() tea.Cmd {
	return tea.Tick(2*time.Second, func(t time.Time) tea.Msg {
		return autoloopTickMsg(t)
	})
}

func (m AutoloopModel) loadLog() tea.Cmd {
	return func() tea.Msg {
		raw, err := m.logReader.ReadLast(200)
		if err != nil {
			return autoloopLogMsg{err: err}
		}
		total, _ := m.logReader.Count()

		var entries []api.AutoloopEntry
		for _, r := range raw {
			var e api.AutoloopEntry
			if err := json.Unmarshal(r, &e); err == nil {
				entries = append(entries, e)
			}
		}
		return autoloopLogMsg{entries: entries, total: total}
	}
}

func (m AutoloopModel) loadNew() tea.Cmd {
	return func() tea.Msg {
		raw, err := m.logReader.ReadNew()
		if err != nil {
			return autoloopLogMsg{err: err}
		}
		if len(raw) == 0 {
			return autoloopLogMsg{entries: nil}
		}
		total, _ := m.logReader.Count()

		var entries []api.AutoloopEntry
		for _, r := range raw {
			var e api.AutoloopEntry
			if err := json.Unmarshal(r, &e); err == nil {
				entries = append(entries, e)
			}
		}
		return autoloopLogMsg{entries: entries, total: total}
	}
}

func (m AutoloopModel) Update(msg tea.Msg) (AutoloopModel, tea.Cmd) {
	switch msg := msg.(type) {
	case autoloopLogMsg:
		m.loading = false
		if msg.err != nil {
			m.err = msg.err
		} else if msg.entries != nil {
			if len(m.entries) == 0 {
				// Initial load
				m.entries = msg.entries
			} else {
				// Append new
				m.entries = append(m.entries, msg.entries...)
			}
			if msg.total > 0 {
				m.total = msg.total
			}
			// Keep last 200
			if len(m.entries) > 200 {
				m.entries = m.entries[len(m.entries)-200:]
			}
			// Update best score and sparkline
			m.scores = nil
			for _, e := range m.entries {
				avg := e.ScoresQuick["avg"]
				if e.ScoresFull != nil {
					if fa, ok := e.ScoresFull["avg"]; ok {
						avg = fa
					}
				}
				m.scores = append(m.scores, avg)
				if e.Accepted && avg > m.bestScore {
					m.bestScore = avg
					m.bestName = e.Name
				}
			}
		}

	case autoloopTickMsg:
		// Check if process is running
		proc := m.procMgr.Get("autoloop")
		m.running = proc != nil && proc.State == internal.ProcessRunning

		return m, tea.Batch(m.loadNew(), m.tick())

	case tea.KeyMsg:
		switch msg.String() {
		case "s":
			if !m.running {
				m.startTime = time.Now()
				err := m.procMgr.Start("autoloop", "python", []string{"autoloop_fast.py"}, nil)
				if err != nil {
					m.err = err
				} else {
					m.running = true
				}
			}
		case "x":
			if m.running {
				m.procMgr.Stop("autoloop")
				m.running = false
			}
		case "r":
			m.loading = true
			m.entries = nil
			return m, m.loadLog()
		}
	}
	return m, nil
}

func (m AutoloopModel) View() string {
	var sb strings.Builder

	title := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render("  ᚹ Autoiterate Optimizer")
	sb.WriteString(title + "\n\n")

	// Status banner
	if m.running {
		uptime := ""
		proc := m.procMgr.Get("autoloop")
		if proc != nil {
			uptime = fmt.Sprintf(" (%s)", proc.Uptime().Round(time.Second))
		}
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00ff41")).
			Bold(true).
			Render(fmt.Sprintf("  ⚙ RUNNING%s", uptime)))
	} else {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a")).
			Render("  ● Stopped"))
	}

	if m.err != nil {
		sb.WriteString("  " + lipgloss.NewStyle().
			Foreground(lipgloss.Color("#ff0033")).
			Render(fmt.Sprintf("Error: %v", m.err)))
	}
	sb.WriteString("\n")

	// Best score banner
	if m.bestScore > 0 {
		sb.WriteString("\n")
		sb.WriteString(lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#00e5ff")).
			Padding(0, 2).
			Render(
				lipgloss.NewStyle().Foreground(lipgloss.Color("#00e5ff")).Bold(true).Render("ᛏ Best Score: ")+
					scoreColor(m.bestScore).Render(fmt.Sprintf("%.3f", m.bestScore))+
					lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a")).Render(fmt.Sprintf("  (%s)", m.bestName)),
			))
		sb.WriteString("\n")
	}

	// Experiment counter
	if m.total > 0 {
		rate := ""
		if m.running && !m.startTime.IsZero() {
			elapsed := time.Since(m.startTime).Seconds()
			if elapsed > 1 {
				r := float64(len(m.entries)) / elapsed * 3600
				rate = fmt.Sprintf("  (~%.0f/hr)", r)
			}
		}
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a")).
			Render(fmt.Sprintf("\n  Experiments: %d%s", m.total, rate)))
		sb.WriteString("\n")
	}

	// Sparkline chart
	if len(m.scores) > 2 {
		spark := components.Sparkline{
			Title:  "  ᚠ Score Convergence",
			Values: m.scores,
			Width:  m.width - 12,
		}
		sb.WriteString("\n" + spark.View())
		sb.WriteString("\n")
	}

	// Last 20 experiments table
	sb.WriteString("\n")
	hdr := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Underline(true)
	sb.WriteString(hdr.Render(fmt.Sprintf("  %-5s %-35s %8s %8s %5s", "ID", "Name", "Avg", "Base", "  ✓")) + "\n")

	maxShow := 20
	if m.height > 0 {
		maxShow = m.height - 20
		if maxShow < 10 {
			maxShow = 10
		}
	}

	// Display newest first
	end := len(m.entries)
	showStart := end - maxShow
	if showStart < 0 {
		showStart = 0
	}
	for i := end - 1; i >= showStart; i-- {
		e := m.entries[i]
		avg := e.ScoresQuick["avg"]
		if e.ScoresFull != nil {
			if fa, ok := e.ScoresFull["avg"]; ok {
				avg = fa
			}
		}

		name := e.Name
		if len(name) > 35 {
			name = name[:34] + "…"
		}

		accepted := " "
		if e.Accepted {
			accepted = lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff41")).Bold(true).Render("✓")
		}

		scoreStr := scoreColor(avg).Render(fmt.Sprintf("%8.3f", avg))
		baseStr := lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a")).Render(fmt.Sprintf("%8.3f", e.BaselineAvg))

		sb.WriteString(fmt.Sprintf("  %-5d %-35s %s %s %5s\n",
			e.ID, name, scoreStr, baseStr, accepted))
	}

	sb.WriteString("\n" + lipgloss.NewStyle().
		Foreground(lipgloss.Color("#3a3a5a")).
		Render("  s start  x stop  r reload"))

	return sb.String()
}

func (m AutoloopModel) SetSize(w, h int) AutoloopModel {
	m.width = w
	m.height = h
	return m
}
