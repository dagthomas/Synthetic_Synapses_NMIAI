package tabs

import (
	"fmt"
	"strings"

	"astar-tui/api"
	"astar-tui/internal"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// BacktestModel runs backtests and compares scores
type BacktestModel struct {
	client  *api.Client
	procMgr *internal.ProcessManager
	width   int
	height  int
	running bool
	output  []string
	results []backtestResult
}

type backtestResult struct {
	Name   string
	Round2 float64
	Round3 float64
	Round4 float64
	Round5 float64
	Avg    float64
}

func NewBacktest(client *api.Client, procMgr *internal.ProcessManager) BacktestModel {
	return BacktestModel{
		client:  client,
		procMgr: procMgr,
	}
}

func (BacktestModel) Title() string    { return "Backtest" }
func (BacktestModel) ShortKey() string { return "7" }
func (BacktestModel) Rune() string     { return "ᛞ" }

func (m BacktestModel) Init() tea.Cmd { return nil }

func (m BacktestModel) Update(msg tea.Msg) (BacktestModel, tea.Cmd) {
	switch msg := msg.(type) {
	case backtestOutputMsg:
		m.output = append(m.output, msg.line)
		if len(m.output) > 100 {
			m.output = m.output[len(m.output)-100:]
		}
		proc := m.procMgr.Get("backtest")
		if proc != nil && proc.State != internal.ProcessRunning {
			m.running = false
		}

	case tea.KeyMsg:
		switch msg.String() {
		case "enter":
			if !m.running {
				return m, m.runBacktest()
			}
		case "x":
			if m.running {
				m.procMgr.Stop("backtest")
				m.running = false
			}
		}
	}
	return m, nil
}

type backtestOutputMsg struct {
	line string
}

func (m *BacktestModel) runBacktest() tea.Cmd {
	m.running = true
	m.output = []string{"[Running backtest...]"}
	return func() tea.Msg {
		script := `from autoexperiment import BacktestHarness; import json; bh = BacktestHarness(); r = bh.run_baseline(); print(json.dumps(r))`
		err := m.procMgr.Start("backtest", "python", []string{"-c", script}, nil)
		if err != nil {
			return backtestOutputMsg{line: fmt.Sprintf("[Error: %v]", err)}
		}
		return nil
	}
}

func (m BacktestModel) View() string {
	var sb strings.Builder

	title := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render("  ᛞ Backtest Runner")
	sb.WriteString(title + "\n\n")

	// ASCII art shield
	sb.WriteString(lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Render(`     ╔═══════════╗
     ║  ᛞ  ᛏ  ᛋ  ║
     ║  BACKTEST  ║
     ╚═══════════╝`) + "\n\n")

	if m.running {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00ff41")).
			Bold(true).
			Render("  ⚙ Running backtest...") + "\n\n")
	} else {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a")).
			Render("  Press Enter to run backtest") + "\n\n")
	}

	// Results table
	if len(m.results) > 0 {
		hdr := lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00e5ff")).
			Bold(true).
			Underline(true)
		sb.WriteString(hdr.Render(fmt.Sprintf("  %-20s %8s %8s %8s %8s %8s", "Name", "R2", "R3", "R4", "R5", "Avg")) + "\n")

		for _, r := range m.results {
			sb.WriteString(fmt.Sprintf("  %-20s %s %s %s %s %s\n",
				r.Name,
				scoreColor(r.Round2).Render(fmt.Sprintf("%8.2f", r.Round2)),
				scoreColor(r.Round3).Render(fmt.Sprintf("%8.2f", r.Round3)),
				scoreColor(r.Round4).Render(fmt.Sprintf("%8.2f", r.Round4)),
				scoreColor(r.Round5).Render(fmt.Sprintf("%8.2f", r.Round5)),
				scoreColor(r.Avg).Render(fmt.Sprintf("%8.2f", r.Avg)),
			))
		}
		sb.WriteString("\n")
	}

	// Output
	if len(m.output) > 0 {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00e5ff")).
			Bold(true).
			Render("  Output:") + "\n")
		maxLines := m.height - 25
		if maxLines < 5 {
			maxLines = 15
		}
		start := 0
		if len(m.output) > maxLines {
			start = len(m.output) - maxLines
		}
		for _, line := range m.output[start:] {
			sb.WriteString("  " + lipgloss.NewStyle().
				Foreground(lipgloss.Color("#c0c8e0")).
				Render(line) + "\n")
		}
	}

	sb.WriteString("\n" + lipgloss.NewStyle().
		Foreground(lipgloss.Color("#3a3a5a")).
		Render("  Enter run  x cancel"))

	return sb.String()
}

func (m BacktestModel) SetSize(w, h int) BacktestModel {
	m.width = w
	m.height = h
	return m
}
