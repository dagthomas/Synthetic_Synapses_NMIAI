package tabs

import (
	"fmt"
	"strings"

	"astar-tui/api"
	"astar-tui/internal"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type submitOutputMsg struct {
	line string
}

// SubmitModel runs the submit pipeline
type SubmitModel struct {
	client      *api.Client
	procMgr     *internal.ProcessManager
	width       int
	height      int
	variants    []submitVariant
	cursor      int
	running     bool
	output      []string
	confirmed   bool
	confirming  bool
}

type submitVariant struct {
	Name  string
	Desc  string
	Flags []string
	Rune  string
}

func NewSubmit(client *api.Client, procMgr *internal.ProcessManager) SubmitModel {
	return SubmitModel{
		client:  client,
		procMgr: procMgr,
		variants: []submitVariant{
			{Name: "Full Pipeline", Desc: "Explore → Predict → Submit", Flags: nil, Rune: "ᛟ"},
			{Name: "Static Prior", Desc: "Submit from initial state only", Flags: []string{"--static-only"}, Rune: "ᛋ"},
			{Name: "Uniform", Desc: "Submit 1/6 baseline", Flags: []string{"--uniform"}, Rune: "ᚠ"},
			{Name: "Dry Run", Desc: "Plan without API calls", Flags: []string{"--dry-run"}, Rune: "ᚱ"},
		},
	}
}

func (SubmitModel) Title() string    { return "Submit" }
func (SubmitModel) ShortKey() string { return "3" }
func (SubmitModel) Rune() string     { return "ᛏ" }

func (m SubmitModel) Init() tea.Cmd { return nil }

func (m SubmitModel) Update(msg tea.Msg) (SubmitModel, tea.Cmd) {
	switch msg := msg.(type) {
	case submitOutputMsg:
		m.output = append(m.output, msg.line)
		if len(m.output) > 200 {
			m.output = m.output[len(m.output)-200:]
		}
		// Check if process is done
		proc := m.procMgr.Get("submit")
		if proc != nil && proc.State != internal.ProcessRunning {
			m.running = false
		}

	case tea.KeyMsg:
		if m.confirming {
			switch msg.String() {
			case "y", "enter":
				m.confirming = false
				m.confirmed = true
				return m, m.runSubmit()
			case "n", "esc":
				m.confirming = false
			case "left", "right":
				// toggle in confirm dialog
			}
			return m, nil
		}

		if m.running {
			if msg.String() == "x" || msg.String() == "ctrl+c" {
				m.procMgr.Stop("submit")
				m.running = false
				m.output = append(m.output, "[Cancelled by user]")
			}
			return m, nil
		}

		switch msg.String() {
		case "up", "k":
			if m.cursor > 0 {
				m.cursor--
			}
		case "down", "j":
			if m.cursor < len(m.variants)-1 {
				m.cursor++
			}
		case "enter":
			m.confirming = true
		case "esc":
			m.output = nil
		}
	}
	return m, nil
}

func (m SubmitModel) runSubmit() tea.Cmd {
	variant := m.variants[m.cursor]
	args := []string{"submit.py"}
	args = append(args, variant.Flags...)

	m.output = []string{fmt.Sprintf("[Starting %s...]", variant.Name)}
	m.running = true

	return func() tea.Msg {
		err := m.procMgr.Start("submit", "python", args, func(line string) {
			// Lines come in via process output
		})
		if err != nil {
			return submitOutputMsg{line: fmt.Sprintf("[Error: %v]", err)}
		}
		return nil
	}
}

func (m SubmitModel) View() string {
	var sb strings.Builder

	title := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render("  ᛏ Submit Pipeline")
	sb.WriteString(title + "\n\n")

	if m.confirming {
		variant := m.variants[m.cursor]
		sb.WriteString(lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#00e5ff")).
			Padding(1, 2).
			Render(
				lipgloss.NewStyle().Foreground(lipgloss.Color("#00e5ff")).Bold(true).Render("⚔ Confirm Submission")+"\n\n"+
					lipgloss.NewStyle().Foreground(lipgloss.Color("#c0c8e0")).Render(fmt.Sprintf("Run %s?\n%s", variant.Name, variant.Desc))+"\n\n"+
					lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff41")).Render("[y]es")+
					"  "+
					lipgloss.NewStyle().Foreground(lipgloss.Color("#ff0033")).Render("[n]o"),
			))
		return sb.String()
	}

	if m.running {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00ff41")).
			Bold(true).
			Render("  ⚙ Running...") + "  " +
			lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a")).Render("(x to cancel)") + "\n\n")

		// Show process output
		proc := m.procMgr.Get("submit")
		if proc != nil {
			lines := proc.GetOutput()
			start := 0
			maxLines := m.height - 10
			if maxLines < 10 {
				maxLines = 20
			}
			if len(lines) > maxLines {
				start = len(lines) - maxLines
			}
			for _, line := range lines[start:] {
				sb.WriteString("  " + lipgloss.NewStyle().
					Foreground(lipgloss.Color("#c0c8e0")).
					Render(line) + "\n")
			}
		}
		return sb.String()
	}

	// Show variants
	for i, v := range m.variants {
		isSelected := i == m.cursor
		var line string
		if isSelected {
			line = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#00e5ff")).
				Bold(true).
				Render(fmt.Sprintf("  ▸ %s %s", v.Rune, v.Name))
			line += lipgloss.NewStyle().
				Foreground(lipgloss.Color("#3a3a5a")).
				Render(fmt.Sprintf("  — %s", v.Desc))
		} else {
			line = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#c0c8e0")).
				Render(fmt.Sprintf("    %s %s", v.Rune, v.Name))
		}
		sb.WriteString(line + "\n")
	}

	// Show last output if any
	if len(m.output) > 0 {
		sb.WriteString("\n" + lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00e5ff")).
			Bold(true).
			Render("  Last Output:") + "\n")
		start := 0
		if len(m.output) > 15 {
			start = len(m.output) - 15
		}
		for _, line := range m.output[start:] {
			sb.WriteString("  " + lipgloss.NewStyle().
				Foreground(lipgloss.Color("#c0c8e0")).
				Render(line) + "\n")
		}
	}

	sb.WriteString("\n" + lipgloss.NewStyle().
		Foreground(lipgloss.Color("#3a3a5a")).
		Render("  ↑↓ select  Enter confirm  Esc clear"))

	return sb.String()
}

func (m SubmitModel) SetSize(w, h int) SubmitModel {
	m.width = w
	m.height = h
	return m
}
