package components

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

var (
	statusStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#c0c8e0")).
			Background(lipgloss.Color("#06060e"))

	statusKeyStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00e5ff")).
			Background(lipgloss.Color("#06060e")).
			Bold(true)

	statusValStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#6a6a8a")).
			Background(lipgloss.Color("#06060e"))

	statusGoodStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00ff41")).
			Background(lipgloss.Color("#06060e")).
			Bold(true)

	statusWarnStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#ff0033")).
			Background(lipgloss.Color("#06060e")).
			Bold(true)

	statusSepStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#2a2a44")).
			Background(lipgloss.Color("#06060e"))

	statusGlitchStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#ff0080")).
				Background(lipgloss.Color("#06060e"))

	statusRuneStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#bf00ff")).
			Background(lipgloss.Color("#06060e"))
)

// StatusBar renders the bottom status bar
type StatusBar struct {
	Width        int
	BudgetUsed   int
	BudgetMax    int
	RunningProcs []string
	ActiveRound  string
	ConnStatus   string // "connected", "disconnected", "loading"
	PaneCount    int    // number of visible panes
}

// View renders the status bar
func (sb StatusBar) View() string {
	var sections []string

	// Left glitch edge
	sections = append(sections, statusGlitchStyle.Render("▓▒░ "))

	// Connection status
	switch sb.ConnStatus {
	case "connected":
		sections = append(sections, statusGoodStyle.Render("◈ ONLINE"))
	case "loading":
		sections = append(sections, statusRuneStyle.Render("◌ SYNC.."))
	default:
		sections = append(sections, statusWarnStyle.Render("✕ OFFLINE"))
	}

	sep := statusSepStyle.Render(" ᛫ ")

	// Budget
	budgetStr := fmt.Sprintf("ᚠ %d/%d", sb.BudgetUsed, sb.BudgetMax)
	if sb.BudgetMax-sb.BudgetUsed <= 10 {
		sections = append(sections, sep+statusWarnStyle.Render(budgetStr))
	} else {
		sections = append(sections, sep+statusKeyStyle.Render(budgetStr))
	}

	// Active round
	if sb.ActiveRound != "" {
		sections = append(sections, sep+statusRuneStyle.Render("ᛞ R:"+sb.ActiveRound))
	}

	// Running processes
	if len(sb.RunningProcs) > 0 {
		procStr := fmt.Sprintf("⚙ %d", len(sb.RunningProcs))
		sections = append(sections, sep+statusGoodStyle.Render(procStr))
	}

	// Pane count
	if sb.PaneCount > 1 {
		paneStr := fmt.Sprintf("⊞ %d", sb.PaneCount)
		sections = append(sections, sep+statusValStyle.Render(paneStr))
	}

	left := strings.Join(sections, "")

	// Right side: keybindings + glitch edge
	right := statusKeyStyle.Render("q") + statusValStyle.Render(":quit") +
		statusSepStyle.Render(" ") +
		statusKeyStyle.Render("?") + statusValStyle.Render(":help") +
		statusSepStyle.Render(" ") +
		statusKeyStyle.Render("0-9") + statusValStyle.Render(":tabs") +
		statusGlitchStyle.Render(" ░▒▓")

	// Calculate padding
	leftW := lipgloss.Width(left)
	rightW := lipgloss.Width(right)
	padding := sb.Width - leftW - rightW
	if padding < 1 {
		padding = 1
	}

	bar := left + statusStyle.Render(strings.Repeat(" ", padding)) + right

	return bar
}
