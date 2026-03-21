package components

import (
	"github.com/charmbracelet/lipgloss"
)

var (
	confirmBoxStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#00e5ff")).
			Padding(1, 2).
			Width(50).
			Align(lipgloss.Center)

	confirmTitleStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#00e5ff")).
				Bold(true)

	confirmMsgStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#c0c8e0"))

	confirmYesStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#0a0a12")).
			Background(lipgloss.Color("#00ff41")).
			Padding(0, 2)

	confirmNoStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#0a0a12")).
			Background(lipgloss.Color("#ff0033")).
			Padding(0, 2)

	confirmSelYesStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#0a0a12")).
				Background(lipgloss.Color("#00ff41")).
				Bold(true).
				Padding(0, 2)

	confirmSelNoStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#0a0a12")).
				Background(lipgloss.Color("#ff0033")).
				Bold(true).
				Padding(0, 2)
)

// ConfirmDialog shows a yes/no confirmation
type ConfirmDialog struct {
	Title    string
	Message  string
	Selected bool // true = Yes, false = No
	Visible  bool
}

// View renders the confirmation dialog
func (cd ConfirmDialog) View() string {
	if !cd.Visible {
		return ""
	}

	title := confirmTitleStyle.Render("⚔ " + cd.Title)
	msg := confirmMsgStyle.Render(cd.Message)

	var yesBtn, noBtn string
	if cd.Selected {
		yesBtn = confirmSelYesStyle.Render("▸ Yes")
		noBtn = confirmNoStyle.Render("  No")
	} else {
		yesBtn = confirmYesStyle.Render("  Yes")
		noBtn = confirmSelNoStyle.Render("▸ No")
	}

	buttons := lipgloss.JoinHorizontal(lipgloss.Center, yesBtn, "  ", noBtn)

	content := lipgloss.JoinVertical(lipgloss.Center,
		title,
		"",
		msg,
		"",
		buttons,
	)

	return confirmBoxStyle.Render(content)
}
