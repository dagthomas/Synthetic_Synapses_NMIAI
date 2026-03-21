package components

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
)

var (
	activeTabStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#0a0a12")).
			Background(lipgloss.Color("#00e5ff")).
			Bold(true).
			Padding(0, 1)

	inactiveTabStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#3a3a5a")).
				Background(lipgloss.Color("#06060e")).
				Padding(0, 1)

	paneTabStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#ff2d95")).
			Background(lipgloss.Color("#0d0d1a")).
			Padding(0, 1)

	tabBarStyle = lipgloss.NewStyle().
			Background(lipgloss.Color("#06060e"))

	runeDecor = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00e5ff")).
			Background(lipgloss.Color("#06060e")).
			Bold(true)

	glitchDecor = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#ff0080")).
			Background(lipgloss.Color("#06060e"))

	tabSepStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#0a0a12")).
			Background(lipgloss.Color("#06060e"))
)

// TabInfo describes a tab
type TabInfo struct {
	Key   string // e.g. "1"
	Title string // e.g. "Dashboard"
	Rune  string // e.g. "ᛟ"
}

// TabBar renders the horizontal tab navigation
type TabBar struct {
	Tabs      []TabInfo
	ActiveIdx int
	PaneTabs  []int // tab indices visible in other (non-focused) panes
	Width     int
}

// NewTabBar creates a tab bar
func NewTabBar(tabs []TabInfo) TabBar {
	return TabBar{Tabs: tabs}
}

func (tb TabBar) isPaneTab(idx int) bool {
	for _, t := range tb.PaneTabs {
		if t == idx {
			return true
		}
	}
	return false
}

// View renders the tab bar
func (tb TabBar) View() string {
	var parts []string

	// Left glitch decoration
	parts = append(parts, glitchDecor.Render("▓▒░"))
	parts = append(parts, runeDecor.Render(" ᛟ "))

	for i, tab := range tb.Tabs {
		label := tab.Rune + " " + tab.Key + ":" + tab.Title
		if i == tb.ActiveIdx {
			parts = append(parts, activeTabStyle.Render(label))
		} else if tb.isPaneTab(i) {
			parts = append(parts, paneTabStyle.Render(label))
		} else {
			parts = append(parts, inactiveTabStyle.Render(label))
		}
		if i < len(tb.Tabs)-1 {
			parts = append(parts, tabSepStyle.Render("│"))
		}
	}

	parts = append(parts, runeDecor.Render(" ᛟ "))
	parts = append(parts, glitchDecor.Render("░▒▓"))

	bar := lipgloss.JoinHorizontal(lipgloss.Top, parts...)

	// Pad to full width
	if tb.Width > 0 {
		barWidth := lipgloss.Width(bar)
		if barWidth < tb.Width {
			leftPad := (tb.Width - barWidth) / 2
			rightPad := tb.Width - barWidth - leftPad
			bar = tabBarStyle.Render(strings.Repeat(" ", leftPad)) + bar + tabBarStyle.Render(strings.Repeat(" ", rightPad))
		}
	}

	return bar
}
