package components

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// Checkbox represents a toggle checkbox
type Checkbox struct {
	Label   string
	Checked bool
	Focused bool
}

// View renders the checkbox
func (cb Checkbox) View() string {
	check := "☐"
	color := lipgloss.Color("#3a3a5a")
	if cb.Checked {
		check = "☑"
		color = lipgloss.Color("#00ff41")
	}
	if cb.Focused {
		color = lipgloss.Color("#00e5ff")
		if cb.Checked {
			check = "☑"
		} else {
			check = "☐"
		}
	}
	style := lipgloss.NewStyle().Foreground(color)
	label := lipgloss.NewStyle().Foreground(lipgloss.Color("#c0c8e0"))
	if cb.Focused {
		label = label.Bold(true)
		return style.Render("▸ "+check) + " " + label.Render(cb.Label)
	}
	return style.Render("  "+check) + " " + label.Render(cb.Label)
}

// Toggle flips the checkbox
func (cb *Checkbox) Toggle() {
	cb.Checked = !cb.Checked
}

// TextInput is a simple text input field
type TextInput struct {
	Label       string
	Value       string
	Placeholder string
	Focused     bool
	Masked      bool // password-style masking
	CursorPos   int
	MaxWidth    int
}

// View renders the text input
func (ti TextInput) View() string {
	labelStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#00e5ff")).Bold(true)
	inputBg := lipgloss.Color("#0d0d1a")
	borderColor := lipgloss.Color("#3a3a5a")
	if ti.Focused {
		borderColor = lipgloss.Color("#00e5ff")
	}

	// Label
	label := labelStyle.Render(ti.Label + ":")

	// Value display
	displayVal := ti.Value
	if ti.Masked && displayVal != "" {
		displayVal = strings.Repeat("•", len(displayVal))
	}

	width := ti.MaxWidth
	if width <= 0 {
		width = 40
	}

	if displayVal == "" && !ti.Focused {
		displayVal = ti.Placeholder
	}

	// Pad to width
	if len(displayVal) < width {
		displayVal += strings.Repeat(" ", width-len(displayVal))
	}
	if len(displayVal) > width {
		displayVal = displayVal[:width]
	}

	inputStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#c0c8e0")).
		Background(inputBg).
		Padding(0, 1)

	if ti.Value == "" && !ti.Focused {
		inputStyle = inputStyle.Foreground(lipgloss.Color("#3a3a5a"))
	}

	border := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(borderColor)

	// Cursor
	if ti.Focused {
		// Insert cursor at position
		pos := ti.CursorPos
		if pos > len(displayVal) {
			pos = len(displayVal)
		}
		before := displayVal[:pos]
		cursor := "█"
		after := ""
		if pos < len(displayVal) {
			after = displayVal[pos+1:]
		}
		displayVal = before + cursor + after
	}

	input := border.Render(inputStyle.Render(displayVal))

	return fmt.Sprintf("  %s\n  %s", label, input)
}

// InsertChar adds a character at cursor position
func (ti *TextInput) InsertChar(ch rune) {
	if ti.CursorPos > len(ti.Value) {
		ti.CursorPos = len(ti.Value)
	}
	ti.Value = ti.Value[:ti.CursorPos] + string(ch) + ti.Value[ti.CursorPos:]
	ti.CursorPos++
}

// Backspace deletes character before cursor
func (ti *TextInput) Backspace() {
	if ti.CursorPos > 0 && len(ti.Value) > 0 {
		ti.Value = ti.Value[:ti.CursorPos-1] + ti.Value[ti.CursorPos:]
		ti.CursorPos--
	}
}

// Delete deletes character at cursor
func (ti *TextInput) Delete() {
	if ti.CursorPos < len(ti.Value) {
		ti.Value = ti.Value[:ti.CursorPos] + ti.Value[ti.CursorPos+1:]
	}
}

// MoveCursorLeft moves cursor left
func (ti *TextInput) MoveCursorLeft() {
	if ti.CursorPos > 0 {
		ti.CursorPos--
	}
}

// MoveCursorRight moves cursor right
func (ti *TextInput) MoveCursorRight() {
	if ti.CursorPos < len(ti.Value) {
		ti.CursorPos++
	}
}

// SelectGroup manages a group of checkboxes/inputs
type SelectGroup struct {
	Title    string
	Items    []string
	Selected int
}

// View renders the select group
func (sg SelectGroup) View() string {
	var sb strings.Builder
	if sg.Title != "" {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00e5ff")).
			Bold(true).
			Render("  "+sg.Title) + "\n")
	}
	for i, item := range sg.Items {
		if i == sg.Selected {
			sb.WriteString(lipgloss.NewStyle().
				Foreground(lipgloss.Color("#00e5ff")).
				Bold(true).
				Render(fmt.Sprintf("  ▸ %s", item)) + "\n")
		} else {
			sb.WriteString(lipgloss.NewStyle().
				Foreground(lipgloss.Color("#c0c8e0")).
				Render(fmt.Sprintf("    %s", item)) + "\n")
		}
	}
	return sb.String()
}
