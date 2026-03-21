package tabs

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"astar-tui/components"
	"astar-tui/internal"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type settingsSavedMsg struct{ err error }

// SettingsModel provides configuration UI
type SettingsModel struct {
	width     int
	height    int
	envConfig *internal.EnvConfig
	baseDir   string

	// Focus state
	focusIdx  int
	maxItems  int
	inputMode bool // true when editing a text field

	// Text inputs
	astarToken   components.TextInput
	googleAPIKey components.TextInput

	// Checkboxes
	autoRefresh  components.Checkbox
	tailLogs     components.Checkbox
	showViewport components.Checkbox
	darkTheme    components.Checkbox
	notifications components.Checkbox
	claudeCLI    components.Checkbox

	// Status
	saved     bool
	saveErr   error
}

func NewSettings(env *internal.EnvConfig, baseDir string) SettingsModel {
	token := ""
	gkey := ""
	if env != nil {
		token = env.AstarToken
		gkey = env.GoogleAPIKey
	}

	return SettingsModel{
		envConfig: env,
		baseDir:   baseDir,
		maxItems:  8,
		astarToken: components.TextInput{
			Label:       "ASTAR_TOKEN",
			Value:       token,
			Placeholder: "paste your API token here",
			Masked:      true,
			MaxWidth:    50,
		},
		googleAPIKey: components.TextInput{
			Label:       "GOOGLE_API_KEY",
			Value:       gkey,
			Placeholder: "paste Gemini API key here",
			Masked:      true,
			MaxWidth:    50,
		},
		autoRefresh:  components.Checkbox{Label: "Auto-refresh dashboard (10s)", Checked: true},
		tailLogs:     components.Checkbox{Label: "Auto-tail log files", Checked: true},
		showViewport: components.Checkbox{Label: "Show viewport overlay in Explorer", Checked: true},
		darkTheme:    components.Checkbox{Label: "Dark Norse theme (always on)", Checked: true},
		notifications: components.Checkbox{Label: "Show notifications on score improvement", Checked: true},
		claudeCLI:    components.Checkbox{Label: "Use Claude Code CLI for Multi Researcher", Checked: true},
	}
}

func (SettingsModel) Title() string    { return "Settings" }
func (SettingsModel) ShortKey() string { return "9" }
func (SettingsModel) Rune() string     { return "ᛁ" }

func (m SettingsModel) Init() tea.Cmd { return nil }

func (m SettingsModel) Update(msg tea.Msg) (SettingsModel, tea.Cmd) {
	switch msg := msg.(type) {
	case settingsSavedMsg:
		m.saveErr = msg.err
		m.saved = msg.err == nil

	case tea.KeyMsg:
		if m.inputMode {
			// Text input mode
			switch msg.String() {
			case "enter", "esc":
				m.inputMode = false
				m.getActiveInput().Focused = false
			case "backspace":
				m.getActiveInput().Backspace()
			case "delete":
				m.getActiveInput().Delete()
			case "left":
				m.getActiveInput().MoveCursorLeft()
			case "right":
				m.getActiveInput().MoveCursorRight()
			default:
				if len(msg.String()) == 1 {
					m.getActiveInput().InsertChar(rune(msg.String()[0]))
				}
			}
			return m, nil
		}

		switch msg.String() {
		case "up", "k":
			if m.focusIdx > 0 {
				m.focusIdx--
			}
		case "down", "j":
			if m.focusIdx < m.maxItems-1 {
				m.focusIdx++
			}
		case "enter", " ":
			switch m.focusIdx {
			case 0: // ASTAR_TOKEN
				m.inputMode = true
				m.astarToken.Focused = true
				m.astarToken.CursorPos = len(m.astarToken.Value)
			case 1: // GOOGLE_API_KEY
				m.inputMode = true
				m.googleAPIKey.Focused = true
				m.googleAPIKey.CursorPos = len(m.googleAPIKey.Value)
			case 2:
				m.autoRefresh.Toggle()
			case 3:
				m.tailLogs.Toggle()
			case 4:
				m.showViewport.Toggle()
			case 5:
				m.notifications.Toggle()
			case 6:
				m.claudeCLI.Toggle()
			case 7: // Save button
				return m, m.saveSettings()
			}
		case "s":
			return m, m.saveSettings()
		}
	}
	return m, nil
}

func (m *SettingsModel) getActiveInput() *components.TextInput {
	switch m.focusIdx {
	case 0:
		return &m.astarToken
	case 1:
		return &m.googleAPIKey
	default:
		return &m.astarToken
	}
}

func (m SettingsModel) saveSettings() tea.Cmd {
	return func() tea.Msg {
		envPath := filepath.Join(m.baseDir, ".env")

		// Read existing .env
		var lines []string
		data, err := os.ReadFile(envPath)
		if err == nil {
			lines = strings.Split(string(data), "\n")
		}

		// Update or add keys
		setEnvLine := func(key, val string) {
			found := false
			for i, line := range lines {
				if strings.HasPrefix(strings.TrimSpace(line), key+"=") {
					lines[i] = key + "=" + val
					found = true
					break
				}
			}
			if !found {
				lines = append(lines, key+"="+val)
			}
		}

		if m.astarToken.Value != "" {
			setEnvLine("ASTAR_TOKEN", m.astarToken.Value)
		}
		if m.googleAPIKey.Value != "" {
			setEnvLine("GOOGLE_API_KEY", m.googleAPIKey.Value)
		}

		content := strings.Join(lines, "\n")
		err = os.WriteFile(envPath, []byte(content), 0644)
		return settingsSavedMsg{err: err}
	}
}

func (m SettingsModel) View() string {
	var sb strings.Builder

	// Viking header
	sb.WriteString(lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render("  ᛁ Settings & Configuration") + "\n")

	sb.WriteString(lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Render("\n  ╔═════════════════════════════╗\n  ║  ᛁ ᚲ ᛟ ᚾ ᚠ ᛁ ᚷ           ║\n  ║  C O N F I G U R E         ║\n  ╚═════════════════════════════╝") + "\n\n")

	// API Keys section
	sb.WriteString(lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render("  ᚠ API Keys") + "\n\n")

	// ASTAR_TOKEN
	m.astarToken.Focused = m.focusIdx == 0
	sb.WriteString(m.astarToken.View() + "\n")

	// GOOGLE_API_KEY
	m.googleAPIKey.Focused = m.focusIdx == 1
	sb.WriteString(m.googleAPIKey.View() + "\n\n")

	// Feature toggles
	sb.WriteString(lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render("  ᛋ Feature Toggles") + "\n\n")

	m.autoRefresh.Focused = m.focusIdx == 2
	sb.WriteString(m.autoRefresh.View() + "\n")

	m.tailLogs.Focused = m.focusIdx == 3
	sb.WriteString(m.tailLogs.View() + "\n")

	m.showViewport.Focused = m.focusIdx == 4
	sb.WriteString(m.showViewport.View() + "\n")

	m.notifications.Focused = m.focusIdx == 5
	sb.WriteString(m.notifications.View() + "\n")

	m.claudeCLI.Focused = m.focusIdx == 6
	sb.WriteString(m.claudeCLI.View() + "\n\n")

	// Save button
	saveStyle := lipgloss.NewStyle().
		Padding(0, 2).
		Foreground(lipgloss.Color("#0a0a12")).
		Background(lipgloss.Color("#00e5ff"))
	if m.focusIdx == 7 {
		saveStyle = saveStyle.Bold(true).Background(lipgloss.Color("#00ff41"))
	}
	sb.WriteString("  " + saveStyle.Render("ᛋ Save to .env") + "\n\n")

	// Status
	if m.saved {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00ff41")).
			Bold(true).
			Render("  ✓ Settings saved to .env") + "\n")
	}
	if m.saveErr != nil {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#ff0033")).
			Render(fmt.Sprintf("  ✕ Save error: %v", m.saveErr)) + "\n")
	}

	// System info
	sb.WriteString("\n" + lipgloss.NewStyle().
		Foreground(lipgloss.Color("#3a3a5a")).
		Render(fmt.Sprintf("  Base dir: %s", m.baseDir)) + "\n")
	sb.WriteString(lipgloss.NewStyle().
		Foreground(lipgloss.Color("#3a3a5a")).
		Render(fmt.Sprintf("  Token set: %v  |  Gemini key set: %v",
			m.astarToken.Value != "", m.googleAPIKey.Value != "")) + "\n")

	sb.WriteString("\n" + lipgloss.NewStyle().
		Foreground(lipgloss.Color("#3a3a5a")).
		Render("  ↑↓ navigate  Enter/Space toggle  s save"))

	return sb.String()
}

func (m SettingsModel) SetSize(w, h int) SettingsModel {
	m.width = w
	m.height = h
	return m
}
