package tabs

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"astar-tui/api"
	"astar-tui/components"
	"astar-tui/internal"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// Dashboard messages
type dashDataMsg struct {
	rounds      []api.Round
	budget      *api.Budget
	myRounds    []api.MyRound
	leaderboard []api.LeaderboardEntry
	err         error
}

type dashTickMsg time.Time

type dashDaemonMsg struct {
	totalExp     int
	bestScore    float64
	bestBoom     float64
	bestNonboom  float64
	expPerHour   float64
	acceptedPct  float64
	scores       []float64 // running best for sparkline
	daemonLog    []string
	bestParams   map[string]float64
	paramsSource string
}

// DashboardModel is the home tab
type DashboardModel struct {
	client      *api.Client
	width       int
	height      int
	dataDir     string
	rounds      []api.Round
	budget      *api.Budget
	myRounds    []api.MyRound
	leaderboard []api.LeaderboardEntry
	lastUpdate  time.Time
	err         error
	loading     bool // true only before first data arrives
	refreshing  bool // true when fetching while old data is still displayed

	// Daemon stats
	autoReader   *internal.JSONLReader
	daemonTotal  int
	daemonBest   float64
	daemonBoom   float64
	daemonNon    float64
	daemonExpH   float64
	daemonAccPct float64
	daemonScores []float64
	daemonLog    []string
	bestParams   map[string]float64
	paramsSource string
}

func NewDashboard(client *api.Client, dataDir string) DashboardModel {
	return DashboardModel{
		client:     client,
		dataDir:    dataDir,
		autoReader: internal.NewJSONLReader(filepath.Join(dataDir, "autoloop_fast_log.jsonl")),
		loading:    true,
	}
}

func (DashboardModel) Title() string    { return "Dashboard" }
func (DashboardModel) ShortKey() string { return "1" }
func (DashboardModel) Rune() string     { return "ᛟ" }

func (m DashboardModel) Init() tea.Cmd {
	return tea.Batch(m.fetchData(), m.fetchDaemon(), m.tick())
}

func (m DashboardModel) tick() tea.Cmd {
	return tea.Tick(10*time.Second, func(t time.Time) tea.Msg {
		return dashTickMsg(t)
	})
}

func (m DashboardModel) fetchData() tea.Cmd {
	return func() tea.Msg {
		msg := dashDataMsg{}

		rounds, err := m.client.GetRounds()
		if err != nil {
			msg.err = err
			return msg
		}
		msg.rounds = rounds

		budget, err := m.client.GetBudget()
		if err == nil {
			msg.budget = budget
		}

		myRounds, err := m.client.GetMyRounds()
		if err == nil {
			msg.myRounds = myRounds
		}

		lb, err := m.client.GetLeaderboard()
		if err == nil {
			msg.leaderboard = lb
		}

		return msg
	}
}

func (m DashboardModel) fetchDaemon() tea.Cmd {
	return func() tea.Msg {
		msg := dashDaemonMsg{}

		// Read autoloop_fast log
		raw, err := m.autoReader.ReadLast(500)
		if err != nil || len(raw) == 0 {
			return msg
		}

		total, _ := m.autoReader.Count()
		msg.totalExp = total

		var firstTS, lastTS time.Time
		var accepted int
		best := 0.0

		for _, r := range raw {
			var e struct {
				Timestamp string             `json:"timestamp"`
				Accepted  bool               `json:"accepted"`
				ScoresFull map[string]float64 `json:"scores_full"`
			}
			if json.Unmarshal(r, &e) != nil {
				continue
			}

			if t, err := time.Parse(time.RFC3339Nano, e.Timestamp); err == nil {
				if firstTS.IsZero() {
					firstTS = t
				}
				lastTS = t
			}

			if e.Accepted {
				accepted++
			}

			avg := e.ScoresFull["avg"]
			if avg > best {
				best = avg
			}
			msg.scores = append(msg.scores, best)

			boom := e.ScoresFull["boom_avg"]
			nonboom := e.ScoresFull["nonboom_avg"]
			if avg > msg.bestScore {
				msg.bestScore = avg
			}
			if boom > msg.bestBoom {
				msg.bestBoom = boom
			}
			if nonboom > msg.bestNonboom {
				msg.bestNonboom = nonboom
			}
		}

		if len(raw) > 0 {
			msg.acceptedPct = float64(accepted) / float64(len(raw)) * 100
		}

		if !firstTS.IsZero() && !lastTS.IsZero() {
			dur := lastTS.Sub(firstTS).Hours()
			if dur > 0 {
				msg.expPerHour = math.Round(float64(len(raw)) / dur)
			}
		}

		// Read daemon.log
		logPath := filepath.Join(m.dataDir, "daemon.log")
		if data, err := os.ReadFile(logPath); err == nil {
			lines := strings.Split(strings.TrimSpace(string(data)), "\n")
			n := 8
			if len(lines) > n {
				lines = lines[len(lines)-n:]
			}
			msg.daemonLog = lines
		}

		// Read best_params.json
		paramsPath := filepath.Join(filepath.Dir(m.dataDir), "best_params.json")
		if data, err := os.ReadFile(paramsPath); err == nil {
			var params struct {
				Source      string  `json:"source"`
				ScoreAvg   float64 `json:"score_avg"`
				PriorW     float64 `json:"prior_w"`
				EmpMax     float64 `json:"emp_max"`
				ExpDamp    float64 `json:"exp_damp"`
				BasePower  float64 `json:"base_power"`
				THigh      float64 `json:"T_high"`
				SmoothAlpha float64 `json:"smooth_alpha"`
				Floor      float64 `json:"floor"`
			}
			if json.Unmarshal(data, &params) == nil {
				msg.bestParams = map[string]float64{
					"prior_w":      params.PriorW,
					"emp_max":      params.EmpMax,
					"exp_damp":     params.ExpDamp,
					"base_power":   params.BasePower,
					"T_high":       params.THigh,
					"smooth_alpha": params.SmoothAlpha,
					"floor":        params.Floor,
				}
				msg.paramsSource = params.Source
				if params.ScoreAvg > msg.bestScore {
					msg.bestScore = params.ScoreAvg
				}
			}
		}

		return msg
	}
}

func (m DashboardModel) Update(msg tea.Msg) (DashboardModel, tea.Cmd) {
	switch msg := msg.(type) {
	case dashDataMsg:
		m.loading = false
		m.refreshing = false
		m.err = msg.err
		if msg.err == nil {
			m.rounds = msg.rounds
			m.budget = msg.budget
			m.myRounds = msg.myRounds
			m.leaderboard = msg.leaderboard
			m.lastUpdate = time.Now()
		}

	case dashDaemonMsg:
		m.daemonTotal = msg.totalExp
		m.daemonBest = msg.bestScore
		m.daemonBoom = msg.bestBoom
		m.daemonNon = msg.bestNonboom
		m.daemonExpH = msg.expPerHour
		m.daemonAccPct = msg.acceptedPct
		m.daemonScores = msg.scores
		m.daemonLog = msg.daemonLog
		m.bestParams = msg.bestParams
		m.paramsSource = msg.paramsSource

	case dashTickMsg:
		m.refreshing = true
		return m, tea.Batch(m.fetchData(), m.fetchDaemon(), m.tick())

	case tea.KeyMsg:
		if msg.String() == "r" {
			m.refreshing = true
			return m, tea.Batch(m.fetchData(), m.fetchDaemon())
		}
	}
	return m, nil
}

func (m DashboardModel) View() string {
	w := m.width
	if w < 40 {
		w = 80
	}

	var sb strings.Builder

	// Viking header
	header := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Align(lipgloss.Center).
		Width(w).
		Render("⚔  ASTAR ISLAND EXPLORER  ⚔\n᛫ ᚨ ᛋ ᛏ ᚨ ᚱ ᛫ ᛁ ᛋ ᛚ ᚨ ᚾ ᛞ ᛫")
	sb.WriteString(header)
	sb.WriteString("\n\n")

	if m.err != nil {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#ff0033")).
			Render(fmt.Sprintf("  ✕ Error: %v", m.err)))
		sb.WriteString("\n\n")
	}

	if m.loading {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a")).
			Render("  ◌ Loading from the Norse realms..."))
		return sb.String()
	}

	// Layout: 3 columns
	panelW := (w - 6) / 3
	if panelW < 25 {
		panelW = 25
	}

	panelStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#00e5ff")).
		Padding(0, 1).
		Width(panelW)

	// Panel 1: Active Round
	panel1 := m.renderRoundPanel(panelW - 4)

	// Panel 2: Leaderboard
	panel2 := m.renderLeaderboard(panelW - 4)

	// Panel 3: My Scores
	panel3 := m.renderMyScores(panelW - 4)

	row := lipgloss.JoinHorizontal(lipgloss.Top,
		panelStyle.Render(panel1),
		" ",
		panelStyle.Render(panel2),
		" ",
		panelStyle.Render(panel3),
	)
	sb.WriteString(row)
	sb.WriteString("\n\n")

	// Score summary with bar chart
	if len(m.myRounds) > 0 {
		sb.WriteString(m.renderScoreSummary(w))
	}

	// Daemon stats panel
	if m.daemonTotal > 0 {
		sb.WriteString("\n")
		sb.WriteString(m.renderDaemonPanel(w))
	}

	// Footer
	refreshIndicator := ""
	if m.refreshing {
		refreshIndicator = "  ↻"
	}
	footer := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#3a3a5a")).
		Render(fmt.Sprintf("  Last updated: %s  ᛫  Press r to refresh%s", m.lastUpdate.Format("15:04:05"), refreshIndicator))
	sb.WriteString("\n" + footer)

	return sb.String()
}

func (m DashboardModel) renderRoundPanel(w int) string {
	title := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render("ᛞ Active Round")

	var sb strings.Builder
	sb.WriteString(title + "\n")

	// Find active round
	var active *api.Round
	for i := range m.rounds {
		if m.rounds[i].Status == "active" {
			active = &m.rounds[i]
			break
		}
	}

	if active == nil {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a")).
			Render("No active round"))
		if len(m.rounds) > 0 {
			// Find highest round_number
			latest := m.rounds[0]
			for _, r := range m.rounds {
				if r.Number > latest.Number {
					latest = r
				}
			}
			sb.WriteString(fmt.Sprintf("\nLatest: R%d (%s)", latest.Number, latest.Status))
		}
	} else {
		label := lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a"))
		value := lipgloss.NewStyle().Foreground(lipgloss.Color("#c0c8e0")).Bold(true)

		sb.WriteString(fmt.Sprintf("\n%s %s", label.Render("Round:"), value.Render(fmt.Sprintf("#%d", active.Number))))
		sb.WriteString(fmt.Sprintf("\n%s %s", label.Render("Map:"), value.Render(fmt.Sprintf("%dx%d", active.MapWidth, active.MapHeight))))
		statusLabel := "● ACTIVE"
		statusColor := lipgloss.Color("#00ff41")
		if active.Status != "active" {
			statusLabel = active.Status
			statusColor = lipgloss.Color("#bf00ff")
		}
		sb.WriteString(fmt.Sprintf("\n%s %s", label.Render("Status:"), lipgloss.NewStyle().Foreground(statusColor).Bold(true).Render(statusLabel)))
	}

	// Budget
	if m.budget != nil {
		sb.WriteString("\n")
		remaining := m.budget.QueriesMax - m.budget.QueriesUsed
		budgetColor := lipgloss.Color("#00ff41")
		if remaining <= 10 {
			budgetColor = lipgloss.Color("#ff0033")
		} else if remaining <= 25 {
			budgetColor = lipgloss.Color("#00e5ff")
		}
		budgetStyle := lipgloss.NewStyle().Foreground(budgetColor).Bold(true)
		sb.WriteString(fmt.Sprintf("\n%s %s",
			lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a")).Render("ᚠ Budget:"),
			budgetStyle.Render(fmt.Sprintf("%d/%d", m.budget.QueriesUsed, m.budget.QueriesMax))))

		// Budget bar
		barW := w - 4
		if barW > 0 {
			used := float64(m.budget.QueriesUsed) / float64(m.budget.QueriesMax)
			filledW := int(used * float64(barW))
			emptyW := barW - filledW
			if emptyW < 0 {
				emptyW = 0
			}
			filled := lipgloss.NewStyle().Foreground(budgetColor).Render(strings.Repeat("█", filledW))
			empty := lipgloss.NewStyle().Foreground(lipgloss.Color("#0d0d1a")).Render(strings.Repeat("░", emptyW))
			sb.WriteString("\n" + filled + empty)
		}
	}

	sb.WriteString(fmt.Sprintf("\n\n%s %d", lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a")).Render("Total rounds:"), len(m.rounds)))

	return sb.String()
}

func (m DashboardModel) renderLeaderboard(w int) string {
	title := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render("ᛏ Leaderboard")

	var sb strings.Builder
	sb.WriteString(title + "\n\n")

	if len(m.leaderboard) == 0 {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a")).
			Render("No leaderboard data"))
		return sb.String()
	}

	// Header
	hdr := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Underline(true)
	sb.WriteString(hdr.Render(fmt.Sprintf("%-4s %-14s %6s", "#", "Team", "Score")) + "\n")

	max := 10
	if len(m.leaderboard) < max {
		max = len(m.leaderboard)
	}

	for i := 0; i < max; i++ {
		entry := m.leaderboard[i]
		name := entry.TeamName
		if len(name) > 14 {
			name = name[:13] + "…"
		}

		rankStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#c0c8e0"))
		if entry.Rank <= 3 {
			medals := []string{"", "🥇", "🥈", "🥉"}
			rankStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#00e5ff")).Bold(true)
			name = medals[entry.Rank] + " " + name
			if len(name) > 16 {
				name = name[:16]
			}
		}

		scoreStr := scoreColor(entry.Score).Render(fmt.Sprintf("%6.1f", entry.Score))
		sb.WriteString(rankStyle.Render(fmt.Sprintf("%-4d", entry.Rank)) +
			lipgloss.NewStyle().Foreground(lipgloss.Color("#c0c8e0")).Render(fmt.Sprintf("%-16s", name)) +
			scoreStr + "\n")
	}

	return sb.String()
}

func (m DashboardModel) renderMyScores(w int) string {
	title := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render("ᛋ My Scores")

	var sb strings.Builder
	sb.WriteString(title + "\n\n")

	if len(m.myRounds) == 0 {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a")).
			Render("No submissions yet"))
		return sb.String()
	}

	hdr := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Underline(true)
	sb.WriteString(hdr.Render(fmt.Sprintf("%-6s %6s %4s", "Round", "Score", "Rank")) + "\n")

	// Show last 8 rounds
	start := 0
	if len(m.myRounds) > 8 {
		start = len(m.myRounds) - 8
	}

	for i := start; i < len(m.myRounds); i++ {
		mr := m.myRounds[i]
		scoreStr := "  -  "
		rankStr := "  -"
		if mr.Score != nil {
			scoreStr = scoreColor(*mr.Score).Render(fmt.Sprintf("%6.1f", *mr.Score))
		}
		if mr.Rank != nil {
			rankStr = fmt.Sprintf("%4d", *mr.Rank)
		}

		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#c0c8e0")).
			Render(fmt.Sprintf("R%-5d", mr.RoundNumber)) +
			scoreStr + " " + rankStr + "\n")
	}

	// Average
	var total float64
	var count int
	for _, mr := range m.myRounds {
		if mr.Score != nil {
			total += *mr.Score
			count++
		}
	}
	if count > 0 {
		avg := total / float64(count)
		sb.WriteString("\n" + lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a")).
			Render("Average: ") +
			scoreColor(avg).Render(fmt.Sprintf("%.2f", avg)))
	}

	return sb.String()
}

func (m DashboardModel) renderDaemonPanel(w int) string {
	panelStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#bf00ff")).
		Padding(0, 1).
		Width(w - 2)

	var sb strings.Builder

	title := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#bf00ff")).
		Bold(true).
		Render("ᚹ DAEMON — Autoloop Stats")
	sb.WriteString(title + "\n\n")

	label := lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a"))
	val := lipgloss.NewStyle().Foreground(lipgloss.Color("#c0c8e0")).Bold(true)
	cyan := lipgloss.NewStyle().Foreground(lipgloss.Color("#00e5ff")).Bold(true)
	gold := lipgloss.NewStyle().Foreground(lipgloss.Color("#ffaa00")).Bold(true)
	green := lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff41")).Bold(true)

	// Stats row
	stats := fmt.Sprintf("  %s %s   %s %s   %s %s   %s %s   %s %s",
		label.Render("Experiments:"), cyan.Render(fmt.Sprintf("%d", m.daemonTotal)),
		label.Render("Best:"), scoreColor(m.daemonBest).Render(fmt.Sprintf("%.2f", m.daemonBest)),
		label.Render("Boom:"), scoreColor(m.daemonBoom).Render(fmt.Sprintf("%.2f", m.daemonBoom)),
		label.Render("Non-boom:"), scoreColor(m.daemonNon).Render(fmt.Sprintf("%.2f", m.daemonNon)),
		label.Render("Exp/hr:"), gold.Render(fmt.Sprintf("%.0f", m.daemonExpH)),
	)
	sb.WriteString(stats + "\n")

	// Acceptance rate
	sb.WriteString(fmt.Sprintf("  %s %s\n",
		label.Render("Accept rate:"),
		green.Render(fmt.Sprintf("%.1f%%", m.daemonAccPct)),
	))

	// Convergence sparkline
	if len(m.daemonScores) > 2 {
		spark := components.Sparkline{
			Title:  "  ᚠ Score Convergence",
			Values: m.daemonScores,
			Width:  w - 10,
		}
		sb.WriteString("\n" + spark.View() + "\n")
	}

	// Best params
	if len(m.bestParams) > 0 {
		sb.WriteString("\n  ")
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#bf00ff")).
			Bold(true).
			Render("Best Parameters"))
		if m.paramsSource != "" {
			sb.WriteString(label.Render(fmt.Sprintf(" (source: %s)", m.paramsSource)))
		}
		sb.WriteString("\n  ")

		paramOrder := []string{"prior_w", "emp_max", "exp_damp", "base_power", "T_high", "smooth_alpha", "floor"}
		for _, k := range paramOrder {
			if v, ok := m.bestParams[k]; ok {
				sb.WriteString(fmt.Sprintf("%s=%s  ",
					label.Render(k),
					val.Render(fmt.Sprintf("%.4f", v)),
				))
			}
		}
		sb.WriteString("\n")
	}

	// Daemon log (last few lines)
	if len(m.daemonLog) > 0 {
		sb.WriteString("\n  ")
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#bf00ff")).
			Bold(true).
			Render("Daemon Log"))
		sb.WriteString("\n")

		logStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a"))
		warnStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#ffaa00"))
		errStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#ff0033"))

		for _, line := range m.daemonLog {
			line = strings.TrimRight(line, "\r\n")
			style := logStyle
			if strings.Contains(line, "WARN") || strings.Contains(line, "warn") {
				style = warnStyle
			} else if strings.Contains(line, "ERROR") || strings.Contains(line, "error") {
				style = errStyle
			}
			sb.WriteString("  " + style.Render(line) + "\n")
		}
	}

	return panelStyle.Render(sb.String())
}

func (m DashboardModel) renderScoreSummary(w int) string {
	// Build sparkline of round scores over time
	var vals []float64
	var labels []string
	for _, mr := range m.myRounds {
		if mr.Score != nil {
			vals = append(vals, *mr.Score)
			labels = append(labels, fmt.Sprintf("R%d", mr.RoundNumber))
		}
	}

	if len(vals) == 0 {
		return ""
	}

	spark := components.Sparkline{
		Title:  "  ᚠ Score History",
		Values: vals,
		Width:  w - 10,
	}

	return spark.View()
}

func (m DashboardModel) SetSize(w, h int) DashboardModel {
	m.width = w
	m.height = h
	return m
}

func scoreColor(score float64) lipgloss.Style {
	switch {
	case score >= 93:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("#00e676")).Bold(true)
	case score >= 90:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff41"))
	case score >= 85:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("#00e5ff"))
	case score >= 80:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("#ffaa00"))
	default:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("#ff0033"))
	}
}
