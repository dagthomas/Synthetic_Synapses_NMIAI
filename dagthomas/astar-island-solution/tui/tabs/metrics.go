package tabs

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"astar-tui/api"
	"astar-tui/components"
	"astar-tui/internal"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// Messages
type metricsTickMsg time.Time

type metricsAutoMsg struct {
	entries []api.AutoloopEntry
	total   int
	err     error
}

type metricsResearchMsg struct {
	entries []api.ResearchEntry
	err     error
}

type metricsAPIMsg struct {
	budget      *api.Budget
	myRounds    []api.MyRound
	leaderboard []api.LeaderboardEntry
	err         error
}

// MetricsModel is the metrics/charts overview tab
type MetricsModel struct {
	client  *api.Client
	width   int
	height  int
	dataDir string

	// Readers
	autoReader     *internal.JSONLReader
	researchReader *internal.JSONLReader
	geminiReader   *internal.JSONLReader

	// Autoloop data
	autoEntries []api.AutoloopEntry
	autoTotal   int
	autoScores  []float64

	// Research data
	researchEntries []api.ResearchEntry

	// API data
	budget      *api.Budget
	myRounds    []api.MyRound
	leaderboard []api.LeaderboardEntry

	// Computed
	expRates   []float64
	lastUpdate time.Time
	loading    bool
}

func NewMetrics(client *api.Client, dataDir string) MetricsModel {
	return MetricsModel{
		client:         client,
		dataDir:        dataDir,
		autoReader:     internal.NewJSONLReader(dataDir + "/autoloop_log.jsonl"),
		researchReader: internal.NewJSONLReader(dataDir + "/adk_research_log.jsonl"),
		geminiReader:   internal.NewJSONLReader(dataDir + "/gemini_research_log.jsonl"),
		loading:        true,
	}
}

func (MetricsModel) Title() string    { return "Metrics" }
func (MetricsModel) ShortKey() string { return "0" }
func (MetricsModel) Rune() string     { return "ᛗ" }

func (m MetricsModel) Init() tea.Cmd {
	return tea.Batch(m.loadAutoloop(), m.loadResearch(), m.fetchAPI(), m.tick())
}

func (m MetricsModel) tick() tea.Cmd {
	return tea.Tick(5*time.Second, func(t time.Time) tea.Msg {
		return metricsTickMsg(t)
	})
}

func (m MetricsModel) loadAutoloop() tea.Cmd {
	return func() tea.Msg {
		raw, err := m.autoReader.ReadLast(500)
		if err != nil {
			return metricsAutoMsg{err: err}
		}
		total, _ := m.autoReader.Count()
		var entries []api.AutoloopEntry
		for _, r := range raw {
			var e api.AutoloopEntry
			if json.Unmarshal(r, &e) == nil {
				entries = append(entries, e)
			}
		}
		return metricsAutoMsg{entries: entries, total: total}
	}
}

func (m MetricsModel) loadResearch() tea.Cmd {
	return func() tea.Msg {
		// Load both ADK and Gemini research logs
		var entries []api.ResearchEntry

		raw, _ := m.researchReader.ReadLast(200)
		for _, r := range raw {
			var e api.ResearchEntry
			if json.Unmarshal(r, &e) == nil {
				e.Normalize()
				entries = append(entries, e)
			}
		}

		rawG, _ := m.geminiReader.ReadLast(200)
		for _, r := range rawG {
			var g api.GeminiResearchEntry
			if json.Unmarshal(r, &g) == nil {
				// Convert to ResearchEntry
				e := api.ResearchEntry{
					ID:        g.ID,
					Timestamp: g.Timestamp,
					Name:      g.Name,
					Hypothesis: g.Hypothesis,
					Model:     "gemini",
					Scores:    g.Scores,
					Elapsed:   g.Elapsed,
					Error:     g.Error,
				}
				if imp, ok := g.Improvement.(float64); ok {
					e.Improvement = imp
				}
				e.Normalize()
				entries = append(entries, e)
			}
		}

		return metricsResearchMsg{entries: entries}
	}
}

func (m MetricsModel) fetchAPI() tea.Cmd {
	return func() tea.Msg {
		msg := metricsAPIMsg{}
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

func (m MetricsModel) Update(msg tea.Msg) (MetricsModel, tea.Cmd) {
	switch msg := msg.(type) {
	case metricsAutoMsg:
		m.loading = false
		if msg.err == nil {
			m.autoEntries = msg.entries
			m.autoTotal = msg.total
			m.computeAutoMetrics()
		}

	case metricsResearchMsg:
		m.loading = false
		if msg.err == nil {
			m.researchEntries = msg.entries
		}

	case metricsAPIMsg:
		m.loading = false
		if msg.err == nil {
			m.budget = msg.budget
			m.myRounds = msg.myRounds
			m.leaderboard = msg.leaderboard
			m.lastUpdate = time.Now()
		}

	case metricsTickMsg:
		return m, tea.Batch(m.loadAutoloop(), m.loadResearch(), m.fetchAPI(), m.tick())

	case tea.KeyMsg:
		if msg.String() == "r" {
			return m, tea.Batch(m.loadAutoloop(), m.loadResearch(), m.fetchAPI())
		}
	}
	return m, nil
}

func (m *MetricsModel) computeAutoMetrics() {
	m.autoScores = nil
	// Track running best score
	best := 0.0
	for _, e := range m.autoEntries {
		avg := avgMapValues(e.ScoresQuick)
		if avg > best {
			best = avg
		}
		m.autoScores = append(m.autoScores, best)
	}

	// Compute experiment rate (exps per minute over sliding windows)
	m.expRates = computeExpRates(m.autoEntries)
}

func (m MetricsModel) View() string {
	w := m.width
	if w < 40 {
		w = 80
	}
	h := m.height
	if h < 10 {
		h = 40
	}

	titleStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Align(lipgloss.Center).
		Width(w)

	var sb strings.Builder
	sb.WriteString(titleStyle.Render("ᛗ  METRICS OVERVIEW  ᛗ"))
	sb.WriteString("\n")

	if m.loading && len(m.autoEntries) == 0 {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a")).
			Render("  ◌ Gathering data from the Norse realms..."))
		return sb.String()
	}

	// 2x2 grid + bottom strip
	panelW := (w - 3) / 2
	if panelW < 30 {
		panelW = 30
	}
	innerW := panelW - 4

	panelStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#00e5ff")).
		Padding(0, 1).
		Width(panelW)

	// Top row: Score Convergence | Experiment Rate
	p1 := m.renderScoreConvergence(innerW)
	p2 := m.renderExperimentRate(innerW)
	topRow := lipgloss.JoinHorizontal(lipgloss.Top,
		panelStyle.Render(p1), " ", panelStyle.Render(p2))
	sb.WriteString(topRow)
	sb.WriteString("\n")

	// Middle row: Research Outcomes | System Overview
	p3 := m.renderResearchOutcomes(innerW)
	p4 := m.renderSystemOverview(innerW)
	midRow := lipgloss.JoinHorizontal(lipgloss.Top,
		panelStyle.Render(p3), " ", panelStyle.Render(p4))
	sb.WriteString(midRow)
	sb.WriteString("\n")

	// Bottom strip: Parameter Heatmap
	bottomStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#00e5ff")).
		Padding(0, 1).
		Width(w - 2)
	p5 := m.renderParamHeatmap(w - 6)
	sb.WriteString(bottomStyle.Render(p5))

	// Footer
	footer := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#3a3a5a")).
		Render(fmt.Sprintf("\n  Last updated: %s  ᛫  Press r to refresh  ᛫  %d experiments total",
			m.lastUpdate.Format("15:04:05"), m.autoTotal))
	sb.WriteString(footer)

	return sb.String()
}

// Panel 1: Score Convergence sparkline
func (m MetricsModel) renderScoreConvergence(w int) string {
	sectionTitle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render("ᚠ SCORE CONVERGENCE")

	if len(m.autoScores) == 0 {
		return sectionTitle + "\n" + lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a")).
			Render("  No experiment data yet")
	}

	spark := components.Sparkline{
		Title:  "",
		Values: m.autoScores,
		Width:  w - 2,
	}

	var sb strings.Builder
	sb.WriteString(sectionTitle + "\n")
	sb.WriteString(spark.View())

	// Acceptance rate
	accepted := 0
	for _, e := range m.autoEntries {
		if e.Accepted {
			accepted++
		}
	}
	total := len(m.autoEntries)
	if total > 0 {
		pct := float64(accepted) / float64(total) * 100
		sb.WriteString(fmt.Sprintf("\n  Accepted: %d/%d (%.1f%%)", accepted, total, pct))
	}

	return sb.String()
}

// Panel 2: Experiment Rate sparkline
func (m MetricsModel) renderExperimentRate(w int) string {
	sectionTitle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render("ᛏ EXPERIMENT RATE")

	if len(m.expRates) == 0 {
		return sectionTitle + "\n" + lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a")).
			Render("  No rate data yet")
	}

	spark := components.Sparkline{
		Title:  "",
		Values: m.expRates,
		Width:  w - 2,
	}

	var sb strings.Builder
	sb.WriteString(sectionTitle + "\n")
	sb.WriteString(spark.View())

	// Current rate
	if len(m.expRates) > 0 {
		current := m.expRates[len(m.expRates)-1]
		sb.WriteString(fmt.Sprintf("\n  Current: %.0f exp/min", current))
	}

	return sb.String()
}

// Panel 3: Research Outcomes — bar chart + sparkline
func (m MetricsModel) renderResearchOutcomes(w int) string {
	sectionTitle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render("ᚹ RESEARCH OUTCOMES")

	if len(m.researchEntries) == 0 {
		return sectionTitle + "\n" + lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a")).
			Render("  No research data yet")
	}

	// Count statuses
	var successCount, errorCount, improvedCount float64
	for _, e := range m.researchEntries {
		switch e.Status {
		case "success":
			successCount++
			if e.Improvement > 0 {
				improvedCount++
			}
		case "error":
			errorCount++
		}
	}

	chart := components.BarChart{
		Title:    "",
		Labels:   []string{"Success", "Error", "Improved"},
		Values:   []float64{successCount, errorCount, improvedCount},
		MaxWidth: w - 16,
		Color:    lipgloss.Color("#00e5ff"),
	}

	var sb strings.Builder
	sb.WriteString(sectionTitle + "\n")
	sb.WriteString(chart.View())

	// Improvement sparkline
	var improvements []float64
	for _, e := range m.researchEntries {
		if e.Improvement != 0 {
			improvements = append(improvements, e.Improvement)
		}
	}
	if len(improvements) > 2 {
		sb.WriteString("\n\n")
		spark := components.Sparkline{
			Title:  "  Improvement Trend",
			Values: improvements,
			Width:  w - 4,
		}
		sb.WriteString(spark.View())
	}

	return sb.String()
}

// Panel 4: System Overview — budget gauge, round scores, leaderboard mini
func (m MetricsModel) renderSystemOverview(w int) string {
	sectionTitle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render("ᛞ SYSTEM OVERVIEW")

	var sb strings.Builder
	sb.WriteString(sectionTitle + "\n\n")

	// Budget gauge
	if m.budget != nil {
		used := m.budget.QueriesUsed
		max := m.budget.QueriesMax
		pct := float64(used) / float64(max)
		barW := w - 20
		if barW < 10 {
			barW = 10
		}
		filledW := int(pct * float64(barW))
		emptyW := barW - filledW
		if emptyW < 0 {
			emptyW = 0
		}

		budgetColor := lipgloss.Color("#00ff41")
		remaining := max - used
		if remaining <= 10 {
			budgetColor = lipgloss.Color("#ff0033")
		} else if remaining <= 25 {
			budgetColor = lipgloss.Color("#00e5ff")
		}

		filled := lipgloss.NewStyle().Foreground(budgetColor).Render(strings.Repeat("█", filledW))
		empty := lipgloss.NewStyle().Foreground(lipgloss.Color("#0d0d1a")).Render(strings.Repeat("░", emptyW))
		sb.WriteString(fmt.Sprintf("  ᚠ Budget [%s%s] %d/%d\n", filled, empty, used, max))
	} else {
		sb.WriteString("  ᚠ Budget: loading...\n")
	}

	// Round scores sparkline
	if len(m.myRounds) > 0 {
		var scores []float64
		for _, mr := range m.myRounds {
			if mr.Score != nil {
				scores = append(scores, *mr.Score)
			}
		}
		if len(scores) > 1 {
			spark := components.Sparkline{
				Title:  "  Round Scores",
				Values: scores,
				Width:  w - 4,
			}
			sb.WriteString("\n" + spark.View())
		}
	}

	// Leaderboard mini (top 5)
	if len(m.leaderboard) > 0 {
		sb.WriteString("\n\n  ")
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00e5ff")).
			Bold(true).
			Render("Leaderboard Top 5"))
		sb.WriteString("\n")

		max := 5
		if len(m.leaderboard) < max {
			max = len(m.leaderboard)
		}

		// Build a mini bar chart
		var labels []string
		var vals []float64
		for i := 0; i < max; i++ {
			e := m.leaderboard[i]
			name := e.TeamName
			if len(name) > 10 {
				name = name[:9] + "…"
			}
			labels = append(labels, fmt.Sprintf("#%d %s", e.Rank, name))
			vals = append(vals, e.Score)
		}
		chart := components.BarChart{
			Labels:   labels,
			Values:   vals,
			MaxWidth: w - 20,
			Color:    lipgloss.Color("#bf00ff"),
		}
		sb.WriteString(chart.View())
	}

	return sb.String()
}

// Bottom panel: Parameter heatmap — top tweaked params with distributions
func (m MetricsModel) renderParamHeatmap(w int) string {
	sectionTitle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render("ᚱ PARAMETER HEATMAP")

	if len(m.autoEntries) == 0 {
		return sectionTitle + "  " + lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a")).
			Render("No parameter data")
	}

	// Gather all parameter values
	paramValues := make(map[string][]float64)
	for _, e := range m.autoEntries {
		for k, v := range e.Params {
			paramValues[k] = append(paramValues[k], v)
		}
	}

	// Sort by frequency (most tweaked = most unique values)
	type paramStat struct {
		name     string
		count    int
		min, max float64
		mean     float64
		stddev   float64
	}

	var stats []paramStat
	for name, vals := range paramValues {
		if len(vals) < 2 {
			continue
		}
		mn, mx := vals[0], vals[0]
		sum := 0.0
		for _, v := range vals {
			if v < mn {
				mn = v
			}
			if v > mx {
				mx = v
			}
			sum += v
		}
		mean := sum / float64(len(vals))

		variance := 0.0
		for _, v := range vals {
			d := v - mean
			variance += d * d
		}
		variance /= float64(len(vals))
		stddev := math.Sqrt(variance)

		stats = append(stats, paramStat{
			name:   name,
			count:  len(vals),
			min:    mn,
			max:    mx,
			mean:   mean,
			stddev: stddev,
		})
	}

	// Sort by stddev (most variation = most interesting)
	sort.Slice(stats, func(i, j int) bool {
		return stats[i].stddev > stats[j].stddev
	})

	// Show top 5
	max := 5
	if len(stats) < max {
		max = len(stats)
	}

	if max == 0 {
		return sectionTitle + "  " + lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a")).
			Render("No varied parameters")
	}

	headerStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Underline(true)

	labelStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#c0c8e0"))
	valueStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#3a3a5a"))
	barStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#ff2d95"))

	var sb strings.Builder
	sb.WriteString(sectionTitle + "\n")
	sb.WriteString(headerStyle.Render(fmt.Sprintf("  %-20s %8s %8s %8s %8s  Distribution", "Parameter", "Min", "Max", "Mean", "StdDev")))
	sb.WriteString("\n")

	barMaxW := w - 60
	if barMaxW < 10 {
		barMaxW = 10
	}

	for i := 0; i < max; i++ {
		s := stats[i]
		name := s.name
		if len(name) > 20 {
			name = name[:19] + "…"
		}

		// Mini distribution bar (normalized stddev)
		relSpread := 0.0
		if s.max != s.min {
			relSpread = s.stddev / (s.max - s.min)
		}
		barLen := int(relSpread * float64(barMaxW))
		if barLen < 1 {
			barLen = 1
		}

		sb.WriteString(fmt.Sprintf("  %s %s %s %s %s  %s\n",
			labelStyle.Render(fmt.Sprintf("%-20s", name)),
			valueStyle.Render(fmt.Sprintf("%8.4f", s.min)),
			valueStyle.Render(fmt.Sprintf("%8.4f", s.max)),
			valueStyle.Render(fmt.Sprintf("%8.4f", s.mean)),
			valueStyle.Render(fmt.Sprintf("%8.4f", s.stddev)),
			barStyle.Render(strings.Repeat("▓", barLen)),
		))
	}

	return sb.String()
}

func (m MetricsModel) SetSize(w, h int) MetricsModel {
	m.width = w
	m.height = h
	return m
}

// --- helpers ---

func avgMapValues(m map[string]float64) float64 {
	if len(m) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range m {
		sum += v
	}
	return sum / float64(len(m))
}

func computeExpRates(entries []api.AutoloopEntry) []float64 {
	if len(entries) < 2 {
		return nil
	}

	// Parse timestamps and compute experiments per minute in sliding windows
	type timed struct {
		t time.Time
	}
	var times []time.Time
	for _, e := range entries {
		t, err := time.Parse(time.RFC3339, e.Timestamp)
		if err != nil {
			// Try other formats
			t, err = time.Parse("2006-01-02T15:04:05", e.Timestamp)
			if err != nil {
				continue
			}
		}
		times = append(times, t)
	}

	if len(times) < 2 {
		return nil
	}

	// Compute rate over sliding windows of 20 entries
	windowSize := 20
	var rates []float64
	for i := windowSize; i < len(times); i++ {
		dt := times[i].Sub(times[i-windowSize]).Minutes()
		if dt > 0 {
			rates = append(rates, float64(windowSize)/dt)
		}
	}

	return rates
}
