package tabs

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"astar-tui/api"
	"astar-tui/components"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type roundsDataMsg struct {
	myRounds []api.MyRound
	err      error
}

type roundsTickMsg time.Time

type analysisMsg struct {
	roundID   string
	seedIndex int
	analysis  *api.Analysis
	err       error
}

// RoundsModel shows all rounds with drill-down to per-seed scores
type RoundsModel struct {
	client      *api.Client
	width       int
	height      int
	myRounds    []api.MyRound
	cursor      int
	err         error
	loading     bool // true only before first data arrives
	refreshing  bool // true when fetching while old data is still displayed
	// Drill-down state
	drillDown   bool
	drillRound  *api.MyRound
	drillSeed   int
	drillScroll int
	analysis    *api.Analysis
	analysisErr error
}

func NewRounds(client *api.Client) RoundsModel {
	return RoundsModel{client: client, loading: true}
}

func (RoundsModel) Title() string    { return "Rounds" }
func (RoundsModel) ShortKey() string { return "2" }
func (RoundsModel) Rune() string     { return "ᚱ" }

func (m RoundsModel) Init() tea.Cmd {
	return tea.Batch(m.fetchRounds(), m.tick())
}

func (m RoundsModel) tick() tea.Cmd {
	return tea.Tick(15*time.Second, func(t time.Time) tea.Msg {
		return roundsTickMsg(t)
	})
}

func (m RoundsModel) fetchRounds() tea.Cmd {
	return func() tea.Msg {
		rounds, err := m.client.GetMyRounds()
		return roundsDataMsg{myRounds: rounds, err: err}
	}
}

func (m RoundsModel) fetchAnalysis(roundID string, seed int) tea.Cmd {
	return func() tea.Msg {
		a, err := m.client.GetAnalysis(roundID, seed)
		return analysisMsg{roundID: roundID, seedIndex: seed, analysis: a, err: err}
	}
}

func (m RoundsModel) Update(msg tea.Msg) (RoundsModel, tea.Cmd) {
	switch msg := msg.(type) {
	case roundsDataMsg:
		m.loading = false
		m.refreshing = false
		m.err = msg.err
		if msg.myRounds != nil {
			m.myRounds = msg.myRounds
		}

	case roundsTickMsg:
		m.refreshing = true
		return m, tea.Batch(m.fetchRounds(), m.tick())

	case analysisMsg:
		m.analysis = msg.analysis
		m.analysisErr = msg.err

	case tea.KeyMsg:
		switch msg.String() {
		case "up", "k":
			if !m.drillDown && m.cursor > 0 {
				m.cursor--
			} else if m.drillDown && m.drillSeed > 0 {
				m.drillSeed--
				m.drillScroll = 0
				if m.drillRound != nil {
					return m, m.fetchAnalysis(m.drillRound.ID, m.drillSeed)
				}
			}
		case "down", "j":
			if !m.drillDown && m.cursor < len(m.myRounds)-1 {
				m.cursor++
			} else if m.drillDown && m.drillRound != nil && m.drillSeed < len(m.drillRound.SeedScores)-1 {
				m.drillSeed++
				m.drillScroll = 0
				return m, m.fetchAnalysis(m.drillRound.ID, m.drillSeed)
			}
		case "pgdown", "ctrl+d":
			if m.drillDown {
				m.drillScroll += m.height / 2
			}
		case "pgup", "ctrl+u":
			if m.drillDown && m.drillScroll > 0 {
				m.drillScroll -= m.height / 2
				if m.drillScroll < 0 {
					m.drillScroll = 0
				}
			}
		case "enter":
			if !m.drillDown && m.cursor < len(m.myRounds) {
				m.drillDown = true
				m.drillRound = &m.myRounds[m.cursor]
				m.drillSeed = 0
				m.drillScroll = 0
				m.analysis = nil
				return m, m.fetchAnalysis(m.drillRound.ID, 0)
			}
		case "esc":
			if m.drillDown {
				m.drillDown = false
				m.drillRound = nil
				m.analysis = nil
				m.drillScroll = 0
			}
		case "r":
			m.refreshing = true
			return m, m.fetchRounds()
		}
	}
	return m, nil
}

func (m RoundsModel) View() string {
	if m.drillDown {
		return m.viewDrillDown()
	}
	return m.viewTable()
}

func (m RoundsModel) viewTable() string {
	var sb strings.Builder

	titleText := "  ᚱ Rounds & Scores"
	if m.refreshing {
		titleText += " ↻"
	}
	title := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render(titleText)
	sb.WriteString(title + "\n\n")

	if m.err != nil {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#ff0033")).
			Render(fmt.Sprintf("  Error: %v", m.err)))
		return sb.String()
	}

	if m.loading {
		sb.WriteString("  ◌ Loading...")
		return sb.String()
	}

	if len(m.myRounds) == 0 {
		sb.WriteString("  No rounds found")
		return sb.String()
	}

	// Header
	hdr := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Underline(true)
	sb.WriteString(hdr.Render(fmt.Sprintf("  %-6s %-10s %8s %6s %6s %8s", "Round", "Status", "Score", "Rank", "Seeds", "Budget")) + "\n")

	for i, mr := range m.myRounds {
		isSelected := i == m.cursor

		scoreStr := "    -   "
		if mr.Score != nil {
			scoreStr = scoreColor(*mr.Score).Render(fmt.Sprintf("%8.2f", *mr.Score))
		}

		rankStr := "   -  "
		if mr.Rank != nil {
			rankStr = fmt.Sprintf("%6d", *mr.Rank)
		}

		statusStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a"))
		if mr.Status == "active" {
			statusStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff41")).Bold(true)
		} else if mr.Status == "completed" {
			statusStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#bf00ff"))
		}

		line := fmt.Sprintf("  R%-5d %s %s %s %6d %8d",
			mr.RoundNumber,
			statusStyle.Render(fmt.Sprintf("%-10s", mr.Status)),
			scoreStr,
			rankStr,
			mr.SeedsSubmitted,
			mr.QueriesUsed,
		)

		if isSelected {
			line = lipgloss.NewStyle().
				Background(lipgloss.Color("#0d0d1a")).
				Render("▸" + line[1:])
		}

		sb.WriteString(line + "\n")
	}

	sb.WriteString("\n" + lipgloss.NewStyle().
		Foreground(lipgloss.Color("#3a3a5a")).
		Render("  ↑↓ navigate  Enter drill-down  r refresh  Esc back"))

	return sb.String()
}

func (m RoundsModel) viewDrillDown() string {
	if m.drillRound == nil {
		return ""
	}

	var sb strings.Builder
	mr := m.drillRound
	w := m.width
	if w < 40 {
		w = 80
	}

	// Header
	title := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render(fmt.Sprintf("  ᛋ Round %d — Seed Analysis", mr.RoundNumber))
	sb.WriteString(title)

	// Round summary stats
	if mr.Score != nil {
		sb.WriteString("  ")
		sb.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a")).Render("Round Score: "))
		sb.WriteString(scoreColor(*mr.Score).Render(fmt.Sprintf("%.2f", *mr.Score)))
	}
	if mr.Rank != nil {
		sb.WriteString("  ")
		sb.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a")).Render("Rank: "))
		sb.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("#c0c8e0")).Bold(true).Render(fmt.Sprintf("#%d", *mr.Rank)))
	}
	sb.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a")).Render(
		fmt.Sprintf("  Queries: %d/%d", mr.QueriesUsed, mr.QueriesMax)))
	sb.WriteString("\n\n")

	// Seed score sparkline
	if len(mr.SeedScores) > 1 {
		spark := components.Sparkline{
			Title:  "  ᚠ Seed Scores",
			Values: mr.SeedScores,
			Width:  w - 14,
		}
		sb.WriteString(spark.View())
		sb.WriteString("\n\n")
	}

	// Seed selector with bar chart
	hdr := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Underline(true)
	sb.WriteString(hdr.Render(fmt.Sprintf("  %-8s %8s  %-30s", "Seed", "Score", "")) + "\n")

	barW := 30
	if w > 80 {
		barW = 40
	}
	for i, score := range mr.SeedScores {
		isSelected := i == m.drillSeed
		scoreStr := scoreColor(score).Render(fmt.Sprintf("%8.2f", score))

		barLen := int(score / 100 * float64(barW))
		if barLen < 0 {
			barLen = 0
		}
		restLen := barW - barLen
		if restLen < 0 {
			restLen = 0
		}
		bar := lipgloss.NewStyle().Foreground(lipgloss.Color("#00e5ff")).Render(strings.Repeat("█", barLen))
		rest := lipgloss.NewStyle().Foreground(lipgloss.Color("#0d0d1a")).Render(strings.Repeat("░", restLen))

		prefix := "  "
		if isSelected {
			prefix = lipgloss.NewStyle().Foreground(lipgloss.Color("#00e5ff")).Bold(true).Render("▸ ")
		}
		sb.WriteString(fmt.Sprintf("%sSeed %-2d %s  %s%s\n", prefix, i, scoreStr, bar, rest))
	}

	// Seed score stats
	if len(mr.SeedScores) > 0 {
		min, max, sum := mr.SeedScores[0], mr.SeedScores[0], 0.0
		for _, s := range mr.SeedScores {
			sum += s
			if s < min {
				min = s
			}
			if s > max {
				max = s
			}
		}
		avg := sum / float64(len(mr.SeedScores))
		variance := 0.0
		for _, s := range mr.SeedScores {
			d := s - avg
			variance += d * d
		}
		stddev := math.Sqrt(variance / float64(len(mr.SeedScores)))

		muted := lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a"))
		sb.WriteString(muted.Render(fmt.Sprintf("  Min: %.2f  Max: %.2f  Avg: %.2f  StdDev: %.2f  Spread: %.2f",
			min, max, avg, stddev, max-min)))
		sb.WriteString("\n")
	}

	// Analysis details
	if m.analysisErr != nil {
		sb.WriteString(fmt.Sprintf("\n  %s %v",
			lipgloss.NewStyle().Foreground(lipgloss.Color("#ff0033")).Render("Analysis error:"),
			m.analysisErr))
	} else if m.analysis != nil {
		sb.WriteString(m.renderAnalysis())
	} else {
		sb.WriteString("\n  " + lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a")).Render("◌ Loading analysis..."))
	}

	sb.WriteString("\n\n" + lipgloss.NewStyle().
		Foreground(lipgloss.Color("#3a3a5a")).
		Render("  ↑↓ switch seed  PgDn/PgUp scroll  Esc back to table"))

	// Apply scroll
	output := sb.String()
	if m.drillScroll > 0 {
		lines := strings.Split(output, "\n")
		if m.drillScroll < len(lines) {
			lines = lines[m.drillScroll:]
		}
		output = strings.Join(lines, "\n")
	}

	return output
}

func (m RoundsModel) renderAnalysis() string {
	a := m.analysis
	if a == nil {
		return ""
	}
	w := m.width
	if w < 40 {
		w = 80
	}

	panelStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#00e5ff")).
		Padding(0, 1)

	goldBold := lipgloss.NewStyle().Foreground(lipgloss.Color("#00e5ff")).Bold(true)
	muted := lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a"))

	var sb strings.Builder
	sb.WriteString("\n")
	sb.WriteString(goldBold.Render(fmt.Sprintf("  ⚔ Seed %d Analysis — Score: ", a.SeedIndex)))
	sb.WriteString(scoreColor(a.Score).Render(fmt.Sprintf("%.2f", a.Score)))
	sb.WriteString("\n")

	// --- Score Distribution Histogram ---
	if len(a.CellScores) > 0 {
		buckets := map[string]float64{
			"95-100": 0, "90-95": 0, "85-90": 0,
			"80-85": 0, "70-80": 0, "<70": 0,
		}
		var allScores []float64
		var sumScore float64
		worstCells := make([]struct {
			x, y  int
			score float64
		}, 0, 10)

		for y, row := range a.CellScores {
			for x, cs := range row {
				allScores = append(allScores, cs)
				sumScore += cs
				switch {
				case cs >= 95:
					buckets["95-100"]++
				case cs >= 90:
					buckets["90-95"]++
				case cs >= 85:
					buckets["85-90"]++
				case cs >= 80:
					buckets["80-85"]++
				case cs >= 70:
					buckets["70-80"]++
				default:
					buckets["<70"]++
				}
				// Track worst cells
				if len(worstCells) < 10 || cs < worstCells[len(worstCells)-1].score {
					worstCells = append(worstCells, struct {
						x, y  int
						score float64
					}{x, y, cs})
					sort.Slice(worstCells, func(i, j int) bool {
						return worstCells[i].score < worstCells[j].score
					})
					if len(worstCells) > 10 {
						worstCells = worstCells[:10]
					}
				}
			}
		}

		avgCell := sumScore / float64(len(allScores))

		// Left panel: histogram + stats
		histChart := components.BarChart{
			Title:    "ᛏ Cell Score Distribution",
			Labels:   []string{"95-100", "90-95", "85-90", "80-85", "70-80", "<70"},
			Values:   []float64{buckets["95-100"], buckets["90-95"], buckets["85-90"], buckets["80-85"], buckets["70-80"], buckets["<70"]},
			MaxWidth: 25,
			Color:    lipgloss.Color("#00e5ff"),
		}
		histContent := histChart.View()
		histContent += "\n\n" + muted.Render(fmt.Sprintf("  Total cells: %d  Avg: %.2f", len(allScores), avgCell))

		// Right panel: worst cells
		var worstSB strings.Builder
		worstSB.WriteString(goldBold.Render("ᚱ Worst Cells") + "\n")
		worstSB.WriteString(muted.Render(fmt.Sprintf("  %-6s %-6s %8s", "X", "Y", "Score")) + "\n")
		for _, wc := range worstCells {
			worstSB.WriteString(fmt.Sprintf("  %-6d %-6d %s\n",
				wc.x, wc.y, scoreColor(wc.score).Render(fmt.Sprintf("%8.2f", wc.score))))
		}

		// Join panels side by side
		panelW := (w - 6) / 2
		if panelW < 30 {
			panelW = 30
		}
		leftPanel := panelStyle.Width(panelW).Render(histContent)
		rightPanel := panelStyle.Width(panelW).Render(worstSB.String())
		sb.WriteString("\n" + lipgloss.JoinHorizontal(lipgloss.Top, leftPanel, " ", rightPanel))
		sb.WriteString("\n")
	}

	// --- Ground Truth Terrain Distribution ---
	if len(a.GroundTruth) > 0 {
		terrainNames := []string{"Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"}
		terrainCounts := make([]float64, 6)
		totalCells := 0

		for _, row := range a.GroundTruth {
			for _, cell := range row {
				if len(cell) >= 6 {
					// Find dominant terrain (argmax)
					maxProb := 0.0
					maxIdx := 0
					for t, p := range cell[:6] {
						if p > maxProb {
							maxProb = p
							maxIdx = t
						}
					}
					terrainCounts[maxIdx]++
					totalCells++
				}
			}
		}

		terrainChart := components.BarChart{
			Title:    "ᛞ Ground Truth Terrain",
			Labels:   terrainNames,
			Values:   terrainCounts,
			MaxWidth: 30,
			Color:    lipgloss.Color("#00ff41"),
		}

		// Prediction confidence analysis
		var confSB strings.Builder
		if len(a.Prediction) > 0 {
			var totalEntropy, maxEntropy float64
			var highConfCount, lowConfCount int
			cellCount := 0

			for _, row := range a.Prediction {
				for _, cell := range row {
					if len(cell) >= 6 {
						// Compute entropy
						entropy := 0.0
						maxProb := 0.0
						for _, p := range cell[:6] {
							if p > 0.01 {
								entropy -= p * math.Log2(p)
							}
							if p > maxProb {
								maxProb = p
							}
						}
						totalEntropy += entropy
						if entropy > maxEntropy {
							maxEntropy = entropy
						}
						if maxProb >= 0.8 {
							highConfCount++
						} else if maxProb < 0.4 {
							lowConfCount++
						}
						cellCount++
					}
				}
			}

			confSB.WriteString(goldBold.Render("ᚹ Prediction Confidence") + "\n\n")
			if cellCount > 0 {
				avgEntropy := totalEntropy / float64(cellCount)
				confSB.WriteString(muted.Render("  Avg entropy: "))
				confSB.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("#c0c8e0")).Render(fmt.Sprintf("%.3f bits", avgEntropy)))
				confSB.WriteString("\n")
				confSB.WriteString(muted.Render("  Max entropy: "))
				confSB.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("#c0c8e0")).Render(fmt.Sprintf("%.3f bits", maxEntropy)))
				confSB.WriteString("\n")
				confSB.WriteString(muted.Render("  High conf (>80%): "))
				confSB.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff41")).Render(
					fmt.Sprintf("%d (%.0f%%)", highConfCount, float64(highConfCount)/float64(cellCount)*100)))
				confSB.WriteString("\n")
				confSB.WriteString(muted.Render("  Low conf  (<40%): "))
				confSB.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("#ff0033")).Render(
					fmt.Sprintf("%d (%.0f%%)", lowConfCount, float64(lowConfCount)/float64(cellCount)*100)))
				confSB.WriteString("\n")

				// Confidence bar
				highPct := float64(highConfCount) / float64(cellCount)
				barW := 30
				highW := int(highPct * float64(barW))
				lowW := barW - highW
				if lowW < 0 {
					lowW = 0
				}
				confSB.WriteString("\n  ")
				confSB.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff41")).Render(strings.Repeat("█", highW)))
				confSB.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("#ff0033")).Render(strings.Repeat("░", lowW)))
			}
		}

		panelW := (w - 6) / 2
		if panelW < 30 {
			panelW = 30
		}
		leftPanel := panelStyle.Width(panelW).Render(terrainChart.View() + "\n" +
			muted.Render(fmt.Sprintf("  Total: %d cells", totalCells)))
		rightPanel := panelStyle.Width(panelW).Render(confSB.String())
		sb.WriteString("\n" + lipgloss.JoinHorizontal(lipgloss.Top, leftPanel, " ", rightPanel))
		sb.WriteString("\n")
	}

	// --- Cell Score Heatmap ---
	if len(a.CellScores) > 0 {
		sb.WriteString("\n" + goldBold.Render("  ᛋ Cell Score Heatmap") + "  ")
		sb.WriteString(muted.Render("(green=good, red=bad)") + "\n")

		for y := 0; y < len(a.CellScores) && y < 40; y++ {
			sb.WriteString("  ")
			for x := 0; x < len(a.CellScores[y]) && x < 40; x++ {
				score := a.CellScores[y][x]
				var color lipgloss.Color
				switch {
				case score >= 95:
					color = "#00e676"
				case score >= 90:
					color = "#00ff41"
				case score >= 85:
					color = "#00e5ff"
				case score >= 80:
					color = "#ffaa00"
				case score >= 70:
					color = "#ff0033"
				default:
					color = "#b71c1c"
				}
				sb.WriteString(lipgloss.NewStyle().Foreground(color).Render("█"))
			}
			sb.WriteString("\n")
		}
	}

	return sb.String()
}

func (m RoundsModel) SetSize(w, h int) RoundsModel {
	m.width = w
	m.height = h
	return m
}
