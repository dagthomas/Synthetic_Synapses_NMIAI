package tabs

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"astar-tui/api"
	"astar-tui/components"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type explorerDataMsg struct {
	detail *api.RoundDetail
	err    error
}

type explorerObsMsg struct {
	obs []api.Observation
	err error
}

type explorerPredMsg struct {
	seedIndex int
	predGrid  [][]int // argmax of prediction tensor (class 0-5)
	err       error
}

// ExplorerModel shows the 40x40 map grid with seed switching
type ExplorerModel struct {
	client    *api.Client
	dataDir   string
	width     int
	height    int
	detail    *api.RoundDetail
	seedIndex int
	grids     [5][][]int // per-seed merged grids
	predGrids [5][][]int // per-seed prediction argmax (class 0-5)
	hasPreds  bool       // true after predictions loaded
	viewportX int
	viewportY int
	showVP    bool
	loading   bool
	err       error
	status    string // transient status message (obs loaded, preds loaded, errors)
	mapGrid   components.MapGrid
}

func NewExplorer(client *api.Client, dataDir string) ExplorerModel {
	return ExplorerModel{
		client:  client,
		dataDir: dataDir,
		loading: true,
		showVP:  true,
	}
}

func (ExplorerModel) Title() string    { return "Explorer" }
func (ExplorerModel) ShortKey() string { return "4" }
func (ExplorerModel) Rune() string     { return "ᚨ" }

func (m ExplorerModel) Init() tea.Cmd {
	return m.fetchDetail()
}

func (m ExplorerModel) fetchDetail() tea.Cmd {
	return func() tea.Msg {
		active, err := m.client.GetActiveRound()
		if err != nil {
			return explorerDataMsg{err: err}
		}
		if active == nil {
			return explorerDataMsg{err: fmt.Errorf("no rounds found")}
		}
		detail, err := m.client.GetRoundDetail(active.ID)
		return explorerDataMsg{detail: detail, err: err}
	}
}

func (m ExplorerModel) Update(msg tea.Msg) (ExplorerModel, tea.Cmd) {
	switch msg := msg.(type) {
	case explorerDataMsg:
		m.loading = false
		m.err = msg.err
		if msg.detail != nil {
			m.detail = msg.detail
			m.buildGrids()
			m.updateMapGrid()
		}

	case explorerObsMsg:
		if msg.err != nil {
			m.status = fmt.Sprintf("Obs error: %v", msg.err)
		} else if len(msg.obs) == 0 {
			dir := ""
			if m.detail != nil {
				dir = filepath.Join(m.dataDir, "rounds", m.detail.ID)
			}
			m.status = fmt.Sprintf("No obs files found in %s", dir)
		} else {
			m.mergeObservations(msg.obs)
			m.updateMapGrid()
			m.status = fmt.Sprintf("Loaded %d observations", len(msg.obs))
		}

	case explorerPredMsg:
		if msg.err != nil {
			m.status = fmt.Sprintf("Pred error: %v", msg.err)
		} else if msg.seedIndex >= 0 && msg.seedIndex < 5 {
			m.predGrids[msg.seedIndex] = msg.predGrid
			m.hasPreds = true
			m.updateMapGrid()
			m.status = fmt.Sprintf("Loaded predictions for seed %d", msg.seedIndex)
		}

	case tea.KeyMsg:
		switch msg.String() {
		case "1":
			m.seedIndex = 0
			m.updateMapGrid()
		case "2":
			m.seedIndex = 1
			m.updateMapGrid()
		case "3":
			m.seedIndex = 2
			m.updateMapGrid()
		case "4":
			m.seedIndex = 3
			m.updateMapGrid()
		case "5":
			m.seedIndex = 4
			m.updateMapGrid()
		case "left", "h":
			if m.viewportX > 0 {
				m.viewportX--
				m.updateMapGrid()
			}
		case "right", "l":
			if m.viewportX < 25 {
				m.viewportX++
				m.updateMapGrid()
			}
		case "up", "k":
			if m.viewportY > 0 {
				m.viewportY--
				m.updateMapGrid()
			}
		case "down", "j":
			if m.viewportY < 25 {
				m.viewportY++
				m.updateMapGrid()
			}
		case "v":
			m.showVP = !m.showVP
			m.updateMapGrid()
		case "r":
			m.loading = true
			return m, m.fetchDetail()
		case "o":
			return m, m.loadObservations()
		case "p":
			return m, m.loadPredictions()
		}
	}
	return m, nil
}

func (m *ExplorerModel) buildGrids() {
	if m.detail == nil {
		return
	}

	for seed := 0; seed < 5 && seed < len(m.detail.InitialStates); seed++ {
		state := m.detail.InitialStates[seed]
		grid := make([][]int, 40)
		for y := 0; y < 40; y++ {
			grid[y] = make([]int, 40)
			for x := 0; x < 40; x++ {
				if y < len(state.Grid) && x < len(state.Grid[y]) {
					grid[y][x] = state.Grid[y][x]
				} else {
					grid[y][x] = -1
				}
			}
		}
		m.grids[seed] = grid
	}
}

func (m *ExplorerModel) mergeObservations(obs []api.Observation) {
	for _, o := range obs {
		if o.SeedIndex < 0 || o.SeedIndex >= 5 || m.grids[o.SeedIndex] == nil {
			continue
		}
		grid := m.grids[o.SeedIndex]
		for dy, row := range o.Grid {
			for dx, cell := range row {
				gy := o.Viewport.Y + dy
				gx := o.Viewport.X + dx
				if gy >= 0 && gy < 40 && gx >= 0 && gx < 40 {
					grid[gy][gx] = cell
				}
			}
		}
	}
}

func (m ExplorerModel) loadObservations() tea.Cmd {
	return func() tea.Msg {
		if m.detail == nil {
			return explorerObsMsg{err: fmt.Errorf("no round loaded")}
		}
		roundDir := filepath.Join(m.dataDir, "rounds", m.detail.ID)
		pattern := filepath.Join(roundDir, "obs_*.json")
		matches, err := filepath.Glob(pattern)
		if err != nil {
			return explorerObsMsg{err: err}
		}

		var allObs []api.Observation
		for _, path := range matches {
			data, err := os.ReadFile(path)
			if err != nil {
				continue
			}
			var obs api.Observation
			if err := json.Unmarshal(data, &obs); err != nil {
				continue
			}
			allObs = append(allObs, obs)
		}
		return explorerObsMsg{obs: allObs}
	}
}

func (m ExplorerModel) loadPredictions() tea.Cmd {
	return func() tea.Msg {
		if m.detail == nil {
			return explorerPredMsg{err: fmt.Errorf("no round loaded")}
		}
		// Fetch analysis for current seed (contains our prediction tensor)
		analysis, err := m.client.GetAnalysis(m.detail.ID, m.seedIndex)
		if err != nil {
			return explorerPredMsg{seedIndex: m.seedIndex, err: err}
		}
		if analysis == nil || len(analysis.Prediction) == 0 {
			return explorerPredMsg{seedIndex: m.seedIndex, err: fmt.Errorf("no predictions available")}
		}
		// Convert [][][]float64 (40x40x6) to [][]int argmax
		predGrid := make([][]int, len(analysis.Prediction))
		for y, row := range analysis.Prediction {
			predGrid[y] = make([]int, len(row))
			for x, probs := range row {
				bestClass := 0
				bestProb := 0.0
				for c, p := range probs {
					if p > bestProb {
						bestProb = p
						bestClass = c
					}
				}
				predGrid[y][x] = bestClass
			}
		}
		return explorerPredMsg{seedIndex: m.seedIndex, predGrid: predGrid}
	}
}

func (m *ExplorerModel) updateMapGrid() {
	grid := m.grids[m.seedIndex]
	mg := components.MapGrid{
		Grid:         grid,
		ViewportX:    m.viewportX,
		ViewportY:    m.viewportY,
		ViewportW:    15,
		ViewportH:    15,
		ShowViewport: m.showVP,
		SeedIndex:    m.seedIndex,
	}
	if m.hasPreds && m.predGrids[m.seedIndex] != nil {
		mg.PredGrid = m.predGrids[m.seedIndex]
	}
	m.mapGrid = mg
}

func (m ExplorerModel) View() string {
	var sb strings.Builder

	title := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render("  ᚨ Astar Island Explorer")
	sb.WriteString(title + "\n")

	if m.detail != nil {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a")).
			Render(fmt.Sprintf("  Round #%d (%s)  ID: %s", m.detail.Number, m.detail.Status, m.detail.ID[:8])) + "\n")
	}
	if m.err != nil {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#ff0033")).
			Render(fmt.Sprintf("  Error: %v", m.err)) + "\n")
	}
	if m.status != "" {
		sb.WriteString(lipgloss.NewStyle().
			Foreground(lipgloss.Color("#bf00ff")).
			Render(fmt.Sprintf("  %s", m.status)) + "\n")
	}
	if m.loading {
		sb.WriteString("  ◌ Loading realm data...\n")
		return sb.String()
	}

	// Seed selector
	sb.WriteString("  ")
	for i := 0; i < 5; i++ {
		if i == m.seedIndex {
			sb.WriteString(lipgloss.NewStyle().
				Foreground(lipgloss.Color("#0a0a12")).
				Background(lipgloss.Color("#bf00ff")).
				Bold(true).
				Padding(0, 1).
				Render(fmt.Sprintf("Seed %d", i)))
		} else {
			sb.WriteString(lipgloss.NewStyle().
				Foreground(lipgloss.Color("#c0c8e0")).
				Background(lipgloss.Color("#0d0d1a")).
				Padding(0, 1).
				Render(fmt.Sprintf("Seed %d", i)))
		}
		sb.WriteString(" ")
	}
	sb.WriteString("\n\n")

	// Map grid + legend side by side
	if m.grids[m.seedIndex] != nil {
		mapView := m.mapGrid.View()
		legend := m.mapGrid.Legend()

		// Coverage
		coverage := m.mapGrid.Coverage()
		coverageStr := lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00e5ff")).
			Bold(true).
			Render(fmt.Sprintf("\n  Coverage: %.0f%%", coverage))

		// Terrain stats
		stats := m.mapGrid.Stats()
		var statsLines []string
		terrains := []int{0, 1, 2, 3, 4, 5, 10}
		terrainNames := map[int]string{0: "Empty", 1: "Settlement", 2: "Port", 3: "Ruin", 4: "Forest", 5: "Mountain", 10: "Ocean"}
		for _, t := range terrains {
			if c, ok := stats[t]; ok && c > 0 {
				statsLines = append(statsLines, fmt.Sprintf("  %s: %d", terrainNames[t], c))
			}
		}
		statsStr := lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a")).Render(strings.Join(statsLines, "\n"))

		// Viewport info
		vpInfo := lipgloss.NewStyle().
			Foreground(lipgloss.Color("#ffaa00")).
			Render(fmt.Sprintf("\n  Viewport: (%d,%d)", m.viewportX, m.viewportY))

		// Settlement details
		var settInfo string
		if m.detail != nil && m.seedIndex < len(m.detail.InitialStates) {
			setts := m.detail.InitialStates[m.seedIndex].Settlements
			alive := 0
			for _, s := range setts {
				if s.Alive {
					alive++
				}
			}
			settInfo = lipgloss.NewStyle().Foreground(lipgloss.Color("#00e5ff")).
				Render(fmt.Sprintf("\n\n  ᛋ Settlements: %d (%d alive)", len(setts), alive))
		}

		// Viewport contents breakdown
		vpStats := m.mapGrid.ViewportStats()
		if vpStats != "" {
			vpStats = "\n\n" + vpStats
		}

		// Prediction comparison
		vpComp := m.mapGrid.ViewportComparison()
		if vpComp != "" {
			vpComp = "\n\n" + vpComp
		}

		rightPanel := legend + coverageStr + vpInfo + "\n" + statsStr + settInfo + vpStats + vpComp

		combined := lipgloss.JoinHorizontal(lipgloss.Top, mapView, "  ", rightPanel)
		sb.WriteString(combined)
	}

	sb.WriteString("\n\n" + lipgloss.NewStyle().
		Foreground(lipgloss.Color("#3a3a5a")).
		Render("  1-5 seed  ←→↑↓ viewport  v toggle VP  o load obs  p load preds  r refresh"))

	return sb.String()
}

func (m ExplorerModel) SetSize(w, h int) ExplorerModel {
	m.width = w
	m.height = h
	return m
}
