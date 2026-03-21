package components

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// Terrain constants matching config.py
const (
	TerrainOcean      = 10
	TerrainPlains     = 11
	TerrainEmpty      = 0
	TerrainSettlement = 1
	TerrainPort       = 2
	TerrainRuin       = 3
	TerrainForest     = 4
	TerrainMountain   = 5
	TerrainUnknown    = -1
)

// Terrain display configs
type terrainDisplay struct {
	Char  string
	Color lipgloss.Color
	Name  string
}

var terrainDisplays = map[int]terrainDisplay{
	TerrainOcean:      {"~", lipgloss.Color("#1a237e"), "Ocean"},
	TerrainPlains:     {"·", lipgloss.Color("#3a3a5a"), "Plains"},
	TerrainEmpty:      {"·", lipgloss.Color("#3a3a5a"), "Empty"},
	TerrainSettlement: {"ᛋ", lipgloss.Color("#00e5ff"), "Settlement"},
	TerrainPort:       {"⚓", lipgloss.Color("#bf00ff"), "Port"},
	TerrainRuin:       {"ᚱ", lipgloss.Color("#ff0033"), "Ruin"},
	TerrainForest:     {"♠", lipgloss.Color("#00ff41"), "Forest"},
	TerrainMountain:   {"▲", lipgloss.Color("#9e9e9e"), "Mountain"},
	TerrainUnknown:    {"░", lipgloss.Color("#0d0d1a"), "Unknown"},
}

// MapGrid renders a 40x40 terrain grid with viewport overlay
type MapGrid struct {
	Grid       [][]int // 40x40 terrain codes (-1 for unknown)
	PredGrid   [][]int // 40x40 prediction argmax (class 0-5), nil if not loaded
	ViewportX  int     // current viewport top-left X
	ViewportY  int     // current viewport top-left Y
	ViewportW  int     // viewport width (15)
	ViewportH  int     // viewport height (15)
	ShowViewport bool
	CellWidth  int  // chars per cell (1 or 2)
	Compact    bool // skip row numbers
	SeedIndex  int
}

var (
	vpBorderStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#ffaa00")).
			Bold(true)

	axisStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a"))

	legendTitleStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00e5ff")).
			Bold(true)
)

// View renders the map grid
func (mg MapGrid) View() string {
	if len(mg.Grid) == 0 {
		return lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a")).
			Render("  No map data loaded")
	}

	h := len(mg.Grid)
	w := 0
	if h > 0 {
		w = len(mg.Grid[0])
	}

	var sb strings.Builder

	// Seed label
	seedLabel := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#bf00ff")).
		Bold(true).
		Render(fmt.Sprintf("  ᛋ Seed %d", mg.SeedIndex))
	sb.WriteString(seedLabel)
	sb.WriteString("\n")

	// Cell width: 2 chars per cell makes the grid appear square in terminal
	cw := mg.CellWidth
	if cw < 2 {
		cw = 2
	}

	// Column numbers (every 5)
	if !mg.Compact {
		sb.WriteString("   ")
		for x := 0; x < w; x++ {
			if x%5 == 0 {
				sb.WriteString(axisStyle.Render(fmt.Sprintf("%-*d", cw*5, x)))
			}
		}
		sb.WriteString("\n")
	}

	// Rows
	for y := 0; y < h; y++ {
		// Row number
		if !mg.Compact {
			sb.WriteString(axisStyle.Render(fmt.Sprintf("%2d ", y)))
		}

		for x := 0; x < w; x++ {
			code := mg.Grid[y][x]
			td, ok := terrainDisplays[code]
			if !ok {
				td = terrainDisplays[TerrainUnknown]
			}

			ch := td.Char
			style := lipgloss.NewStyle().Foreground(td.Color)

			// Highlight viewport border
			if mg.ShowViewport && mg.isViewportBorder(x, y) {
				style = vpBorderStyle
				ch = "□"
			}

			// Pad cell to cw characters for square aspect ratio
			cell := ch + strings.Repeat(" ", cw-1)
			sb.WriteString(style.Render(cell))
		}

		if !mg.Compact {
			sb.WriteString(axisStyle.Render(fmt.Sprintf(" %d", y)))
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

func (mg MapGrid) isViewportBorder(x, y int) bool {
	vx, vy := mg.ViewportX, mg.ViewportY
	vw, vh := mg.ViewportW, mg.ViewportH
	if vw == 0 {
		vw = 15
	}
	if vh == 0 {
		vh = 15
	}

	inX := x >= vx && x < vx+vw
	inY := y >= vy && y < vy+vh

	// Top or bottom border
	if inX && (y == vy || y == vy+vh-1) {
		return true
	}
	// Left or right border
	if inY && (x == vx || x == vx+vw-1) {
		return true
	}
	return false
}

// Legend renders the terrain legend
func (mg MapGrid) Legend() string {
	var sb strings.Builder
	sb.WriteString(legendTitleStyle.Render("  ᚱ Terrain Legend"))
	sb.WriteString("\n")

	order := []int{TerrainOcean, TerrainEmpty, TerrainSettlement, TerrainPort, TerrainRuin, TerrainForest, TerrainMountain, TerrainUnknown}
	for _, code := range order {
		td := terrainDisplays[code]
		style := lipgloss.NewStyle().Foreground(td.Color)
		sb.WriteString(fmt.Sprintf("  %s %s\n", style.Render(td.Char), td.Name))
	}

	return sb.String()
}

// Coverage calculates what % of cells are observed
func (mg MapGrid) Coverage() float64 {
	if len(mg.Grid) == 0 {
		return 0
	}
	total := 0
	observed := 0
	for _, row := range mg.Grid {
		for _, cell := range row {
			total++
			if cell != TerrainUnknown {
				observed++
			}
		}
	}
	if total == 0 {
		return 0
	}
	return float64(observed) / float64(total) * 100
}

// Stats computes terrain counts
func (mg MapGrid) Stats() map[int]int {
	counts := make(map[int]int)
	for _, row := range mg.Grid {
		for _, cell := range row {
			counts[cell]++
		}
	}
	return counts
}

// ViewportStats returns terrain counts and description for cells inside the viewport
func (mg MapGrid) ViewportStats() string {
	if len(mg.Grid) == 0 || !mg.ShowViewport {
		return ""
	}
	vw, vh := mg.ViewportW, mg.ViewportH
	if vw == 0 {
		vw = 15
	}
	if vh == 0 {
		vh = 15
	}

	counts := make(map[int]int)
	total := 0
	observed := 0
	for y := mg.ViewportY; y < mg.ViewportY+vh && y < len(mg.Grid); y++ {
		for x := mg.ViewportX; x < mg.ViewportX+vw && x < len(mg.Grid[y]); x++ {
			cell := mg.Grid[y][x]
			counts[cell]++
			total++
			if cell != TerrainUnknown {
				observed++
			}
		}
	}

	var sb strings.Builder
	sb.WriteString(legendTitleStyle.Render("  ᚨ Viewport Contents"))
	sb.WriteString(fmt.Sprintf(" (%d,%d)→(%d,%d)\n",
		mg.ViewportX, mg.ViewportY,
		mg.ViewportX+vw-1, mg.ViewportY+vh-1))

	order := []int{TerrainOcean, TerrainEmpty, TerrainSettlement, TerrainPort, TerrainRuin, TerrainForest, TerrainMountain, TerrainUnknown}
	for _, code := range order {
		c := counts[code]
		if c == 0 {
			continue
		}
		td := terrainDisplays[code]
		style := lipgloss.NewStyle().Foreground(td.Color)
		pct := float64(c) / float64(total) * 100
		bar := strings.Repeat("█", int(pct/5))
		sb.WriteString(fmt.Sprintf("  %s %-11s %3d %s\n",
			style.Render(td.Char), td.Name, c,
			style.Render(bar)))
	}

	if total > 0 {
		obsPct := float64(observed) / float64(total) * 100
		sb.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a")).
			Render(fmt.Sprintf("  Observed: %d/%d (%.0f%%)", observed, total, obsPct)))
	}

	return sb.String()
}

// terrainToClass maps terrain grid codes to prediction class indices
// Prediction classes: 0=empty, 1=settlement, 2=port, 3=ruin, 4=forest, 5=mountain
func terrainToClass(terrain int) int {
	switch terrain {
	case TerrainEmpty, TerrainPlains, TerrainOcean:
		return 0
	case TerrainSettlement:
		return 1
	case TerrainPort:
		return 2
	case TerrainRuin:
		return 3
	case TerrainForest:
		return 4
	case TerrainMountain:
		return 5
	default:
		return -1 // unknown
	}
}

// classNames for prediction classes
var classNames = []string{"Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"}
var classColors = []lipgloss.Color{
	lipgloss.Color("#3a3a5a"), // empty
	lipgloss.Color("#00e5ff"), // settlement
	lipgloss.Color("#bf00ff"), // port
	lipgloss.Color("#ff0033"), // ruin
	lipgloss.Color("#00ff41"), // forest
	lipgloss.Color("#9e9e9e"), // mountain
}

// ViewportComparison shows observed vs predicted terrain in the viewport
func (mg MapGrid) ViewportComparison() string {
	if len(mg.Grid) == 0 || len(mg.PredGrid) == 0 || !mg.ShowViewport {
		return ""
	}
	vw, vh := mg.ViewportW, mg.ViewportH
	if vw == 0 {
		vw = 15
	}
	if vh == 0 {
		vh = 15
	}

	// Count observed classes and predicted classes within viewport
	obsCounts := make([]int, 6)
	predCounts := make([]int, 6)
	matches := 0
	mismatches := 0
	unknown := 0

	for y := mg.ViewportY; y < mg.ViewportY+vh && y < len(mg.Grid); y++ {
		for x := mg.ViewportX; x < mg.ViewportX+vw && x < len(mg.Grid[y]); x++ {
			terrain := mg.Grid[y][x]
			obsClass := terrainToClass(terrain)

			predClass := -1
			if y < len(mg.PredGrid) && x < len(mg.PredGrid[y]) {
				predClass = mg.PredGrid[y][x]
			}

			if obsClass < 0 {
				unknown++
				continue
			}
			obsCounts[obsClass]++
			if predClass >= 0 && predClass < 6 {
				predCounts[predClass]++
			}

			if obsClass == predClass {
				matches++
			} else if predClass >= 0 {
				mismatches++
			}
		}
	}

	total := matches + mismatches
	var sb strings.Builder

	sb.WriteString(legendTitleStyle.Render("  ⚔ Observed vs Predicted"))
	sb.WriteString("\n")

	if total == 0 {
		sb.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a")).
			Render("  No comparable cells"))
		return sb.String()
	}

	// Accuracy
	acc := float64(matches) / float64(total) * 100
	accColor := lipgloss.Color("#00ff41")
	if acc < 80 {
		accColor = lipgloss.Color("#ff0033")
	} else if acc < 90 {
		accColor = lipgloss.Color("#00e5ff")
	}
	sb.WriteString(fmt.Sprintf("  Match: %s  (%d/%d",
		lipgloss.NewStyle().Foreground(accColor).Bold(true).Render(fmt.Sprintf("%.0f%%", acc)),
		matches, total))
	if unknown > 0 {
		sb.WriteString(fmt.Sprintf(", %d unobserved", unknown))
	}
	sb.WriteString(")\n\n")

	// Per-class comparison table
	hdr := lipgloss.NewStyle().Foreground(lipgloss.Color("#00e5ff")).Underline(true)
	sb.WriteString(hdr.Render(fmt.Sprintf("  %-11s %4s %4s %5s", "Class", "Obs", "Pred", "Diff")) + "\n")

	for c := 0; c < 6; c++ {
		obs := obsCounts[c]
		pred := predCounts[c]
		if obs == 0 && pred == 0 {
			continue
		}
		diff := pred - obs
		style := lipgloss.NewStyle().Foreground(classColors[c])
		diffStr := fmt.Sprintf("%+d", diff)
		diffStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#3a3a5a"))
		if diff > 0 {
			diffStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#bf00ff")) // over-predicted
		} else if diff < 0 {
			diffStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#ff0033")) // under-predicted
		}
		sb.WriteString(fmt.Sprintf("  %s %4d %4d %s\n",
			style.Render(fmt.Sprintf("%-11s", classNames[c])),
			obs, pred,
			diffStyle.Render(fmt.Sprintf("%5s", diffStr))))
	}

	// Blindspot detection: classes we observe but under-predict
	var blindspots []string
	for c := 0; c < 6; c++ {
		if obsCounts[c] > 0 && predCounts[c] < obsCounts[c] {
			deficit := obsCounts[c] - predCounts[c]
			pct := float64(deficit) / float64(obsCounts[c]) * 100
			if pct >= 20 {
				blindspots = append(blindspots,
					fmt.Sprintf("  %s: %d missing (%.0f%%)",
						lipgloss.NewStyle().Foreground(classColors[c]).Render(classNames[c]),
						deficit, pct))
			}
		}
	}
	if len(blindspots) > 0 {
		sb.WriteString("\n" + lipgloss.NewStyle().Foreground(lipgloss.Color("#ff0033")).Bold(true).
			Render("  ⚠ Blindspots") + "\n")
		for _, bs := range blindspots {
			sb.WriteString(bs + "\n")
		}
	}

	return sb.String()
}

// HeatmapChar returns a heat-colored block based on score 0-100
func HeatmapChar(score float64) string {
	var color lipgloss.Color
	switch {
	case score >= 95:
		color = lipgloss.Color("#00e676")
	case score >= 90:
		color = lipgloss.Color("#00ff41")
	case score >= 85:
		color = lipgloss.Color("#00e5ff")
	case score >= 80:
		color = lipgloss.Color("#ffaa00")
	case score >= 70:
		color = lipgloss.Color("#ff0033")
	default:
		color = lipgloss.Color("#b71c1c")
	}
	return lipgloss.NewStyle().Foreground(color).Render("█")
}
