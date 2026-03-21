package components

import (
	"fmt"
	"math"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

var sparkBlocks = []string{"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"}

var (
	sparkLabelStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a"))

	sparkBarStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00e5ff"))

	sparkBestStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00ff41")).
			Bold(true)

	sparkAxisStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#3a3a5a"))
)

// Sparkline renders a text-based score chart
type Sparkline struct {
	Title  string
	Values []float64
	Width  int
	Height int // in lines (each line = one row of block chars)
}

// View renders the sparkline
func (s Sparkline) View() string {
	if len(s.Values) == 0 {
		return sparkLabelStyle.Render("  No data yet...")
	}

	width := s.Width
	if width <= 0 {
		width = 60
	}

	// Take last `width` values
	vals := s.Values
	if len(vals) > width {
		vals = vals[len(vals)-width:]
	}

	// Find min/max for scaling
	minVal, maxVal := vals[0], vals[0]
	for _, v := range vals {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	// Add some padding to range
	rangeVal := maxVal - minVal
	if rangeVal < 0.1 {
		rangeVal = 1.0
		minVal -= 0.5
	}

	// Build sparkline string
	var spark strings.Builder
	bestIdx := 0
	bestVal := vals[0]
	for i, v := range vals {
		if v > bestVal {
			bestVal = v
			bestIdx = i
		}
		normalized := (v - minVal) / rangeVal
		idx := int(math.Round(normalized * float64(len(sparkBlocks)-1)))
		if idx < 0 {
			idx = 0
		}
		if idx >= len(sparkBlocks) {
			idx = len(sparkBlocks) - 1
		}
		if i == bestIdx && i == len(vals)-1 {
			spark.WriteString(sparkBestStyle.Render(sparkBlocks[idx]))
		} else {
			spark.WriteString(sparkBarStyle.Render(sparkBlocks[idx]))
		}
	}

	// Build the chart
	var lines []string

	title := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#00e5ff")).
		Bold(true).
		Render(s.Title)

	lines = append(lines, title)

	// Axis labels
	maxLabel := sparkAxisStyle.Render(fmt.Sprintf("%.1f", maxVal))
	minLabel := sparkAxisStyle.Render(fmt.Sprintf("%.1f", minVal))

	lines = append(lines, maxLabel+" "+sparkAxisStyle.Render("┐"))
	lines = append(lines, "      "+spark.String())
	lines = append(lines, minLabel+" "+sparkAxisStyle.Render("┘")+"  "+
		sparkBestStyle.Render(fmt.Sprintf("best: %.2f", bestVal))+
		sparkLabelStyle.Render(fmt.Sprintf("  (%d pts)", len(vals))))

	return strings.Join(lines, "\n")
}

// MultiSparkline renders multiple named sparklines stacked
type MultiSparkline struct {
	Lines []NamedSeries
	Width int
}

// NamedSeries is a named data series
type NamedSeries struct {
	Name   string
	Values []float64
	Color  lipgloss.Color
}

// View renders stacked sparklines
func (ms MultiSparkline) View() string {
	var parts []string
	for _, series := range ms.Lines {
		sl := Sparkline{
			Title:  series.Name,
			Values: series.Values,
			Width:  ms.Width,
		}
		parts = append(parts, sl.View())
	}
	return strings.Join(parts, "\n\n")
}

// BarChart renders a horizontal bar chart
type BarChart struct {
	Title   string
	Labels  []string
	Values  []float64
	MaxWidth int
	Color   lipgloss.Color
}

// View renders the bar chart
func (bc BarChart) View() string {
	if len(bc.Values) == 0 {
		return ""
	}

	maxWidth := bc.MaxWidth
	if maxWidth <= 0 {
		maxWidth = 30
	}

	maxVal := bc.Values[0]
	for _, v := range bc.Values {
		if v > maxVal {
			maxVal = v
		}
	}
	if maxVal <= 0 {
		maxVal = 1
	}

	color := bc.Color
	if color == "" {
		color = lipgloss.Color("#00e5ff")
	}

	barStyle := lipgloss.NewStyle().Foreground(color)
	labelStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#c0c8e0")).Width(12).Align(lipgloss.Right)

	var lines []string
	if bc.Title != "" {
		lines = append(lines, lipgloss.NewStyle().
			Foreground(lipgloss.Color("#00e5ff")).
			Bold(true).
			Render(bc.Title))
	}

	for i, v := range bc.Values {
		label := ""
		if i < len(bc.Labels) {
			label = bc.Labels[i]
		}
		barLen := int(v / maxVal * float64(maxWidth))
		if barLen < 1 && v > 0 {
			barLen = 1
		}
		bar := barStyle.Render(strings.Repeat("█", barLen))
		valStr := sparkLabelStyle.Render(fmt.Sprintf(" %.1f", v))
		lines = append(lines, labelStyle.Render(label)+" "+bar+valStr)
	}

	return strings.Join(lines, "\n")
}
