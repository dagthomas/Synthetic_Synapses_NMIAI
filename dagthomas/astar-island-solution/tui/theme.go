package main

import "github.com/charmbracelet/lipgloss"

// Cyberpunk × Norse color palette
var (
	ColorBg        = lipgloss.Color("#0a0a12")
	ColorBgPanel   = lipgloss.Color("#0d0d1a")
	ColorBgBar     = lipgloss.Color("#06060e")
	ColorFg        = lipgloss.Color("#c0c8e0")
	ColorGold      = lipgloss.Color("#00e5ff") // Primary: neon cyan
	ColorMuted     = lipgloss.Color("#3a3a5a")
	ColorSuccess   = lipgloss.Color("#00ff41") // Matrix green
	ColorDanger    = lipgloss.Color("#ff0033") // Glitch red
	ColorInfo      = lipgloss.Color("#bf00ff") // Electric purple
	ColorSettlement = lipgloss.Color("#00e5ff") // Neon cyan
	ColorPort      = lipgloss.Color("#ff2d95") // Hot pink
	ColorRuin      = lipgloss.Color("#ff0033") // Glitch red
	ColorForest    = lipgloss.Color("#00ff41") // Matrix green
	ColorMountain  = lipgloss.Color("#9d4edd") // Purple
	ColorEmpty     = lipgloss.Color("#2a2a44") // Dark
	ColorOcean     = lipgloss.Color("#0044aa") // Deep neon blue
	ColorUnknown   = lipgloss.Color("#1a1a2e") // Void
	ColorAccent2   = lipgloss.Color("#ff2d95") // Hot pink
	ColorWarm      = lipgloss.Color("#ffaa00") // Amber warning
	ColorNeonCyan  = lipgloss.Color("#00ffff") // Pure neon cyan
	ColorNeonPink  = lipgloss.Color("#ff0080") // Pure neon pink
)

// Rune decorations — Norse glyphs in neon context
const (
	RuneTopLeft     = "ᛟ"
	RuneTopRight    = "ᛟ"
	RuneSeparator   = "᛫"
	RuneArrow       = "ᚱ"
	RuneShield      = "ᛞ"
	RuneSun         = "ᛋ"
	RuneWealth      = "ᚠ"
	RuneJoy         = "ᚹ"
	RuneVictory     = "ᛏ"
	RuneHorse       = "ᛖ"
	RuneWater       = "ᛚ"
	RuneIce         = "ᛁ"
	RuneHagalaz     = "ᚺ"
	RuneNeed        = "ᚾ"
	RuneBirch       = "ᛒ"
	RuneAnsuz       = "ᚨ"
	RuneKenaz       = "ᚲ"
	RuneGebo        = "ᚷ"
	RuneEhwaz       = "ᛖ"
	RuneDagaz       = "ᛞ"
	RuneOthala      = "ᛟ"
	RuneBorder      = "─"
	RuneCornerTL    = "┌"
	RuneCornerTR    = "┐"
	RuneCornerBL    = "└"
	RuneCornerBR    = "┘"
	RuneVertical    = "│"
	GlitchBlock     = "█"
	GlitchHalf      = "▓"
	GlitchLight     = "░"
	CyberArrow      = "►"
	CyberDot        = "◈"
	CyberDiamond    = "◇"
)

// Cyberpunk × Norse ASCII banner
const VikingBanner = `
  ░▒▓ ᛟ ᚨ ᛋ ᛏ ᚨ ᚱ ᛫ ᛁ ᛋ ᛚ ᚨ ᚾ ᛞ ᛟ ▓▒░
  ╔══════════════════════════════════╗
  ║  ⛵  ASTAR ISLAND  //EXPLORER  ⚔  ║
  ╚══════════════════════════════════╝`

// Compact header runes
const HeaderRunes = " ░▒▓ ᛟ᛫ᚨ᛫ᛋ᛫ᛏ᛫ᚨ᛫ᚱ᛫ᛁ᛫ᛋ᛫ᛚ᛫ᚨ᛫ᚾ᛫ᛞ᛫ᛟ ▓▒░ "

// Shared styles
var (
	StyleApp = lipgloss.NewStyle().
			Background(ColorBg)

	StyleTitle = lipgloss.NewStyle().
			Foreground(ColorNeonCyan).
			Bold(true)

	StyleSubtitle = lipgloss.NewStyle().
			Foreground(ColorFg).
			Italic(true)

	StyleMuted = lipgloss.NewStyle().
			Foreground(ColorMuted)

	StyleSuccess = lipgloss.NewStyle().
			Foreground(ColorSuccess).
			Bold(true)

	StyleDanger = lipgloss.NewStyle().
			Foreground(ColorDanger).
			Bold(true)

	StyleInfo = lipgloss.NewStyle().
			Foreground(ColorInfo)

	StyleGold = lipgloss.NewStyle().
			Foreground(ColorNeonCyan)

	StyleBorder = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(ColorNeonCyan).
			Padding(0, 1)

	StylePanel = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#1a3a5a")).
			Padding(0, 1)

	StyleActiveTab = lipgloss.NewStyle().
			Foreground(ColorBg).
			Background(ColorNeonCyan).
			Bold(true).
			Padding(0, 1)

	StyleInactiveTab = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#4a4a6a")).
			Background(ColorBgBar).
			Padding(0, 1)

	StyleStatusBar = lipgloss.NewStyle().
			Foreground(ColorFg).
			Background(ColorBgBar).
			Padding(0, 1)

	StyleKeyHint = lipgloss.NewStyle().
			Foreground(ColorNeonCyan).
			Background(ColorBgBar)

	StyleTableHeader = lipgloss.NewStyle().
			Foreground(ColorNeonCyan).
			Bold(true).
			Underline(true)

	StyleTableRow = lipgloss.NewStyle().
			Foreground(ColorFg)

	StyleHighlight = lipgloss.NewStyle().
			Foreground(ColorNeonPink).
			Bold(true)

	StyleScoreGood = lipgloss.NewStyle().
			Foreground(ColorSuccess).
			Bold(true)

	StyleScoreBad = lipgloss.NewStyle().
			Foreground(ColorDanger)

	StyleSeedLabel = lipgloss.NewStyle().
			Foreground(ColorInfo).
			Bold(true)
)

// Terrain styling — cyberpunk neon on void
func TerrainStyle(code int) lipgloss.Style {
	switch code {
	case 10: // Ocean
		return lipgloss.NewStyle().Foreground(ColorOcean)
	case 11: // Plains
		return lipgloss.NewStyle().Foreground(lipgloss.Color("#1a4a2a"))
	case 0: // Empty
		return lipgloss.NewStyle().Foreground(ColorEmpty)
	case 1: // Settlement
		return lipgloss.NewStyle().Foreground(ColorNeonCyan).Bold(true)
	case 2: // Port
		return lipgloss.NewStyle().Foreground(ColorNeonPink).Bold(true)
	case 3: // Ruin
		return lipgloss.NewStyle().Foreground(ColorDanger)
	case 4: // Forest
		return lipgloss.NewStyle().Foreground(ColorSuccess)
	case 5: // Mountain
		return lipgloss.NewStyle().Foreground(ColorInfo).Bold(true)
	default: // Unknown
		return lipgloss.NewStyle().Foreground(ColorUnknown)
	}
}

// Terrain character — Norse runes in the grid
func TerrainChar(code int) string {
	switch code {
	case 10:
		return "~" // Ocean
	case 11:
		return "·" // Plains
	case 0:
		return "░" // Empty — cyberpunk void
	case 1:
		return "ᛋ" // Settlement (Sol rune)
	case 2:
		return "⚓" // Port
	case 3:
		return "ᚱ" // Ruin (Raido rune)
	case 4:
		return "♠" // Forest
	case 5:
		return "▲" // Mountain
	default:
		return "?" // Unknown
	}
}

func TerrainName(code int) string {
	switch code {
	case 10:
		return "Ocean"
	case 11:
		return "Plains"
	case 0:
		return "Empty"
	case 1:
		return "Settlement"
	case 2:
		return "Port"
	case 3:
		return "Ruin"
	case 4:
		return "Forest"
	case 5:
		return "Mountain"
	default:
		return "Unknown"
	}
}

// Score color based on value — cyberpunk neon gradient
func ScoreStyle(score float64) lipgloss.Style {
	switch {
	case score >= 93:
		return lipgloss.NewStyle().Foreground(lipgloss.Color("#00ffff")).Bold(true) // Pure neon cyan
	case score >= 90:
		return lipgloss.NewStyle().Foreground(ColorSuccess)
	case score >= 85:
		return lipgloss.NewStyle().Foreground(ColorInfo)
	case score >= 80:
		return lipgloss.NewStyle().Foreground(ColorWarm)
	default:
		return lipgloss.NewStyle().Foreground(ColorDanger)
	}
}
