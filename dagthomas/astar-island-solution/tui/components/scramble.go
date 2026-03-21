package components

import (
	"math/rand"
	"strings"
	"time"
	"unicode/utf8"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// Glitch character pools — Norse runes + cyberpunk symbols + ASCII junk
const (
	runePool  = "ᛟᚨᛋᛏᚱᛁᛚᚾᛞᚹᚷᛒᚺᚲᛖᛗᚠ"
	cyberPool = "░▒▓█╬╫╪┼╳◈◇▪▫●○"
	asciiPool = "!@#$%^&*<>{}[]|/\\~`0123456789"
)

var (
	glitchBright = lipgloss.NewStyle().Foreground(lipgloss.Color("#00e5ff"))
	glitchDim    = lipgloss.NewStyle().Foreground(lipgloss.Color("#0a2a3a"))
)

// ScrambleTickMsg advances the scramble animation one step
type ScrambleTickMsg struct{}

// ScrambleModel handles hacker-style text reveal.
// Characters appear as random glyphs, then scramble through 3-5 wrong chars,
// then resolve to the real character — sequentially, one char every tick.
type ScrambleModel struct {
	target    []rune   // final text as runes
	cursor    int      // next char index to begin decoding
	settled   int      // chars fully decoded (cursor - scrambleLen behind)
	done      bool
	active    bool
	rng       *rand.Rand
	pools     []rune   // combined glyph pool
	initGlyph []rune   // initial random glyph per char (shown before scramble reaches it)
	scrambles []int    // per-char: how many scramble cycles remaining (0 = settled)
}

const (
	scrambleCycles = 5  // each char cycles through 5 random glyphs before settling
	tickInterval   = 15 // ms between each character starting to decode
)

// NewScramble creates a new scramble animator
func NewScramble() ScrambleModel {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	// Build combined pool
	var pools []rune
	for _, r := range runePool {
		pools = append(pools, r)
	}
	for _, r := range cyberPool {
		pools = append(pools, r)
	}
	for _, r := range asciiPool {
		pools = append(pools, r)
	}
	return ScrambleModel{
		rng:   rng,
		pools: pools,
	}
}

// SetTarget starts a new scramble animation for the given text
func (s ScrambleModel) SetTarget(text string) ScrambleModel {
	runes := []rune(text)
	n := len(runes)

	s.target = runes
	s.cursor = 0
	s.settled = 0
	s.done = false
	s.active = true

	// Pre-fill every non-whitespace char with a random glyph
	s.initGlyph = make([]rune, n)
	s.scrambles = make([]int, n)
	for i, ch := range runes {
		if ch == '\n' || ch == '\r' || ch == ' ' || ch == '\t' {
			s.initGlyph[i] = ch
			s.scrambles[i] = 0
		} else {
			s.initGlyph[i] = s.pools[s.rng.Intn(len(s.pools))]
			s.scrambles[i] = -1 // not yet started
		}
	}

	return s
}

// Tick advances the scramble by one step:
// - Start decoding the next character (set its scramble counter)
// - Decrement all active scramble counters (re-randomize each tick)
// - Any counter hitting 0 = that char is now settled
func (s ScrambleModel) Tick() ScrambleModel {
	if s.done || !s.active {
		return s
	}

	n := len(s.target)

	// Advance cursor: start decoding the next char
	// Process multiple chars per tick for speed (2 chars per 15ms tick = ~130 chars/sec)
	for i := 0; i < 3; i++ {
		if s.cursor < n {
			ch := s.target[s.cursor]
			if ch == '\n' || ch == '\r' || ch == ' ' || ch == '\t' {
				s.scrambles[s.cursor] = 0
			} else {
				s.scrambles[s.cursor] = scrambleCycles
			}
			s.cursor++
		}
	}

	// Tick all active scrambles: decrement counter, re-randomize glyph
	allSettled := true
	for i := 0; i < n; i++ {
		if s.scrambles[i] > 0 {
			s.scrambles[i]--
			// Re-randomize the display glyph while still scrambling
			if s.scrambles[i] > 0 {
				s.initGlyph[i] = s.pools[s.rng.Intn(len(s.pools))]
			}
			allSettled = false
		} else if s.scrambles[i] == -1 {
			allSettled = false // not yet started
		}
	}

	if allSettled && s.cursor >= n {
		s.done = true
		s.active = false
	}

	return s
}

// ScrambleTickCmd returns the tea.Cmd to schedule the next scramble frame
func ScrambleTickCmd() tea.Cmd {
	return tea.Tick(time.Millisecond*tickInterval, func(t time.Time) tea.Msg {
		return ScrambleTickMsg{}
	})
}

// IsDone returns whether the animation has finished
func (s ScrambleModel) IsDone() bool {
	return s.done || !s.active
}

// IsActive returns whether the scramble is currently running
func (s ScrambleModel) IsActive() bool {
	return s.active
}

// View returns the current scrambled text.
func (s ScrambleModel) View(content string) string {
	if !s.active && s.done {
		return content // animation finished
	}
	if !s.active {
		return content // not started
	}

	n := len(s.target)
	if n == 0 {
		return content
	}

	// Estimate byte capacity
	var sb strings.Builder
	sb.Grow(utf8.UTFMax * n)

	for i, ch := range s.target {
		sc := s.scrambles[i]

		if ch == '\n' || ch == '\r' || ch == ' ' || ch == '\t' {
			// Whitespace — always pass through
			sb.WriteRune(ch)
		} else if sc == 0 {
			// Settled — show real character
			sb.WriteRune(ch)
		} else if sc > 0 {
			// Actively scrambling — show random glyph in bright neon
			sb.WriteString(glitchBright.Render(string(s.initGlyph[i])))
		} else {
			// sc == -1: not yet reached by cursor — show initial random glyph dimmed
			sb.WriteString(glitchDim.Render(string(s.initGlyph[i])))
		}
	}

	return sb.String()
}
