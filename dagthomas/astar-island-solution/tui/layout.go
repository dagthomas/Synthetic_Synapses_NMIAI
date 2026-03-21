package main

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

// LayoutKind identifies the type of layout node.
type LayoutKind int

const (
	LayoutLeaf   LayoutKind = iota
	LayoutVSplit            // vertical divider — panes side-by-side
	LayoutHSplit            // horizontal divider — panes stacked
)

// Direction for pane navigation.
type Direction int

const (
	DirLeft Direction = iota
	DirRight
	DirUp
	DirDown
)

// LayoutNode is a binary split tree. Leaves hold a tab index; internal nodes
// hold a split direction and ratio.
type LayoutNode struct {
	Kind   LayoutKind
	Left   *LayoutNode
	Right  *LayoutNode
	TabIdx int
	Ratio  float64
	parent *LayoutNode
}

// NewLayout creates a single-pane layout showing tab 0.
func NewLayout() *LayoutNode {
	return &LayoutNode{Kind: LayoutLeaf, TabIdx: 0, Ratio: 0.5}
}

// IsLeaf returns true if this node displays a tab.
func (n *LayoutNode) IsLeaf() bool {
	return n.Kind == LayoutLeaf
}

// Split converts this leaf into a split with two children.
// Returns the left/top child (keeps focus position).
func (n *LayoutNode) Split(kind LayoutKind) *LayoutNode {
	if !n.IsLeaf() {
		return n
	}
	left := &LayoutNode{Kind: LayoutLeaf, TabIdx: n.TabIdx, Ratio: 0.5, parent: n}
	right := &LayoutNode{Kind: LayoutLeaf, TabIdx: n.TabIdx, Ratio: 0.5, parent: n}
	n.Kind = kind
	n.Left = left
	n.Right = right
	n.Ratio = 0.5
	n.TabIdx = -1
	return left
}

// Close removes this leaf and promotes its sibling. Returns a leaf to focus.
func (n *LayoutNode) Close() *LayoutNode {
	p := n.parent
	if p == nil {
		return n
	}
	var sibling *LayoutNode
	if p.Left == n {
		sibling = p.Right
	} else {
		sibling = p.Left
	}
	p.Kind = sibling.Kind
	p.TabIdx = sibling.TabIdx
	p.Ratio = sibling.Ratio
	p.Left = sibling.Left
	p.Right = sibling.Right
	if p.Left != nil {
		p.Left.parent = p
	}
	if p.Right != nil {
		p.Right.parent = p
	}
	return p.FirstLeaf()
}

// Navigate returns the nearest leaf in the given direction, or nil.
func (n *LayoutNode) Navigate(dir Direction) *LayoutNode {
	child := n
	p := n.parent
	for p != nil {
		switch dir {
		case DirRight:
			if p.Kind == LayoutVSplit && child == p.Left {
				return p.Right.edgeLeaf(DirLeft)
			}
		case DirLeft:
			if p.Kind == LayoutVSplit && child == p.Right {
				return p.Left.edgeLeaf(DirRight)
			}
		case DirDown:
			if p.Kind == LayoutHSplit && child == p.Left {
				return p.Right.edgeLeaf(DirUp)
			}
		case DirUp:
			if p.Kind == LayoutHSplit && child == p.Right {
				return p.Left.edgeLeaf(DirDown)
			}
		}
		child = p
		p = p.parent
	}
	return nil
}

// edgeLeaf descends to the leaf on the entry side of a subtree.
func (n *LayoutNode) edgeLeaf(from Direction) *LayoutNode {
	if n.IsLeaf() {
		return n
	}
	switch from {
	case DirLeft, DirUp:
		return n.Left.edgeLeaf(from)
	default:
		return n.Right.edgeLeaf(from)
	}
}

// FirstLeaf returns the leftmost/topmost leaf.
func (n *LayoutNode) FirstLeaf() *LayoutNode {
	if n.IsLeaf() {
		return n
	}
	return n.Left.FirstLeaf()
}

// Leaves collects all leaves in order.
func (n *LayoutNode) Leaves() []*LayoutNode {
	if n.IsLeaf() {
		return []*LayoutNode{n}
	}
	return append(n.Left.Leaves(), n.Right.Leaves()...)
}

// LeafCount returns the number of panes.
func (n *LayoutNode) LeafCount() int {
	if n.IsLeaf() {
		return 1
	}
	return n.Left.LeafCount() + n.Right.LeafCount()
}

// Resize adjusts the parent split's ratio. Positive delta always grows the
// focused pane regardless of whether it is the left or right child.
func (n *LayoutNode) Resize(delta float64) {
	p := n.parent
	if p == nil {
		return
	}
	if n == p.Right {
		delta = -delta
	}
	r := p.Ratio + delta
	if r < 0.15 {
		r = 0.15
	}
	if r > 0.85 {
		r = 0.85
	}
	p.Ratio = r
}

// splitDim divides total into two parts with a 1-unit divider between them.
func splitDim(total int, ratio float64) (int, int) {
	left := int(float64(total) * ratio)
	if left < 1 {
		left = 1
	}
	right := total - left - 1
	if right < 1 {
		right = 1
		left = total - 2
		if left < 1 {
			left = 1
		}
	}
	return left, right
}

// --- pane accent colors ---

// PaneAccents are foreground accent colors for distinguishing panes in split layouts.
// Used for borders, dividers, and header text — background stays dark.
var PaneAccents = []lipgloss.Color{
	"#00e5ff", // neon cyan (primary)
	"#ff0080", // neon pink
	"#bf00ff", // electric purple
	"#00ff41", // matrix green
	"#ffaa00", // amber
	"#ff0033", // glitch red
	"#00ffaa", // neon teal
	"#ff6ec7", // hot magenta
	"#44aaff", // electric blue
}

// PaneAccent returns the accent color for a given leaf index.
func PaneAccent(leafIdx int) lipgloss.Color {
	return PaneAccents[leafIdx%len(PaneAccents)]
}

// --- rendering helpers ---

var (
	dividerMutedStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#1a1a2e"))
	dividerFocusStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#00e5ff"))
)

func renderVDivider(h int, focused bool) string {
	st := dividerMutedStyle
	if focused {
		st = dividerFocusStyle
	}
	lines := make([]string, h)
	for i := range lines {
		if i == 0 {
			lines[i] = st.Render("┬")
		} else if i == h-1 {
			lines[i] = st.Render("┴")
		} else if focused && i == h/2 {
			lines[i] = st.Render("◈")
		} else if focused && (i == h/2-1 || i == h/2+1) {
			lines[i] = st.Render("║")
		} else {
			lines[i] = st.Render("│")
		}
	}
	return strings.Join(lines, "\n")
}

func renderHDivider(w int, focused bool) string {
	st := dividerMutedStyle
	if focused {
		st = dividerFocusStyle
	}
	if w <= 2 {
		return st.Render(strings.Repeat("─", w))
	}
	mid := w / 2
	var sb strings.Builder
	for i := range w {
		if i == 0 {
			sb.WriteString("├")
		} else if i == w-1 {
			sb.WriteString("┤")
		} else if focused && i == mid {
			sb.WriteString("◈")
		} else if focused && (i == mid-1 || i == mid+1) {
			sb.WriteString("═")
		} else {
			sb.WriteString("─")
		}
	}
	return st.Render(sb.String())
}

// renderVDividerAccent renders a vertical divider with a specific accent color.
func renderVDividerAccent(h int, color lipgloss.Color) string {
	st := lipgloss.NewStyle().Foreground(color)
	lines := make([]string, h)
	for i := range lines {
		if i == 0 {
			lines[i] = st.Render("┬")
		} else if i == h-1 {
			lines[i] = st.Render("┴")
		} else if i == h/2 {
			lines[i] = st.Render("◈")
		} else if i == h/2-1 || i == h/2+1 {
			lines[i] = st.Render("║")
		} else {
			lines[i] = st.Render("│")
		}
	}
	return strings.Join(lines, "\n")
}

// renderHDividerAccent renders a horizontal divider with a specific accent color.
func renderHDividerAccent(w int, color lipgloss.Color) string {
	st := lipgloss.NewStyle().Foreground(color)
	if w <= 2 {
		return st.Render(strings.Repeat("─", w))
	}
	mid := w / 2
	var sb strings.Builder
	for i := range w {
		if i == 0 {
			sb.WriteString("├")
		} else if i == w-1 {
			sb.WriteString("┤")
		} else if i == mid {
			sb.WriteString("◈")
		} else if i == mid-1 || i == mid+1 {
			sb.WriteString("═")
		} else {
			sb.WriteString("─")
		}
	}
	return st.Render(sb.String())
}

// padToSize ensures content is exactly w columns by h lines.
func padToSize(content string, w, h int) string {
	if w <= 0 || h <= 0 {
		return ""
	}
	lines := strings.Split(content, "\n")
	if len(lines) > h {
		lines = lines[:h]
	}
	for i, line := range lines {
		lw := lipgloss.Width(line)
		if lw < w {
			lines[i] = line + strings.Repeat(" ", w-lw)
		} else if lw > w {
			lines[i] = ansi.Truncate(line, w, "")
		}
	}
	empty := strings.Repeat(" ", w)
	for len(lines) < h {
		lines = append(lines, empty)
	}
	return strings.Join(lines, "\n")
}
