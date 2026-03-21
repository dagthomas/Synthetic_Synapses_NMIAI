package main

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// LayoutState is the serializable form of a LayoutNode tree.
type LayoutState struct {
	Kind     LayoutKind   `json:"kind"`
	TabIdx   int          `json:"tab_idx,omitempty"`
	Ratio    float64      `json:"ratio"`
	Left     *LayoutState `json:"left,omitempty"`
	Right    *LayoutState `json:"right,omitempty"`
	Focused  []int        `json:"focused,omitempty"` // path to focused leaf (only on root)
}

// layoutToState converts a LayoutNode tree to serializable state.
func layoutToState(node *LayoutNode, focused *LayoutNode) LayoutState {
	s := LayoutState{
		Kind:  node.Kind,
		Ratio: node.Ratio,
	}
	if node.IsLeaf() {
		s.TabIdx = node.TabIdx
	} else {
		left := layoutToState(node.Left, focused)
		right := layoutToState(node.Right, focused)
		s.Left = &left
		s.Right = &right
	}
	// Encode focused path on root
	if focused != nil {
		s.Focused = focusedPath(node, focused)
	}
	return s
}

// focusedPath returns the path from root to the focused leaf (0 = left, 1 = right).
func focusedPath(root, target *LayoutNode) []int {
	if root == target {
		return []int{}
	}
	if root.IsLeaf() {
		return nil
	}
	if p := focusedPath(root.Left, target); p != nil {
		return append([]int{0}, p...)
	}
	if p := focusedPath(root.Right, target); p != nil {
		return append([]int{1}, p...)
	}
	return nil
}

// stateToLayout converts serialized state back to a LayoutNode tree.
func stateToLayout(s *LayoutState) *LayoutNode {
	if s == nil {
		return NewLayout()
	}
	node := &LayoutNode{
		Kind:  s.Kind,
		Ratio: s.Ratio,
	}
	if s.Kind == LayoutLeaf {
		node.TabIdx = s.TabIdx
	} else {
		node.Left = stateToLayout(s.Left)
		node.Right = stateToLayout(s.Right)
		node.Left.parent = node
		node.Right.parent = node
	}
	return node
}

// resolveFocused follows a path to find the focused node.
func resolveFocused(root *LayoutNode, path []int) *LayoutNode {
	node := root
	for _, step := range path {
		if node.IsLeaf() {
			return node
		}
		if step == 0 {
			node = node.Left
		} else {
			node = node.Right
		}
	}
	return node
}

// SaveLayout writes the layout state to a JSON file.
func SaveLayout(dataDir string, layout *LayoutNode, focused *LayoutNode) error {
	state := layoutToState(layout, focused)
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	path := filepath.Join(dataDir, "tui_layout.json")
	return os.WriteFile(path, data, 0644)
}

// LoadLayout reads the layout state from a JSON file.
// Returns nil, nil if the file doesn't exist.
func LoadLayout(dataDir string) (*LayoutNode, *LayoutNode, error) {
	path := filepath.Join(dataDir, "tui_layout.json")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil, nil
		}
		return nil, nil, err
	}

	var state LayoutState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, nil, err
	}

	layout := stateToLayout(&state)
	focused := resolveFocused(layout, state.Focused)
	return layout, focused, nil
}
