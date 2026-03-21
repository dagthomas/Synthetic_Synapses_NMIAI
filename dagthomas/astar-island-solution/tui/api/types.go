package api

// Round represents a competition round
type Round struct {
	ID          string  `json:"id"`
	Number      int     `json:"round_number"`
	Status      string  `json:"status"`
	MapWidth    int     `json:"map_width"`
	MapHeight   int     `json:"map_height"`
	StartedAt   string  `json:"started_at"`
	ClosesAt    string  `json:"closes_at"`
	EventDate   string  `json:"event_date"`
	RoundWeight float64 `json:"round_weight"`
}

// RoundDetail is a full round with initial states
type RoundDetail struct {
	ID            string         `json:"id"`
	Number        int            `json:"round_number"`
	Status        string         `json:"status"`
	MapWidth      int            `json:"map_width"`
	MapHeight     int            `json:"map_height"`
	StartedAt     string         `json:"started_at"`
	ClosesAt      string         `json:"closes_at"`
	EventDate     string         `json:"event_date"`
	RoundWeight   float64        `json:"round_weight"`
	InitialStates []InitialState `json:"initial_states"`
}

// InitialState is the starting grid + settlements for one seed
type InitialState struct {
	Grid        [][]int      `json:"grid"`
	Settlements []Settlement `json:"settlements"`
}

// Settlement describes a settlement on the map
type Settlement struct {
	X          int     `json:"x"`
	Y          int     `json:"y"`
	Population float64 `json:"population"`
	Food       float64 `json:"food"`
	Wealth     float64 `json:"wealth"`
	Defense    float64 `json:"defense"`
	HasPort    bool    `json:"has_port"`
	Alive      bool    `json:"alive"`
	OwnerID    int     `json:"owner_id"`
}

// Budget is the query budget info
type Budget struct {
	QueriesUsed int `json:"queries_used"`
	QueriesMax  int `json:"queries_max"`
}

// MyRound is a round with the user's scores
type MyRound struct {
	ID             string    `json:"id"`
	RoundNumber    int       `json:"round_number"`
	Status         string    `json:"status"`
	Score          *float64  `json:"round_score"`
	Rank           *int      `json:"rank"`
	SeedScores     []float64 `json:"seed_scores"`
	SeedsSubmitted int       `json:"seeds_submitted"`
	SeedsCount     int       `json:"seeds_count"`
	QueriesUsed    int       `json:"queries_used"`
	QueriesMax     int       `json:"queries_max"`
}

// Analysis is the post-round ground truth comparison
type Analysis struct {
	RoundID    string           `json:"round_id"`
	SeedIndex  int              `json:"seed_index"`
	Score      float64          `json:"score"`
	GroundTruth [][][]float64   `json:"ground_truth"`
	Prediction  [][][]float64   `json:"prediction"`
	CellScores  [][]float64     `json:"cell_scores"`
}

// LeaderboardEntry is one row of the leaderboard
type LeaderboardEntry struct {
	Rank               int     `json:"rank"`
	TeamName           string  `json:"team_name"`
	Score              float64 `json:"weighted_score"`
	RoundsParticipated int     `json:"rounds_participated"`
	HotStreakScore     float64 `json:"hot_streak_score"`
}

// Observation is a single viewport query result
type Observation struct {
	SeedIndex   int          `json:"seed_index"`
	QueryNum    int          `json:"query_num"`
	Viewport    Viewport     `json:"viewport"`
	Grid        [][]int      `json:"grid"`
	Settlements []Settlement `json:"settlements"`
}

// Viewport describes the observation window
type Viewport struct {
	X int `json:"x"`
	Y int `json:"y"`
	W int `json:"w"`
	H int `json:"h"`
}

// AutoloopEntry is one experiment from autoloop_log.jsonl
type AutoloopEntry struct {
	ID          int                    `json:"id"`
	Timestamp   string                 `json:"timestamp"`
	Name        string                 `json:"name"`
	Params      map[string]float64     `json:"params"`
	ScoresQuick map[string]float64     `json:"scores_quick"`
	ScoresFull  map[string]float64     `json:"scores_full"`
	Accepted    bool                   `json:"accepted"`
	BaselineAvg float64                `json:"baseline_avg"`
	Elapsed     float64                `json:"elapsed"`
}

// ResearchEntry is one experiment from research logs
type ResearchEntry struct {
	ID          int                `json:"id"`
	Timestamp   string             `json:"timestamp"`
	Name        string             `json:"name"`
	Status      string             `json:"status"`
	Hypothesis  string             `json:"hypothesis"`
	Model       string             `json:"model"`
	Scores      map[string]float64 `json:"scores"`
	Improvement float64            `json:"improvement"`
	Warnings    []string           `json:"warnings"`
	Elapsed     float64            `json:"elapsed"`
	Error       string             `json:"error"`
	Code        string             `json:"code"`
	Timings     *ResearchTimings   `json:"timings"`
}

// ResearchTimings from multi_researcher
type ResearchTimings struct {
	Analysis float64 `json:"analysis"`
	Code     float64 `json:"code"`
	Backtest float64 `json:"backtest"`
	Total    float64 `json:"total"`
}

// Normalize fills in derived fields for entries from different log formats
func (e *ResearchEntry) Normalize() {
	// Derive status from error field if missing
	if e.Status == "" {
		if e.Error == "" {
			e.Status = "success"
		} else {
			e.Status = "error"
		}
	}
	// Use timings.total if elapsed is missing
	if e.Elapsed == 0 && e.Timings != nil {
		e.Elapsed = e.Timings.Total
	}
}

// GeminiResearchEntry from gemini_research_log.jsonl
type GeminiResearchEntry struct {
	ID              int                `json:"id"`
	Timestamp       string             `json:"timestamp"`
	Name            string             `json:"name"`
	Hypothesis      string             `json:"hypothesis"`
	Scores          map[string]float64 `json:"scores"`
	Improvement     interface{}        `json:"improvement"`  // can be string or float
	Elapsed         float64            `json:"elapsed"`
	Error           string             `json:"error"`
	ProposalSummary string             `json:"proposal_summary"`
}
