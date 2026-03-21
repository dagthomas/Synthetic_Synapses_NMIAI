package api

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const (
	BaseURL    = "https://api.ainm.no"
	BasePath   = "/astar-island"
	MaxRetries = 5
	Timeout    = 10 * time.Second
)

// Client is the Go equivalent of client.py AstarIslandClient
type Client struct {
	token      string
	httpClient *http.Client
	baseURL    string
}

// NewClient creates a new API client
func NewClient(token string) *Client {
	return &Client{
		token: token,
		httpClient: &http.Client{
			Timeout: Timeout,
		},
		baseURL: BaseURL + BasePath,
	}
}

func (c *Client) get(path string) ([]byte, error) {
	var lastErr error
	for attempt := range MaxRetries {
		req, err := http.NewRequest("GET", c.baseURL+path, nil)
		if err != nil {
			return nil, fmt.Errorf("creating request: %w", err)
		}
		req.Header.Set("Authorization", "Bearer "+c.token)

		resp, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = err
			time.Sleep(time.Duration(1<<attempt) * time.Second)
			continue
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode == 429 && attempt < MaxRetries-1 {
			wait := time.Duration(1<<attempt+1) * time.Second
			time.Sleep(wait)
			continue
		}
		if resp.StatusCode >= 400 {
			return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
		}
		return body, nil
	}
	return nil, fmt.Errorf("max retries exceeded: %v", lastErr)
}

// GetRounds lists all rounds
func (c *Client) GetRounds() ([]Round, error) {
	data, err := c.get("/rounds")
	if err != nil {
		return nil, err
	}
	var rounds []Round
	if err := json.Unmarshal(data, &rounds); err != nil {
		return nil, fmt.Errorf("%w (response: %.200s)", err, string(data))
	}
	return rounds, nil
}

// GetActiveRound returns the currently active round, or the newest round as fallback
func (c *Client) GetActiveRound() (*Round, error) {
	rounds, err := c.GetRounds()
	if err != nil {
		return nil, err
	}
	for _, r := range rounds {
		if r.Status == "active" {
			return &r, nil
		}
	}
	// No active round — return the one with the highest round_number
	if len(rounds) > 0 {
		best := &rounds[0]
		for i := range rounds {
			if rounds[i].Number > best.Number {
				best = &rounds[i]
			}
		}
		return best, nil
	}
	return nil, nil
}

// GetRoundDetail returns full round detail including initial states
func (c *Client) GetRoundDetail(roundID string) (*RoundDetail, error) {
	data, err := c.get("/rounds/" + roundID)
	if err != nil {
		return nil, err
	}
	var detail RoundDetail
	return &detail, json.Unmarshal(data, &detail)
}

// GetBudget returns the query budget
func (c *Client) GetBudget() (*Budget, error) {
	data, err := c.get("/budget")
	if err != nil {
		return nil, err
	}
	var budget Budget
	return &budget, json.Unmarshal(data, &budget)
}

// GetMyRounds returns rounds with scores
func (c *Client) GetMyRounds() ([]MyRound, error) {
	data, err := c.get("/my-rounds")
	if err != nil {
		return nil, err
	}
	var rounds []MyRound
	if err := json.Unmarshal(data, &rounds); err != nil {
		// API might return an error object instead of an array
		return nil, fmt.Errorf("%w (response: %.200s)", err, string(data))
	}
	return rounds, nil
}

// GetMyPredictions returns predictions for a round
func (c *Client) GetMyPredictions(roundID string) (json.RawMessage, error) {
	data, err := c.get("/my-predictions/" + roundID)
	if err != nil {
		return nil, err
	}
	return data, nil
}

// GetAnalysis returns post-round ground truth comparison
func (c *Client) GetAnalysis(roundID string, seedIndex int) (*Analysis, error) {
	data, err := c.get(fmt.Sprintf("/analysis/%s/%d", roundID, seedIndex))
	if err != nil {
		return nil, err
	}
	var analysis Analysis
	return &analysis, json.Unmarshal(data, &analysis)
}

// GetLeaderboard returns the leaderboard
func (c *Client) GetLeaderboard() ([]LeaderboardEntry, error) {
	data, err := c.get("/leaderboard")
	if err != nil {
		return nil, err
	}
	var entries []LeaderboardEntry
	return entries, json.Unmarshal(data, &entries)
}

// Verify checks that the token is valid
func (c *Client) Verify() error {
	_, err := c.GetRounds()
	return err
}
