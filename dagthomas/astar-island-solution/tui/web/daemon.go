package web

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"astar-tui/internal"
)

// handleDaemonStatus returns last N lines from daemon.log
func (s *Server) handleDaemonStatus(w http.ResponseWriter, r *http.Request) {
	logPath := filepath.Join(s.dataDir, "daemon.log")
	data, err := os.ReadFile(logPath)
	if err != nil {
		writeJSON(w, []string{})
		return
	}

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	n := 50
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}

	writeJSON(w, lines)
}

// handleDaemonParams returns current best_params.json
func (s *Server) handleDaemonParams(w http.ResponseWriter, r *http.Request) {
	paramsPath := filepath.Join(filepath.Dir(s.dataDir), "best_params.json")
	data, err := os.ReadFile(paramsPath)
	if err != nil {
		writeJSON(w, map[string]any{})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(data)
}

// handleDaemonAutoloop returns aggregated autoloop stats
func (s *Server) handleDaemonAutoloop(w http.ResponseWriter, r *http.Request) {
	reader, ok := s.logReaders["autoloop_fast"]
	if !ok {
		writeJSON(w, map[string]any{"total_experiments": 0})
		return
	}

	entries, err := reader.ReadAll()
	if err != nil || len(entries) == 0 {
		writeJSON(w, map[string]any{"total_experiments": 0})
		return
	}

	type autoloopStats struct {
		TotalExperiments  int                `json:"total_experiments"`
		BestScore         float64            `json:"best_score"`
		BestBoom          float64            `json:"best_boom"`
		BestNonboom       float64            `json:"best_nonboom"`
		ExperimentsPerHr  float64            `json:"experiments_per_hour"`
		LastImprovementID int                `json:"last_improvement_id"`
		AcceptedCount     int                `json:"accepted_count"`
		TopParams         []json.RawMessage  `json:"top_params"`
	}

	stats := autoloopStats{}
	stats.TotalExperiments = len(entries)

	var accepted []json.RawMessage
	var firstTS, lastTS time.Time

	for _, raw := range entries {
		var entry struct {
			ID         int                    `json:"id"`
			Timestamp  string                 `json:"timestamp"`
			Accepted   bool                   `json:"accepted"`
			ScoresFull map[string]float64     `json:"scores_full"`
			BaselineAvg float64               `json:"baseline_avg"`
		}
		if err := json.Unmarshal(raw, &entry); err != nil {
			continue
		}

		if t, err := time.Parse(time.RFC3339Nano, entry.Timestamp); err == nil {
			if firstTS.IsZero() {
				firstTS = t
			}
			lastTS = t
		}

		if entry.Accepted {
			stats.AcceptedCount++
			accepted = append(accepted, raw)
		}

		avg := entry.ScoresFull["avg"]
		boom := entry.ScoresFull["boom_avg"]
		nonboom := entry.ScoresFull["nonboom_avg"]

		if avg > stats.BestScore {
			stats.BestScore = avg
			stats.LastImprovementID = entry.ID
		}
		if boom > stats.BestBoom {
			stats.BestBoom = boom
		}
		if nonboom > stats.BestNonboom {
			stats.BestNonboom = nonboom
		}
	}

	// Experiments per hour
	if !firstTS.IsZero() && !lastTS.IsZero() {
		dur := lastTS.Sub(firstTS).Hours()
		if dur > 0 {
			stats.ExperimentsPerHr = math.Round(float64(stats.TotalExperiments) / dur)
		}
	}

	// Last 20 accepted
	if len(accepted) > 20 {
		stats.TopParams = accepted[len(accepted)-20:]
	} else {
		stats.TopParams = accepted
	}
	if stats.TopParams == nil {
		stats.TopParams = []json.RawMessage{}
	}

	writeJSON(w, stats)
}

// handleDaemonScores returns per-round score/regime from calibration data
func (s *Server) handleDaemonScores(w http.ResponseWriter, r *http.Request) {
	type roundScore struct {
		RoundNumber   int     `json:"round_number"`
		AvgScore      float64 `json:"avg_score"`
		Regime        string  `json:"regime"`
		SettlementPct float64 `json:"settlement_pct"`
	}

	calDir := filepath.Join(s.dataDir, "calibration")
	roundDirs, err := filepath.Glob(filepath.Join(calDir, "round*"))
	if err != nil {
		writeJSON(w, []roundScore{})
		return
	}

	re := regexp.MustCompile(`round(\d+)`)
	var results []roundScore

	for _, rd := range roundDirs {
		base := filepath.Base(rd)
		m := re.FindStringSubmatch(base)
		if m == nil {
			continue
		}
		roundNum := 0
		fmt.Sscanf(m[1], "%d", &roundNum)

		// Read all analysis seeds
		seedFiles, _ := filepath.Glob(filepath.Join(rd, "analysis_seed_*.json"))
		if len(seedFiles) == 0 {
			continue
		}

		var totalScore, totalSett float64
		count := 0

		for _, sf := range seedFiles {
			data, err := os.ReadFile(sf)
			if err != nil {
				continue
			}
			var analysis struct {
				Score       float64       `json:"score"`
				GroundTruth [][][]float64 `json:"ground_truth"`
			}
			if err := json.Unmarshal(data, &analysis); err != nil {
				continue
			}

			totalScore += analysis.Score

			// Compute settlement %
			if len(analysis.GroundTruth) > 0 {
				var settSum float64
				cells := 0
				for y := range analysis.GroundTruth {
					for x := range analysis.GroundTruth[y] {
						if len(analysis.GroundTruth[y][x]) >= 2 {
							settSum += analysis.GroundTruth[y][x][1]
							cells++
						}
					}
				}
				if cells > 0 {
					totalSett += settSum / float64(cells) * 100
				}
			}
			count++
		}

		if count == 0 {
			continue
		}

		avgScore := totalScore / float64(count)
		avgSett := totalSett / float64(count)

		regime := "moderate"
		if avgSett < 5 {
			regime = "collapse"
		} else if avgSett >= 18 {
			regime = "boom"
		}

		results = append(results, roundScore{
			RoundNumber:   roundNum,
			AvgScore:      math.Round(avgScore*100) / 100,
			Regime:        regime,
			SettlementPct: math.Round(avgSett*100) / 100,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].RoundNumber < results[j].RoundNumber
	})

	writeJSON(w, results)
}

// handleDaemonAutoloopStream streams new autoloop entries via SSE
func (s *Server) handleDaemonAutoloopStream(w http.ResponseWriter, r *http.Request) {
	reader, ok := s.logReaders["autoloop_fast"]
	if !ok {
		writeError(w, 404, "autoloop_fast log reader not found")
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, 500, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	flusher.Flush()

	streamReader := internal.NewJSONLReader(reader.Path())
	streamReader.ReadAll() // seek to end

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
			entries, err := streamReader.ReadNew()
			if err != nil {
				log.Printf("SSE read error for daemon autoloop: %v", err)
				continue
			}
			for _, entry := range entries {
				data, _ := json.Marshal(entry)
				fmt.Fprintf(w, "data: %s\n\n", data)
			}
			if len(entries) > 0 {
				flusher.Flush()
			}
		}
	}
}

