package web

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"astar-tui/api"
	"astar-tui/internal"
)

// ProcessDef defines a process that can be started
type ProcessDef struct {
	Command string
	Args    []string
}

// Server is the HTTP API server for the web dashboard
type Server struct {
	client      *api.Client
	procMgr     *internal.ProcessManager
	dataDir     string
	logReaders  map[string]*internal.JSONLReader
	processDefs map[string]ProcessDef
	imagen      *ImagenService
}

// NewServer creates a new web API server
func NewServer(client *api.Client, procMgr *internal.ProcessManager, dataDir string, env *internal.EnvConfig) *Server {
	s := &Server{
		client:  client,
		procMgr: procMgr,
		dataDir: dataDir,
		logReaders: map[string]*internal.JSONLReader{
			"autoloop":      internal.NewJSONLReader(filepath.Join(dataDir, "autoloop_log.jsonl")),
			"autoloop_fast": internal.NewJSONLReader(filepath.Join(dataDir, "autoloop_fast_log.jsonl")),
			"adk":           internal.NewJSONLReader(filepath.Join(dataDir, "adk_research_log.jsonl")),
			"gemini":        internal.NewJSONLReader(filepath.Join(dataDir, "gemini_research_log.jsonl")),
			"multi":         internal.NewJSONLReader(filepath.Join(dataDir, "multi_research_log.jsonl")),
			"history":       internal.NewJSONLReader(filepath.Join(dataDir, "adk_agent_history.jsonl")),
		},
		processDefs: map[string]ProcessDef{
			"autoloop": {Command: "python", Args: []string{"autoloop_fast.py"}},
			"gemini":   {Command: "python", Args: []string{"gemini_researcher.py"}},
			"adk":      {Command: "python", Args: []string{"adk_researcher.py"}},
			"multi":    {Command: "python", Args: []string{"multi_researcher.py"}},
		},
		imagen: NewImagenService(env.GoogleAPIKey, filepath.Join(dataDir, "imagen")),
	}
	return s
}

// ListenAndServe starts the HTTP server
func (s *Server) ListenAndServe(addr string) error {
	mux := http.NewServeMux()

	// API proxy endpoints
	mux.HandleFunc("GET /api/rounds", s.handleRounds)
	mux.HandleFunc("GET /api/rounds/{id}", s.handleRoundDetail)
	mux.HandleFunc("GET /api/rounds/{id}/seeds/{idx}/analysis", s.handleAnalysis)
	mux.HandleFunc("GET /api/local/analysis/{roundNum}/{idx}", s.handleLocalAnalysis)
	mux.HandleFunc("GET /api/budget", s.handleBudget)
	mux.HandleFunc("GET /api/my-rounds", s.handleMyRounds)
	mux.HandleFunc("GET /api/leaderboard", s.handleLeaderboard)

	// Local data endpoints
	mux.HandleFunc("GET /api/logs/{source}", s.handleLogs)
	mux.HandleFunc("GET /api/logs/{source}/stream", s.handleLogStream)
	mux.HandleFunc("GET /api/processes", s.handleProcesses)
	mux.HandleFunc("POST /api/processes/{name}/start", s.handleProcessStart)
	mux.HandleFunc("POST /api/processes/{name}/stop", s.handleProcessStop)
	mux.HandleFunc("GET /api/metrics", s.handleMetrics)
	mux.HandleFunc("GET /api/explorer/{roundId}/observations", s.handleExplorerObservations)

	// Daemon live data endpoints
	mux.HandleFunc("GET /api/daemon/status", s.handleDaemonStatus)
	mux.HandleFunc("GET /api/daemon/params", s.handleDaemonParams)
	mux.HandleFunc("GET /api/daemon/autoloop", s.handleDaemonAutoloop)
	mux.HandleFunc("GET /api/daemon/scores", s.handleDaemonScores)
	mux.HandleFunc("GET /api/daemon/autoloop/stream", s.handleDaemonAutoloopStream)

	// Imagen endpoints
	mux.HandleFunc("POST /api/imagen/generate", s.handleImagenGenerate)
	mux.HandleFunc("GET /api/imagen/gallery", s.handleImagenGallery)
	mux.HandleFunc("GET /api/imagen/images/{filename}", s.handleImagenServe)
	mux.HandleFunc("DELETE /api/imagen/images/{filename}", s.handleImagenDelete)

	handler := corsMiddleware(mux)

	log.Printf("Web API server listening on %s", addr)
	return http.ListenAndServe(addr, handler)
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == "OPTIONS" {
			w.WriteHeader(204)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("JSON encode error: %v", err)
	}
}

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

// --- API proxy handlers ---

func (s *Server) handleRounds(w http.ResponseWriter, r *http.Request) {
	rounds, err := s.client.GetRounds()
	if err != nil {
		writeError(w, 502, err.Error())
		return
	}
	writeJSON(w, rounds)
}

func (s *Server) handleRoundDetail(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	detail, err := s.client.GetRoundDetail(id)
	if err != nil {
		writeError(w, 502, err.Error())
		return
	}
	writeJSON(w, detail)
}

func (s *Server) handleAnalysis(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	idxStr := r.PathValue("idx")
	idx, err := strconv.Atoi(idxStr)
	if err != nil {
		writeError(w, 400, "invalid seed index")
		return
	}
	analysis, err := s.client.GetAnalysis(id, idx)
	if err != nil {
		writeError(w, 502, err.Error())
		return
	}
	writeJSON(w, analysis)
}

func (s *Server) handleLocalAnalysis(w http.ResponseWriter, r *http.Request) {
	roundNum := r.PathValue("roundNum")
	idxStr := r.PathValue("idx")
	// Read from local calibration data
	path := filepath.Join("data", "calibration", "round"+roundNum, "analysis_seed_"+idxStr+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		writeError(w, 404, fmt.Sprintf("local analysis not found: %s", path))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(data)
}

func (s *Server) handleBudget(w http.ResponseWriter, r *http.Request) {
	budget, err := s.client.GetBudget()
	if err != nil {
		writeError(w, 502, err.Error())
		return
	}
	writeJSON(w, budget)
}

func (s *Server) handleMyRounds(w http.ResponseWriter, r *http.Request) {
	rounds, err := s.client.GetMyRounds()
	if err != nil {
		writeError(w, 502, err.Error())
		return
	}
	writeJSON(w, rounds)
}

func (s *Server) handleLeaderboard(w http.ResponseWriter, r *http.Request) {
	entries, err := s.client.GetLeaderboard()
	if err != nil {
		writeError(w, 502, err.Error())
		return
	}
	writeJSON(w, entries)
}

// --- Local data handlers ---

func (s *Server) handleLogs(w http.ResponseWriter, r *http.Request) {
	source := r.PathValue("source")
	reader, ok := s.logReaders[source]
	if !ok {
		writeError(w, 404, fmt.Sprintf("unknown log source: %s", source))
		return
	}

	n := 200
	if lastStr := r.URL.Query().Get("last"); lastStr != "" {
		if parsed, err := strconv.Atoi(lastStr); err == nil && parsed > 0 {
			n = parsed
		}
	}

	entries, err := reader.ReadLast(n)
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}
	if entries == nil {
		entries = []json.RawMessage{}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(entries)
}

func (s *Server) handleProcesses(w http.ResponseWriter, r *http.Request) {
	all := s.procMgr.All()
	type procInfo struct {
		Name   string `json:"name"`
		State  string `json:"state"`
		Uptime string `json:"uptime,omitempty"`
		Lines  int    `json:"output_lines"`
	}
	result := make([]procInfo, 0, len(all))
	for name, proc := range all {
		info := procInfo{
			Name:  name,
			State: proc.State.String(),
			Lines: len(proc.GetOutput()),
		}
		if proc.State == internal.ProcessRunning {
			info.Uptime = proc.Uptime().Round(time.Second).String()
		}
		result = append(result, info)
	}

	// Check for externally-started processes via log file activity
	running := make(map[string]bool)
	for _, p := range result {
		if p.State == "Running" {
			running[p.Name] = true
		}
	}

	// Detect externally-started Python processes by log file recency
	osRunning := detectExternalProcesses(s.dataDir)

	// Merge: for each process def, prefer managed state, fall back to OS detection
	for name := range s.processDefs {
		if running[name] {
			continue // already reported as Running by procMgr
		}
		if osRunning[name] {
			// Found externally — also check if we already added a Stopped entry
			found := false
			for i, p := range result {
				if p.Name == name {
					result[i].State = "Running"
					found = true
					break
				}
			}
			if !found {
				result = append(result, procInfo{Name: name, State: "Running"})
			}
		} else {
			// Not running anywhere
			alreadyAdded := false
			for _, p := range result {
				if p.Name == name {
					alreadyAdded = true
					break
				}
			}
			if !alreadyAdded {
				result = append(result, procInfo{Name: name, State: "Stopped"})
			}
		}
	}

	// Also add daemon if detected
	if osRunning["daemon"] {
		found := false
		for _, p := range result {
			if p.Name == "daemon" {
				found = true
				break
			}
		}
		if !found {
			result = append(result, procInfo{Name: "daemon", State: "Running"})
		}
	}

	writeJSON(w, result)
}

// detectExternalProcesses checks if log/output files are being actively written
// to detect externally-started processes.
func detectExternalProcesses(dataDir string) map[string]bool {
	found := make(map[string]bool)

	type logCheck struct {
		file      string
		nodeID    string
		threshold time.Duration
	}

	// Map log files to process node IDs with appropriate thresholds
	checks := []logCheck{
		{"autoloop_fast_output.log", "autoloop", 2 * time.Minute},
		{"gemini_researcher_output.log", "gemini", 10 * time.Minute},
		{"multi_researcher_output.log", "multi", 10 * time.Minute},
		{"daemon.log", "daemon", 45 * time.Minute},
		{"daemon_output.log", "daemon", 45 * time.Minute},
	}

	now := time.Now()

	for _, check := range checks {
		path := filepath.Join(dataDir, check.file)
		info, err := os.Stat(path)
		if err != nil {
			continue
		}
		if now.Sub(info.ModTime()) < check.threshold {
			found[check.nodeID] = true
		}
	}

	return found
}

func (s *Server) handleProcessStart(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	def, ok := s.processDefs[name]
	if !ok {
		writeError(w, 404, fmt.Sprintf("unknown process: %s", name))
		return
	}

	err := s.procMgr.Start(name, def.Command, def.Args, nil)
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}
	writeJSON(w, map[string]string{"status": "started", "name": name})
}

func (s *Server) handleProcessStop(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	s.procMgr.Stop(name)
	writeJSON(w, map[string]string{"status": "stopped", "name": name})
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	type metrics struct {
		AutoloopCount int     `json:"autoloop_count"`
		ADKCount      int     `json:"adk_count"`
		GeminiCount   int     `json:"gemini_count"`
		MultiCount    int     `json:"multi_count"`
		BestScore     float64 `json:"best_score"`
		QueriesUsed   int     `json:"queries_used"`
		QueriesMax    int     `json:"queries_max"`
	}

	m := metrics{}

	if c, err := s.logReaders["autoloop"].Count(); err == nil {
		m.AutoloopCount = c
	}
	if c, err := s.logReaders["adk"].Count(); err == nil {
		m.ADKCount = c
	}
	if c, err := s.logReaders["gemini"].Count(); err == nil {
		m.GeminiCount = c
	}
	if c, err := s.logReaders["multi"].Count(); err == nil {
		m.MultiCount = c
	}

	if budget, err := s.client.GetBudget(); err == nil {
		m.QueriesUsed = budget.QueriesUsed
		m.QueriesMax = budget.QueriesMax
	}

	if myRounds, err := s.client.GetMyRounds(); err == nil {
		for _, mr := range myRounds {
			if mr.Score != nil && *mr.Score > m.BestScore {
				m.BestScore = *mr.Score
			}
		}
	}

	writeJSON(w, m)
}

func (s *Server) handleExplorerObservations(w http.ResponseWriter, r *http.Request) {
	roundID := r.PathValue("roundId")
	// Validate roundID to prevent path traversal
	if strings.ContainsAny(roundID, `/\..`) {
		writeError(w, 400, "invalid round ID")
		return
	}
	obsDir := filepath.Join(s.dataDir, "rounds", roundID)

	pattern := filepath.Join(obsDir, "obs_*.json")
	files, err := filepath.Glob(pattern)
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}

	var observations []json.RawMessage
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		observations = append(observations, json.RawMessage(data))
	}
	if observations == nil {
		observations = []json.RawMessage{}
	}

	writeJSON(w, observations)
}

// --- Imagen handlers ---

func (s *Server) handleImagenGenerate(w http.ResponseWriter, r *http.Request) {
	// Limit body to 10MB
	r.Body = http.MaxBytesReader(w, r.Body, 10<<20)

	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, 400, "invalid request body: "+err.Error())
		return
	}

	// Strip data URL prefix if present
	if idx := strings.Index(req.Image, ","); idx != -1 && strings.HasPrefix(req.Image, "data:") {
		req.Image = req.Image[idx+1:]
	}

	if req.Image == "" {
		writeError(w, 400, "image is required")
		return
	}

	img, err := s.imagen.Generate(req.Image, req.Prompt)
	if err != nil {
		log.Printf("Imagen generate error: %v", err)
		writeError(w, 502, err.Error())
		return
	}

	if err := s.imagen.SaveImage(img); err != nil {
		log.Printf("Imagen save error: %v", err)
		writeError(w, 500, "failed to save generated image: "+err.Error())
		return
	}

	writeJSON(w, img)
}

func (s *Server) handleImagenGallery(w http.ResponseWriter, r *http.Request) {
	entries, err := s.imagen.ListGallery()
	if err != nil {
		writeError(w, 500, err.Error())
		return
	}
	if entries == nil {
		entries = []GalleryEntry{}
	}
	writeJSON(w, entries)
}

func (s *Server) handleImagenServe(w http.ResponseWriter, r *http.Request) {
	filename := r.PathValue("filename")
	// Prevent path traversal
	if strings.ContainsAny(filename, `/\`) || strings.Contains(filename, "..") {
		writeError(w, 400, "invalid filename")
		return
	}
	path := filepath.Join(s.imagen.galleryDir, filename)
	http.ServeFile(w, r, path)
}

func (s *Server) handleImagenDelete(w http.ResponseWriter, r *http.Request) {
	filename := r.PathValue("filename")
	if strings.ContainsAny(filename, `/\`) || strings.Contains(filename, "..") {
		writeError(w, 400, "invalid filename")
		return
	}
	if err := s.imagen.DeleteImage(filename); err != nil {
		writeError(w, 500, err.Error())
		return
	}
	writeJSON(w, map[string]string{"status": "deleted"})
}
