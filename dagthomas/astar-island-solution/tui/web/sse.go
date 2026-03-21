package web

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"astar-tui/internal"
)

// handleLogStream streams new log entries via SSE
func (s *Server) handleLogStream(w http.ResponseWriter, r *http.Request) {
	source := r.PathValue("source")
	reader, ok := s.logReaders[source]
	if !ok {
		writeError(w, 404, fmt.Sprintf("unknown log source: %s", source))
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

	// Create a dedicated reader for this SSE connection so we track offset independently
	streamReader := internal.NewJSONLReader(reader.Path())
	// Initialize offset to end of file so we only get new entries
	streamReader.ReadAll()

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
			entries, err := streamReader.ReadNew()
			if err != nil {
				log.Printf("SSE read error for %s: %v", source, err)
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
