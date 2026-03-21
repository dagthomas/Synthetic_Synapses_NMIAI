package internal

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
)

// EnvConfig holds configuration loaded from .env
type EnvConfig struct {
	AstarToken   string
	GoogleAPIKey string
	GCPProjectID string
	GCPLocation  string // defaults to "us-central1"
}

// LoadEnv parses a .env file and returns config.
// Looks for .env in the parent directory of the executable (astar-island-solution/.env)
func LoadEnv(tuiDir string) (*EnvConfig, error) {
	envPath := filepath.Join(tuiDir, "..", ".env")
	return LoadEnvFile(envPath)
}

// LoadEnvFile parses a specific .env file
func LoadEnvFile(path string) (*EnvConfig, error) {
	cfg := &EnvConfig{}

	f, err := os.Open(path)
	if err != nil {
		// Try environment variables as fallback
		cfg.AstarToken = os.Getenv("ASTAR_TOKEN")
		cfg.GoogleAPIKey = os.Getenv("GOOGLE_API_KEY")
		if cfg.AstarToken != "" {
			return cfg, nil
		}
		return cfg, err
	}
	defer f.Close()

	vars := make(map[string]string)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			val := strings.TrimSpace(parts[1])
			// Remove surrounding quotes if present
			val = strings.Trim(val, `"'`)
			vars[key] = val
		}
	}

	cfg.AstarToken = vars["ASTAR_TOKEN"]
	cfg.GoogleAPIKey = vars["GOOGLE_API_KEY"]
	cfg.GCPProjectID = vars["GCP_PROJECT_ID"]
	cfg.GCPLocation = vars["GCP_LOCATION"]

	// Env vars override file
	if v := os.Getenv("ASTAR_TOKEN"); v != "" {
		cfg.AstarToken = v
	}
	if v := os.Getenv("GOOGLE_API_KEY"); v != "" {
		cfg.GoogleAPIKey = v
	}
	if v := os.Getenv("GCP_PROJECT_ID"); v != "" {
		cfg.GCPProjectID = v
	}
	if v := os.Getenv("GCP_LOCATION"); v != "" {
		cfg.GCPLocation = v
	}

	// Defaults
	if cfg.GCPLocation == "" {
		cfg.GCPLocation = "us-central1"
	}

	return cfg, nil
}
