package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"astar-tui/api"
	"astar-tui/internal"
	"astar-tui/web"

	tea "github.com/charmbracelet/bubbletea"
)

func main() {
	webMode := flag.Bool("web", false, "Start web API server instead of TUI")
	webAddr := flag.String("web-addr", ":8090", "Web API server address")
	flag.Parse()

	// Find our working directory (parent of tui/)
	exeDir, err := os.Getwd()
	if err != nil {
		exeDir = "."
	}

	// Determine the astar-island-solution root
	// If we're in tui/, go up one level
	baseDir := exeDir
	if filepath.Base(exeDir) == "tui" {
		baseDir = filepath.Dir(exeDir)
	}

	dataDir := filepath.Join(baseDir, "data")

	// Load .env
	env, err := internal.LoadEnv(exeDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: Could not load .env: %v\n", err)
	}

	if env == nil || env.AstarToken == "" {
		fmt.Fprintln(os.Stderr, `
   ░▒▓█ ERROR █▓▒░
   ╔══════════════════════════════════════╗
   ║  ✕  ASTAR_TOKEN not found!          ║
   ║  Set it in .env or environment       ║
   ╚══════════════════════════════════════╝`)
		os.Exit(1)
	}

	// Print startup banner
	fmt.Print(`
   ░▒▓████████████████████████████████████████▓▒░

      ᛟ ᚨ ᛋ ᛏ ᚨ ᚱ ᛫ ᛁ ᛋ ᛚ ᚨ ᚾ ᛞ ᛟ

      █▀▀█ ▄▀▀▀ ▀▀█▀▀ █▀▀█ █▀▀█
      █▄▄█ ▀▀▀█   █   █▄▄█ █▄▄▀
      █  █ ▀▀▀▀   ▀   █  █ █  █

      ᛁ ᛋ ᛚ ᚨ ᚾ ᛞ  //EXPLORER v2.0

   ░▒▓████████████████████████████████████████▓▒░

   ◈ Initializing neural rune interface...

`)

	// Create API client
	client := api.NewClient(env.AstarToken)

	// Verify connectivity
	if err := client.Verify(); err != nil {
		fmt.Fprintf(os.Stderr, "   ✕ API connection failed: %v\n", err)
		fmt.Fprintln(os.Stderr, "   Starting in offline mode...")
	}

	// Create process manager
	procMgr := internal.NewProcessManager(baseDir)

	if *webMode {
		// Start web API server
		fmt.Printf("   Starting web API server on %s\n", *webAddr)
		srv := web.NewServer(client, procMgr, dataDir, env)
		if err := srv.ListenAndServe(*webAddr); err != nil {
			fmt.Fprintf(os.Stderr, "Web server error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	// Create app
	app := NewApp(client, procMgr, env, dataDir)

	// Run Bubble Tea
	p := tea.NewProgram(app, tea.WithAltScreen(), tea.WithMouseCellMotion())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
