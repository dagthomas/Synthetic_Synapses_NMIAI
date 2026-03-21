package internal

import (
	"bufio"
	"fmt"
	"io"
	"os/exec"
	"sync"
	"time"
)

// ProcessState represents the state of a managed process
type ProcessState int

const (
	ProcessStopped ProcessState = iota
	ProcessRunning
	ProcessFailed
)

func (s ProcessState) String() string {
	switch s {
	case ProcessRunning:
		return "Running"
	case ProcessFailed:
		return "Failed"
	default:
		return "Stopped"
	}
}

// ManagedProcess wraps a subprocess with streaming output
type ManagedProcess struct {
	Name      string
	State     ProcessState
	StartedAt time.Time
	cmd       *exec.Cmd
	output    []string
	mu        sync.Mutex
	done      chan struct{}
	onOutput  func(line string) // callback for new output
}

// ProcessManager manages multiple subprocesses
type ProcessManager struct {
	processes map[string]*ManagedProcess
	workDir   string
	mu        sync.Mutex
}

// NewProcessManager creates a process manager
func NewProcessManager(workDir string) *ProcessManager {
	return &ProcessManager{
		processes: make(map[string]*ManagedProcess),
		workDir:   workDir,
	}
}

// Start launches a subprocess
func (pm *ProcessManager) Start(name string, command string, args []string, onOutput func(string)) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	// Stop existing process with same name
	if existing, ok := pm.processes[name]; ok && existing.State == ProcessRunning {
		existing.Stop()
	}

	cmd := exec.Command(command, args...)
	cmd.Dir = pm.workDir

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("stderr pipe: %w", err)
	}

	proc := &ManagedProcess{
		Name:      name,
		State:     ProcessRunning,
		StartedAt: time.Now(),
		cmd:       cmd,
		done:      make(chan struct{}),
		onOutput:  onOutput,
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start: %w", err)
	}

	// Stream stdout and stderr
	go proc.streamReader(stdout)
	go proc.streamReader(stderr)

	// Wait for completion
	go func() {
		err := cmd.Wait()
		proc.mu.Lock()
		if err != nil {
			proc.State = ProcessFailed
			proc.addLine(fmt.Sprintf("[Process exited with error: %v]", err))
		} else {
			proc.State = ProcessStopped
			proc.addLine("[Process completed]")
		}
		proc.mu.Unlock()
		close(proc.done)
	}()

	pm.processes[name] = proc
	return nil
}

func (p *ManagedProcess) streamReader(r io.Reader) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		p.mu.Lock()
		p.addLine(line)
		p.mu.Unlock()
		if p.onOutput != nil {
			p.onOutput(line)
		}
	}
}

func (p *ManagedProcess) addLine(line string) {
	p.output = append(p.output, line)
	// Keep last 500 lines
	if len(p.output) > 500 {
		p.output = p.output[len(p.output)-500:]
	}
}

// Stop kills the process
func (p *ManagedProcess) Stop() {
	if p.cmd != nil && p.cmd.Process != nil {
		p.cmd.Process.Kill()
	}
}

// GetOutput returns the output lines
func (p *ManagedProcess) GetOutput() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	cp := make([]string, len(p.output))
	copy(cp, p.output)
	return cp
}

// Uptime returns how long the process has been running
func (p *ManagedProcess) Uptime() time.Duration {
	if p.State == ProcessRunning {
		return time.Since(p.StartedAt)
	}
	return 0
}

// Stop stops a named process
func (pm *ProcessManager) Stop(name string) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	if proc, ok := pm.processes[name]; ok {
		proc.Stop()
	}
}

// Get returns a process by name
func (pm *ProcessManager) Get(name string) *ManagedProcess {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	return pm.processes[name]
}

// Running returns names of all running processes
func (pm *ProcessManager) Running() []string {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	var names []string
	for name, proc := range pm.processes {
		if proc.State == ProcessRunning {
			names = append(names, name)
		}
	}
	return names
}

// All returns all processes
func (pm *ProcessManager) All() map[string]*ManagedProcess {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	cp := make(map[string]*ManagedProcess)
	for k, v := range pm.processes {
		cp[k] = v
	}
	return cp
}
