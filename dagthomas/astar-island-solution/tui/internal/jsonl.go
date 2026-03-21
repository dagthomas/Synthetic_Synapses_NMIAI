package internal

import (
	"bufio"
	"encoding/json"
	"errors"
	"os"
	"sync"
)

// JSONLReader reads JSONL files with tail/incremental support
type JSONLReader struct {
	path   string
	offset int64
	mu     sync.Mutex
}

// NewJSONLReader creates a reader for a JSONL file
func NewJSONLReader(path string) *JSONLReader {
	return &JSONLReader{path: path}
}

// ReadAll reads all entries from the file
func (r *JSONLReader) ReadAll() ([]json.RawMessage, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	f, err := os.Open(r.path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()

	var entries []json.RawMessage
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024) // 10MB max line
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		cp := make([]byte, len(line))
		copy(cp, line)
		entries = append(entries, json.RawMessage(cp))
	}

	info, _ := f.Stat()
	if info != nil {
		r.offset = info.Size()
	}
	return entries, scanner.Err()
}

// ReadLast reads the last n entries from the file
func (r *JSONLReader) ReadLast(n int) ([]json.RawMessage, error) {
	all, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(all) <= n {
		return all, nil
	}
	return all[len(all)-n:], nil
}

// ReadNew reads entries added since the last read
func (r *JSONLReader) ReadNew() ([]json.RawMessage, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	f, err := os.Open(r.path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return nil, err
	}

	if info.Size() <= r.offset {
		// File hasn't grown (or was truncated)
		if info.Size() < r.offset {
			r.offset = 0 // File was truncated, start over
		}
		return nil, nil
	}

	_, err = f.Seek(r.offset, 0)
	if err != nil {
		return nil, err
	}

	var entries []json.RawMessage
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		cp := make([]byte, len(line))
		copy(cp, line)
		entries = append(entries, json.RawMessage(cp))
	}

	r.offset = info.Size()
	return entries, scanner.Err()
}

// Path returns the file path
func (r *JSONLReader) Path() string {
	return r.path
}

// Count returns the total number of entries
func (r *JSONLReader) Count() (int, error) {
	f, err := os.Open(r.path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return 0, nil
		}
		return 0, err
	}
	defer f.Close()

	count := 0
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)
	for scanner.Scan() {
		if len(scanner.Bytes()) > 0 {
			count++
		}
	}
	return count, scanner.Err()
}
