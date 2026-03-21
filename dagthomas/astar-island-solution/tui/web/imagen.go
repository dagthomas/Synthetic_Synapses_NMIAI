package web

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ImagenService handles Gemini image generation and gallery storage.
type ImagenService struct {
	apiKey     string
	galleryDir string
}

// NewImagenService creates a new ImagenService.
func NewImagenService(apiKey, galleryDir string) *ImagenService {
	if err := os.MkdirAll(galleryDir, 0o755); err != nil {
		log.Printf("Warning: could not create imagen gallery dir: %v", err)
	}
	return &ImagenService{
		apiKey:     apiKey,
		galleryDir: galleryDir,
	}
}

// --- Types ---

// GenerateRequest is the HTTP request body from the frontend.
type GenerateRequest struct {
	Image  string `json:"image"`  // base64 PNG, no data: prefix
	Prompt string `json:"prompt"` // optional override prompt
}

// GeneratedImage is a generated image with metadata.
type GeneratedImage struct {
	ID        string    `json:"id"`
	Filename  string    `json:"filename"`
	ImageB64  string    `json:"image_base64"`
	Prompt    string    `json:"prompt"`
	CreatedAt time.Time `json:"created_at"`
}

// GalleryEntry is a gallery listing (no image data, just metadata + URL).
type GalleryEntry struct {
	ID        string    `json:"id"`
	Filename  string    `json:"filename"`
	Prompt    string    `json:"prompt"`
	CreatedAt time.Time `json:"created_at"`
	URL       string    `json:"url"`
}

// gallerySidecar is the JSON sidecar stored alongside each generated image.
type gallerySidecar struct {
	ID        string    `json:"id"`
	Filename  string    `json:"filename"`
	Prompt    string    `json:"prompt"`
	CreatedAt time.Time `json:"created_at"`
}

// --- Gemini Image Generation API ---

const defaultPrompt = "Transform the attached image into a photorealistic render. " +
	"Keep all elements, layout, terrain shapes, water placement, settlements, forests, and mountains " +
	"exactly where they are in the original image — preserve the composition and structure faithfully. " +
	"Make every element look realistic: grass becomes lush real grass, water becomes crystal clear realistic water, " +
	"mountains become detailed rocky peaks with snow caps, forests become dense realistic trees, " +
	"settlements become small realistic villages with buildings. " +
	"Add global illumination, volumetric god rays, cinematic golden hour lighting, " +
	"and make the island float dramatically in the sky surrounded by clouds below."

const geminiModel = "gemini-3.1-flash-image-preview"

// Gemini API request/response types
type geminiRequest struct {
	Contents         []geminiContent  `json:"contents"`
	GenerationConfig geminiGenConfig  `json:"generationConfig"`
}

type geminiContent struct {
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text       string           `json:"text,omitempty"`
	InlineData *geminiInlineData `json:"inlineData,omitempty"`
}

type geminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type geminiGenConfig struct {
	ResponseModalities []string         `json:"responseModalities"`
	ImageConfig        *geminiImageCfg  `json:"imageConfig,omitempty"`
}

type geminiImageCfg struct {
	AspectRatio string `json:"aspectRatio,omitempty"`
}

type geminiResponse struct {
	Candidates []geminiCandidate `json:"candidates"`
	Error      *geminiError      `json:"error,omitempty"`
}

type geminiCandidate struct {
	Content geminiContent `json:"content"`
}

type geminiError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// Generate calls the Gemini API with a reference image to generate a new scene.
func (s *ImagenService) Generate(referenceImageB64, prompt string) (*GeneratedImage, error) {
	if s.apiKey == "" {
		return nil, fmt.Errorf("GOOGLE_API_KEY not configured in .env")
	}

	if prompt == "" {
		prompt = defaultPrompt
	}

	// Build request — send reference image + text prompt as multimodal input
	reqBody := geminiRequest{
		Contents: []geminiContent{
			{
				Parts: []geminiPart{
					{Text: prompt},
					{InlineData: &geminiInlineData{
						MimeType: "image/png",
						Data:     referenceImageB64,
					}},
				},
			},
		},
		GenerationConfig: geminiGenConfig{
			ResponseModalities: []string{"TEXT", "IMAGE"},
			ImageConfig: &geminiImageCfg{
				AspectRatio: "16:9",
			},
		},
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	url := fmt.Sprintf(
		"https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent",
		geminiModel,
	)

	req, err := http.NewRequest("POST", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("x-goog-api-key", s.apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gemini request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("gemini returned %d: %s", resp.StatusCode, string(respBody))
	}

	// Log truncated response for debugging
	logBody := string(respBody)
	if len(logBody) > 2000 {
		logBody = logBody[:2000] + "...(truncated)"
	}
	log.Printf("Gemini response (status %d): %s", resp.StatusCode, logBody)

	var gResp geminiResponse
	if err := json.Unmarshal(respBody, &gResp); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	if gResp.Error != nil {
		return nil, fmt.Errorf("gemini error %d: %s", gResp.Error.Code, gResp.Error.Message)
	}

	// Find the image part in the response
	var imageB64 string
	for _, candidate := range gResp.Candidates {
		log.Printf("Candidate has %d parts", len(candidate.Content.Parts))
		for i, part := range candidate.Content.Parts {
			log.Printf("  Part %d: text=%q, hasInlineData=%v", i, part.Text, part.InlineData != nil)
			if part.InlineData != nil {
				log.Printf("    InlineData mime=%s, dataLen=%d", part.InlineData.MimeType, len(part.InlineData.Data))
				if strings.HasPrefix(part.InlineData.MimeType, "image/") {
					imageB64 = part.InlineData.Data
				}
			}
		}
		if imageB64 != "" {
			break
		}
	}

	if imageB64 == "" {
		return nil, fmt.Errorf("gemini returned no image in response (candidates=%d)", len(gResp.Candidates))
	}

	// Build result
	id := generateID()
	now := time.Now()
	filename := fmt.Sprintf("gen_%s_%s.png", now.Format("20060102_150405"), id)

	return &GeneratedImage{
		ID:        id,
		Filename:  filename,
		ImageB64:  imageB64,
		Prompt:    prompt,
		CreatedAt: now,
	}, nil
}

// --- Storage ---

// SaveImage saves a generated image and its metadata sidecar to disk.
func (s *ImagenService) SaveImage(img *GeneratedImage) error {
	// Decode base64 to PNG bytes
	pngData, err := base64.StdEncoding.DecodeString(img.ImageB64)
	if err != nil {
		return fmt.Errorf("decode image: %w", err)
	}

	pngPath := filepath.Join(s.galleryDir, img.Filename)
	if err := os.WriteFile(pngPath, pngData, 0o644); err != nil {
		return fmt.Errorf("write image: %w", err)
	}

	// Write metadata sidecar
	sidecar := gallerySidecar{
		ID:        img.ID,
		Filename:  img.Filename,
		Prompt:    img.Prompt,
		CreatedAt: img.CreatedAt,
	}
	sidecarBytes, err := json.MarshalIndent(sidecar, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal sidecar: %w", err)
	}

	jsonPath := strings.TrimSuffix(pngPath, ".png") + ".json"
	if err := os.WriteFile(jsonPath, sidecarBytes, 0o644); err != nil {
		return fmt.Errorf("write sidecar: %w", err)
	}

	return nil
}

// DeleteImage removes a generated image and its sidecar from disk.
func (s *ImagenService) DeleteImage(filename string) error {
	pngPath := filepath.Join(s.galleryDir, filename)
	jsonPath := strings.TrimSuffix(pngPath, ".png") + ".json"
	os.Remove(jsonPath) // ignore error, sidecar may not exist
	if err := os.Remove(pngPath); err != nil {
		return fmt.Errorf("delete image: %w", err)
	}
	return nil
}

// ListGallery returns all gallery entries sorted by timestamp descending.
func (s *ImagenService) ListGallery() ([]GalleryEntry, error) {
	pattern := filepath.Join(s.galleryDir, "gen_*.json")
	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}

	var entries []GalleryEntry
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		var sc gallerySidecar
		if err := json.Unmarshal(data, &sc); err != nil {
			continue
		}
		entries = append(entries, GalleryEntry{
			ID:        sc.ID,
			Filename:  sc.Filename,
			Prompt:    sc.Prompt,
			CreatedAt: sc.CreatedAt,
			URL:       "/api/imagen/images/" + sc.Filename,
		})
	}

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].CreatedAt.After(entries[j].CreatedAt)
	})

	return entries, nil
}

// --- Helpers ---

func generateID() string {
	b := make([]byte, 6)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}
