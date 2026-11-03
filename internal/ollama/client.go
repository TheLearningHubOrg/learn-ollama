// Package ollama is a small client for the Ollama REST API. It uses only the
// standard library so that every request and response shape stays visible.
// The official Go client lives at github.com/ollama/ollama/api; this package
// mirrors a subset of it for teaching.
package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// DefaultHost is where a local Ollama server listens unless OLLAMA_HOST says otherwise.
const DefaultHost = "http://127.0.0.1:11434"

// Client talks to one Ollama server.
type Client struct {
	BaseURL string
	HTTP    *http.Client
}

// New returns a client for baseURL, for example "http://127.0.0.1:11434".
// A bare "host:port" (the form OLLAMA_HOST often uses) gets an http:// prefix.
func New(baseURL string) *Client {
	if baseURL == "" {
		baseURL = DefaultHost
	}
	if !strings.HasPrefix(baseURL, "http://") && !strings.HasPrefix(baseURL, "https://") {
		baseURL = "http://" + baseURL
	}
	// Generation can take minutes on a CPU, so the client timeout is generous.
	// Per-request deadlines come from the context instead.
	httpClient := &http.Client{Timeout: 10 * time.Minute}
	return &Client{BaseURL: strings.TrimRight(baseURL, "/"), HTTP: httpClient}
}

// Message is one chat turn.
type Message struct {
	Role     string `json:"role"`
	Content  string `json:"content"`
	Thinking string `json:"thinking,omitempty"`
}

// ChatRequest is the body of POST /api/chat.
type ChatRequest struct {
	Model     string          `json:"model"`
	Messages  []Message       `json:"messages"`
	Stream    *bool           `json:"stream,omitempty"`
	Format    json.RawMessage `json:"format,omitempty"`
	Options   map[string]any  `json:"options,omitempty"`
	KeepAlive string          `json:"keep_alive,omitempty"`
}

// ChatResponse is one object of the /api/chat reply. A streamed reply is a
// sequence of these; only the last one has Done set and carries the timings.
type ChatResponse struct {
	Model              string  `json:"model"`
	Message            Message `json:"message"`
	Done               bool    `json:"done"`
	DoneReason         string  `json:"done_reason,omitempty"`
	TotalDuration      int64   `json:"total_duration,omitempty"`
	LoadDuration       int64   `json:"load_duration,omitempty"`
	PromptEvalCount    int     `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64   `json:"prompt_eval_duration,omitempty"`
	EvalCount          int     `json:"eval_count,omitempty"`
	EvalDuration       int64   `json:"eval_duration,omitempty"`
}

// TokensPerSecond is the decode speed: generated tokens divided by the time
// spent generating them. Durations from Ollama are in nanoseconds.
func (r ChatResponse) TokensPerSecond() float64 {
	if r.EvalDuration <= 0 {
		return 0
	}
	return float64(r.EvalCount) / (float64(r.EvalDuration) / 1e9)
}

// ModelInfo is one entry of GET /api/tags.
type ModelInfo struct {
	Name    string       `json:"name"`
	Size    int64        `json:"size"`
	Digest  string       `json:"digest"`
	Details ModelDetails `json:"details"`
}

// ModelDetails describes the weights behind a model tag.
type ModelDetails struct {
	Format            string `json:"format"`
	Family            string `json:"family"`
	ParameterSize     string `json:"parameter_size"`
	QuantizationLevel string `json:"quantization_level"`
}

// APIError is a non-2xx reply. Ollama sends errors as {"error": "..."}.
type APIError struct {
	Status  int
	Message string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("ollama: HTTP %d: %s", e.Status, e.Message)
}

// Bool returns a pointer to b, for the optional Stream field.
func Bool(b bool) *bool { return &b }

// Version calls GET /api/version.
func (c *Client) Version(ctx context.Context) (string, error) {
	var out struct {
		Version string `json:"version"`
	}
	if err := c.getJSON(ctx, "/api/version", &out); err != nil {
		return "", err
	}
	return out.Version, nil
}

// List calls GET /api/tags and returns the models stored on the server.
func (c *Client) List(ctx context.Context) ([]ModelInfo, error) {
	var out struct {
		Models []ModelInfo `json:"models"`
	}
	if err := c.getJSON(ctx, "/api/tags", &out); err != nil {
		return nil, err
	}
	return out.Models, nil
}

// Chat sends a non-streaming chat request and returns the complete reply.
func (c *Client) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	req.Stream = Bool(false)
	resp, err := c.do(ctx, http.MethodPost, "/api/chat", req)
	if err != nil {
		return ChatResponse{}, err
	}
	defer resp.Body.Close()
	var out ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return ChatResponse{}, fmt.Errorf("ollama: decode chat reply: %w", err)
	}
	return out, nil
}

// ChatStream sends a streaming chat request. Ollama answers with NDJSON: one
// JSON object per line. fn is called for every object in order; the last one
// has Done set. Returning an error from fn stops reading and closes the
// connection, which tells the server to stop generating.
func (c *Client) ChatStream(ctx context.Context, req ChatRequest, fn func(ChatResponse) error) error {
	req.Stream = Bool(true)
	resp, err := c.do(ctx, http.MethodPost, "/api/chat", req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	// A single line can hold a long tool call or thinking block, so raise the
	// 64 KiB default limit.
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var chunk struct {
			ChatResponse
			Error string `json:"error"`
		}
		if err := json.Unmarshal(line, &chunk); err != nil {
			return fmt.Errorf("ollama: decode stream line: %w", err)
		}
		// Errors that happen after the 200 status line arrive in-band.
		if chunk.Error != "" {
			return &APIError{Status: http.StatusOK, Message: chunk.Error}
		}
		if err := fn(chunk.ChatResponse); err != nil {
			return err
		}
	}
	return scanner.Err()
}

func (c *Client) getJSON(ctx context.Context, path string, out any) error {
	resp, err := c.do(ctx, http.MethodGet, path, nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return fmt.Errorf("ollama: decode %s: %w", path, err)
	}
	return nil
}

func (c *Client) do(ctx context.Context, method, path string, body any) (*http.Response, error) {
	var payload io.Reader
	if body != nil {
		buf, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		payload = bytes.NewReader(buf)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, payload)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
		var e struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(data, &e) != nil || e.Error == "" {
			e.Error = strings.TrimSpace(string(data))
		}
		return nil, &APIError{Status: resp.StatusCode, Message: e.Error}
	}
	return resp, nil
}
