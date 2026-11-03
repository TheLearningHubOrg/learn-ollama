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

// Message is one chat turn. Assistant turns may carry ToolCalls; the
// results go back as Role "tool" messages with ToolName set.
type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	Thinking  string     `json:"thinking,omitempty"`
	Images    []string   `json:"images,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	ToolName  string     `json:"tool_name,omitempty"`
}

// ToolCall is a function call the model asks the application to run. In the
// native API Arguments is a JSON object (the OpenAI-compatible API sends a
// string instead).
type ToolCall struct {
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction names the function and carries its arguments.
type ToolCallFunction struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// Tool describes one function the model may call. Parameters is a JSON schema.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction is the schema half of a Tool.
type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// ChatRequest is the body of POST /api/chat. Think is a bool for most
// thinking models and "low", "medium" or "high" for models with levels.
type ChatRequest struct {
	Model     string          `json:"model"`
	Messages  []Message       `json:"messages"`
	Tools     []Tool          `json:"tools,omitempty"`
	Stream    *bool           `json:"stream,omitempty"`
	Format    json.RawMessage `json:"format,omitempty"`
	Options   map[string]any  `json:"options,omitempty"`
	KeepAlive string          `json:"keep_alive,omitempty"`
	Think     any             `json:"think,omitempty"`
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
	return readNDJSON(resp.Body, fn)
}

// EmbedRequest is the body of POST /api/embed. Input is always sent as an
// array so one request embeds a whole batch.
type EmbedRequest struct {
	Model     string   `json:"model"`
	Input     []string `json:"input"`
	Truncate  *bool    `json:"truncate,omitempty"`
	KeepAlive string   `json:"keep_alive,omitempty"`
}

// EmbedResponse holds one vector per input, in input order. Ollama returns
// L2-normalised vectors, so a dot product is the cosine similarity.
type EmbedResponse struct {
	Model           string      `json:"model"`
	Embeddings      [][]float32 `json:"embeddings"`
	PromptEvalCount int         `json:"prompt_eval_count,omitempty"`
}

// Embed calls POST /api/embed.
func (c *Client) Embed(ctx context.Context, req EmbedRequest) (EmbedResponse, error) {
	resp, err := c.do(ctx, http.MethodPost, "/api/embed", req)
	if err != nil {
		return EmbedResponse{}, err
	}
	defer resp.Body.Close()
	var out EmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return EmbedResponse{}, fmt.Errorf("ollama: decode embed reply: %w", err)
	}
	if len(out.Embeddings) != len(req.Input) {
		return EmbedResponse{}, fmt.Errorf("ollama: asked for %d embeddings, got %d", len(req.Input), len(out.Embeddings))
	}
	return out, nil
}

// RunningModel is one entry of GET /api/ps: a model loaded in memory now.
// SizeVRAM equal to Size means the whole model is on the GPU.
type RunningModel struct {
	Name          string    `json:"name"`
	Model         string    `json:"model"`
	Size          int64     `json:"size"`
	SizeVRAM      int64     `json:"size_vram"`
	Digest        string    `json:"digest"`
	ExpiresAt     time.Time `json:"expires_at"`
	ContextLength int       `json:"context_length,omitempty"`
}

// GPUFraction is the share of the model held in VRAM, from 0 to 1.
func (m RunningModel) GPUFraction() float64 {
	if m.Size <= 0 {
		return 0
	}
	return float64(m.SizeVRAM) / float64(m.Size)
}

// Running calls GET /api/ps.
func (c *Client) Running(ctx context.Context) ([]RunningModel, error) {
	var out struct {
		Models []RunningModel `json:"models"`
	}
	if err := c.getJSON(ctx, "/api/ps", &out); err != nil {
		return nil, err
	}
	return out.Models, nil
}

// PullProgress is one line of the POST /api/pull stream. Layer downloads
// report Digest, Total and Completed; the last line has Status "success".
type PullProgress struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
}

// Pull downloads a model and reports progress through fn. Pulling a model that
// is already up to date only re-checks the manifest.
func (c *Client) Pull(ctx context.Context, model string, fn func(PullProgress) error) error {
	resp, err := c.do(ctx, http.MethodPost, "/api/pull", map[string]any{"model": model, "stream": true})
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	var last PullProgress
	err = readNDJSON(resp.Body, func(p PullProgress) error {
		last = p
		if fn != nil {
			return fn(p)
		}
		return nil
	})
	if err != nil {
		return err
	}
	if last.Status != "success" {
		return fmt.Errorf("ollama: pull %s ended with status %q", model, last.Status)
	}
	return nil
}

// readNDJSON decodes one JSON object per line into T and calls fn for each.
// A line with an "error" key is an error that happened after the 200 status
// line was sent, so it is returned as an APIError.
func readNDJSON[T any](body io.Reader, fn func(T) error) error {
	scanner := bufio.NewScanner(body)
	// A single line can hold a long tool call or thinking block, so raise the
	// 64 KiB default limit.
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var probe struct {
			Error string `json:"error"`
		}
		if err := json.Unmarshal(line, &probe); err != nil {
			return fmt.Errorf("ollama: decode stream line: %w", err)
		}
		if probe.Error != "" {
			return &APIError{Status: http.StatusOK, Message: probe.Error}
		}
		var item T
		if err := json.Unmarshal(line, &item); err != nil {
			return fmt.Errorf("ollama: decode stream line: %w", err)
		}
		if err := fn(item); err != nil {
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
