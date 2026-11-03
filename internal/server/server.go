// Package server exposes the gateway's HTTP API. It sits in front of an
// Ollama server and adds a health check, input validation, a Modelfile linter
// and a memory estimator.
package server

import (
	"crypto/subtle"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"

	"github.com/TheLearningHubOrg/learn-ollama/internal/metrics"
	"github.com/TheLearningHubOrg/learn-ollama/internal/modelfile"
	"github.com/TheLearningHubOrg/learn-ollama/internal/ollama"
	"github.com/TheLearningHubOrg/learn-ollama/internal/sizing"
)

const maxBody = 1 << 20 // 1 MiB

// Server holds the gateway's dependencies. Metrics and Token are optional:
// with Metrics set the gateway serves GET /metrics, and with Token set every
// route except /health needs "Authorization: Bearer <Token>".
type Server struct {
	Ollama       *ollama.Client
	DefaultModel string
	Metrics      *metrics.Registry
	Token        string
}

// Routes returns the HTTP handler. Method-qualified patterns need Go 1.22+.
func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	handle := func(pattern string, h http.HandlerFunc) {
		if s.Metrics != nil {
			h = s.Metrics.Wrap(pattern, h)
		}
		mux.HandleFunc(pattern, h)
	}
	handle("GET /health", s.health)
	handle("GET /api/models", s.models)
	handle("POST /api/ask", s.ask)
	handle("POST /api/modelfile/lint", s.lint)
	handle("GET /api/estimate", s.estimate)
	if s.Metrics != nil {
		mux.HandleFunc("GET /metrics", s.metrics)
	}
	if s.Token == "" {
		return mux
	}
	return RequireToken(s.Token, mux, "/health")
}

// RequireToken rejects requests without "Authorization: Bearer <token>".
// Paths listed in open stay reachable so probes keep working. The comparison
// runs in constant time so response timing does not leak the token.
func RequireToken(token string, next http.Handler, open ...string) http.Handler {
	want := []byte(token)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for _, p := range open {
			if r.URL.Path == p {
				next.ServeHTTP(w, r)
				return
			}
		}
		got, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
		if !ok || subtle.ConstantTimeCompare([]byte(got), want) != 1 {
			w.Header().Set("WWW-Authenticate", `Bearer realm="learn-ollama"`)
			writeError(w, http.StatusUnauthorized, "missing or invalid bearer token")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// metrics renders Prometheus text. It polls /api/ps on every scrape so the
// loaded-model gauges are current; a failed poll sets ollama_up to 0 instead
// of failing the scrape.
func (s *Server) metrics(w http.ResponseWriter, r *http.Request) {
	running, err := s.Ollama.Running(r.Context())
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	s.Metrics.Write(w, running, err == nil)
}

// health answers without calling Ollama, so a container orchestrator can
// tell "gateway is up" apart from "model server is up". Add ?deep=1 to check
// Ollama too.
func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	if r.URL.Query().Get("deep") == "" {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
		return
	}
	version, err := s.Ollama.Version(r.Context())
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "degraded", "error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "ollama": version})
}

func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	models, err := s.Ollama.List(r.Context())
	if err != nil {
		writeUpstreamError(w, err)
		return
	}
	type row struct {
		Name         string `json:"name"`
		Parameters   string `json:"parameters"`
		Quantization string `json:"quantization"`
		SizeGB       string `json:"size_gb"`
	}
	rows := make([]row, 0, len(models))
	for _, m := range models {
		rows = append(rows, row{
			Name:         m.Name,
			Parameters:   m.Details.ParameterSize,
			Quantization: m.Details.QuantizationLevel,
			SizeGB:       strconv.FormatFloat(float64(m.Size)/1e9, 'f', 2, 64),
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": rows})
}

type askRequest struct {
	Model  string `json:"model"`
	System string `json:"system"`
	Prompt string `json:"prompt"`
	NumCtx int    `json:"num_ctx"`
}

type askResponse struct {
	Model            string  `json:"model"`
	Answer           string  `json:"answer"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	TokensPerSecond  float64 `json:"tokens_per_second"`
	LoadMillis       int64   `json:"load_ms"`
}

func (s *Server) ask(w http.ResponseWriter, r *http.Request) {
	var in askRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBody))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&in); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	if strings.TrimSpace(in.Prompt) == "" {
		writeError(w, http.StatusBadRequest, "prompt is required")
		return
	}
	if in.Model == "" {
		in.Model = s.DefaultModel
	}

	var msgs []ollama.Message
	if in.System != "" {
		msgs = append(msgs, ollama.Message{Role: "system", Content: in.System})
	}
	msgs = append(msgs, ollama.Message{Role: "user", Content: in.Prompt})

	req := ollama.ChatRequest{Model: in.Model, Messages: msgs}
	if in.NumCtx > 0 {
		req.Options = map[string]any{"num_ctx": in.NumCtx}
	}

	resp, err := s.Ollama.Chat(r.Context(), req)
	if err != nil {
		writeUpstreamError(w, err)
		return
	}
	if s.Metrics != nil {
		s.Metrics.AddTokens(resp.Model, resp.PromptEvalCount, resp.EvalCount)
	}
	writeJSON(w, http.StatusOK, askResponse{
		Model:            resp.Model,
		Answer:           resp.Message.Content,
		PromptTokens:     resp.PromptEvalCount,
		CompletionTokens: resp.EvalCount,
		TokensPerSecond:  roundTo(resp.TokensPerSecond(), 1),
		LoadMillis:       resp.LoadDuration / 1e6,
	})
}

func (s *Server) lint(w http.ResponseWriter, r *http.Request) {
	m, err := modelfile.Parse(http.MaxBytesReader(w, r.Body, maxBody))
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	warnings := modelfile.Lint(m)
	if warnings == nil {
		warnings = []modelfile.Warning{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"modelfile": m, "warnings": warnings})
}

// estimate sizes a deployment, for example
// /api/estimate?model=llama3.1:8b&quant=Q4_K_M&ctx=8192&parallel=2&kv=q8_0&vram=12
func (s *Server) estimate(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	name := q.Get("model")
	arch, ok := sizing.Presets[name]
	if !ok {
		writeError(w, http.StatusBadRequest, "unknown model preset "+strconv.Quote(name))
		return
	}
	ctxLen, err1 := intParam(q.Get("ctx"), 4096)
	parallel, err2 := intParam(q.Get("parallel"), 1)
	if err := errors.Join(err1, err2); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	req := sizing.Request{
		Arch:          arch,
		Quant:         valueOr(q.Get("quant"), "Q4_K_M"),
		ContextLength: ctxLen,
		Parallel:      parallel,
		KVCacheType:   q.Get("kv"),
	}
	est, err := req.Estimate()
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	out := map[string]any{"model": name, "estimate": est}
	if v := q.Get("vram"); v != "" {
		vram, err := strconv.ParseFloat(v, 64)
		if err != nil {
			writeError(w, http.StatusBadRequest, "vram must be a number of GiB")
			return
		}
		out["gpu_layers"] = sizing.GPULayers(est, arch.Layers, vram)
		out["total_layers"] = arch.Layers
	}
	writeJSON(w, http.StatusOK, out)
}

func writeUpstreamError(w http.ResponseWriter, err error) {
	var apiErr *ollama.APIError
	if errors.As(err, &apiErr) && apiErr.Status == http.StatusNotFound {
		// Ollama answers 404 when the model has not been pulled.
		writeError(w, http.StatusNotFound, apiErr.Message)
		return
	}
	writeError(w, http.StatusBadGateway, "ollama request failed: "+err.Error())
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil && !errors.Is(err, io.ErrClosedPipe) {
		log.Printf("write response: %v", err)
	}
}

func intParam(raw string, def int) (int, error) {
	if raw == "" {
		return def, nil
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		return 0, errors.New("expected a positive integer, got " + strconv.Quote(raw))
	}
	return n, nil
}

func valueOr(v, def string) string {
	if v == "" {
		return def
	}
	return v
}

func roundTo(v float64, places int) float64 {
	p := 1.0
	for i := 0; i < places; i++ {
		p *= 10
	}
	return float64(int64(v*p+0.5)) / p
}
