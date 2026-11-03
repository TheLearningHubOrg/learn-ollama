// Package metrics records gateway traffic and renders it in the Prometheus
// text exposition format. Ollama has no /metrics endpoint of its own, so the
// gateway is where request, latency and token counters come from. It uses
// only the standard library so the format stays visible; a real service
// would normally use the Prometheus client library.
package metrics

import (
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/TheLearningHubOrg/learn-ollama/internal/ollama"
)

// Buckets are latency histogram upper bounds in seconds. LLM requests range
// from milliseconds (embeddings) to minutes (long generations on a CPU).
var Buckets = []float64{0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120}

type reqKey struct {
	route  string
	status int
}

type histogram struct {
	counts []uint64 // one per bucket, non-cumulative
	sum    float64
	count  uint64
}

// Registry holds the gateway's counters. The zero value is not usable; call New.
type Registry struct {
	mu         sync.Mutex
	requests   map[reqKey]uint64
	latency    map[string]*histogram
	tokens     map[string]map[string]uint64 // model -> kind -> count
	inFlight   int64
	startedAt  time.Time
	nowForTest func() time.Time
}

// New returns an empty registry.
func New() *Registry {
	return &Registry{
		requests:  map[reqKey]uint64{},
		latency:   map[string]*histogram{},
		tokens:    map[string]map[string]uint64{},
		startedAt: time.Now(),
	}
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (s *statusRecorder) WriteHeader(code int) {
	s.status = code
	s.ResponseWriter.WriteHeader(code)
}

// Flush keeps streaming handlers working through the wrapper.
func (s *statusRecorder) Flush() {
	if f, ok := s.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// Wrap instruments a handler under a fixed route label. Use the route
// pattern, never the raw URL path, or every distinct path becomes a new time
// series.
func (r *Registry) Wrap(route string, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		r.mu.Lock()
		r.inFlight++
		r.mu.Unlock()
		next(rec, req)
		r.mu.Lock()
		r.inFlight--
		r.mu.Unlock()
		r.observe(route, rec.status, time.Since(start))
	}
}

func (r *Registry) observe(route string, status int, d time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.requests[reqKey{route, status}]++
	h := r.latency[route]
	if h == nil {
		h = &histogram{counts: make([]uint64, len(Buckets))}
		r.latency[route] = h
	}
	sec := d.Seconds()
	for i, ub := range Buckets {
		if sec <= ub {
			h.counts[i]++
			break
		}
	}
	h.sum += sec
	h.count++
}

// AddTokens counts prompt and completion tokens for a model, taken from
// prompt_eval_count and eval_count in Ollama's final response.
func (r *Registry) AddTokens(model string, prompt, completion int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	m := r.tokens[model]
	if m == nil {
		m = map[string]uint64{}
		r.tokens[model] = m
	}
	m["prompt"] += uint64(max(prompt, 0))
	m["completion"] += uint64(max(completion, 0))
}

// Write renders all metrics. running is the current /api/ps result; pass nil
// when Ollama could not be reached, and ollama_up becomes 0.
func (r *Registry) Write(w io.Writer, running []ollama.RunningModel, ollamaUp bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	fmt.Fprintln(w, "# HELP gateway_requests_total HTTP requests handled, by route and status.")
	fmt.Fprintln(w, "# TYPE gateway_requests_total counter")
	keys := make([]reqKey, 0, len(r.requests))
	for k := range r.requests {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].route != keys[j].route {
			return keys[i].route < keys[j].route
		}
		return keys[i].status < keys[j].status
	})
	for _, k := range keys {
		fmt.Fprintf(w, "gateway_requests_total{route=%q,status=\"%d\"} %d\n", k.route, k.status, r.requests[k])
	}

	fmt.Fprintln(w, "# HELP gateway_request_duration_seconds Request latency, by route.")
	fmt.Fprintln(w, "# TYPE gateway_request_duration_seconds histogram")
	for _, route := range sortedKeys(r.latency) {
		h := r.latency[route]
		var cum uint64
		for i, ub := range Buckets {
			cum += h.counts[i]
			fmt.Fprintf(w, "gateway_request_duration_seconds_bucket{route=%q,le=%q} %d\n", route, formatFloat(ub), cum)
		}
		fmt.Fprintf(w, "gateway_request_duration_seconds_bucket{route=%q,le=\"+Inf\"} %d\n", route, h.count)
		fmt.Fprintf(w, "gateway_request_duration_seconds_sum{route=%q} %s\n", route, formatFloat(h.sum))
		fmt.Fprintf(w, "gateway_request_duration_seconds_count{route=%q} %d\n", route, h.count)
	}

	fmt.Fprintln(w, "# HELP gateway_tokens_total Tokens processed, by model and kind (prompt or completion).")
	fmt.Fprintln(w, "# TYPE gateway_tokens_total counter")
	for _, model := range sortedKeys(r.tokens) {
		for _, kind := range sortedKeys(r.tokens[model]) {
			fmt.Fprintf(w, "gateway_tokens_total{model=%q,kind=%q} %d\n", model, kind, r.tokens[model][kind])
		}
	}

	fmt.Fprintln(w, "# HELP gateway_in_flight_requests Requests being handled now.")
	fmt.Fprintln(w, "# TYPE gateway_in_flight_requests gauge")
	fmt.Fprintf(w, "gateway_in_flight_requests %d\n", r.inFlight)

	up := 0
	if ollamaUp {
		up = 1
	}
	fmt.Fprintln(w, "# HELP ollama_up Whether the gateway reached Ollama's /api/ps on this scrape.")
	fmt.Fprintln(w, "# TYPE ollama_up gauge")
	fmt.Fprintf(w, "ollama_up %d\n", up)

	fmt.Fprintln(w, "# HELP ollama_model_loaded_bytes Memory held by a loaded model, by location.")
	fmt.Fprintln(w, "# TYPE ollama_model_loaded_bytes gauge")
	now := time.Now
	if r.nowForTest != nil {
		now = r.nowForTest
	}
	for _, m := range running {
		fmt.Fprintf(w, "ollama_model_loaded_bytes{model=%q,location=\"vram\"} %d\n", m.Name, m.SizeVRAM)
		fmt.Fprintf(w, "ollama_model_loaded_bytes{model=%q,location=\"ram\"} %d\n", m.Name, m.Size-m.SizeVRAM)
	}
	fmt.Fprintln(w, "# HELP ollama_model_expires_in_seconds Time until an idle model is unloaded.")
	fmt.Fprintln(w, "# TYPE ollama_model_expires_in_seconds gauge")
	for _, m := range running {
		fmt.Fprintf(w, "ollama_model_expires_in_seconds{model=%q} %s\n", m.Name, formatFloat(m.ExpiresAt.Sub(now()).Seconds()))
	}
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func formatFloat(f float64) string { return strconv.FormatFloat(f, 'g', -1, 64) }
