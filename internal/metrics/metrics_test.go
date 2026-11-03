package metrics

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/TheLearningHubOrg/learn-ollama/internal/ollama"
)

func TestWrapCountsStatusAndLatency(t *testing.T) {
	r := New()
	ok := r.Wrap("POST /api/ask", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	bad := r.Wrap("POST /api/ask", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusBadRequest) })
	for range 2 {
		ok(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/api/ask", nil))
	}
	bad(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/api/ask", nil))
	r.AddTokens("llama3.2:3b", 12, 30)

	var sb strings.Builder
	r.Write(&sb, nil, false)
	out := sb.String()
	for _, want := range []string{
		`gateway_requests_total{route="POST /api/ask",status="200"} 2`,
		`gateway_requests_total{route="POST /api/ask",status="400"} 1`,
		`gateway_request_duration_seconds_bucket{route="POST /api/ask",le="+Inf"} 3`,
		`gateway_request_duration_seconds_count{route="POST /api/ask"} 3`,
		`gateway_tokens_total{model="llama3.2:3b",kind="completion"} 30`,
		`gateway_tokens_total{model="llama3.2:3b",kind="prompt"} 12`,
		`gateway_in_flight_requests 0`,
		`ollama_up 0`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
}

func TestBucketsAreCumulative(t *testing.T) {
	r := New()
	r.observe("x", 200, 50*time.Millisecond)
	r.observe("x", 200, 3*time.Second)
	var sb strings.Builder
	r.Write(&sb, nil, true)
	out := sb.String()
	if !strings.Contains(out, `le="0.1"} 1`) || !strings.Contains(out, `le="5"} 2`) || !strings.Contains(out, `le="2.5"} 1`) {
		t.Fatalf("buckets not cumulative:\n%s", out)
	}
}

func TestRunningModels(t *testing.T) {
	r := New()
	now := time.Date(2026, 11, 3, 10, 0, 0, 0, time.UTC)
	r.nowForTest = func() time.Time { return now }
	var sb strings.Builder
	r.Write(&sb, []ollama.RunningModel{{Name: "llama3.1:8b", Size: 100, SizeVRAM: 75, ExpiresAt: now.Add(90 * time.Second)}}, true)
	out := sb.String()
	for _, want := range []string{
		`ollama_up 1`,
		`ollama_model_loaded_bytes{model="llama3.1:8b",location="vram"} 75`,
		`ollama_model_loaded_bytes{model="llama3.1:8b",location="ram"} 25`,
		`ollama_model_expires_in_seconds{model="llama3.1:8b"} 90`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
}
