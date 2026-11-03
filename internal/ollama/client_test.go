package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fakeOllama answers like a real Ollama server for the endpoints the client uses.
func fakeOllama(t *testing.T) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/version", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"version":"0.0.0-test"}`)
	})
	mux.HandleFunc("GET /api/tags", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"models":[{"name":"llama3.2:3b","size":2019393189,"digest":"abc",
			"details":{"format":"gguf","family":"llama","parameter_size":"3.2B","quantization_level":"Q4_K_M"}}]}`)
	})
	mux.HandleFunc("POST /api/chat", func(w http.ResponseWriter, r *http.Request) {
		var req ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, `{"error":"bad json"}`, http.StatusBadRequest)
			return
		}
		if req.Model == "missing" {
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"model \"missing\" not found, try pulling it first"}`)
			return
		}
		if req.Stream != nil && !*req.Stream {
			fmt.Fprint(w, `{"model":"llama3.2:3b","message":{"role":"assistant","content":"pong"},
				"done":true,"done_reason":"stop","eval_count":50,"eval_duration":1000000000,"prompt_eval_count":12}`)
			return
		}
		// Streaming: one JSON object per line, final object carries the stats.
		w.Header().Set("Content-Type", "application/x-ndjson")
		for _, tok := range []string{"po", "ng"} {
			fmt.Fprintf(w, `{"model":"llama3.2:3b","message":{"role":"assistant","content":%q},"done":false}`+"\n", tok)
		}
		fmt.Fprint(w, `{"model":"llama3.2:3b","message":{"role":"assistant","content":""},"done":true,"eval_count":2,"eval_duration":100000000}`+"\n")
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func TestNewAddsScheme(t *testing.T) {
	cases := map[string]string{
		"":                       DefaultHost,
		"0.0.0.0:11434":          "http://0.0.0.0:11434",
		"http://ollama:11434/":   "http://ollama:11434",
		"https://llm.example.io": "https://llm.example.io",
	}
	for in, want := range cases {
		if got := New(in).BaseURL; got != want {
			t.Errorf("New(%q).BaseURL = %q, want %q", in, got, want)
		}
	}
}

func TestVersionAndList(t *testing.T) {
	c := New(fakeOllama(t).URL)
	ctx := context.Background()

	v, err := c.Version(ctx)
	if err != nil || v != "0.0.0-test" {
		t.Fatalf("Version() = %q, %v", v, err)
	}
	models, err := c.List(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 || models[0].Details.QuantizationLevel != "Q4_K_M" {
		t.Fatalf("List() = %+v", models)
	}
}

func TestChatNonStreaming(t *testing.T) {
	c := New(fakeOllama(t).URL)
	resp, err := c.Chat(context.Background(), ChatRequest{
		Model:    "llama3.2:3b",
		Messages: []Message{{Role: "user", Content: "ping"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Content != "pong" || !resp.Done {
		t.Fatalf("unexpected reply %+v", resp)
	}
	if tps := resp.TokensPerSecond(); math.Abs(tps-50) > 1e-9 {
		t.Fatalf("TokensPerSecond() = %v, want 50", tps)
	}
}

func TestChatStreamConcatenatesChunks(t *testing.T) {
	c := New(fakeOllama(t).URL)
	var sb strings.Builder
	var last ChatResponse
	err := c.ChatStream(context.Background(), ChatRequest{Model: "llama3.2:3b"}, func(r ChatResponse) error {
		sb.WriteString(r.Message.Content)
		last = r
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if sb.String() != "pong" {
		t.Fatalf("stream text = %q, want pong", sb.String())
	}
	if !last.Done || last.EvalCount != 2 {
		t.Fatalf("final chunk = %+v", last)
	}
}

func TestChatStreamStopsWhenCallbackFails(t *testing.T) {
	c := New(fakeOllama(t).URL)
	stop := errors.New("enough")
	calls := 0
	err := c.ChatStream(context.Background(), ChatRequest{Model: "llama3.2:3b"}, func(ChatResponse) error {
		calls++
		return stop
	})
	if !errors.Is(err, stop) || calls != 1 {
		t.Fatalf("err = %v, calls = %d", err, calls)
	}
}

func TestMissingModelIsAPIError(t *testing.T) {
	c := New(fakeOllama(t).URL)
	_, err := c.Chat(context.Background(), ChatRequest{Model: "missing"})
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %v", err)
	}
	if apiErr.Status != http.StatusNotFound || !strings.Contains(apiErr.Message, "not found") {
		t.Fatalf("unexpected error %+v", apiErr)
	}
}

func TestInBandStreamError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"error":"runner process has terminated"}`+"\n")
	}))
	defer srv.Close()
	err := New(srv.URL).ChatStream(context.Background(), ChatRequest{Model: "x"}, func(ChatResponse) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "runner process has terminated") {
		t.Fatalf("expected in-band error, got %v", err)
	}
}
