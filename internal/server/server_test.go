package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/TheLearningHubOrg/learn-ollama/internal/metrics"
	"github.com/TheLearningHubOrg/learn-ollama/internal/ollama"
)

// newGateway starts a fake Ollama and a gateway pointed at it.
func newGateway(t *testing.T) *httptest.Server {
	t.Helper()
	upstream := http.NewServeMux()
	upstream.HandleFunc("GET /api/version", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"version":"0.0.0-test"}`)
	})
	upstream.HandleFunc("GET /api/tags", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"models":[{"name":"llama3.2:3b","size":2000000000,
			"details":{"parameter_size":"3.2B","quantization_level":"Q4_K_M"}}]}`)
	})
	upstream.HandleFunc("POST /api/chat", func(w http.ResponseWriter, r *http.Request) {
		var req ollama.ChatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Model == "missing" {
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"model 'missing' not found"}`)
			return
		}
		reply := map[string]any{
			"model":             req.Model,
			"message":           map[string]string{"role": "assistant", "content": "echo: " + req.Messages[len(req.Messages)-1].Content},
			"done":              true,
			"prompt_eval_count": len(req.Messages),
			"eval_count":        30,
			"eval_duration":     int64(1.5e9),
			"load_duration":     int64(2e8),
		}
		_ = json.NewEncoder(w).Encode(reply)
	})
	ollamaSrv := httptest.NewServer(upstream)
	t.Cleanup(ollamaSrv.Close)

	gw := &Server{Ollama: ollama.New(ollamaSrv.URL), DefaultModel: "llama3.2:3b"}
	srv := httptest.NewServer(gw.Routes())
	t.Cleanup(srv.Close)
	return srv
}

func call(t *testing.T, method, url, body string) (int, map[string]any) {
	t.Helper()
	req, err := http.NewRequest(method, url, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("response is not JSON: %s", raw)
	}
	return resp.StatusCode, out
}

func TestHealth(t *testing.T) {
	srv := newGateway(t)
	if code, body := call(t, "GET", srv.URL+"/health", ""); code != 200 || body["status"] != "ok" {
		t.Fatalf("GET /health = %d %v", code, body)
	}
	if code, body := call(t, "GET", srv.URL+"/health?deep=1", ""); code != 200 || body["ollama"] != "0.0.0-test" {
		t.Fatalf("GET /health?deep=1 = %d %v", code, body)
	}
}

func TestDeepHealthReportsDeadOllama(t *testing.T) {
	gw := &Server{Ollama: ollama.New("http://127.0.0.1:1")}
	srv := httptest.NewServer(gw.Routes())
	defer srv.Close()
	if code, body := call(t, "GET", srv.URL+"/health?deep=1", ""); code != 503 || body["status"] != "degraded" {
		t.Fatalf("GET /health?deep=1 = %d %v", code, body)
	}
}

func TestModels(t *testing.T) {
	srv := newGateway(t)
	code, body := call(t, "GET", srv.URL+"/api/models", "")
	if code != 200 {
		t.Fatalf("status %d: %v", code, body)
	}
	models := body["models"].([]any)
	first := models[0].(map[string]any)
	if first["quantization"] != "Q4_K_M" || first["size_gb"] != "2.00" {
		t.Fatalf("unexpected model row %v", first)
	}
}

func TestAsk(t *testing.T) {
	srv := newGateway(t)
	code, body := call(t, "POST", srv.URL+"/api/ask", `{"prompt":"hello","system":"be brief"}`)
	if code != 200 {
		t.Fatalf("status %d: %v", code, body)
	}
	if body["answer"] != "echo: hello" || body["model"] != "llama3.2:3b" {
		t.Fatalf("unexpected body %v", body)
	}
	if body["tokens_per_second"] != 20.0 || body["prompt_tokens"] != 2.0 || body["load_ms"] != 200.0 {
		t.Fatalf("unexpected stats %v", body)
	}
}

func TestAskValidation(t *testing.T) {
	srv := newGateway(t)
	cases := []struct {
		body string
		want int
	}{
		{`{"prompt":"  "}`, 400},
		{`not json`, 400},
		{`{"prompt":"hi","temperature":2}`, 400},
		{`{"prompt":"hi","model":"missing"}`, 404},
	}
	for _, c := range cases {
		if code, body := call(t, "POST", srv.URL+"/api/ask", c.body); code != c.want {
			t.Errorf("POST %s = %d %v, want %d", c.body, code, body, c.want)
		}
	}
}

func TestLintEndpoint(t *testing.T) {
	srv := newGateway(t)
	code, body := call(t, "POST", srv.URL+"/api/modelfile/lint", "FROM llama3.2\nPARAMETER num_ctx 512\n")
	if code != 200 {
		t.Fatalf("status %d: %v", code, body)
	}
	if warnings := body["warnings"].([]any); len(warnings) != 1 {
		t.Fatalf("warnings = %v", warnings)
	}
	if code, _ := call(t, "POST", srv.URL+"/api/modelfile/lint", "PARAMETER num_ctx 512\n"); code != 400 {
		t.Fatalf("missing FROM should be 400, got %d", code)
	}
}

func TestEstimateEndpoint(t *testing.T) {
	srv := newGateway(t)
	code, body := call(t, "GET", srv.URL+"/api/estimate?model=llama3.1:8b&quant=Q4_K_M&ctx=8192&vram=8", "")
	if code != 200 {
		t.Fatalf("status %d: %v", code, body)
	}
	est := body["estimate"].(map[string]any)
	if est["kv_cache_gib"] != 1.0 {
		t.Fatalf("estimate = %v", est)
	}
	if body["gpu_layers"] != 32.0 {
		t.Fatalf("an 8B Q4_K_M with 8K context should fit in 8 GiB, got %v layers", body["gpu_layers"])
	}
	if code, _ := call(t, "GET", srv.URL+"/api/estimate?model=nope", ""); code != 400 {
		t.Fatalf("unknown preset should be 400, got %d", code)
	}
}

func TestTokenRequiredExceptHealth(t *testing.T) {
	gw := &Server{Ollama: ollama.New("http://127.0.0.1:1"), Token: "s3cret"}
	srv := httptest.NewServer(gw.Routes())
	defer srv.Close()

	if code, _ := call(t, "GET", srv.URL+"/health", ""); code != 200 {
		t.Fatalf("/health must stay open, got %d", code)
	}
	if code, _ := call(t, "GET", srv.URL+"/api/estimate?model=llama3.1:8b", ""); code != 401 {
		t.Fatalf("no token should be 401, got %d", code)
	}
	for token, want := range map[string]int{"wrong": 401, "s3cret": 200} {
		req, _ := http.NewRequest("GET", srv.URL+"/api/estimate?model=llama3.1:8b", nil)
		req.Header.Set("Authorization", "Bearer "+token)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != want {
			t.Errorf("token %q: status %d, want %d", token, resp.StatusCode, want)
		}
	}
}

func TestMetricsEndpoint(t *testing.T) {
	upstream := http.NewServeMux()
	upstream.HandleFunc("POST /api/chat", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"model":"llama3.2:3b","message":{"role":"assistant","content":"hi"},"done":true,
			"prompt_eval_count":7,"eval_count":3,"eval_duration":1000000}`)
	})
	upstream.HandleFunc("GET /api/ps", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"models":[{"name":"llama3.2:3b","size":3000,"size_vram":3000,"expires_at":"2099-01-01T00:00:00Z"}]}`)
	})
	ollamaSrv := httptest.NewServer(upstream)
	defer ollamaSrv.Close()

	gw := &Server{Ollama: ollama.New(ollamaSrv.URL), DefaultModel: "llama3.2:3b", Metrics: metrics.New()}
	srv := httptest.NewServer(gw.Routes())
	defer srv.Close()
	if code, body := call(t, "POST", srv.URL+"/api/ask", `{"prompt":"hello"}`); code != 200 {
		t.Fatalf("ask = %d %v", code, body)
	}
	resp, err := http.Get(srv.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	text := string(raw)
	for _, want := range []string{
		`gateway_requests_total{route="POST /api/ask",status="200"} 1`,
		`gateway_tokens_total{model="llama3.2:3b",kind="prompt"} 7`,
		`ollama_up 1`,
		`ollama_model_loaded_bytes{model="llama3.2:3b",location="vram"} 3000`,
	} {
		if !strings.Contains(text, want) {
			t.Errorf("missing %q in:\n%s", want, text)
		}
	}
}
