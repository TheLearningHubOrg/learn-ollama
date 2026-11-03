package rag

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/TheLearningHubOrg/learn-ollama/internal/ollama"
)

func TestChunkOverlap(t *testing.T) {
	chunks, err := Chunk("a b c d e f g", 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"a b c", "c d e", "e f g"}
	if strings.Join(chunks, "|") != strings.Join(want, "|") {
		t.Fatalf("Chunk() = %q, want %q", chunks, want)
	}
}

func TestChunkRejectsBadOverlap(t *testing.T) {
	if _, err := Chunk("a b", 2, 2); err == nil {
		t.Fatal("expected error for overlap >= size")
	}
}

// keywordEmbedder is a fake embedding model: one dimension per keyword, so
// similarity is predictable in tests.
type keywordEmbedder struct{ keywords []string }

func (k keywordEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, len(k.keywords)+1)
		v[len(k.keywords)] = 0.01 // keep every vector non-zero
		lower := strings.ToLower(t)
		for j, kw := range k.keywords {
			if strings.Contains(lower, kw) {
				v[j] = 1
			}
		}
		out[i] = v
	}
	return out, nil
}

func TestSearchRanksRelevantDocFirst(t *testing.T) {
	emb := keywordEmbedder{keywords: []string{"password", "backup", "production", "hardware key"}}
	docs := map[string]string{
		"sec-4.1": "Passwords must be at least 16 characters and use MFA.",
		"ret-8.2": "Database backups are kept on immutable storage for 7 years.",
		"rem-2.0": "Production access requires a company-issued hardware key.",
	}
	ix := NewIndex("fake")
	ctx := context.Background()
	for _, id := range []string{"sec-4.1", "ret-8.2", "rem-2.0"} {
		vecs, _ := emb.Embed(ctx, []string{docs[id]})
		if err := ix.Add(id, docs[id], vecs[0]); err != nil {
			t.Fatal(err)
		}
	}
	q, _ := emb.Embed(ctx, []string{"What do I need for production access? A hardware key?"})
	hits, err := ix.Search(q[0], 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 2 || hits[0].ID != "rem-2.0" {
		t.Fatalf("Search() = %+v", hits)
	}
	if hits[0].Score <= hits[1].Score {
		t.Fatalf("hits not sorted: %+v", hits)
	}
}

func TestAddRejectsDimensionMismatch(t *testing.T) {
	ix := NewIndex("fake")
	_ = ix.Add("a", "x", []float32{1, 0})
	if err := ix.Add("b", "y", []float32{1, 0, 0}); err == nil {
		t.Fatal("expected dimension mismatch error")
	}
}

func TestNormalizeUnitLength(t *testing.T) {
	v := Normalize([]float32{3, 4})
	if math.Abs(float64(Dot(v, v))-1) > 1e-6 {
		t.Fatalf("|v|^2 = %v, want 1", Dot(v, v))
	}
}

func TestBuildPromptRespectsBudget(t *testing.T) {
	hits := []Hit{{ID: "a", Text: strings.Repeat("x", 40)}, {ID: "b", Text: strings.Repeat("y", 40)}}
	p := BuildPrompt("q?", hits, 60)
	if !strings.Contains(p, "[a]") || strings.Contains(p, "[b]") {
		t.Fatalf("budget not applied:\n%s", p)
	}
	if !strings.HasSuffix(p, "Question: q?") {
		t.Fatalf("question missing:\n%s", p)
	}
}

func TestRecallAtK(t *testing.T) {
	results := [][]string{{"a", "b"}, {"c", "d"}, {"e", "f"}, {"g", "h"}}
	relevant := []string{"a", "d", "x", "g"}
	if got := RecallAtK(results, relevant, 1); got != 0.5 {
		t.Fatalf("recall@1 = %v, want 0.5", got)
	}
	if got := RecallAtK(results, relevant, 2); got != 0.75 {
		t.Fatalf("recall@2 = %v, want 0.75", got)
	}
}

func TestOllamaEmbedderAddsPrefix(t *testing.T) {
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ollama.EmbedRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		seen = req.Input
		vecs := make([][]float32, len(req.Input))
		for i := range vecs {
			vecs[i] = []float32{1, 0}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"embeddings": vecs})
	}))
	defer srv.Close()
	e := OllamaEmbedder{Client: ollama.New(srv.URL), Model: "nomic-embed-text", Prefix: "search_query: "}
	if _, err := e.Embed(context.Background(), []string{"hello"}); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 1 || seen[0] != "search_query: hello" {
		t.Fatalf("server saw %q", seen)
	}
}
