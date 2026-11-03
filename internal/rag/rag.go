// Package rag is a minimal retrieval-augmented generation toolkit: a word
// chunker, an in-memory vector index ranked by dot product, a prompt builder
// with a context budget, and recall@k for evaluating retrieval.
//
// It keeps everything in memory so the mechanics stay visible. Swap the Index
// for pgvector, sqlite-vec or another store once the corpus outgrows RAM.
package rag

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/TheLearningHubOrg/learn-ollama/internal/ollama"
)

// Chunk splits text into pieces of at most size words. Neighbouring chunks
// share overlap words so a sentence cut at a boundary still appears whole in
// one of them.
func Chunk(text string, size, overlap int) ([]string, error) {
	if size <= 0 {
		return nil, errors.New("rag: chunk size must be positive")
	}
	if overlap < 0 || overlap >= size {
		return nil, errors.New("rag: overlap must be in [0, size)")
	}
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil, nil
	}
	step := size - overlap
	var chunks []string
	for start := 0; start < len(words); start += step {
		end := min(start+size, len(words))
		chunks = append(chunks, strings.Join(words[start:end], " "))
		if end == len(words) {
			break
		}
	}
	return chunks, nil
}

// Embedder turns texts into vectors. OllamaEmbedder is the real one; tests
// use a fake.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// OllamaEmbedder calls /api/embed. Prefix is prepended to every text, for
// models trained with task prefixes such as "search_document: " and
// "search_query: " (nomic-embed-text).
type OllamaEmbedder struct {
	Client *ollama.Client
	Model  string
	Prefix string
}

// Embed embeds all texts in one batch request.
func (e OllamaEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	in := make([]string, len(texts))
	for i, t := range texts {
		in[i] = e.Prefix + t
	}
	resp, err := e.Client.Embed(ctx, ollama.EmbedRequest{Model: e.Model, Input: in})
	if err != nil {
		return nil, err
	}
	return resp.Embeddings, nil
}

// Doc is one stored chunk.
type Doc struct {
	ID   string
	Text string
	Vec  []float32
}

// Hit is a search result.
type Hit struct {
	ID    string
	Text  string
	Score float32
}

// Index is an in-memory vector index. Model records which embedding model
// produced the vectors, because vectors from different models are not
// comparable.
type Index struct {
	Model string
	dim   int
	docs  []Doc
}

// NewIndex returns an empty index for vectors from model.
func NewIndex(model string) *Index { return &Index{Model: model} }

// Len is the number of stored chunks.
func (ix *Index) Len() int { return len(ix.docs) }

// Add stores a chunk. The vector is normalised so Search can use a plain dot
// product as cosine similarity. All vectors must have the same dimension.
func (ix *Index) Add(id, text string, vec []float32) error {
	if len(vec) == 0 {
		return errors.New("rag: empty vector")
	}
	if ix.dim == 0 {
		ix.dim = len(vec)
	}
	if len(vec) != ix.dim {
		return fmt.Errorf("rag: vector for %q has %d dims, index has %d", id, len(vec), ix.dim)
	}
	ix.docs = append(ix.docs, Doc{ID: id, Text: text, Vec: Normalize(vec)})
	return nil
}

// Search returns the k chunks most similar to q, best first. This is an
// exact (brute-force) scan: O(n·d) per query, fine up to tens of thousands of
// chunks. Approximate indexes such as HNSW trade a little recall for speed.
func (ix *Index) Search(q []float32, k int) ([]Hit, error) {
	if len(q) != ix.dim {
		return nil, fmt.Errorf("rag: query has %d dims, index has %d", len(q), ix.dim)
	}
	q = Normalize(q)
	hits := make([]Hit, 0, len(ix.docs))
	for _, d := range ix.docs {
		hits = append(hits, Hit{ID: d.ID, Text: d.Text, Score: Dot(q, d.Vec)})
	}
	sort.SliceStable(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score })
	if k < len(hits) {
		hits = hits[:k]
	}
	return hits, nil
}

// Normalize returns v scaled to unit length (a zero vector is returned as is).
func Normalize(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return v
	}
	inv := float32(1 / math.Sqrt(sum))
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

// Dot is the dot product of two equal-length vectors.
func Dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// BuildPrompt joins retrieved chunks into a user message, best first, and
// stops before the context exceeds maxChars. Character budgets are a cheap
// stand-in for token budgets: about 4 characters per token for English text.
func BuildPrompt(question string, hits []Hit, maxChars int) string {
	var b strings.Builder
	b.WriteString("Context:\n")
	used := 0
	for _, h := range hits {
		entry := fmt.Sprintf("[%s] %s\n", h.ID, h.Text)
		if used+len(entry) > maxChars {
			break
		}
		b.WriteString(entry)
		used += len(entry)
	}
	b.WriteString("\nQuestion: ")
	b.WriteString(question)
	return b.String()
}

// RecallAtK is the share of questions whose relevant chunk ID appears in the
// first k results. results[i] are the ranked IDs returned for question i and
// relevant[i] is the ID that should have been found.
func RecallAtK(results [][]string, relevant []string, k int) float64 {
	if len(results) == 0 || len(results) != len(relevant) {
		return 0
	}
	found := 0
	for i, ids := range results {
		for j := 0; j < len(ids) && j < k; j++ {
			if ids[j] == relevant[i] {
				found++
				break
			}
		}
	}
	return float64(found) / float64(len(results))
}
