// Command bench load-tests an Ollama server with streaming chat requests and
// prints time to first token, per-request decode speed and aggregate
// throughput.
//
//	go run ./cmd/bench -model llama3.2:3b -n 32 -c 4
//
// Run it once with -warm to load the model first, so the numbers measure
// serving rather than loading.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/TheLearningHubOrg/learn-ollama/internal/bench"
	"github.com/TheLearningHubOrg/learn-ollama/internal/ollama"
)

func main() {
	host := flag.String("host", envOr("OLLAMA_HOST", ollama.DefaultHost), "Ollama base URL")
	model := flag.String("model", "llama3.2:3b", "model to test")
	n := flag.Int("n", 16, "total requests")
	c := flag.Int("c", 4, "concurrent requests")
	prompt := flag.String("prompt", "Explain how a B-tree index speeds up a range query.", "prompt text")
	numPredict := flag.Int("num-predict", 128, "tokens to generate per request")
	numCtx := flag.Int("num-ctx", 0, "context length (0 keeps the server default)")
	warm := flag.Bool("warm", true, "send one request first so load time is excluded")
	flag.Parse()

	client := ollama.New(*host)
	opts := map[string]any{"num_predict": *numPredict, "temperature": 0}
	if *numCtx > 0 {
		// Every request must use the same num_ctx, or the scheduler reloads the model.
		opts["num_ctx"] = *numCtx
	}
	req := ollama.ChatRequest{
		Model:    *model,
		Messages: []ollama.Message{{Role: "user", Content: *prompt}},
		Options:  opts,
	}

	ctx := context.Background()
	if *warm {
		warmReq := req
		warmReq.Options = map[string]any{"num_predict": 1}
		if *numCtx > 0 {
			warmReq.Options["num_ctx"] = *numCtx
		}
		if _, err := client.Chat(ctx, warmReq); err != nil {
			log.Fatalf("warm-up failed: %v", err)
		}
	}

	samples, wall := bench.Run(ctx, *n, *c, func(ctx context.Context) bench.Sample {
		return one(ctx, client, req)
	})
	fmt.Println(bench.Summarize(samples, wall, *c))
	for _, s := range samples {
		if s.Err != nil {
			fmt.Fprintln(os.Stderr, "error:", s.Err)
		}
	}
}

func one(ctx context.Context, client *ollama.Client, req ollama.ChatRequest) bench.Sample {
	var s bench.Sample
	start := time.Now()
	err := client.ChatStream(ctx, req, func(r ollama.ChatResponse) error {
		if s.TTFT == 0 && r.Message.Content != "" {
			s.TTFT = time.Since(start)
		}
		if r.Done {
			s.EvalCount = r.EvalCount
			s.EvalDuration = time.Duration(r.EvalDuration)
		}
		return nil
	})
	s.Total = time.Since(start)
	s.Err = err
	return s
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
