// Command eval runs a golden set of prompts against a model and fails when
// the pass rate drops below a threshold. Use it as a regression gate before
// promoting a new model tag, quantization or Ollama version.
//
//	go run ./examples/eval -model llama3.2:3b -cases examples/eval/cases.jsonl -min 0.9
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/TheLearningHubOrg/learn-ollama/internal/ollama"
)

// Case is one line of the JSONL golden set. A case passes when the answer
// contains every string in MustContain, none in MustNotContain, and matches
// Regex when one is given.
type Case struct {
	ID             string   `json:"id"`
	System         string   `json:"system,omitempty"`
	Prompt         string   `json:"prompt"`
	MustContain    []string `json:"must_contain,omitempty"`
	MustNotContain []string `json:"must_not_contain,omitempty"`
	Regex          string   `json:"regex,omitempty"`
}

func main() {
	model := flag.String("model", "llama3.2:3b", "model tag under test")
	path := flag.String("cases", "examples/eval/cases.jsonl", "JSONL golden set")
	minRate := flag.Float64("min", 0.9, "minimum pass rate, 0-1")
	flag.Parse()

	cases, err := load(*path)
	if err != nil {
		log.Fatal(err)
	}
	client := ollama.New(os.Getenv("OLLAMA_HOST"))
	passed := 0
	for _, c := range cases {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		answer, err := ask(ctx, client, *model, c)
		cancel()
		ok, why := err == nil, ""
		if err != nil {
			why = err.Error()
		} else {
			ok, why = check(c, answer)
		}
		if ok {
			passed++
			fmt.Printf("PASS %s\n", c.ID)
		} else {
			fmt.Printf("FAIL %s: %s\n", c.ID, why)
		}
	}
	rate := float64(passed) / float64(len(cases))
	fmt.Printf("\n%s: %d/%d passed (%.0f%%), threshold %.0f%%\n", *model, passed, len(cases), rate*100, *minRate*100)
	if rate < *minRate {
		os.Exit(1)
	}
}

func ask(ctx context.Context, client *ollama.Client, model string, c Case) (string, error) {
	var msgs []ollama.Message
	if c.System != "" {
		msgs = append(msgs, ollama.Message{Role: "system", Content: c.System})
	}
	msgs = append(msgs, ollama.Message{Role: "user", Content: c.Prompt})
	resp, err := client.Chat(ctx, ollama.ChatRequest{
		Model:    model,
		Messages: msgs,
		// Fixed temperature and seed make runs comparable on one machine.
		// They do not guarantee identical text across GPUs or versions.
		Options: map[string]any{"temperature": 0, "seed": 7, "num_predict": 256},
	})
	if err != nil {
		return "", err
	}
	return resp.Message.Content, nil
}

func check(c Case, answer string) (bool, string) {
	lower := strings.ToLower(answer)
	for _, s := range c.MustContain {
		if !strings.Contains(lower, strings.ToLower(s)) {
			return false, fmt.Sprintf("missing %q", s)
		}
	}
	for _, s := range c.MustNotContain {
		if strings.Contains(lower, strings.ToLower(s)) {
			return false, fmt.Sprintf("contains forbidden %q", s)
		}
	}
	if c.Regex != "" {
		re, err := regexp.Compile(c.Regex)
		if err != nil {
			return false, "bad regex: " + err.Error()
		}
		if !re.MatchString(answer) {
			return false, "does not match " + c.Regex
		}
	}
	return true, ""
}

func load(path string) ([]Case, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var cases []Case
	sc := bufio.NewScanner(f)
	for line := 1; sc.Scan(); line++ {
		text := strings.TrimSpace(sc.Text())
		if text == "" {
			continue
		}
		var c Case
		if err := json.Unmarshal([]byte(text), &c); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		cases = append(cases, c)
	}
	if len(cases) == 0 {
		return nil, fmt.Errorf("%s: no cases", path)
	}
	return cases, sc.Err()
}
