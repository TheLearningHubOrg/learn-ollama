package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/TheLearningHubOrg/learn-ollama/internal/ollama"
)

// scripted replays canned replies and records every request it receives.
type scripted struct {
	replies  []ollama.Message
	requests []ollama.ChatRequest
}

func (s *scripted) Chat(_ context.Context, req ollama.ChatRequest) (ollama.ChatResponse, error) {
	s.requests = append(s.requests, req)
	if len(s.replies) == 0 {
		return ollama.ChatResponse{}, errors.New("no more replies")
	}
	msg := s.replies[0]
	s.replies = s.replies[1:]
	return ollama.ChatResponse{Message: msg, Done: true}, nil
}

func call(name, args string) ollama.Message {
	return ollama.Message{Role: "assistant", ToolCalls: []ollama.ToolCall{{
		Function: ollama.ToolCallFunction{Name: name, Arguments: json.RawMessage(args)},
	}}}
}

func orderTool() Tool {
	return Tool{
		Def: ollama.Tool{Type: "function", Function: ollama.ToolFunction{
			Name:        "get_order_status",
			Description: "Look up the shipping status of an order by numeric id.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"]}`),
		}},
		Run: func(_ context.Context, args json.RawMessage) (string, error) {
			var in struct {
				ID int `json:"id"`
			}
			if err := json.Unmarshal(args, &in); err != nil {
				return "", err
			}
			if in.ID != 42 {
				return "", errors.New("order not found")
			}
			return `{"status":"shipped"}`, nil
		},
	}
}

func TestLoopRunsToolThenAnswers(t *testing.T) {
	llm := &scripted{replies: []ollama.Message{
		call("get_order_status", `{"id":42}`),
		{Role: "assistant", Content: "Order 42 has shipped.", Thinking: "the tool said shipped"},
	}}
	a := &Agent{LLM: llm, Model: "qwen3:8b", Tools: []Tool{orderTool()}}

	answer, transcript, steps, err := a.Run(context.Background(), []ollama.Message{{Role: "user", Content: "Where is order 42?"}})
	if err != nil {
		t.Fatal(err)
	}
	if answer != "Order 42 has shipped." {
		t.Fatalf("answer = %q", answer)
	}
	if len(steps) != 1 || steps[0].Result != `{"status":"shipped"}` {
		t.Fatalf("steps = %+v", steps)
	}
	// user, assistant(tool_calls), tool, assistant(final)
	if len(transcript) != 4 || transcript[2].Role != "tool" || transcript[2].ToolName != "get_order_status" {
		t.Fatalf("transcript = %+v", transcript)
	}
	if transcript[3].Thinking != "" {
		t.Fatal("thinking text should not be kept in the transcript")
	}
	// The second request must carry the tool result back to the model.
	second := llm.requests[1].Messages
	if second[len(second)-1].Content != `{"status":"shipped"}` {
		t.Fatalf("tool result not sent back: %+v", second)
	}
	if len(llm.requests[0].Tools) != 1 {
		t.Fatal("tool definitions not sent")
	}
}

func TestToolErrorsAreShownToModel(t *testing.T) {
	llm := &scripted{replies: []ollama.Message{
		call("get_order_status", `{"id":7}`),
		call("delete_everything", `{}`),
		{Role: "assistant", Content: "I could not find order 7."},
	}}
	a := &Agent{LLM: llm, Model: "m", Tools: []Tool{orderTool()}}
	_, transcript, steps, err := a.Run(context.Background(), []ollama.Message{{Role: "user", Content: "order 7?"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(steps) != 2 || steps[0].Err == nil || !strings.Contains(steps[1].Err.Error(), "unknown tool") {
		t.Fatalf("steps = %+v", steps)
	}
	if !strings.HasPrefix(transcript[2].Content, "error: order not found") {
		t.Fatalf("tool error not forwarded: %q", transcript[2].Content)
	}
}

func TestStepLimit(t *testing.T) {
	replies := make([]ollama.Message, 10)
	for i := range replies {
		replies[i] = call("get_order_status", `{"id":42}`)
	}
	a := &Agent{LLM: &scripted{replies: replies}, Model: "m", Tools: []Tool{orderTool()}, MaxSteps: 3}
	_, _, steps, err := a.Run(context.Background(), nil)
	if !errors.Is(err, ErrStepLimit) || len(steps) != 3 {
		t.Fatalf("err = %v, steps = %d", err, len(steps))
	}
}

func TestToolTimeoutAndTruncation(t *testing.T) {
	slow := Tool{
		Def: ollama.Tool{Type: "function", Function: ollama.ToolFunction{Name: "slow"}},
		Run: func(ctx context.Context, _ json.RawMessage) (string, error) {
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(time.Second):
				return "late", nil
			}
		},
	}
	big := Tool{
		Def: ollama.Tool{Type: "function", Function: ollama.ToolFunction{Name: "big"}},
		Run: func(context.Context, json.RawMessage) (string, error) { return strings.Repeat("x", 100), nil },
	}
	llm := &scripted{replies: []ollama.Message{
		{Role: "assistant", ToolCalls: []ollama.ToolCall{
			{Function: ollama.ToolCallFunction{Name: "slow", Arguments: json.RawMessage(`{}`)}},
			{Function: ollama.ToolCallFunction{Name: "big", Arguments: json.RawMessage(`{}`)}},
		}},
		{Role: "assistant", Content: "done"},
	}}
	a := &Agent{LLM: llm, Model: "m", Tools: []Tool{slow, big}, ToolTimeout: 20 * time.Millisecond, MaxResultBytes: 10}
	_, _, steps, err := a.Run(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !errors.Is(steps[0].Err, context.DeadlineExceeded) {
		t.Fatalf("slow tool err = %v", steps[0].Err)
	}
	if steps[1].Result != strings.Repeat("x", 10)+"\n[truncated]" {
		t.Fatalf("big tool result = %q", steps[1].Result)
	}
}
