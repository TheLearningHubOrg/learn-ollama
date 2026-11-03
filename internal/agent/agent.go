// Package agent runs a tool-calling loop against Ollama's /api/chat. The model
// never executes anything itself: it returns tool_calls, the loop runs the
// matching Go functions and sends the results back as role "tool" messages
// until the model answers in plain text or the step limit is reached.
package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/TheLearningHubOrg/learn-ollama/internal/ollama"
)

// ErrStepLimit is returned when the model keeps calling tools past MaxSteps.
var ErrStepLimit = errors.New("agent: step limit reached")

// Chatter is the one method the loop needs; *ollama.Client satisfies it.
type Chatter interface {
	Chat(ctx context.Context, req ollama.ChatRequest) (ollama.ChatResponse, error)
}

// Tool pairs the schema the model sees with the function that runs it.
type Tool struct {
	Def ollama.Tool
	Run func(ctx context.Context, args json.RawMessage) (string, error)
}

// Agent holds the loop's configuration.
type Agent struct {
	LLM            Chatter
	Model          string
	Tools          []Tool
	MaxSteps       int           // model calls per Run; default 5
	ToolTimeout    time.Duration // per tool call; default 10s
	MaxResultBytes int           // tool output sent back to the model; default 4 KiB
	Options        map[string]any
}

// Step records one tool execution, for logging and tests.
type Step struct {
	Tool   string
	Args   string
	Result string
	Err    error
}

// Run sends messages to the model and executes tool calls until the model
// replies without one. It returns the final text, the full transcript and the
// tool steps taken.
func (a *Agent) Run(ctx context.Context, messages []ollama.Message) (string, []ollama.Message, []Step, error) {
	maxSteps := a.MaxSteps
	if maxSteps <= 0 {
		maxSteps = 5
	}
	defs := make([]ollama.Tool, 0, len(a.Tools))
	byName := make(map[string]Tool, len(a.Tools))
	for _, t := range a.Tools {
		defs = append(defs, t.Def)
		byName[t.Def.Function.Name] = t
	}

	var steps []Step
	for i := 0; i < maxSteps; i++ {
		resp, err := a.LLM.Chat(ctx, ollama.ChatRequest{
			Model:    a.Model,
			Messages: messages,
			Tools:    defs,
			Options:  a.Options,
		})
		if err != nil {
			return "", messages, steps, err
		}
		msg := resp.Message
		// Reasoning text is for the user or the logs, not for the next turn:
		// sending it back wastes context and some templates reject it.
		msg.Thinking = ""
		messages = append(messages, msg)
		if len(msg.ToolCalls) == 0 {
			return msg.Content, messages, steps, nil
		}
		for _, call := range msg.ToolCalls {
			step := a.runOne(ctx, byName, call)
			steps = append(steps, step)
			content := step.Result
			if step.Err != nil {
				// The model sees the failure and can retry with other
				// arguments or explain the problem to the user.
				content = "error: " + step.Err.Error()
			}
			messages = append(messages, ollama.Message{Role: "tool", ToolName: step.Tool, Content: content})
		}
	}
	return "", messages, steps, ErrStepLimit
}

func (a *Agent) runOne(ctx context.Context, tools map[string]Tool, call ollama.ToolCall) Step {
	name := call.Function.Name
	step := Step{Tool: name, Args: string(call.Function.Arguments)}
	tool, ok := tools[name]
	if !ok {
		step.Err = fmt.Errorf("unknown tool %q", name)
		return step
	}
	if !json.Valid(call.Function.Arguments) {
		step.Err = errors.New("arguments are not valid JSON")
		return step
	}
	timeout := a.ToolTimeout
	if timeout <= 0 {
		timeout = 10 * time.Second
	}
	tctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	out, err := tool.Run(tctx, call.Function.Arguments)
	if err != nil {
		step.Err = err
		return step
	}
	limit := a.MaxResultBytes
	if limit <= 0 {
		limit = 4096
	}
	if len(out) > limit {
		out = out[:limit] + "\n[truncated]"
	}
	step.Result = out
	return step
}
