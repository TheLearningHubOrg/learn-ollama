// Package modelfile parses and lints Ollama Modelfiles.
//
// A Modelfile is a list of instructions, one per line: FROM, PARAMETER,
// SYSTEM, TEMPLATE, ADAPTER, LICENSE and MESSAGE. Instruction names are case
// insensitive. A value may be wrapped in double quotes, or in triple double
// quotes (""") to span several lines. Lines starting with # are comments.
package modelfile

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
)

// ErrNoFrom is returned when a Modelfile has no FROM instruction.
var ErrNoFrom = errors.New("modelfile: no FROM instruction")

// Parameter is one PARAMETER line.
type Parameter struct {
	Name  string `json:"name"`
	Value string `json:"value"`
	Line  int    `json:"line"`
}

// Message is one MESSAGE line: a few-shot turn baked into the model.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Modelfile is the parsed form of a Modelfile.
type Modelfile struct {
	From       string      `json:"from"`
	Adapters   []string    `json:"adapters,omitempty"`
	System     string      `json:"system,omitempty"`
	Template   string      `json:"template,omitempty"`
	Licenses   []string    `json:"licenses,omitempty"`
	Parameters []Parameter `json:"parameters,omitempty"`
	Messages   []Message   `json:"messages,omitempty"`
}

// ParseError points at the line that could not be parsed.
type ParseError struct {
	Line int
	Msg  string
}

func (e *ParseError) Error() string {
	return fmt.Sprintf("modelfile: line %d: %s", e.Line, e.Msg)
}

// Parse reads a Modelfile.
func Parse(r io.Reader) (*Modelfile, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	m := &Modelfile{}
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		start := lineNo
		name, rest, _ := strings.Cut(line, " ")
		rest = strings.TrimSpace(rest)

		// PARAMETER and MESSAGE carry a key before the value.
		key := ""
		upper := strings.ToUpper(name)
		if upper == "PARAMETER" || upper == "MESSAGE" {
			key, rest, _ = strings.Cut(rest, " ")
			rest = strings.TrimSpace(rest)
		}

		value, err := readValue(rest, scanner, &lineNo)
		if err != nil {
			return nil, &ParseError{Line: start, Msg: err.Error()}
		}

		switch upper {
		case "FROM":
			m.From = value
		case "ADAPTER":
			m.Adapters = append(m.Adapters, value)
		case "SYSTEM":
			m.System = value
		case "TEMPLATE":
			m.Template = value
		case "LICENSE":
			m.Licenses = append(m.Licenses, value)
		case "PARAMETER":
			if key == "" || value == "" {
				return nil, &ParseError{Line: start, Msg: "PARAMETER needs a name and a value"}
			}
			m.Parameters = append(m.Parameters, Parameter{Name: strings.ToLower(key), Value: value, Line: start})
		case "MESSAGE":
			role := strings.ToLower(key)
			if role != "system" && role != "user" && role != "assistant" {
				return nil, &ParseError{Line: start, Msg: fmt.Sprintf("MESSAGE role %q must be system, user or assistant", key)}
			}
			m.Messages = append(m.Messages, Message{Role: role, Content: value})
		default:
			return nil, &ParseError{Line: start, Msg: fmt.Sprintf("unknown instruction %q", name)}
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if m.From == "" {
		return nil, ErrNoFrom
	}
	return m, nil
}

// readValue unquotes a value. A value that opens with """ may continue on
// the following lines until the closing """.
func readValue(rest string, scanner *bufio.Scanner, lineNo *int) (string, error) {
	const triple = `"""`
	if strings.HasPrefix(rest, triple) {
		body := strings.TrimPrefix(rest, triple)
		if end := strings.Index(body, triple); end >= 0 {
			return body[:end], nil
		}
		parts := []string{body}
		for scanner.Scan() {
			*lineNo++
			text := scanner.Text()
			if end := strings.Index(text, triple); end >= 0 {
				parts = append(parts, text[:end])
				return strings.Trim(strings.Join(parts, "\n"), "\n"), nil
			}
			parts = append(parts, text)
		}
		return "", errors.New(`unterminated """ block`)
	}
	if len(rest) >= 2 && strings.HasPrefix(rest, `"`) && strings.HasSuffix(rest, `"`) {
		return rest[1 : len(rest)-1], nil
	}
	return rest, nil
}

// String renders the Modelfile back to text in a canonical order.
func (m *Modelfile) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "FROM %s\n", m.From)
	for _, a := range m.Adapters {
		fmt.Fprintf(&b, "ADAPTER %s\n", a)
	}
	for _, p := range m.Parameters {
		v := p.Value
		if strings.ContainsAny(v, " \t") {
			v = strconv.Quote(v)
		}
		fmt.Fprintf(&b, "PARAMETER %s %s\n", p.Name, v)
	}
	if m.Template != "" {
		fmt.Fprintf(&b, "TEMPLATE \"\"\"%s\"\"\"\n", m.Template)
	}
	if m.System != "" {
		fmt.Fprintf(&b, "SYSTEM \"\"\"%s\"\"\"\n", m.System)
	}
	for _, msg := range m.Messages {
		fmt.Fprintf(&b, "MESSAGE %s \"\"\"%s\"\"\"\n", msg.Role, msg.Content)
	}
	for _, l := range m.Licenses {
		fmt.Fprintf(&b, "LICENSE \"\"\"%s\"\"\"\n", l)
	}
	return b.String()
}

// Warning is a lint finding. The Modelfile still builds, but probably not
// the way the author intended.
type Warning struct {
	Line    int    `json:"line,omitempty"`
	Message string `json:"message"`
}

type kind int

const (
	kindInt kind = iota
	kindFloat
	kindBool
	kindString
)

// knownParameters lists the runtime options a Modelfile can set and the type
// each one expects.
var knownParameters = map[string]kind{
	"num_ctx":           kindInt,
	"num_batch":         kindInt,
	"num_gpu":           kindInt,
	"num_thread":        kindInt,
	"num_keep":          kindInt,
	"num_predict":       kindInt,
	"repeat_last_n":     kindInt,
	"top_k":             kindInt,
	"seed":              kindInt,
	"temperature":       kindFloat,
	"top_p":             kindFloat,
	"min_p":             kindFloat,
	"typical_p":         kindFloat,
	"repeat_penalty":    kindFloat,
	"presence_penalty":  kindFloat,
	"frequency_penalty": kindFloat,
	"use_mmap":          kindBool,
	"stop":              kindString,
}

// Lint checks parameter names, types and ranges, and flags common mistakes.
func Lint(m *Modelfile) []Warning {
	var out []Warning
	seen := map[string]int{}
	for _, p := range m.Parameters {
		k, ok := knownParameters[p.Name]
		if !ok {
			out = append(out, Warning{Line: p.Line, Message: fmt.Sprintf("unknown parameter %q", p.Name)})
			continue
		}
		if prev, dup := seen[p.Name]; dup && p.Name != "stop" {
			out = append(out, Warning{Line: p.Line, Message: fmt.Sprintf("%s is also set on line %d; the last value wins", p.Name, prev)})
		}
		seen[p.Name] = p.Line

		switch k {
		case kindInt:
			n, err := strconv.Atoi(p.Value)
			if err != nil {
				out = append(out, Warning{Line: p.Line, Message: fmt.Sprintf("%s expects an integer, got %q", p.Name, p.Value)})
			} else if p.Name == "num_ctx" && n < 2048 {
				out = append(out, Warning{Line: p.Line, Message: "num_ctx below 2048 truncates most chat histories and RAG prompts"})
			}
		case kindFloat:
			f, err := strconv.ParseFloat(p.Value, 64)
			if err != nil {
				out = append(out, Warning{Line: p.Line, Message: fmt.Sprintf("%s expects a number, got %q", p.Name, p.Value)})
			} else if p.Name == "temperature" && (f < 0 || f > 2) {
				out = append(out, Warning{Line: p.Line, Message: "temperature outside 0-2 is almost never useful"})
			} else if (p.Name == "top_p" || p.Name == "min_p") && (f < 0 || f > 1) {
				out = append(out, Warning{Line: p.Line, Message: p.Name + " must be between 0 and 1"})
			}
		case kindBool:
			if _, err := strconv.ParseBool(p.Value); err != nil {
				out = append(out, Warning{Line: p.Line, Message: fmt.Sprintf("%s expects true or false, got %q", p.Name, p.Value)})
			}
		}
	}
	if m.Template != "" && !strings.Contains(m.Template, "{{") {
		out = append(out, Warning{Message: "TEMPLATE has no {{ }} actions, so every prompt renders the same text"})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Line < out[j].Line })
	return out
}

// Param returns the last value set for name, and whether it was set.
func (m *Modelfile) Param(name string) (string, bool) {
	for i := len(m.Parameters) - 1; i >= 0; i-- {
		if m.Parameters[i].Name == name {
			return m.Parameters[i].Value, true
		}
	}
	return "", false
}
