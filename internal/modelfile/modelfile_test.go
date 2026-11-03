package modelfile

import (
	"errors"
	"os"
	"strings"
	"testing"
)

const coder = `# A coding assistant built on a base model
FROM qwen2.5-coder:7b
PARAMETER num_ctx 8192
PARAMETER temperature 0.1
PARAMETER stop "<|im_end|>"
SYSTEM """
You review Go code.
Answer with a diff when you suggest a change.
"""
MESSAGE user "Is fmt.Println safe for concurrent use?"
MESSAGE assistant "Yes. Each call writes to os.Stdout with a single Write."
`

func TestParseCoder(t *testing.T) {
	m, err := Parse(strings.NewReader(coder))
	if err != nil {
		t.Fatal(err)
	}
	if m.From != "qwen2.5-coder:7b" {
		t.Errorf("From = %q", m.From)
	}
	if v, _ := m.Param("num_ctx"); v != "8192" {
		t.Errorf("num_ctx = %q", v)
	}
	if v, _ := m.Param("stop"); v != "<|im_end|>" {
		t.Errorf("stop = %q, want quotes removed", v)
	}
	want := "You review Go code.\nAnswer with a diff when you suggest a change."
	if m.System != want {
		t.Errorf("System = %q, want %q", m.System, want)
	}
	if len(m.Messages) != 2 || m.Messages[1].Role != "assistant" {
		t.Errorf("Messages = %+v", m.Messages)
	}
	if w := Lint(m); len(w) != 0 {
		t.Errorf("Lint() = %+v, want no warnings", w)
	}
}

func TestParseSingleLineTripleQuotes(t *testing.T) {
	m, err := Parse(strings.NewReader(`FROM llama3.2
SYSTEM """Be brief."""`))
	if err != nil {
		t.Fatal(err)
	}
	if m.System != "Be brief." {
		t.Errorf("System = %q", m.System)
	}
}

func TestInstructionsAreCaseInsensitive(t *testing.T) {
	m, err := Parse(strings.NewReader("from llama3.2\nparameter TEMPERATURE 0.5\n"))
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := m.Param("temperature"); !ok || v != "0.5" {
		t.Errorf("temperature = %q, %v", v, ok)
	}
}

func TestParseErrors(t *testing.T) {
	cases := map[string]string{
		"no from":         "PARAMETER temperature 0.2\n",
		"unknown":         "FROM llama3.2\nRUN echo hi\n",
		"bad role":        "FROM llama3.2\nMESSAGE tool \"hi\"\n",
		"unterminated":    "FROM llama3.2\nSYSTEM \"\"\"never closed\n",
		"parameter value": "FROM llama3.2\nPARAMETER temperature\n",
	}
	for name, src := range cases {
		if _, err := Parse(strings.NewReader(src)); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	if _, err := Parse(strings.NewReader("# only a comment\n")); !errors.Is(err, ErrNoFrom) {
		t.Errorf("expected ErrNoFrom, got %v", err)
	}
	_, err := Parse(strings.NewReader("FROM x\nRUN y\n"))
	var pe *ParseError
	if !errors.As(err, &pe) || pe.Line != 2 {
		t.Errorf("expected ParseError on line 2, got %v", err)
	}
}

func TestLintFindsMistakes(t *testing.T) {
	m, err := Parse(strings.NewReader(`FROM llama3.2
PARAMETER num_ctx 1024
PARAMETER temperature 3
PARAMETER temprature 0.2
PARAMETER top_k forty
PARAMETER num_ctx 4096
TEMPLATE "plain text"
`))
	if err != nil {
		t.Fatal(err)
	}
	got := Lint(m)
	wantFragments := []string{"has no {{ }}", "below 2048", "temperature outside", "unknown parameter", "expects an integer", "last value wins"}
	if len(got) != len(wantFragments) {
		t.Fatalf("Lint() returned %d warnings: %+v", len(got), got)
	}
	for i, frag := range wantFragments {
		if !strings.Contains(got[i].Message, frag) {
			t.Errorf("warning %d = %q, want it to mention %q", i, got[i].Message, frag)
		}
	}
}

func TestRoundTrip(t *testing.T) {
	m, err := Parse(strings.NewReader(coder))
	if err != nil {
		t.Fatal(err)
	}
	again, err := Parse(strings.NewReader(m.String()))
	if err != nil {
		t.Fatalf("re-parse: %v\n%s", err, m.String())
	}
	if again.System != m.System || len(again.Parameters) != len(m.Parameters) {
		t.Fatalf("round trip changed the Modelfile:\n%s", m.String())
	}
}

// The example Modelfiles shipped in the repo must parse and lint clean.
func TestExampleModelfiles(t *testing.T) {
	for _, path := range []string{
		"../../examples/modelfiles/code-reviewer.Modelfile",
		"../../examples/modelfiles/json-extractor.Modelfile",
		"../../examples/modelfiles/import-gguf.Modelfile",
	} {
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		m, err := Parse(f)
		f.Close()
		if err != nil {
			t.Errorf("%s: %v", path, err)
			continue
		}
		if w := Lint(m); len(w) != 0 {
			t.Errorf("%s: unexpected warnings %+v", path, w)
		}
	}
}
