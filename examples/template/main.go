// Command template renders a ChatML-style chat template with Go's
// text/template, the engine Ollama uses for the TEMPLATE instruction, and
// prints the exact prompt string the model receives.
// Run: go run ./examples/template
package main

import (
	"os"
	"text/template"
)

// Message mirrors the fields Ollama exposes for each entry of .Messages.
type Message struct {
	Role    string
	Content string
}

// Data is a subset of what Ollama passes to a template. The real value also
// carries .System, .Tools, .Prompt and .Response.
type Data struct {
	Messages []Message
}

const chatML = `{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
`

func main() {
	tmpl := template.Must(template.New("chatml").Parse(chatML))
	data := Data{Messages: []Message{
		{Role: "system", Content: "You are a terse assistant."},
		{Role: "user", Content: "What is 2+2?"},
		{Role: "assistant", Content: "4"},
		{Role: "user", Content: "And doubled?"},
	}}
	if err := tmpl.Execute(os.Stdout, data); err != nil {
		panic(err)
	}
}
