// Command rag is a complete local RAG pipeline in Go: chunk, embed, index,
// retrieve and answer, with every call going to a local Ollama server.
//
//	ollama pull nomic-embed-text && ollama pull llama3.2:3b
//	go run ./examples/rag "What do I need to access production?"
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/TheLearningHubOrg/learn-ollama/internal/ollama"
	"github.com/TheLearningHubOrg/learn-ollama/internal/rag"
)

const (
	embedModel = "nomic-embed-text"
	chatModel  = "llama3.2:3b"
)

var corpus = map[string]string{
	"security-4.1":  "Passwords must be at least 16 characters long. Multi-factor authentication is mandatory for every employee account.",
	"retention-8.2": "Database backups are written to immutable object storage and kept for seven years. Restores are tested every quarter.",
	"remote-2.0":    "Production access from outside the office requires a company-issued hardware security key and an approved change ticket.",
	"oncall-3.4":    "The on-call engineer acknowledges a P1 page within five minutes and opens an incident channel within fifteen.",
}

func main() {
	question := "What do I need to access production?"
	if len(os.Args) > 1 {
		question = strings.Join(os.Args[1:], " ")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client := ollama.New(os.Getenv("OLLAMA_HOST"))
	docEmb := rag.OllamaEmbedder{Client: client, Model: embedModel, Prefix: "search_document: "}
	queryEmb := rag.OllamaEmbedder{Client: client, Model: embedModel, Prefix: "search_query: "}

	// 1. Chunk every document and embed all chunks in one batch request.
	var ids, texts []string
	for id, doc := range corpus {
		chunks, err := rag.Chunk(doc, 60, 10)
		if err != nil {
			log.Fatal(err)
		}
		for i, c := range chunks {
			ids = append(ids, fmt.Sprintf("%s#%d", id, i))
			texts = append(texts, c)
		}
	}
	vecs, err := docEmb.Embed(ctx, texts)
	if err != nil {
		log.Fatalf("embed corpus: %v", err)
	}
	index := rag.NewIndex(embedModel)
	for i := range ids {
		if err := index.Add(ids[i], texts[i], vecs[i]); err != nil {
			log.Fatal(err)
		}
	}

	// 2. Embed the question with the query prefix and retrieve the top chunks.
	qv, err := queryEmb.Embed(ctx, []string{question})
	if err != nil {
		log.Fatalf("embed question: %v", err)
	}
	hits, err := index.Search(qv[0], 2)
	if err != nil {
		log.Fatal(err)
	}
	for _, h := range hits {
		fmt.Printf("retrieved %-16s score=%.3f\n", h.ID, h.Score)
	}

	// 3. Answer only from the retrieved context, citing chunk IDs.
	resp, err := client.Chat(ctx, ollama.ChatRequest{
		Model: chatModel,
		Messages: []ollama.Message{
			{Role: "system", Content: "Answer only from the context. Cite the [id] of every chunk you use. " +
				"If the answer is not in the context, reply: Not in the policy documents."},
			{Role: "user", Content: rag.BuildPrompt(question, hits, 6000)},
		},
		Options: map[string]any{"temperature": 0, "num_ctx": 4096},
	})
	if err != nil {
		log.Fatalf("chat: %v", err)
	}
	fmt.Println()
	fmt.Println(resp.Message.Content)
}
