// Command server runs the learn-ollama gateway: a small HTTP service in front
// of an Ollama server.
//
//	PORT          port to listen on (default 8080)
//	OLLAMA_HOST   where Ollama listens (default http://127.0.0.1:11434)
//	OLLAMA_MODEL  model used when a request names none (default llama3.2:3b)
//	GATEWAY_TOKEN bearer token required on every route except /health (unset: no auth)
package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/TheLearningHubOrg/learn-ollama/internal/metrics"
	"github.com/TheLearningHubOrg/learn-ollama/internal/ollama"
	"github.com/TheLearningHubOrg/learn-ollama/internal/server"
)

func main() {
	port := envOr("PORT", "8080")
	host := envOr("OLLAMA_HOST", ollama.DefaultHost)

	srv := &server.Server{
		Ollama:       ollama.New(host),
		DefaultModel: envOr("OLLAMA_MODEL", "llama3.2:3b"),
		Metrics:      metrics.New(),
		Token:        os.Getenv("GATEWAY_TOKEN"),
	}
	if srv.Token == "" {
		log.Print("GATEWAY_TOKEN is not set: the gateway accepts unauthenticated requests")
	}

	// Bind 0.0.0.0 so the port is reachable from outside a container.
	httpServer := &http.Server{
		Addr:              "0.0.0.0:" + port,
		Handler:           srv.Routes(),
		ReadHeaderTimeout: 10 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		log.Printf("gateway listening on %s, Ollama at %s", httpServer.Addr, host)
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatal(err)
		}
	}()

	<-ctx.Done()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		log.Printf("shutdown: %v", err)
	}
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
