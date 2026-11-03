# learn-ollama gateway: a Go service compiled to a static binary on a
# distroless base. Ollama itself runs in its own container (see compose.yaml).

# ---- build ------------------------------------------------------------------
FROM golang:1.26-alpine AS build
WORKDIR /src
# Manifests first so the module layer is cached between source edits.
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/app ./cmd/server

# ---- runtime: no shell, no package manager, runs as uid 65532 ---------------
FROM gcr.io/distroless/static-debian12:nonroot AS runtime
COPY --from=build /out/app /app
ENV PORT=8080 \
    OLLAMA_HOST=http://ollama:11434
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/app"]
