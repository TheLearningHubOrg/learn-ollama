// Command ggufinfo prints the header of a GGUF file: architecture,
// quantization, context length, layer shape, tensor types and a memory
// estimate. It works on any .gguf file, including the blobs Ollama stores
// under ~/.ollama/models/blobs.
//
//	go run ./cmd/ggufinfo ./my-model-Q4_K_M.gguf
//	go run ./cmd/ggufinfo -ctx 8192 -parallel 2 ~/.ollama/models/blobs/sha256-...
package main

import (
	"flag"
	"fmt"
	"os"
	"sort"

	"github.com/TheLearningHubOrg/learn-ollama/internal/gguf"
	"github.com/TheLearningHubOrg/learn-ollama/internal/sizing"
)

func main() {
	ctxLen := flag.Int("ctx", 4096, "context length to size the KV cache for")
	parallel := flag.Int("parallel", 1, "parallel request slots (OLLAMA_NUM_PARALLEL)")
	kvType := flag.String("kv", "f16", "KV cache type: f16, q8_0 or q4_0")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: ggufinfo [-ctx N] [-parallel N] [-kv f16|q8_0|q4_0] FILE.gguf")
		os.Exit(2)
	}

	fh, err := os.Open(flag.Arg(0))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer fh.Close()
	f, err := gguf.Read(fh)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	arch := f.Architecture()
	name, _ := f.String("general.name")
	fmt.Printf("name           %s\n", name)
	fmt.Printf("architecture   %s\n", arch)
	fmt.Printf("gguf version   %d\n", f.Version)
	fmt.Printf("file type      %s\n", f.FileType())
	fmt.Printf("parameters     %.2fB\n", float64(f.ParameterCount())/1e9)

	layers, _ := f.ArchUint("block_count")
	trained, _ := f.ArchUint("context_length")
	heads, _ := f.ArchUint("attention.head_count")
	kvHeads, ok := f.ArchUint("attention.head_count_kv")
	if !ok {
		kvHeads = heads // no GQA: every head has its own K and V
	}
	embd, _ := f.ArchUint("embedding_length")
	headDim, ok := f.ArchUint("attention.key_length")
	if !ok && heads > 0 {
		headDim = embd / heads
	}
	fmt.Printf("layers         %d\n", layers)
	fmt.Printf("trained ctx    %d tokens\n", trained)
	fmt.Printf("heads          %d query, %d kv, head_dim %d\n", heads, kvHeads, headDim)

	fmt.Println("tensor types")
	counts := f.TensorTypeCounts()
	types := make([]string, 0, len(counts))
	for t := range counts {
		types = append(types, t)
	}
	sort.Strings(types)
	for _, t := range types {
		fmt.Printf("  %-6s %d tensors\n", t, counts[t])
	}

	if layers == 0 || kvHeads == 0 || headDim == 0 {
		return
	}
	req := sizing.Request{
		Arch: sizing.Architecture{
			ParamsBillion: float64(f.ParameterCount()) / 1e9,
			Layers:        int(layers),
			KVHeads:       int(kvHeads),
			HeadDim:       int(headDim),
		},
		Quant:         f.FileType(),
		ContextLength: *ctxLen,
		Parallel:      *parallel,
		KVCacheType:   *kvType,
	}
	est, err := req.Estimate()
	if err != nil {
		fmt.Printf("estimate       skipped: %v\n", err)
		return
	}
	fmt.Printf("estimate       weights %.2f GiB + KV %.2f GiB + overhead %.2f GiB = %.2f GiB\n",
		est.WeightsGiB, est.KVCacheGiB, est.OverheadGiB, est.TotalGiB)
}
