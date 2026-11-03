// Package sizing estimates how much memory a quantized model needs in Ollama
// and how fast it can decode. The numbers are planning estimates: real usage
// also depends on the architecture, the compute graph and the GPU driver.
package sizing

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

const gib = 1 << 30

// BitsPerWeight is the approximate effective size of common GGUF types,
// including block scales and the higher-precision tensors that mixed
// k-quants keep. Values are typical for Llama-style models.
var BitsPerWeight = map[string]float64{
	"F16":    16,
	"BF16":   16,
	"Q8_0":   8.5,
	"Q6_K":   6.56,
	"Q5_K_M": 5.7,
	"Q5_K_S": 5.5,
	"Q4_K_M": 4.85,
	"Q4_K_S": 4.6,
	"Q4_0":   4.55,
	"Q3_K_M": 3.9,
	"Q2_K":   3.0,
}

// KVBytesPerElement is the storage cost of one cached key or value element
// for each OLLAMA_KV_CACHE_TYPE. q8_0 and q4_0 store blocks of 32 values plus
// one 16-bit scale.
var KVBytesPerElement = map[string]float64{
	"f16":  2,
	"q8_0": 34.0 / 32.0,
	"q4_0": 18.0 / 32.0,
}

// Architecture holds the shape numbers that drive the KV cache size. They
// are in the GGUF metadata and in `ollama show` output.
type Architecture struct {
	ParamsBillion float64 `json:"params_billion"`
	Layers        int     `json:"layers"`
	KVHeads       int     `json:"kv_heads"`
	HeadDim       int     `json:"head_dim"`
}

// Presets are a few models used throughout the guide.
var Presets = map[string]Architecture{
	"llama3.2:3b":  {ParamsBillion: 3.21, Layers: 28, KVHeads: 8, HeadDim: 128},
	"llama3.1:8b":  {ParamsBillion: 8.03, Layers: 32, KVHeads: 8, HeadDim: 128},
	"qwen2.5:7b":   {ParamsBillion: 7.62, Layers: 28, KVHeads: 4, HeadDim: 128},
	"llama3.1:70b": {ParamsBillion: 70.6, Layers: 80, KVHeads: 8, HeadDim: 128},
}

// Request describes one deployment to size.
type Request struct {
	Arch          Architecture
	Quant         string
	ContextLength int
	Parallel      int
	KVCacheType   string
}

// Estimate is the result, in GiB.
type Estimate struct {
	WeightsGiB  float64 `json:"weights_gib"`
	KVCacheGiB  float64 `json:"kv_cache_gib"`
	OverheadGiB float64 `json:"overhead_gib"`
	TotalGiB    float64 `json:"total_gib"`
}

// OverheadGiB is a rough allowance for compute buffers and the runtime.
const OverheadGiB = 1.0

// Estimate computes weights + KV cache + overhead.
//
// KV cache bytes = 2 (K and V) x layers x kv_heads x head_dim x context x
// parallel slots x bytes per element. Ollama allocates context for every
// parallel slot, so OLLAMA_NUM_PARALLEL multiplies this term.
func (r Request) Estimate() (Estimate, error) {
	bpw, ok := BitsPerWeight[strings.ToUpper(r.Quant)]
	if !ok {
		return Estimate{}, fmt.Errorf("unknown quantization %q (known: %s)", r.Quant, strings.Join(Quantizations(), ", "))
	}
	kvType := strings.ToLower(r.KVCacheType)
	if kvType == "" {
		kvType = "f16"
	}
	kvBytes, ok := KVBytesPerElement[kvType]
	if !ok {
		return Estimate{}, fmt.Errorf("unknown KV cache type %q (use f16, q8_0 or q4_0)", r.KVCacheType)
	}
	if r.ContextLength <= 0 || r.Arch.Layers <= 0 || r.Arch.ParamsBillion <= 0 {
		return Estimate{}, fmt.Errorf("context length, layers and parameter count must be positive")
	}
	parallel := r.Parallel
	if parallel <= 0 {
		parallel = 1
	}

	weights := r.Arch.ParamsBillion * 1e9 * bpw / 8
	kv := 2 * float64(r.Arch.Layers) * float64(r.Arch.KVHeads) * float64(r.Arch.HeadDim) *
		float64(r.ContextLength) * float64(parallel) * kvBytes

	e := Estimate{
		WeightsGiB:  round2(weights / gib),
		KVCacheGiB:  round2(kv / gib),
		OverheadGiB: OverheadGiB,
	}
	e.TotalGiB = round2(e.WeightsGiB + e.KVCacheGiB + e.OverheadGiB)
	return e, nil
}

// GPULayers returns how many of the model's layers fit in vramGiB, the same
// decision Ollama's scheduler makes before it starts a runner. Layers that do
// not fit run on the CPU from system RAM.
func GPULayers(e Estimate, layers int, vramGiB float64) int {
	if layers <= 0 {
		return 0
	}
	perLayer := (e.WeightsGiB + e.KVCacheGiB) / float64(layers)
	usable := vramGiB - e.OverheadGiB
	if usable <= 0 || perLayer <= 0 {
		return 0
	}
	n := int(math.Floor(usable / perLayer))
	if n > layers {
		return layers
	}
	return n
}

// MaxDecodeTokensPerSecond is the memory-bandwidth ceiling for decoding: each
// generated token streams every weight once, so the ceiling is bandwidth
// divided by model size. Real systems reach roughly 50-80% of it.
func MaxDecodeTokensPerSecond(bandwidthGBps, modelGB float64) float64 {
	if modelGB <= 0 {
		return 0
	}
	return round2(bandwidthGBps / modelGB)
}

// Quantizations lists the known quantization names in sorted order.
func Quantizations() []string {
	names := make([]string, 0, len(BitsPerWeight))
	for name := range BitsPerWeight {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func round2(v float64) float64 { return math.Round(v*100) / 100 }
