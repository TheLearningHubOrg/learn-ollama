package sizing

import "testing"

func TestKVCacheMatchesFormula(t *testing.T) {
	// Llama 3.1 8B: 32 layers x 8 KV heads x 128 dims. At 8192 tokens in f16:
	// 2 * 32 * 8 * 128 * 8192 * 2 bytes = 1 GiB exactly.
	e, err := Request{Arch: Presets["llama3.1:8b"], Quant: "Q4_K_M", ContextLength: 8192}.Estimate()
	if err != nil {
		t.Fatal(err)
	}
	if e.KVCacheGiB != 1.0 {
		t.Fatalf("KV cache = %v GiB, want 1.0", e.KVCacheGiB)
	}
	// 8.03e9 weights * 4.85 bits / 8 = about 4.53 GiB, close to the 4.9 GB download.
	if e.WeightsGiB < 4.4 || e.WeightsGiB > 4.7 {
		t.Fatalf("weights = %v GiB, want about 4.5", e.WeightsGiB)
	}
	if e.TotalGiB != round2(e.WeightsGiB+e.KVCacheGiB+OverheadGiB) {
		t.Fatalf("total %v does not add up", e.TotalGiB)
	}
}

func TestParallelAndKVQuantization(t *testing.T) {
	base := Request{Arch: Presets["llama3.1:8b"], Quant: "Q4_K_M", ContextLength: 8192}

	four := base
	four.Parallel = 4
	e4, _ := four.Estimate()
	if e4.KVCacheGiB != 4.0 {
		t.Fatalf("4 slots: KV = %v GiB, want 4.0", e4.KVCacheGiB)
	}

	q8 := base
	q8.KVCacheType = "q8_0"
	e8, _ := q8.Estimate()
	if e8.KVCacheGiB != 0.53 {
		t.Fatalf("q8_0 KV = %v GiB, want 0.53", e8.KVCacheGiB)
	}
}

func TestEstimateRejectsBadInput(t *testing.T) {
	bad := []Request{
		{Arch: Presets["llama3.2:3b"], Quant: "Q9_X", ContextLength: 4096},
		{Arch: Presets["llama3.2:3b"], Quant: "Q4_K_M", ContextLength: 0},
		{Arch: Presets["llama3.2:3b"], Quant: "Q4_K_M", ContextLength: 4096, KVCacheType: "fp8"},
	}
	for _, r := range bad {
		if _, err := r.Estimate(); err == nil {
			t.Errorf("expected an error for %+v", r)
		}
	}
}

func TestGPULayers(t *testing.T) {
	e, _ := Request{Arch: Presets["llama3.1:70b"], Quant: "Q4_K_M", ContextLength: 4096}.Estimate()
	layers := Presets["llama3.1:70b"].Layers

	if got := GPULayers(e, layers, 24); got <= 0 || got >= layers {
		t.Fatalf("70B on a 24 GiB card should be a partial offload, got %d of %d layers", got, layers)
	}
	if got := GPULayers(e, layers, 80); got != layers {
		t.Fatalf("70B Q4_K_M should fit in 80 GiB, got %d of %d layers", got, layers)
	}
	if got := GPULayers(e, layers, 0.5); got != 0 {
		t.Fatalf("0.5 GiB should hold no layers, got %d", got)
	}
}

func TestMaxDecodeTokensPerSecond(t *testing.T) {
	// About 1008 GB/s of memory bandwidth and a 4.9 GB model.
	if got := MaxDecodeTokensPerSecond(1008, 4.9); got != 205.71 {
		t.Fatalf("ceiling = %v, want 205.71", got)
	}
	if got := MaxDecodeTokensPerSecond(100, 0); got != 0 {
		t.Fatalf("zero-size model should return 0, got %v", got)
	}
}
