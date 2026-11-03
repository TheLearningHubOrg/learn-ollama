// Command quantize round-trips one block of weights through the Q8_0 and
// Q4_0 schemes that GGUF uses, and prints the storage cost and the error.
// Run: go run ./examples/quantize
package main

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// blockSize is the number of weights that share one scale in Q8_0 and Q4_0.
const blockSize = 32

// quantizeQ8 stores each weight as an int8 next to one scale d = amax/127.
// Storage: 32 bytes of int8 + 2 bytes of fp16 scale = 34 bytes per block.
func quantizeQ8(w []float32) (d float32, q []int8) {
	var amax float32
	for _, x := range w {
		amax = max(amax, abs(x))
	}
	d = amax / 127
	q = make([]int8, len(w))
	if d == 0 {
		return d, q
	}
	for i, x := range w {
		q[i] = int8(math.Round(float64(x / d)))
	}
	return d, q
}

func dequantizeQ8(d float32, q []int8) []float32 {
	out := make([]float32, len(q))
	for i, v := range q {
		out[i] = d * float32(v)
	}
	return out
}

// quantizeQ4 follows llama.cpp's Q4_0: find the weight with the largest
// magnitude (keeping its sign), set d = that / -8, and store each weight as
// a 4-bit value 0..15 that decodes as (q - 8) * d.
// Storage: 16 bytes of nibbles + 2 bytes of fp16 scale = 18 bytes per block.
func quantizeQ4(w []float32) (d float32, q []uint8) {
	var amax, signedMax float32
	for _, x := range w {
		if abs(x) > amax {
			amax, signedMax = abs(x), x
		}
	}
	d = signedMax / -8
	q = make([]uint8, len(w))
	if d == 0 {
		return d, q
	}
	inv := 1 / d
	for i, x := range w {
		q[i] = uint8(min(15, int(x*inv+8.5)))
	}
	return d, q
}

func dequantizeQ4(d float32, q []uint8) []float32 {
	out := make([]float32, len(q))
	for i, v := range q {
		out[i] = (float32(v) - 8) * d
	}
	return out
}

func rmse(a, b []float32) float64 {
	var sum float64
	for i := range a {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}
	return math.Sqrt(sum / float64(len(a)))
}

func abs(x float32) float32 { return float32(math.Abs(float64(x))) }

func main() {
	// Trained weights are roughly normal around zero with a small spread.
	rng := rand.New(rand.NewPCG(42, 7))
	w := make([]float32, blockSize)
	for i := range w {
		w[i] = float32(rng.NormFloat64() * 0.02)
	}

	d8, q8 := quantizeQ8(w)
	r8 := dequantizeQ8(d8, q8)
	d4, q4 := quantizeQ4(w)
	r4 := dequantizeQ4(d4, q4)

	fmt.Printf("%-6s %8s %14s %12s\n", "type", "bytes", "bits/weight", "rmse")
	fmt.Printf("%-6s %8d %14.2f %12s\n", "F16", blockSize*2, 16.0, "0")
	fmt.Printf("%-6s %8d %14.2f %12.6f\n", "Q8_0", blockSize+2, float64((blockSize+2)*8)/blockSize, rmse(w, r8))
	fmt.Printf("%-6s %8d %14.2f %12.6f\n", "Q4_0", blockSize/2+2, float64((blockSize/2+2)*8)/blockSize, rmse(w, r4))
	fmt.Println()
	fmt.Println("first four weights: original -> Q8_0 -> Q4_0")
	for i := 0; i < 4; i++ {
		fmt.Printf("  %+.5f -> %+.5f -> %+.5f\n", w[i], r8[i], r4[i])
	}
}
