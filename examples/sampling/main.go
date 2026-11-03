// Command sampling shows how temperature, top_k and top_p reshape a
// next-token distribution. Run: go run ./examples/sampling
package main

import (
	"fmt"
	"math"
	"sort"
)

type candidate struct {
	token string
	logit float64
	prob  float64
}

// softmax turns logits into probabilities. Dividing by the temperature first
// sharpens the distribution (t < 1) or flattens it (t > 1).
func softmax(cs []candidate, temperature float64) {
	maxLogit := math.Inf(-1)
	for _, c := range cs {
		maxLogit = math.Max(maxLogit, c.logit)
	}
	var sum float64
	for i := range cs {
		cs[i].prob = math.Exp((cs[i].logit - maxLogit) / temperature)
		sum += cs[i].prob
	}
	for i := range cs {
		cs[i].prob /= sum
	}
}

// topK keeps the k most likely candidates. cs must be sorted, most likely first.
func topK(cs []candidate, k int) []candidate {
	if k > 0 && k < len(cs) {
		return cs[:k]
	}
	return cs
}

// topP keeps the smallest prefix whose probability mass reaches p.
func topP(cs []candidate, p float64) []candidate {
	var mass float64
	for i, c := range cs {
		mass += c.prob
		if mass >= p {
			return cs[:i+1]
		}
	}
	return cs
}

func show(label string, cs []candidate) {
	fmt.Printf("%-22s", label)
	for _, c := range cs {
		fmt.Printf(" %s=%.2f", c.token, c.prob)
	}
	fmt.Println()
}

func main() {
	// Made-up logits for the token after "The capital of France is".
	base := []candidate{
		{token: "Paris", logit: 6.0},
		{token: "Lyon", logit: 4.5},
		{token: "France", logit: 4.0},
		{token: "the", logit: 2.5},
		{token: "banana", logit: 0.5},
	}
	sort.Slice(base, func(i, j int) bool { return base[i].logit > base[j].logit })

	for _, t := range []float64{0.2, 0.8, 1.5} {
		cs := append([]candidate(nil), base...)
		softmax(cs, t)
		show(fmt.Sprintf("temperature %.1f", t), cs)
	}

	cs := append([]candidate(nil), base...)
	softmax(cs, 0.8)
	cs = topK(cs, 3)
	softmax(cs, 0.8) // renormalise over the survivors
	show("temp 0.8 + top_k 3", cs)

	cs = append([]candidate(nil), base...)
	softmax(cs, 0.8)
	cs = topP(cs, 0.9)
	softmax(cs, 0.8)
	show("temp 0.8 + top_p 0.9", cs)
}
