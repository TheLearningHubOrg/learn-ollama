// Package bench measures an Ollama deployment the way users feel it: time to
// first token (TTFT), per-request decode speed and aggregate throughput under
// a fixed level of concurrency.
package bench

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// Sample is the result of one request.
type Sample struct {
	TTFT         time.Duration // request start to first content token
	Total        time.Duration // request start to final chunk
	EvalCount    int           // generated tokens, from the final chunk
	EvalDuration time.Duration // decode time reported by Ollama
	Err          error
}

// DecodeRate is tokens per second during decode for this request.
func (s Sample) DecodeRate() float64 {
	if s.EvalDuration <= 0 {
		return 0
	}
	return float64(s.EvalCount) / s.EvalDuration.Seconds()
}

// Summary aggregates samples from one run.
type Summary struct {
	Requests      int
	Errors        int
	Concurrency   int
	Wall          time.Duration
	TTFTP50       time.Duration
	TTFTP95       time.Duration
	TotalP95      time.Duration
	DecodeP50     float64 // tokens/s per request, median
	AggregateRate float64 // all generated tokens / wall time
}

// Run executes n requests with at most concurrency in flight and returns the
// samples in completion order plus the wall-clock time of the whole run.
func Run(ctx context.Context, n, concurrency int, fn func(context.Context) Sample) ([]Sample, time.Duration) {
	if concurrency <= 0 {
		concurrency = 1
	}
	samples := make([]Sample, 0, n)
	var mu sync.Mutex
	var wg sync.WaitGroup
	sem := make(chan struct{}, concurrency)
	start := time.Now()
	for i := 0; i < n; i++ {
		sem <- struct{}{}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			s := fn(ctx)
			mu.Lock()
			samples = append(samples, s)
			mu.Unlock()
		}()
	}
	wg.Wait()
	return samples, time.Since(start)
}

// Summarize computes percentiles over successful samples.
func Summarize(samples []Sample, wall time.Duration, concurrency int) Summary {
	sum := Summary{Requests: len(samples), Wall: wall, Concurrency: concurrency}
	var ttft, total []time.Duration
	var rates []float64
	tokens := 0
	for _, s := range samples {
		if s.Err != nil {
			sum.Errors++
			continue
		}
		ttft = append(ttft, s.TTFT)
		total = append(total, s.Total)
		rates = append(rates, s.DecodeRate())
		tokens += s.EvalCount
	}
	sum.TTFTP50 = Percentile(ttft, 50)
	sum.TTFTP95 = Percentile(ttft, 95)
	sum.TotalP95 = Percentile(total, 95)
	sort.Float64s(rates)
	if len(rates) > 0 {
		sum.DecodeP50 = rates[nearestRank(len(rates), 50)]
	}
	if wall > 0 {
		sum.AggregateRate = float64(tokens) / wall.Seconds()
	}
	return sum
}

// Percentile uses the nearest-rank method: the smallest value with at least
// p percent of the data at or below it. It sorts a copy of d.
func Percentile(d []time.Duration, p float64) time.Duration {
	if len(d) == 0 {
		return 0
	}
	c := append([]time.Duration(nil), d...)
	sort.Slice(c, func(i, j int) bool { return c[i] < c[j] })
	return c[nearestRank(len(c), p)]
}

func nearestRank(n int, p float64) int {
	r := int(math.Ceil(p/100*float64(n))) - 1
	return max(0, min(r, n-1))
}

// String formats the summary as a small report.
func (s Summary) String() string {
	return fmt.Sprintf(
		"requests=%d errors=%d concurrency=%d wall=%s\n"+
			"ttft p50=%s p95=%s  total p95=%s\n"+
			"decode p50=%.1f tok/s per request  aggregate=%.1f tok/s",
		s.Requests, s.Errors, s.Concurrency, s.Wall.Round(time.Millisecond),
		s.TTFTP50.Round(time.Millisecond), s.TTFTP95.Round(time.Millisecond), s.TotalP95.Round(time.Millisecond),
		s.DecodeP50, s.AggregateRate)
}
