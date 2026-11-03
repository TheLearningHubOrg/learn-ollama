package bench

import (
	"context"
	"errors"
	"math"
	"sync/atomic"
	"testing"
	"time"
)

func TestPercentileNearestRank(t *testing.T) {
	d := []time.Duration{50, 10, 40, 20, 30}
	if got := Percentile(d, 50); got != 30 {
		t.Fatalf("p50 = %v, want 30", got)
	}
	if got := Percentile(d, 95); got != 50 {
		t.Fatalf("p95 = %v, want 50", got)
	}
	if got := Percentile(nil, 50); got != 0 {
		t.Fatalf("empty p50 = %v", got)
	}
	if d[0] != 50 {
		t.Fatal("Percentile must not reorder its input")
	}
}

func TestSummarizeSkipsErrors(t *testing.T) {
	samples := []Sample{
		{TTFT: 100 * time.Millisecond, Total: time.Second, EvalCount: 40, EvalDuration: time.Second},
		{TTFT: 300 * time.Millisecond, Total: 2 * time.Second, EvalCount: 60, EvalDuration: 2 * time.Second},
		{Err: errors.New("503 server busy")},
	}
	s := Summarize(samples, 2*time.Second, 2)
	if s.Errors != 1 || s.Requests != 3 {
		t.Fatalf("counts = %+v", s)
	}
	if s.TTFTP95 != 300*time.Millisecond {
		t.Fatalf("ttft p95 = %v", s.TTFTP95)
	}
	if math.Abs(s.AggregateRate-50) > 1e-9 { // 100 tokens / 2 s
		t.Fatalf("aggregate = %v, want 50", s.AggregateRate)
	}
	if s.DecodeP50 != 30 { // rates 40 and 30, median by nearest rank is 30
		t.Fatalf("decode p50 = %v, want 30", s.DecodeP50)
	}
}

func TestRunBoundsConcurrency(t *testing.T) {
	var inFlight, peak atomic.Int32
	samples, _ := Run(context.Background(), 12, 3, func(context.Context) Sample {
		n := inFlight.Add(1)
		for {
			p := peak.Load()
			if n <= p || peak.CompareAndSwap(p, n) {
				break
			}
		}
		time.Sleep(5 * time.Millisecond)
		inFlight.Add(-1)
		return Sample{EvalCount: 1, EvalDuration: time.Millisecond}
	})
	if len(samples) != 12 {
		t.Fatalf("samples = %d", len(samples))
	}
	if peak.Load() > 3 {
		t.Fatalf("peak concurrency %d exceeds 3", peak.Load())
	}
}
