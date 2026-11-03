package gguf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"strings"
	"testing"
)

// writer builds GGUF headers byte by byte, the way a converter would.
type writer struct{ bytes.Buffer }

func (w *writer) u32(v uint32) { _ = binary.Write(&w.Buffer, binary.LittleEndian, v) }
func (w *writer) u64(v uint64) { _ = binary.Write(&w.Buffer, binary.LittleEndian, v) }
func (w *writer) str(s string) { w.u64(uint64(len(s))); w.WriteString(s) }

func (w *writer) kvString(key, val string) {
	w.str(key)
	w.u32(uint32(TypeString))
	w.str(val)
}

func (w *writer) kvU32(key string, val uint32) {
	w.str(key)
	w.u32(uint32(TypeUint32))
	w.u32(val)
}

func (w *writer) tensor(name string, typ uint32, offset uint64, dims ...uint64) {
	w.str(name)
	w.u32(uint32(len(dims)))
	for _, d := range dims {
		w.u64(d)
	}
	w.u32(typ)
	w.u64(offset)
}

// sample is a tiny llama-style header: 5 metadata keys and 3 tensors.
func sample() []byte {
	var w writer
	w.WriteString(Magic)
	w.u32(3) // version
	w.u64(3) // tensors
	w.u64(5) // metadata keys
	w.kvString("general.architecture", "llama")
	w.kvU32("general.file_type", 15) // Q4_K_M
	w.kvU32("llama.block_count", 28)
	w.kvU32("llama.context_length", 131072)
	// tokenizer.ggml.tokens: an array of 10 strings; only the first 8 are kept.
	w.str("tokenizer.ggml.tokens")
	w.u32(uint32(TypeArray))
	w.u32(uint32(TypeString))
	w.u64(10)
	for _, tok := range strings.Split("<s> </s> a b c d e f g h", " ") {
		w.str(tok)
	}
	w.tensor("token_embd.weight", 12, 0, 3072, 128256) // Q4_K
	w.tensor("blk.0.attn_v.weight", 14, 1<<20, 3072, 1024)
	w.tensor("blk.0.attn_norm.weight", 0, 2<<20, 3072)
	return w.Bytes()
}

func TestReadSample(t *testing.T) {
	f, err := Read(bytes.NewReader(sample()))
	if err != nil {
		t.Fatal(err)
	}
	if f.Version != 3 || f.Architecture() != "llama" || f.FileType() != "Q4_K_M" {
		t.Fatalf("header = v%d %q %q", f.Version, f.Architecture(), f.FileType())
	}
	if n, ok := f.ArchUint("block_count"); !ok || n != 28 {
		t.Fatalf("block_count = %d, %v", n, ok)
	}
	if f.Alignment() != DefaultAlignment {
		t.Fatalf("alignment = %d", f.Alignment())
	}
	toks := f.Metadata["tokenizer.ggml.tokens"].(Array)
	if toks.Len != 10 || len(toks.Values) != keptArrayItems || toks.Values[0] != "<s>" {
		t.Fatalf("tokens = %+v", toks)
	}
	if len(f.Tensors) != 3 || f.Tensors[1].TypeName() != "Q6_K" {
		t.Fatalf("tensors = %+v", f.Tensors)
	}
	want := uint64(3072*128256 + 3072*1024 + 3072)
	if got := f.ParameterCount(); got != want {
		t.Fatalf("ParameterCount() = %d, want %d", got, want)
	}
	counts := f.TensorTypeCounts()
	if counts["Q4_K"] != 1 || counts["Q6_K"] != 1 || counts["F32"] != 1 {
		t.Fatalf("type counts = %v", counts)
	}
}

func TestRejectsBadMagic(t *testing.T) {
	_, err := Read(strings.NewReader("GGML\x03\x00\x00\x00"))
	if !errors.Is(err, ErrNotGGUF) {
		t.Fatalf("err = %v, want ErrNotGGUF", err)
	}
}

func TestRejectsOldVersion(t *testing.T) {
	var w writer
	w.WriteString(Magic)
	w.u32(1)
	if _, err := Read(bytes.NewReader(w.Bytes())); err == nil || !strings.Contains(err.Error(), "version 1") {
		t.Fatalf("err = %v", err)
	}
}

func TestTruncatedFileIsUnexpectedEOF(t *testing.T) {
	full := sample()
	for _, cut := range []int{6, 20, 40, len(full) - 3} {
		_, err := Read(bytes.NewReader(full[:cut]))
		if !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Errorf("cut at %d: err = %v, want unexpected EOF", cut, err)
		}
	}
}

func TestRejectsHugeString(t *testing.T) {
	var w writer
	w.WriteString(Magic)
	w.u32(3)
	w.u64(0)
	w.u64(1)
	w.u64(1 << 40) // a key that claims to be 1 TiB long
	if _, err := Read(bytes.NewReader(w.Bytes())); err == nil || !strings.Contains(err.Error(), "exceeds limit") {
		t.Fatalf("err = %v", err)
	}
}
