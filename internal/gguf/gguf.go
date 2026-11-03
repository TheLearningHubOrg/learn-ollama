// Package gguf reads the header of a GGUF file: the format version, the
// metadata key-value pairs and the tensor directory. It stops before the
// tensor data, so inspecting a 40 GB model reads only a few megabytes.
//
// Layout (all integers little-endian):
//
//	magic        4 bytes  "GGUF"
//	version      uint32   3 for every current writer
//	tensor_count uint64
//	kv_count     uint64
//	kv_count x   { key: string, type: uint32, value }
//	tensor_count x { name: string, n_dims: uint32, dims: n_dims x uint64,
//	                 type: uint32, offset: uint64 }
//	padding to general.alignment (default 32), then tensor data
//
// A string is a uint64 byte length followed by UTF-8 bytes, with no
// terminator. An array is an element type (uint32), a uint64 count and the
// elements.
package gguf

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
)

// Magic is the first four bytes of every GGUF file.
const Magic = "GGUF"

// DefaultAlignment applies when general.alignment is absent.
const DefaultAlignment = 32

// Limits that protect the reader from corrupt or hostile files.
const (
	maxStringLen   = 16 << 20 // a chat template is a few KB; 16 MiB is generous
	maxCount       = 1 << 24  // tensors, keys or array elements
	maxDims        = 4        // ggml tensors have at most four dimensions
	keptArrayItems = 8        // arrays keep their first items only; Len has the full size
)

// ValueType is the type tag stored before every metadata value.
type ValueType uint32

// Metadata value types, in the order the format defines them.
const (
	TypeUint8 ValueType = iota
	TypeInt8
	TypeUint16
	TypeInt16
	TypeUint32
	TypeInt32
	TypeFloat32
	TypeBool
	TypeString
	TypeArray
	TypeUint64
	TypeInt64
	TypeFloat64
)

// Array is a metadata array. Tokenizer vocabularies hold 100k+ strings, so
// only the first few are kept in Values.
type Array struct {
	ElemType ValueType `json:"elem_type"`
	Len      uint64    `json:"len"`
	Values   []any     `json:"values"`
}

// TensorInfo is one entry of the tensor directory.
type TensorInfo struct {
	Name   string   `json:"name"`
	Dims   []uint64 `json:"dims"`
	Type   uint32   `json:"type"`
	Offset uint64   `json:"offset"`
}

// Elements is the number of weights in the tensor.
func (t TensorInfo) Elements() uint64 {
	n := uint64(1)
	for _, d := range t.Dims {
		n *= d
	}
	return n
}

// TypeName is the ggml type of the tensor, for example Q4_K or F16.
func (t TensorInfo) TypeName() string { return TensorTypeName(t.Type) }

// File is the parsed header.
type File struct {
	Version  uint32         `json:"version"`
	Metadata map[string]any `json:"metadata"`
	Tensors  []TensorInfo   `json:"tensors"`
}

// ErrNotGGUF means the magic bytes are wrong.
var ErrNotGGUF = errors.New("gguf: not a GGUF file (bad magic)")

// Read parses a GGUF header from r.
func Read(r io.Reader) (*File, error) {
	d := &decoder{r: bufio.NewReaderSize(r, 1<<16)}

	var magic [4]byte
	if _, err := io.ReadFull(d.r, magic[:]); err != nil {
		return nil, fmt.Errorf("gguf: read magic: %w", err)
	}
	if string(magic[:]) != Magic {
		return nil, ErrNotGGUF
	}
	f := &File{Version: d.u32(), Metadata: map[string]any{}}
	if d.err == nil && (f.Version < 2 || f.Version > 3) {
		// Version 1 used 32-bit counts and is long gone from model hubs.
		return nil, fmt.Errorf("gguf: unsupported version %d", f.Version)
	}
	tensorCount := d.count()
	kvCount := d.count()

	for i := uint64(0); i < kvCount && d.err == nil; i++ {
		key := d.str()
		f.Metadata[key] = d.value(ValueType(d.u32()))
	}
	for i := uint64(0); i < tensorCount && d.err == nil; i++ {
		t := TensorInfo{Name: d.str()}
		n := d.u32()
		if n > maxDims {
			d.fail(fmt.Errorf("tensor %q has %d dimensions", t.Name, n))
			break
		}
		t.Dims = make([]uint64, n)
		for j := range t.Dims {
			t.Dims[j] = d.u64()
		}
		t.Type = d.u32()
		t.Offset = d.u64()
		f.Tensors = append(f.Tensors, t)
	}
	if d.err != nil {
		return nil, d.err
	}
	return f, nil
}

// String returns a string metadata value.
func (f *File) String(key string) (string, bool) {
	s, ok := f.Metadata[key].(string)
	return s, ok
}

// Uint returns an unsigned integer metadata value of any width.
func (f *File) Uint(key string) (uint64, bool) {
	switch v := f.Metadata[key].(type) {
	case uint8:
		return uint64(v), true
	case uint16:
		return uint64(v), true
	case uint32:
		return uint64(v), true
	case uint64:
		return v, true
	case int32:
		if v >= 0 {
			return uint64(v), true
		}
	case int64:
		if v >= 0 {
			return uint64(v), true
		}
	}
	return 0, false
}

// Architecture is general.architecture, for example "llama" or "qwen2".
// Per-architecture keys use it as a prefix: llama.block_count.
func (f *File) Architecture() string {
	a, _ := f.String("general.architecture")
	return a
}

// ArchUint reads "<architecture>.<suffix>", for example ArchUint("block_count").
func (f *File) ArchUint(suffix string) (uint64, bool) {
	return f.Uint(f.Architecture() + "." + suffix)
}

// Alignment is general.alignment or the default of 32 bytes.
func (f *File) Alignment() uint64 {
	if a, ok := f.Uint("general.alignment"); ok && a > 0 {
		return a
	}
	return DefaultAlignment
}

// ParameterCount sums the elements of every tensor.
func (f *File) ParameterCount() uint64 {
	var n uint64
	for _, t := range f.Tensors {
		n += t.Elements()
	}
	return n
}

// TensorTypeCounts counts tensors per ggml type. A Q4_K_M file shows mostly
// Q4_K with some Q6_K, plus F32 norms: the "mixed" in mixed quantization.
func (f *File) TensorTypeCounts() map[string]int {
	out := map[string]int{}
	for _, t := range f.Tensors {
		out[t.TypeName()]++
	}
	return out
}

// Keys returns the metadata keys in sorted order.
func (f *File) Keys() []string {
	keys := make([]string, 0, len(f.Metadata))
	for k := range f.Metadata {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// FileType names general.file_type, the overall quantization of the file.
func (f *File) FileType() string {
	v, ok := f.Uint("general.file_type")
	if !ok {
		return "unknown"
	}
	if name, ok := fileTypes[v]; ok {
		return name
	}
	return fmt.Sprintf("file_type %d", v)
}

var fileTypes = map[uint64]string{
	0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
	10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
	15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 32: "BF16",
}

var tensorTypes = map[uint32]string{
	0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
	9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
	15: "Q8_K", 30: "BF16",
}

// TensorTypeName names a ggml tensor type id.
func TensorTypeName(t uint32) string {
	if name, ok := tensorTypes[t]; ok {
		return name
	}
	return fmt.Sprintf("type%d", t)
}

// decoder keeps the first error so the parsing code stays linear.
type decoder struct {
	r   *bufio.Reader
	err error
	buf [8]byte
}

func (d *decoder) fail(err error) {
	if d.err == nil {
		if errors.Is(err, io.EOF) {
			err = io.ErrUnexpectedEOF
		}
		d.err = fmt.Errorf("gguf: %w", err)
	}
}

func (d *decoder) read(n int) []byte {
	if d.err != nil {
		return d.buf[:n]
	}
	if _, err := io.ReadFull(d.r, d.buf[:n]); err != nil {
		d.fail(err)
	}
	return d.buf[:n]
}

func (d *decoder) u8() uint8   { return d.read(1)[0] }
func (d *decoder) u16() uint16 { return binary.LittleEndian.Uint16(d.read(2)) }
func (d *decoder) u32() uint32 { return binary.LittleEndian.Uint32(d.read(4)) }
func (d *decoder) u64() uint64 { return binary.LittleEndian.Uint64(d.read(8)) }

func (d *decoder) count() uint64 {
	n := d.u64()
	if n > maxCount {
		d.fail(fmt.Errorf("count %d exceeds limit", n))
		return 0
	}
	return n
}

func (d *decoder) str() string {
	n := d.u64()
	if d.err != nil {
		return ""
	}
	if n > maxStringLen {
		d.fail(fmt.Errorf("string of %d bytes exceeds limit", n))
		return ""
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(d.r, b); err != nil {
		d.fail(err)
		return ""
	}
	return string(b)
}

func (d *decoder) value(t ValueType) any {
	switch t {
	case TypeUint8:
		return d.u8()
	case TypeInt8:
		return int8(d.u8())
	case TypeUint16:
		return d.u16()
	case TypeInt16:
		return int16(d.u16())
	case TypeUint32:
		return d.u32()
	case TypeInt32:
		return int32(d.u32())
	case TypeFloat32:
		return math.Float32frombits(d.u32())
	case TypeBool:
		return d.u8() != 0
	case TypeString:
		return d.str()
	case TypeUint64:
		return d.u64()
	case TypeInt64:
		return int64(d.u64())
	case TypeFloat64:
		return math.Float64frombits(d.u64())
	case TypeArray:
		elem := ValueType(d.u32())
		if elem == TypeArray {
			d.fail(errors.New("nested arrays are not supported"))
			return nil
		}
		n := d.count()
		arr := Array{ElemType: elem, Len: n}
		for i := uint64(0); i < n && d.err == nil; i++ {
			v := d.value(elem)
			if i < keptArrayItems {
				arr.Values = append(arr.Values, v)
			}
		}
		return arr
	default:
		d.fail(fmt.Errorf("unknown value type %d", t))
		return nil
	}
}
