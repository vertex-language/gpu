// Block-quantized formats: a block of Size elements in Bytes bytes, small
// integers and a shared scale, laid out as ggml lays them -- the bytes of a
// GGUF tensor, used as they are. Each format is a struct with no stored
// properties, a tag type: a kernel generic over one (linalg.Gemv of
// quantized weights) is specialized to it at compile time, as
// llama.cpp's templated kernels are, and the value passed carries nothing.
//
// Decoding is exact: a 4- or 8-bit integer times a float16 scale is a
// float32 with no rounding, so every device decodes to the same bits as
// ggml's dequantize_row.
import "gpu"

/// Block is a block-quantized format. A row of n elements (n a multiple of
/// Size) is n / Size blocks, one after another.
public protocol Block {
    init()
    /// Size is how many elements a block holds.
    static func Size() -> int
    /// Bytes is how many bytes a block takes.
    static func Bytes() -> int
    /// Decode is element j of the block whose first byte is at.
    static func Decode(_ b: gpu.Span<uint8>, _ at: int, _ j: int) -> float32
    /// Dot is the block at `at` dotted with Size() floats of x from xi, the
    /// scale applied once: the inner step of a quantized Gemv.
    static func Dot(_ b: gpu.Span<uint8>, _ at: int, _ x: gpu.Span<float32>, _ xi: int) -> float32
}

/// _scale is the float16 scale at a block's first two bytes.
@inlinable public func _scale(_ b: gpu.Span<uint8>, _ at: int) -> float32 {
    return float32(float16(bitPattern: uint16(b[at]) | uint16(b[at + 1]) << 8))
}

/// Q4_0 is ggml's q4_0: a float16 scale d, then 16 bytes of 4-bit
/// quants. Element j < 16 is (low nibble of byte j - 8)·d, and element
/// j + 16 is (high nibble of byte j - 8)·d.
public struct Q4_0: Block {
    public init() {}
    @inlinable public static func Size() -> int { return 32 }
    @inlinable public static func Bytes() -> int { return 18 }
    @inlinable public static func Decode(_ b: gpu.Span<uint8>, _ at: int, _ j: int) -> float32 {
        let q = b[at + 2 + (j & 15)]
        let n = j < 16 ? q & 0x0F : q >> 4
        return float32(int32(n) - 8) * _scale(b, at)
    }
    @inlinable public static func Dot(_ b: gpu.Span<uint8>, _ at: int, _ x: gpu.Span<float32>, _ xi: int) -> float32 {
        var s: float32 = 0
        var j = 0
        while j < 16 {
            let q = b[at + 2 + j]
            s = s + float32(int32(q & 0x0F) - 8) * x[xi + j] + float32(int32(q >> 4) - 8) * x[xi + j + 16]
            j += 1
        }
        return s * _scale(b, at)
    }
}

/// Q8_0 is ggml's q8_0: a float16 scale d, then 32 int8 quants. Element
/// j is q[j]·d.
public struct Q8_0: Block {
    public init() {}
    @inlinable public static func Size() -> int { return 32 }
    @inlinable public static func Bytes() -> int { return 34 }
    @inlinable public static func Decode(_ b: gpu.Span<uint8>, _ at: int, _ j: int) -> float32 {
        return float32(int8(bitPattern: b[at + 2 + j])) * _scale(b, at)
    }
    @inlinable public static func Dot(_ b: gpu.Span<uint8>, _ at: int, _ x: gpu.Span<float32>, _ xi: int) -> float32 {
        var s: float32 = 0
        var j = 0
        while j < 32 {
            s = s + float32(int8(bitPattern: b[at + 2 + j])) * x[xi + j]
            j += 1
        }
        return s * _scale(b, at)
    }
}

@inlinable public func _dequantizeKernel<B: Block>(_ f: B, _ b: gpu.Span<uint8>, _ at: int, _ y: gpu.MutableSpan<float32>, _ n: int) kernel {
    let i = gpu.Index.x
    if i < n {
        y[i] = B.Decode(b, at + i / B.Size() * B.Bytes(), i % B.Size())
    }
}

/// Dequantize decodes count elements of the format, starting at byte `at`
/// of b (a block boundary), into y. count is a multiple of the format's
/// Size(). What an embedding lookup of quantized rows is:
/// `dtype.Dequantize(w, dtype.Q4_0(), at: row * rowBytes, count: n, into: x)`.
@inlinable public func Dequantize<B: Block>(_ b: gpu.Buffer<uint8>, _ format: B, at: int = 0, count: int, into y: gpu.Buffer<float32>) async throws {
    if count == 0 {
        return
    }
    try await _dequantizeKernel.Launch(format, b, at, y, count, over: count)
}
