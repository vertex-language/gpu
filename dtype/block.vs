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
    /// GGMLType is the format's number in ggml and GGUF (2 for q4_0, 8 for
    /// q8_0): what picks a kernel tuned to it.
    static func GGMLType() -> int
    /// Size is how many elements a block holds.
    static func Size() -> int
    /// Bytes is how many bytes a block takes.
    static func Bytes() -> int
    /// Decode is element j of the block whose first byte is at.
    static func Decode(_ b: gpu.Span<uint8>, _ at: int, _ j: int) -> float32
    /// Dot is the block whose first byte is at block dotted with the
    /// Size() floats from x, the scale applied once: the inner step of a
    /// quantized Gemv. It reads unchecked; the caller has checked that the
    /// block and the floats are there.
    static func Dot(_ block: UnsafeMutablePointer<uint8>, _ x: UnsafeMutablePointer<float32>) -> float32
}

/// _scale is the float16 scale at a block's first two bytes.
@inlinable public func _scale(_ b: gpu.Span<uint8>, _ at: int) -> float32 {
    return float32(float16(bitPattern: uint16(b[at]) | uint16(b[at + 1]) << 8))
}

/// _scaleAt is the float16 scale at a block's first two bytes, which lie
/// 2-byte aligned: every block format's size is even.
@inlinable public func _scaleAt(_ p: UnsafeMutablePointer<uint8>) -> float32 {
    return float32(float16(bitPattern: UnsafePointer<uint16>(UnsafeRawPointer(p)).pointee))
}

/// Q4_0 is ggml's q4_0: a float16 scale d, then 16 bytes of 4-bit
/// quants. Element j < 16 is (low nibble of byte j - 8)·d, and element
/// j + 16 is (high nibble of byte j - 8)·d.
public struct Q4_0: Block {
    public init() {}
    @inlinable public static func GGMLType() -> int { return 2 }
    @inlinable public static func Size() -> int { return 32 }
    @inlinable public static func Bytes() -> int { return 18 }
    @inlinable public static func Decode(_ b: gpu.Span<uint8>, _ at: int, _ j: int) -> float32 {
        let q = b[at + 2 + (j & 15)]
        let n = j < 16 ? q & 0x0F : q >> 4
        return float32(int32(n) - 8) * _scale(b, at)
    }
    @inlinable public static func Dot(_ block: UnsafeMutablePointer<uint8>, _ v: UnsafeMutablePointer<float32>) -> float32 {
        // Blocks are 18 bytes, so their quants start 2-byte aligned: two
        // bytes a load, four elements from each.
        let q = UnsafePointer<uint16>(UnsafeRawPointer(block + 2))
        var s: float32 = 0
        var j = 0
        while j < 8 {
            let two = q[j]
            let e = j &* 2
            s = s + float32(int32(two & 0x0F) &- 8) * v[e] + float32(int32((two >> 4) & 0x0F) &- 8) * v[e &+ 16]
                  + float32(int32((two >> 8) & 0x0F) &- 8) * v[e &+ 1] + float32(int32(two >> 12) &- 8) * v[e &+ 17]
            j = j &+ 1
        }
        return s * _scaleAt(block)
    }
}

/// Q8_0 is ggml's q8_0: a float16 scale d, then 32 int8 quants. Element
/// j is q[j]·d.
public struct Q8_0: Block {
    public init() {}
    @inlinable public static func GGMLType() -> int { return 8 }
    @inlinable public static func Size() -> int { return 32 }
    @inlinable public static func Bytes() -> int { return 34 }
    @inlinable public static func Decode(_ b: gpu.Span<uint8>, _ at: int, _ j: int) -> float32 {
        return float32(int8(bitPattern: b[at + 2 + j])) * _scale(b, at)
    }
    @inlinable public static func Dot(_ block: UnsafeMutablePointer<uint8>, _ v: UnsafeMutablePointer<float32>) -> float32 {
        let q = block + 2
        var s: float32 = 0
        var j = 0
        while j < 32 {
            s = s + float32(int8(bitPattern: q[j])) * v[j]
            j = j &+ 1
        }
        return s * _scaleAt(block)
    }
}

/// _scaleMinK4 is the 6-bit scale and min of sub-block j (0 to 7) of a
/// k-quant block, packed in its 12 scale bytes (ggml's get_scale_min_k4).
@inlinable public func _scaleMinK4(_ j: int, _ q: UnsafeMutablePointer<uint8>) -> (int32, int32) {
    if j < 4 {
        return (int32(q[j] & 63), int32(q[j &+ 4] & 63))
    }
    return (int32((q[j &+ 4] & 0xF) | ((q[j &- 4] >> 6) << 4)), int32((q[j &+ 4] >> 4) | ((q[j] >> 6) << 4)))
}

/// Q4_K is ggml's q4_K: 256 elements in 144 bytes -- a float16 scale d and
/// min dmin, 12 bytes of 6-bit scales and mins for 8 sub-blocks of 32,
/// then 128 bytes of 4-bit quants. Sub-block pair j (0 to 3) is 32 bytes of
/// quants: element l of the first is (low nibble of byte l)·d·sc - dmin·m,
/// of the second (its high nibble)·d·sc' - dmin·m'.
public struct Q4_K: Block {
    public init() {}
    @inlinable public static func GGMLType() -> int { return 12 }
    @inlinable public static func Size() -> int { return 256 }
    @inlinable public static func Bytes() -> int { return 144 }
    @inlinable public static func Decode(_ b: gpu.Span<uint8>, _ at: int, _ e: int) -> float32 {
        let block = b._base + at
        let j = e / 64, l = e % 64
        let (sc, m) = _scaleMinK4(2 * j + (l < 32 ? 0 : 1), block + 4)
        let q = block[16 + j * 32 + (l % 32)]
        let n = l < 32 ? q & 0xF : q >> 4
        // ggml's order: d·sc and dmin·m, then d1·q - m1, each exact but
        // the last, which rounds once.
        return _scaleAt(block) * float32(sc) * float32(n) - _scaleAt(block + 2) * float32(m)
    }
    @inlinable public static func Dot(_ block: UnsafeMutablePointer<uint8>, _ v: UnsafeMutablePointer<float32>) -> float32 {
        let d = _scaleAt(block), dmin = _scaleAt(block + 2)
        let scales = block + 4
        var total: float32 = 0
        var j = 0
        while j < 4 {
            let q = block + (16 &+ j &* 32)
            let x = v + j &* 64
            var lo: float32 = 0, hi: float32 = 0, xlo: float32 = 0, xhi: float32 = 0
            var l = 0
            while l < 32 {
                let byte = q[l]
                let a = x[l], c = x[l &+ 32]
                lo = lo + float32(byte & 0xF) * a
                hi = hi + float32(byte >> 4) * c
                xlo = xlo + a
                xhi = xhi + c
                l = l &+ 1
            }
            let (s1, m1) = _scaleMinK4(j &* 2, scales)
            let (s2, m2) = _scaleMinK4(j &* 2 &+ 1, scales)
            total = total + d * (float32(s1) * lo + float32(s2) * hi) - dmin * (float32(m1) * xlo + float32(m2) * xhi)
            j = j &+ 1
        }
        return total
    }
}

/// Q6_K is ggml's q6_K: 256 elements in 210 bytes -- 128 bytes of each
/// quant's low 4 bits, 64 of its high 2, 16 int8 scales (one a sub-block
/// of 16), and a float16 scale d last. An element is d·sc·(q - 32).
public struct Q6_K: Block {
    public init() {}
    @inlinable public static func GGMLType() -> int { return 14 }
    @inlinable public static func Size() -> int { return 256 }
    @inlinable public static func Bytes() -> int { return 210 }
    @inlinable public static func Decode(_ b: gpu.Span<uint8>, _ at: int, _ e: int) -> float32 {
        let block = b._base + at
        let half = e / 128, r = e % 128, quarter = r / 32, l = r % 32
        let ql = block + half * 64, qh = block + (128 + half * 32)
        let sc = int8(bitPattern: block[192 + half * 8 + l / 16 + quarter * 2])
        let low = quarter % 2 == 0 ? ql[l] : ql[l + 32]
        let q = int32(quarter < 2 ? low & 0xF : low >> 4) | int32((qh[l] >> uint8(quarter * 2)) & 3) << 4
        // ggml's order: d·sc, then ·q.
        return _scaleAt(block + 208) * float32(sc) * float32(q - 32)
    }
    @inlinable public static func Dot(_ block: UnsafeMutablePointer<uint8>, _ v: UnsafeMutablePointer<float32>) -> float32 {
        let d = _scaleAt(block + 208)
        var total: float32 = 0
        var half = 0
        while half < 2 {
            let ql = block + half &* 64, qh = block + (128 &+ half &* 32), sc = block + (192 &+ half &* 8)
            let x = v + half &* 128
            var s: float32 = 0
            var g = 0
            while g < 2 {
                // Sub-blocks of 16: l in 16·g ..< 16·(g + 1), in each quarter.
                var a0: float32 = 0, a1: float32 = 0, a2: float32 = 0, a3: float32 = 0
                var l = g &* 16
                let end = l &+ 16
                while l < end {
                    let lo0 = ql[l], lo1 = ql[l &+ 32], hi = qh[l]
                    a0 = a0 + float32((int32(lo0 & 0xF) | int32(hi & 3) << 4) &- 32) * x[l]
                    a1 = a1 + float32((int32(lo1 & 0xF) | int32((hi >> 2) & 3) << 4) &- 32) * x[l &+ 32]
                    a2 = a2 + float32((int32(lo0 >> 4) | int32((hi >> 4) & 3) << 4) &- 32) * x[l &+ 64]
                    a3 = a3 + float32((int32(lo1 >> 4) | int32(hi >> 6) << 4) &- 32) * x[l &+ 96]
                    l = l &+ 1
                }
                s = s + float32(int8(bitPattern: sc[g])) * a0 + float32(int8(bitPattern: sc[g &+ 2])) * a1
                      + float32(int8(bitPattern: sc[g &+ 4])) * a2 + float32(int8(bitPattern: sc[g &+ 6])) * a3
                g = g &+ 1
            }
            total = total + d * s
            half = half &+ 1
        }
        return total
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
