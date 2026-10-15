// Gemm: a model's linear layer over many tokens at once -- a prompt, a
// batch of sequences, a speculative draft -- with the weights in any
// block format, decoded as they are read.
//
// Gemv reads every weight once a token; Gemm reads each tile of weights
// once for 32 tokens, so a prompt costs a few passes over the weights
// where token-at-a-time costs one a token. A workgroup computes a 32 x 32
// tile of Y: 32 tokens by 32 output columns, walking k 32 at a time.
// Each step, its 128 work-items decode the weight tile into shared
// storage (B.Decode, so every format takes the one kernel) and load the
// tokens' tile beside it; then each work-item sums a 4 x 2 patch of the
// tile. Sums are taken in float32 in the order of k.
import (
    "gpu"
    "gpu/dtype"
)

@inlinable public func _gemmBlockKernel<B: dtype.Block>(_ f: B, _ w: gpu.Span<uint8>, _ x: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>,
                                                         _ m: int, _ n: int, _ k: int, _ ldy: int, _ col0: int, _ accumulate: bool) kernel {
    let ws = gpu.Shared<float32>(count: 1056)  // 32 weight rows x 33: a column read is no bank conflict
    let xs = gpu.Shared<float32>(count: 1056)  // 32 tokens x 33
    let t = gpu.LocalIndex.x
    let tx = t & 7                              // weight rows tx·4 ..< tx·4 + 4 of the tile
    let ty = t >> 3                             // tokens ty·2 ..< ty·2 + 2
    let row0 = gpu.GroupIndex.x &* 32           // the tile's first weight row (output column)
    let tok0 = gpu.GroupIndex.y &* 32           // its first token
    let size = B.Size(), bytes = B.Bytes()
    let rowBytes = k / size &* bytes
    var a00: float32 = 0, a01: float32 = 0, a10: float32 = 0, a11: float32 = 0
    var a20: float32 = 0, a21: float32 = 0, a30: float32 = 0, a31: float32 = 0
    var k0 = 0
    while k0 < k {
        var e = t
        while e < 1024 {
            let r = e >> 5
            let c = e & 31
            let kk = k0 &+ c
            var wr = row0 &+ r
            if wr >= m { wr = m &- 1 }
            var v: float32 = 0
            if kk < k {
                v = B.Decode(w, wr &* rowBytes &+ kk / size &* bytes, kk % size)
            }
            ws[r &* 33 &+ c] = v
            var tr = tok0 &+ r
            if tr >= n { tr = n &- 1 }
            xs[r &* 33 &+ c] = kk < k ? x[tr &* k &+ kk] : 0
            e = e &+ 128
        }
        gpu.Barrier()
        var c = 0
        while c < 32 {
            let x0 = xs[(ty &* 2) &* 33 &+ c]
            let x1 = xs[(ty &* 2 &+ 1) &* 33 &+ c]
            let w0 = ws[(tx &* 4) &* 33 &+ c]
            let w1 = ws[(tx &* 4 &+ 1) &* 33 &+ c]
            let w2 = ws[(tx &* 4 &+ 2) &* 33 &+ c]
            let w3 = ws[(tx &* 4 &+ 3) &* 33 &+ c]
            a00 = a00 + w0 * x0
            a01 = a01 + w0 * x1
            a10 = a10 + w1 * x0
            a11 = a11 + w1 * x1
            a20 = a20 + w2 * x0
            a21 = a21 + w2 * x1
            a30 = a30 + w3 * x0
            a31 = a31 + w3 * x1
            c = c &+ 1
        }
        gpu.Barrier()
        k0 = k0 &+ 32
    }
    _gemmStore(y, tok0 &+ ty &* 2, row0 &+ tx &* 4, m, n, ldy, col0, accumulate, a00, a10, a20, a30)
    _gemmStore(y, tok0 &+ ty &* 2 &+ 1, row0 &+ tx &* 4, m, n, ldy, col0, accumulate, a01, a11, a21, a31)
}

/// _gemmStore writes four columns of one token's row of Y, those inside it.
@inlinable public func _gemmStore(_ y: gpu.MutableSpan<float32>, _ tok: int, _ col: int, _ m: int, _ n: int, _ ldy: int, _ col0: int,
                                  _ accumulate: bool, _ v0: float32, _ v1: float32, _ v2: float32, _ v3: float32) {
    if tok >= n {
        return
    }
    let base = tok &* ldy &+ col0 &+ col
    if col < m { y[base] = accumulate ? y[base] + v0 : v0 }
    if col &+ 1 < m { y[base &+ 1] = accumulate ? y[base &+ 1] + v1 : v1 }
    if col &+ 2 < m { y[base &+ 2] = accumulate ? y[base &+ 2] + v2 : v2 }
    if col &+ 3 < m { y[base &+ 3] = accumulate ? y[base &+ 3] + v3 : v3 }
}

/// Gemm is Y = X · Wᵀ for weights in a block format: X is n rows of k
/// floats (n tokens), W is m rows of k in the format -- a GGUF tensor's
/// bytes as they are -- and Y is n rows of m. With accumulate, Y = X·Wᵀ + Y:
/// a residual added in the same pass. A linear layer over n tokens.
@inlinable public func Gemm<B: dtype.Block>(_ w: gpu.Buffer<uint8>, _ format: B, _ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>,
                                            m: int, n: int, k: int, accumulate: bool = false) async throws {
    try await Gemm([w], rows: [m], format, x, into: y, n: n, k: k, accumulate: accumulate)
}

/// Gemm is Y = X · [W0; W1; W2]ᵀ for weights of one format stacked by rows
/// -- a layer's Q, K and V, or gate and up -- each token's outputs one
/// part after another, as Gemv writes one token's. A row of Y is stride
/// floats apart (0: the parts' rows, packed), and the outputs start at
/// column: so several Gemms of different formats fill one row side by
/// side.
@inlinable public func Gemm<B: dtype.Block>(_ parts: [gpu.Buffer<uint8>], rows: [int], _ format: B, _ x: gpu.Buffer<float32>,
                                            into y: gpu.Buffer<float32>, n: int, k: int, accumulate: bool = false,
                                            stride: int = 0, column: int = 0) async throws {
    if k % B.Size() != 0 {
        fatalError("linalg.Gemm: rows of \(k) are not whole blocks of \(B.Size())")
    }
    if parts.count != rows.count || parts.isEmpty {
        throw gpu.DeviceError.badCount
    }
    var m = 0
    for r in rows { m += r }
    if n == 0 || m == 0 { return }
    let rowBytes = k / B.Size() * B.Bytes()
    let ldy = stride == 0 ? m : stride
    if x.count < n * k || column + m > ldy || y.count < (n - 1) * ldy + column + m {
        throw gpu.DeviceError.badCount
    }
    var col0 = column
    for i in 0..<parts.count {
        if parts[i].count < rows[i] * rowBytes {
            throw gpu.DeviceError.badCount
        }
        if rows[i] > 0 {
            try await _gemmBlockKernel.Launch(format, parts[i], x, y, rows[i], n, k, ldy, col0, accumulate,
                                              over: ((rows[i] + 31) / 32 * 128, (n + 31) / 32, 1), workgroup: (128, 1, 1))
        }
        col0 += rows[i]
    }
}
