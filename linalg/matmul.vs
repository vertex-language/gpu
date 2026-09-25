// Dense linear algebra over row-major buffers: Matmul with a fused
// epilogue, Gemv, Transpose.
//
// Matmul is tiled through shared storage: each workgroup computes a 16x16
// block of C, walking K sixteen at a time, and each work-item one element
// of it. Every element's sum is taken in the order of k, so a result is
// the same on every device that rounds each multiply and add, and differs
// only where a device fuses them. The sum is taken in the type's
// dtype.Number.Accumulator -- float32 for a half, as every half GEMM
// does -- and the epilogue too, so C is rounded once, at the end.
import "gpu"
import "gpu/dtype"
import "gpu/parallel"

/// Activation is applied to each element of a Matmul's result, after
/// scale, bias and residual. GELU and SiLU wait on math inside kernels.
public enum Activation {
    case None
    case ReLU

    public var _code: int32 {
        switch self {
        case .None: return 0
        case .ReLU: return 1
        }
    }
}

/// _at is element (row, col) of a row-major matrix with cols columns,
/// read transposed where t holds.
@inlinable public func _at<T: dtype.Number>(_ x: gpu.Span<T>, _ base: int, _ row: int, _ col: int, _ rows: int, _ cols: int, _ t: bool) -> T {
    if t {
        return x[base + col * rows + row]
    }
    return x[base + row * cols + col]
}

@inlinable public func _matmulKernel<T: dtype.Number>(
    _ a: gpu.Span<T>, _ b: gpu.Span<T>, _ c: gpu.MutableSpan<T>,
    _ bias: gpu.Span<T>, _ residual: gpu.Span<T>,
    _ m: int, _ n: int, _ k: int,
    _ strideA: int, _ strideB: int, _ strideC: int,
    _ ta: bool, _ tb: bool,
    _ scale: T, _ hasBias: bool, _ hasResidual: bool, _ activation: int32
) kernel {
    let ta16 = gpu.Shared<T>(count: 256)
    let tb16 = gpu.Shared<T>(count: 256)
    let lx = gpu.LocalIndex.x
    let ly = gpu.LocalIndex.y
    let col = gpu.GroupIndex.x * 16 + lx
    let row = gpu.GroupIndex.y * 16 + ly
    let batch = gpu.GroupIndex.z
    let baseA = batch * strideA
    let baseB = batch * strideB
    var sum: T.Accumulator = 0
    var t = 0
    while t < k {
        // A's rows x t..t+16, and B's t..t+16 x columns, zero past the edges.
        let ak = t + lx
        ta16[ly * 16 + lx] = row < m && ak < k ? _at(a, baseA, row, ak, m, k, ta) : 0
        let bk = t + ly
        tb16[ly * 16 + lx] = bk < k && col < n ? _at(b, baseB, bk, col, k, n, tb) : 0
        gpu.Barrier()
        var i = 0
        while i < 16 {
            sum = sum + T.Widen(ta16[ly * 16 + i]) * T.Widen(tb16[i * 16 + lx])
            i += 1
        }
        gpu.Barrier()
        t += 16
    }
    if row < m && col < n {
        let at = batch * strideC + row * n + col
        var v = sum * T.Widen(scale)
        if hasBias {
            v = v + T.Widen(bias[col])
        }
        if hasResidual {
            v = v + T.Widen(residual[at])
        }
        if activation == 1 && v < 0 {
            v = 0
        }
        c[at] = T.Narrow(v)
    }
}

/// Shape is a matrix product's problem: C (m x n) = A (m x k) · B (k x n),
/// batch of them laid end to end, with A or B read transposed -- stored
/// k x m, or n x k.
public struct Shape {
    public var m: int
    public var n: int
    public var k: int
    public var batch: int
    public var transposeA: bool
    public var transposeB: bool

    public init(m: int, n: int, k: int, batch: int = 1, transposeA: bool = false, transposeB: bool = false) {
        self.m = m
        self.n = n
        self.k = k
        self.batch = batch
        self.transposeA = transposeA
        self.transposeB = transposeB
    }
}

/// Epilogue is what Matmul does to each element of A · B before it writes
/// it: C = activation(scale · (A · B) + bias[column] + residual). It is
/// applied inside the kernel that computes the product, so it costs no
/// extra pass over C.
public struct Epilogue<T: dtype.Number> {
    public var scale: T
    public var bias: gpu.Buffer<T>?
    public var residual: gpu.Buffer<T>?
    public var activation: Activation

    public init(scale: T, bias: gpu.Buffer<T>?, residual: gpu.Buffer<T>?, activation: Activation) {
        self.scale = scale
        self.bias = bias
        self.residual = residual
        self.activation = activation
    }
}

/// Matmul is C = A · B for the problem shape: all row-major.
@inlinable public func Matmul<T: dtype.Number>(_ a: gpu.Buffer<T>, _ b: gpu.Buffer<T>, into c: gpu.Buffer<T>, _ shape: Shape) async throws {
    try await Matmul(a, b, into: c, shape, Epilogue<T>(scale: 1, bias: nil, residual: nil, activation: .None))
}

/// Matmul is C = epilogue(A · B) for the problem shape.
@inlinable public func Matmul<T: dtype.Number>(_ a: gpu.Buffer<T>, _ b: gpu.Buffer<T>, into c: gpu.Buffer<T>, _ s: Shape, _ e: Epilogue<T>) async throws {
    if s.m == 0 || s.n == 0 || s.batch == 0 {
        return
    }
    try await _matmulKernel.Launch(a, b, c, e.bias ?? c, e.residual ?? c, s.m, s.n, s.k, s.m * s.k, s.k * s.n, s.m * s.n,
                                   s.transposeA, s.transposeB, e.scale, e.bias != nil, e.residual != nil, e.activation._code,
                                   over: ((s.n + 15) / 16 * 16, (s.m + 15) / 16 * 16, s.batch), workgroup: (16, 16, 1))
}

@inlinable public func _gemvKernel<T: dtype.Number>(_ a: gpu.Span<T>, _ x: gpu.Span<T>, _ y: gpu.MutableSpan<T>, _ m: int, _ k: int, _ accumulate: bool) kernel {
    // A row a workgroup: each work-item sums a strided part of it, and the
    // group adds the parts in a fixed order.
    let row = gpu.GroupIndex.x
    var part: T.Accumulator = 0
    var j = parallel.GroupRank()
    while j < k {
        part = part + T.Widen(a[row * k + j]) * T.Widen(x[j])
        j += parallel.GroupCount()
    }
    let total = parallel.GroupSum(part)
    if parallel.GroupRank() == 0 && row < m {
        y[row] = T.Narrow(accumulate ? total + T.Widen(y[row]) : total)
    }
}

/// Gemv is y = A · x: A is m x k, row-major; x has k elements and y m.
/// With accumulate, y = A · x + y: a residual added in the same pass.
/// The matrix-vector product decoding a model is made of.
@inlinable public func Gemv<T: dtype.Number>(_ a: gpu.Buffer<T>, _ x: gpu.Buffer<T>, into y: gpu.Buffer<T>, m: int, k: int, accumulate: bool = false) async throws {
    if m == 0 {
        return
    }
    try await _gemvKernel.Launch(a, x, y, m, k, accumulate, over: m * 128, workgroup: 128)
}

@inlinable public func _gemvBlockKernel<B: dtype.Block>(_ f: B, _ w: gpu.Span<uint8>, _ x: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>, _ m: int, _ k: int, _ per: int, _ accumulate: bool) kernel {
    // A row takes `per` lanes of a wave (a power of 2), so a short row does
    // not leave most of a wave idle: its lanes dot strided blocks of the
    // row, decoding them in place, and add the parts with butterfly
    // shuffles inside their segment -- no barrier, no shared storage. On
    // the CPU device a wave is one work-item, which takes its whole row.
    let lanes = gpu.Wave.Size
    let sub = gpu.Wave.Lane % per
    let row = (gpu.GroupIndex.x * (128 / lanes) + gpu.LocalIndex.x / lanes) * (lanes / per) + gpu.Wave.Lane / per
    let blocks = k / B.Size()
    let bytes = B.Bytes(), size = B.Size()
    let base = row &* blocks &* bytes
    var part: float32 = 0
    if row < m {
        var j = sub
        // Gemv checked that w holds m rows of blocks and x has k floats, so
        // each block and its floats are in range: B.Dot reads them unchecked.
        while j < blocks {
            part = part + B.Dot(w, base &+ j &* bytes, x, j &* size)
            j = j &+ per
        }
    }
    // Every lane takes part in the shuffles, a row past m's too.
    var offset = per / 2
    while offset > 0 {
        part = part + gpu.Wave.ShuffleXor(part, mask: offset)
        offset = offset / 2
    }
    if sub == 0 && row < m {
        y[row] = accumulate ? part + y[row] : part
    }
}

/// Gemv is y = W · x with W block-quantized: m rows of k elements, each
/// row k / format.Size() blocks of the format, one after another -- a GGUF
/// tensor's bytes as they are. x and y are float32; the sum is taken in
/// float32. With accumulate, y = W · x + y. What decoding a quantized
/// model is made of.
@inlinable public func Gemv<B: dtype.Block>(_ w: gpu.Buffer<uint8>, _ format: B, _ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, m: int, k: int, accumulate: bool = false) async throws {
    if m == 0 {
        return
    }
    if k % B.Size() != 0 || w.count < m * (k / B.Size()) * B.Bytes() || x.count < k || y.count < m {
        fatalError("linalg.Gemv: \(m) rows of \(k) in \(w.count) bytes, x of \(x.count), y of \(y.count)")
    }
    // Lanes a row: about two blocks each, a power of 2, at most a wave.
    let wave = y.Device.WaveSize
    var per = 1
    while per < wave && per * 2 <= k / B.Size() {
        per *= 2
    }
    let rows = 128 / wave * (wave / per)
    try await _gemvBlockKernel.Launch(format, w, x, y, m, k, per, accumulate, over: (m + rows - 1) / rows * 128, workgroup: 128)
}

@inlinable public func _transposeKernel<T: dtype.Number>(_ x: gpu.Span<T>, _ y: gpu.MutableSpan<T>, _ rows: int, _ cols: int) kernel {
    let tile = gpu.Shared<T>(count: 272) // 16 x 17: a column read is no bank conflict
    let lx = gpu.LocalIndex.x
    let ly = gpu.LocalIndex.y
    let c = gpu.GroupIndex.x * 16 + lx
    let r = gpu.GroupIndex.y * 16 + ly
    if r < rows && c < cols {
        tile[ly * 17 + lx] = x[r * cols + c]
    }
    gpu.Barrier()
    let oc = gpu.GroupIndex.y * 16 + lx
    let orow = gpu.GroupIndex.x * 16 + ly
    if orow < cols && oc < rows {
        y[orow * rows + oc] = tile[lx * 17 + ly]
    }
}

/// Transpose writes the rows x cols row-major matrix x into y as its
/// cols x rows transpose.
@inlinable public func Transpose<T: dtype.Number>(_ x: gpu.Buffer<T>, into y: gpu.Buffer<T>, rows: int, cols: int) async throws {
    if rows == 0 || cols == 0 {
        return
    }
    try await _transposeKernel.Launch(x, y, rows, cols,
                                      over: ((cols + 15) / 16 * 16, (rows + 15) / 16 * 16), workgroup: (16, 16))
}
