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
import (
    "gpu"
    "gpu/dtype"
    "gpu/parallel"
)

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
    // group adds the parts in a fixed order. For element types other than
    // float32, whose Gemv is _gemvF32Kernel.
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

// ---- matrix-vector products of stacked weights ----
//
// Each Gemv below is y = [W0; W1; W2] · x: the rows of up to three weights
// of one format, one after another, in one launch -- the Q, K and V
// projections of one input, or gate and up, fused without copying the
// weights together. A row finds its part by where it falls: rows below m0
// are W0's, the next m1 W1's, the rest W2's.

/// _rowAt is where row r of the stacked parts begins; a row past m reads
/// row m - 1 again (and writes nothing).
@inlinable public func _rowAt<E>(_ w0: gpu.Span<E>, _ w1: gpu.Span<E>, _ w2: gpu.Span<E>, _ r: int, _ m0: int, _ m1: int, _ m: int, _ row: int) -> UnsafeMutablePointer<E> {
    let c = r < m ? r : m &- 1
    if c < m0 { return w0._base + c &* row }
    if c < m0 &+ m1 { return w1._base + (c &- m0) &* row }
    return w2._base + (c &- m0 &- m1) &* row
}

/// _reduce adds a row's parts across the `per` lanes that hold them.
@inlinable public func _reduce(_ v: float32, _ per: int) -> float32 {
    var part = v
    var offset = per / 2
    while offset > 0 {
        part = part + gpu.Wave.ShuffleXor(part, mask: offset)
        offset = offset / 2
    }
    return part
}

/// _lanes is how many lanes of a wave a row of n units takes: about two
/// units each, a power of 2, at most a wave -- so a short row leaves few
/// lanes idle.
@inlinable public func _lanes(_ units: int, _ wave: int) -> int {
    var per = 1
    while per < wave && per * 2 <= units {
        per *= 2
    }
    return per
}

@inlinable public func _gemvF32Kernel(_ a0: gpu.Span<float32>, _ a1: gpu.Span<float32>, _ a2: gpu.Span<float32>, _ x: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>,
                                      _ m0: int, _ m1: int, _ m: int, _ k: int, _ per: int, _ accumulate: bool) kernel {
    // A row takes `per` lanes of a wave, which read it four floats at a
    // time and add their parts by shuffles: no barrier, no shared storage.
    // On the CPU device a wave is one work-item, which takes its whole row.
    let lanes = gpu.Wave.Size
    let sub = gpu.Wave.Lane % per
    let row = (gpu.GroupIndex.x * (128 / lanes) + gpu.LocalIndex.x / lanes) * (lanes / per) + gpu.Wave.Lane / per
    let p = _rowAt(a0, a1, a2, row, m0, m1, m, k)
    let v = x._base
    var part: float32 = 0
    var j = sub &* 4
    let step = per &* 4
    let whole = k & ~3
    while j < whole {
        part = part + p[j] * v[j] + p[j &+ 1] * v[j &+ 1] + p[j &+ 2] * v[j &+ 2] + p[j &+ 3] * v[j &+ 3]
        j = j &+ step
    }
    if sub == 0 {
        var t = whole
        while t < k {
            part = part + p[t] * v[t]
            t = t &+ 1
        }
    }
    let total = _reduce(part, per)
    if sub == 0 && row < m {
        y[row] = accumulate ? total + y[row] : total
    }
}

/// Gemv is y = A · x: A is m x k, row-major; x has k elements and y m.
/// With accumulate, y = A · x + y: a residual added in the same pass.
/// The matrix-vector product decoding a model is made of.
@inlinable public func Gemv<T: dtype.Number>(_ a: gpu.Buffer<T>, _ x: gpu.Buffer<T>, into y: gpu.Buffer<T>, m: int, k: int, accumulate: bool = false) async throws {
    if m == 0 {
        return
    }
    if T.self == float32.self {
        try await Gemv([a.View(as: float32.self)], rows: [m], x.View(as: float32.self), into: y.View(as: float32.self), k: k, accumulate: accumulate)
        return
    }
    try await _gemvKernel.Launch(a, x, y, m, k, accumulate, over: m * 128, workgroup: 128)
}

/// Gemv is y = [A0; A1; A2] · x for float32 weights stacked by rows: parts
/// holds one to three weights of rows[i] rows of k each.
public func Gemv(_ parts: [gpu.Buffer<float32>], rows: [int], _ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, k: int, accumulate: bool = false) async throws {
    let (m0, m1, m) = try _stack(parts.map { $0.count }, rows, k, 1, x.count, y.count)
    if m == 0 { return }
    let wave = y.Device.WaveSize
    let per = _lanes(k / 4, wave)
    let group = 128 / wave * (wave / per)
    try await _gemvF32Kernel.Launch(parts[0], parts[parts.count > 1 ? 1 : 0], parts[parts.count > 2 ? 2 : 0], x, y, m0, m1, m, k, per, accumulate,
                                    over: (m + group - 1) / group * 128, workgroup: 128)
}

/// _stack checks stacked parts of rows[i] rows, each row `unit` elements
/// per k, against x and y, and is (m0, m1, m).
public func _stack(_ counts: [int], _ rows: [int], _ k: int, _ bytesPerRow: int, _ xCount: int, _ yCount: int) throws -> (int, int, int) {
    if counts.count == 0 || counts.count > 3 || counts.count != rows.count {
        fatalError("linalg.Gemv: 1 to 3 parts, each with its row count")
    }
    var m = 0
    for i in 0..<counts.count {
        if counts[i] < rows[i] * (bytesPerRow == 1 ? k : bytesPerRow) {
            fatalError("linalg.Gemv: part \(i) of \(counts[i]) elements holds fewer than \(rows[i]) rows")
        }
        m += rows[i]
    }
    if xCount < k || yCount < m {
        fatalError("linalg.Gemv: \(m) rows of \(k): x of \(xCount), y of \(yCount)")
    }
    return (rows[0], rows.count > 1 ? rows[1] : 0, m)
}

@inlinable public func _gemvBlockKernel<B: dtype.Block>(_ f: B, _ w0: gpu.Span<uint8>, _ w1: gpu.Span<uint8>, _ w2: gpu.Span<uint8>, _ x: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>,
                                                        _ m0: int, _ m1: int, _ m: int, _ k: int, _ per: int, _ accumulate: bool) kernel {
    // A row takes `per` lanes of a wave, which dot strided blocks of it,
    // decoding them in place, and add their parts by shuffles. On the CPU
    // device a wave is one work-item, which takes its whole row.
    let lanes = gpu.Wave.Size
    let sub = gpu.Wave.Lane % per
    let row = (gpu.GroupIndex.x * (128 / lanes) + gpu.LocalIndex.x / lanes) * (lanes / per) + gpu.Wave.Lane / per
    let blocks = k / B.Size()
    let bytes = B.Bytes(), size = B.Size()
    let p = _rowAt(w0, w1, w2, row, m0, m1, m, blocks &* bytes)
    let v = x._base
    var part: float32 = 0
    var j = sub
    // Gemv checked the parts and x, so each block and its floats are
    // there: B.Dot reads them unchecked.
    while j < blocks {
        part = part + B.Dot(p + j &* bytes, v + j &* size)
        j = j &+ per
    }
    let total = _reduce(part, per)
    if sub == 0 && row < m {
        y[row] = accumulate ? total + y[row] : total
    }
}

@inlinable public func _gemvQ4_0Kernel(_ w0: gpu.Span<uint8>, _ w1: gpu.Span<uint8>, _ w2: gpu.Span<uint8>, _ x: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>,
                                       _ m0: int, _ m1: int, _ m: int, _ k: int, _ accumulate: bool) kernel {
    // llama.cpp's q4_0 matrix-vector product (mul_vec_q_n_f32_impl), on
    // a GPU with waves of 32. A wave takes 4 rows, so each x it loads is
    // used four times; a lane takes half a block, so a wave covers 16
    // blocks side by side. Nibbles are masked in place, not shifted, with
    // x scaled to match (1, 1/256, 1/16, 1/4096), and a block's -8 offset
    // is one multiply of the sum of its x: d·(Σq·x - 8·Σx).
    let lane = gpu.Wave.Lane
    let r0 = (gpu.GroupIndex.x * 2 + gpu.LocalIndex.x / 32) * 4
    if r0 >= m { return }
    let nb = k / 32
    let rowBytes = nb &* 18
    let p0 = _rowAt(w0, w1, w2, r0, m0, m1, m, rowBytes), p1 = _rowAt(w0, w1, w2, r0 &+ 1, m0, m1, m, rowBytes)
    let p2 = _rowAt(w0, w1, w2, r0 &+ 2, m0, m1, m, rowBytes), p3 = _rowAt(w0, w1, w2, r0 &+ 3, m0, m1, m, rowBytes)
    let il = (lane % 2) &* 8
    var s0: float32 = 0, s1: float32 = 0, s2: float32 = 0, s3: float32 = 0
    var ib = lane / 2
    while ib < nb {
        let yb = x._base + (ib &* 32 &+ il)
        let off = ib &* 18
        let q0 = UnsafePointer<uint16>(UnsafeRawPointer(p0 + (off &+ 2 &+ il)))
        let q1 = UnsafePointer<uint16>(UnsafeRawPointer(p1 + (off &+ 2 &+ il)))
        let q2 = UnsafePointer<uint16>(UnsafeRawPointer(p2 + (off &+ 2 &+ il)))
        let q3 = UnsafePointer<uint16>(UnsafeRawPointer(p3 + (off &+ 2 &+ il)))
        var sumy: float32 = 0
        var a0: float32 = 0, a1: float32 = 0, a2: float32 = 0, a3: float32 = 0
        var i = 0
        while i < 8 {
            let ya = yb[i], yb1 = yb[i &+ 1], yc = yb[i &+ 16], yd = yb[i &+ 17]
            sumy = sumy + ya + yb1 + yc + yd
            let y0 = ya, y1 = yb1 / 256, y2 = yc / 16, y3 = yd / 4096
            let h = i / 2
            let t0 = q0[h], t1 = q1[h], t2 = q2[h], t3 = q3[h]
            a0 = a0 + y0 * float32(t0 & 0x000F) + y1 * float32(t0 & 0x0F00) + y2 * float32(t0 & 0x00F0) + y3 * float32(t0 & 0xF000)
            a1 = a1 + y0 * float32(t1 & 0x000F) + y1 * float32(t1 & 0x0F00) + y2 * float32(t1 & 0x00F0) + y3 * float32(t1 & 0xF000)
            a2 = a2 + y0 * float32(t2 & 0x000F) + y1 * float32(t2 & 0x0F00) + y2 * float32(t2 & 0x00F0) + y3 * float32(t2 & 0xF000)
            a3 = a3 + y0 * float32(t3 & 0x000F) + y1 * float32(t3 & 0x0F00) + y2 * float32(t3 & 0x00F0) + y3 * float32(t3 & 0xF000)
            i = i &+ 2
        }
        s0 = s0 + dtype._scaleAt(p0 + off) * (sumy * -8 + a0)
        s1 = s1 + dtype._scaleAt(p1 + off) * (sumy * -8 + a1)
        s2 = s2 + dtype._scaleAt(p2 + off) * (sumy * -8 + a2)
        s3 = s3 + dtype._scaleAt(p3 + off) * (sumy * -8 + a3)
        ib = ib &+ 16
    }
    let t0 = gpu.Wave.Sum(s0), t1 = gpu.Wave.Sum(s1), t2 = gpu.Wave.Sum(s2), t3 = gpu.Wave.Sum(s3)
    if lane == 0 {
        y[r0] = accumulate ? t0 + y[r0] : t0
        if r0 + 1 < m { y[r0 + 1] = accumulate ? t1 + y[r0 + 1] : t1 }
        if r0 + 2 < m { y[r0 + 2] = accumulate ? t2 + y[r0 + 2] : t2 }
        if r0 + 3 < m { y[r0 + 3] = accumulate ? t3 + y[r0 + 3] : t3 }
    }
}

@inlinable public func _gemvQ8_0Kernel(_ w0: gpu.Span<uint8>, _ w1: gpu.Span<uint8>, _ w2: gpu.Span<uint8>, _ x: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>,
                                       _ m0: int, _ m1: int, _ m: int, _ k: int, _ accumulate: bool) kernel {
    // llama.cpp's q8_0 matrix-vector product (kernel_mul_mv_q8_0_f32), on
    // a GPU with waves of 32. A group of four waves takes 2 rows; a lane
    // takes 8 quants, a quarter of a block, so a wave covers 8 blocks and
    // the group 32, side by side. Each x is loaded once for both rows. The
    // four waves' sums meet in shared storage.
    let parts = gpu.Shared<float32>(count: 8)
    let lane = gpu.Wave.Lane
    let wave = gpu.LocalIndex.x / 32
    let r0 = gpu.GroupIndex.x * 2
    let nb = k / 32
    let rowBytes = nb &* 34
    let p0 = _rowAt(w0, w1, w2, r0, m0, m1, m, rowBytes), p1 = _rowAt(w0, w1, w2, r0 &+ 1, m0, m1, m, rowBytes)
    let il = (lane % 4) &* 8
    var s0: float32 = 0, s1: float32 = 0
    var ib = wave &* 8 &+ lane / 4
    while ib < nb {
        let yb = x._base + (ib &* 32 &+ il)
        let off = ib &* 34
        let q0 = p0 + (off &+ 2 &+ il), q1 = p1 + (off &+ 2 &+ il)
        var a0: float32 = 0, a1: float32 = 0
        var i = 0
        while i < 8 {
            let v = yb[i]
            a0 = a0 + float32(int8(bitPattern: q0[i])) * v
            a1 = a1 + float32(int8(bitPattern: q1[i])) * v
            i = i &+ 1
        }
        s0 = s0 + a0 * dtype._scaleAt(p0 + off)
        s1 = s1 + a1 * dtype._scaleAt(p1 + off)
        ib = ib &+ 32
    }
    let t0 = gpu.Wave.Sum(s0), t1 = gpu.Wave.Sum(s1)
    if lane == 0 {
        parts[wave &* 2] = t0
        parts[wave &* 2 &+ 1] = t1
    }
    gpu.Barrier()
    if gpu.LocalIndex.x == 0 {
        let u0 = parts[0] + parts[2] + parts[4] + parts[6]
        let u1 = parts[1] + parts[3] + parts[5] + parts[7]
        if r0 < m { y[r0] = accumulate ? u0 + y[r0] : u0 }
        if r0 + 1 < m { y[r0 + 1] = accumulate ? u1 + y[r0 + 1] : u1 }
    }
}

@inlinable public func _gemvQ4_KKernel(_ w0: gpu.Span<uint8>, _ w1: gpu.Span<uint8>, _ w2: gpu.Span<uint8>, _ x: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>,
                                       _ m0: int, _ m1: int, _ m: int, _ k: int, _ accumulate: bool) kernel {
    // llama.cpp's q4_K matrix-vector product (kernel_mul_mv_q4_K_f32), on
    // a GPU with waves of 32. A wave takes 2 rows; 8 lanes take a block of
    // 256, so a wave covers 4 blocks side by side. A lane takes 32 of a
    // block's elements: 8 of each of the sub-blocks 2iq, 2iq+1, 2iq+4 and
    // 2iq+5 (iq = 0 or 1), at 8·ir (ir = 0 to 3). Scales and mins are
    // unpacked from their 6 bits two at a time with 16-bit masks; nibbles
    // are masked in place against x scaled to match, and the mins are one
    // multiply each of a sum of x.
    let lane = gpu.Wave.Lane
    let r0 = (gpu.GroupIndex.x &* 2 &+ (gpu.LocalIndex.x &>> 5)) &* 2
    if r0 >= m { return }
    let nb = k &>> 8
    let rowBytes = nb &* 144
    let pa = _rowAt(w0, w1, w2, r0, m0, m1, m, rowBytes), pb = _rowAt(w0, w1, w2, r0 &+ 1, m0, m1, m, rowBytes)
    // The lane's place, by its bits: lanes are never negative.
    let ix = lane &>> 3, it = lane & 7
    let iq = it &>> 2, ir = it & 3
    var sa: float32 = 0, sb: float32 = 0
    var ib = ix
    while ib < nb {
        let y4 = x._base + (ib &* 256 &+ 64 &* iq &+ 8 &* ir)
        var sy0: float32 = 0, sy1: float32 = 0, sy2: float32 = 0, sy3: float32 = 0
        // Accumulators: row a and row b, each four of the low chunk (1)
        // and four of the high (2).
        var a10: float32 = 0, a11: float32 = 0, a12: float32 = 0, a13: float32 = 0
        var a20: float32 = 0, a21: float32 = 0, a22: float32 = 0, a23: float32 = 0
        var b10: float32 = 0, b11: float32 = 0, b12: float32 = 0, b13: float32 = 0
        var b20: float32 = 0, b21: float32 = 0, b22: float32 = 0, b23: float32 = 0
        let ba = pa + ib &* 144, bb = pb + ib &* 144
        let qa1 = UnsafePointer<uint16>(UnsafeRawPointer(ba + (16 &+ 32 &* iq &+ 8 &* ir)))
        let qb1 = UnsafePointer<uint16>(UnsafeRawPointer(bb + (16 &+ 32 &* iq &+ 8 &* ir)))
        var i = 0
        while i < 4 {
            let e = 2 &* i
            let l0 = y4[e], l1 = y4[e &+ 1], l8 = y4[e &+ 32], l9 = y4[e &+ 33]
            let h0 = y4[e &+ 128], h1 = y4[e &+ 129], h8 = y4[e &+ 160], h9 = y4[e &+ 161]
            sy0 = sy0 + l0 + l1
            sy1 = sy1 + l8 + l9
            sy2 = sy2 + h0 + h1
            sy3 = sy3 + h8 + h9
            let ta1 = qa1[i], ta2 = qa1[i &+ 32], tb1 = qb1[i], tb2 = qb1[i &+ 32]
            a10 = a10 + l0 * float32(ta1 & 0x000F)
            a11 = a11 + l1 * float32(ta1 & 0x0F00)
            a12 = a12 + l8 * float32(ta1 & 0x00F0)
            a13 = a13 + l9 * float32(ta1 & 0xF000)
            a20 = a20 + h0 * float32(ta2 & 0x000F)
            a21 = a21 + h1 * float32(ta2 & 0x0F00)
            a22 = a22 + h8 * float32(ta2 & 0x00F0)
            a23 = a23 + h9 * float32(ta2 & 0xF000)
            b10 = b10 + l0 * float32(tb1 & 0x000F)
            b11 = b11 + l1 * float32(tb1 & 0x0F00)
            b12 = b12 + l8 * float32(tb1 & 0x00F0)
            b13 = b13 + l9 * float32(tb1 & 0xF000)
            b20 = b20 + h0 * float32(tb2 & 0x000F)
            b21 = b21 + h1 * float32(tb2 & 0x0F00)
            b22 = b22 + h8 * float32(tb2 & 0x00F0)
            b23 = b23 + h9 * float32(tb2 & 0xF000)
            i = i &+ 1
        }
        sa = sa + _q4KBlock(ba, iq, a10, a11, a12, a13, a20, a21, a22, a23, sy0, sy1, sy2, sy3)
        sb = sb + _q4KBlock(bb, iq, b10, b11, b12, b13, b20, b21, b22, b23, sy0, sy1, sy2, sy3)
        ib = ib &+ 4
    }
    let ta = gpu.Wave.Sum(sa), tb = gpu.Wave.Sum(sb)
    if lane == 0 {
        y[r0] = accumulate ? ta + y[r0] : ta
        if r0 + 1 < m { y[r0 + 1] = accumulate ? tb + y[r0 + 1] : tb }
    }
}

@inlinable public func _gemvQ6_KKernel(_ w0: gpu.Span<uint8>, _ w1: gpu.Span<uint8>, _ w2: gpu.Span<uint8>, _ x: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>,
                                       _ m0: int, _ m1: int, _ m: int, _ k: int, _ accumulate: bool) kernel {
    // llama.cpp's q6_K matrix-vector product (kernel_mul_mv_q6_K_f32), on
    // a GPU with waves of 32. A wave takes 2 rows; 16 lanes take a block of
    // 256, so a wave covers 2 blocks side by side. A lane takes 4
    // consecutive elements of each quarter of one half of the block: its
    // low 4 bits from ql, its high 2 from qh, one scale a quarter.
    let lane = gpu.Wave.Lane
    let r0 = (gpu.GroupIndex.x &* 2 &+ (gpu.LocalIndex.x &>> 5)) &* 2
    if r0 >= m { return }
    let nb = k &>> 8
    let rowBytes = nb &* 210
    let pa = _rowAt(w0, w1, w2, r0, m0, m1, m, rowBytes), pb = _rowAt(w0, w1, w2, r0 &+ 1, m0, m1, m, rowBytes)
    // The lane's place, by its bits: lanes are never negative.
    let tid = lane &>> 1, ix = lane & 1
    let ip = tid &>> 3, il = tid & 7
    let l0 = 4 &* il
    let isc = 8 &* ip &+ (l0 &>> 4)
    var sa: float32 = 0, sb: float32 = 0
    var ib = ix
    while ib < nb {
        let yb = x._base + (ib &* 256 &+ 128 &* ip &+ l0)
        let ba = pa + ib &* 210, bb = pb + ib &* 210
        var a0: float32 = 0, a1: float32 = 0, a2: float32 = 0, a3: float32 = 0
        var b0: float32 = 0, b1: float32 = 0, b2: float32 = 0, b3: float32 = 0
        let qla = ba + (64 &* ip &+ l0), qha = ba + (128 &+ 32 &* ip &+ l0)
        let qlb = bb + (64 &* ip &+ l0), qhb = bb + (128 &+ 32 &* ip &+ l0)
        var l = 0
        while l < 4 {
            let y0 = yb[l], y1 = yb[l &+ 32], y2 = yb[l &+ 64], y3 = yb[l &+ 96]
            let q1 = qla[l], q2 = qla[l &+ 32], h = qha[l]
            a0 = a0 + y0 * float32((int32(q1 & 0xF) | int32(h & 0x03) << 4) &- 32)
            a1 = a1 + y1 * float32((int32(q2 & 0xF) | int32(h & 0x0C) << 2) &- 32)
            a2 = a2 + y2 * float32((int32(q1 >> 4) | int32(h & 0x30)) &- 32)
            a3 = a3 + y3 * float32((int32(q2 >> 4) | int32(h & 0xC0) >> 2) &- 32)
            let r1 = qlb[l], r2 = qlb[l &+ 32], g = qhb[l]
            b0 = b0 + y0 * float32((int32(r1 & 0xF) | int32(g & 0x03) << 4) &- 32)
            b1 = b1 + y1 * float32((int32(r2 & 0xF) | int32(g & 0x0C) << 2) &- 32)
            b2 = b2 + y2 * float32((int32(r1 >> 4) | int32(g & 0x30)) &- 32)
            b3 = b3 + y3 * float32((int32(r2 >> 4) | int32(g & 0xC0) >> 2) &- 32)
            l = l &+ 1
        }
        let sca = ba + (192 &+ isc), scb = bb + (192 &+ isc)
        sa = sa + dtype._scaleAt(ba + 208) * (a0 * float32(int8(bitPattern: sca[0])) + a1 * float32(int8(bitPattern: sca[2]))
                                           + a2 * float32(int8(bitPattern: sca[4])) + a3 * float32(int8(bitPattern: sca[6])))
        sb = sb + dtype._scaleAt(bb + 208) * (b0 * float32(int8(bitPattern: scb[0])) + b1 * float32(int8(bitPattern: scb[2]))
                                           + b2 * float32(int8(bitPattern: scb[4])) + b3 * float32(int8(bitPattern: scb[6])))
        ib = ib &+ 2
    }
    let ta = gpu.Wave.Sum(sa), tb = gpu.Wave.Sum(sb)
    if lane == 0 {
        y[r0] = accumulate ? ta + y[r0] : ta
        if r0 + 1 < m { y[r0 + 1] = accumulate ? tb + y[r0 + 1] : tb }
    }
}

/// _q4KBlock is a lane's part of a q4_K block's dot product: its sums,
/// scaled by the four sub-blocks' scales and mins, unpacked as llama.cpp's
/// kernel does.
@inlinable public func _q4KBlock(_ block: UnsafeMutablePointer<uint8>, _ iq: int,
                                 _ a10: float32, _ a11: float32, _ a12: float32, _ a13: float32,
                                 _ a20: float32, _ a21: float32, _ a22: float32, _ a23: float32,
                                 _ sy0: float32, _ sy1: float32, _ sy2: float32, _ sy3: float32) -> float32 {
    let sc = UnsafePointer<uint16>(UnsafeRawPointer(block + 4)) + iq
    let s0 = sc[0], s2 = sc[2], s4 = sc[4]
    let v0 = s0 & 0x3f3f
    let v1 = s2 & 0x3f3f
    let v2 = (s4 & 0x0f0f) | ((s0 & 0xc0c0) >> 2)
    let v3 = ((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)
    let c0 = float32(v0 & 0xFF), c1 = float32(v0 >> 8), m0 = float32(v1 & 0xFF), m1 = float32(v1 >> 8)
    let c4 = float32(v2 & 0xFF), c5 = float32(v2 >> 8), m4 = float32(v3 & 0xFF), m5 = float32(v3 >> 8)
    let d = dtype._scaleAt(block), dmin = dtype._scaleAt(block + 2)
    return d * ((a10 + a11 / 256) * c0 + (a12 + a13 / 256) * c1 / 16 + (a20 + a21 / 256) * c4 + (a22 + a23 / 256) * c5 / 16)
         - dmin * (sy0 * m0 + sy1 * m1 + sy2 * m4 + sy3 * m5)
}

/// Gemv is y = W · x with W block-quantized: m rows of k elements, each
/// row k / format.Size() blocks of the format, one after another -- a GGUF
/// tensor's bytes as they are. x and y are float32; the sum is taken in
/// float32. With accumulate, y = W · x + y. What decoding a quantized
/// model is made of.
@inlinable public func Gemv<B: dtype.Block>(_ w: gpu.Buffer<uint8>, _ format: B, _ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, m: int, k: int, accumulate: bool = false) async throws {
    try await Gemv([w], rows: [m], format, x, into: y, k: k, accumulate: accumulate)
}

/// Gemv is y = [W0; W1; W2] · x for block-quantized weights of one format
/// stacked by rows: parts holds one to three weights of rows[i] rows of k.
/// On a GPU with waves of 32, q4_0 and q8_0 take kernels tuned to them
/// after llama.cpp's; every format takes the kernel over its Block.
@inlinable public func Gemv<B: dtype.Block>(_ parts: [gpu.Buffer<uint8>], rows: [int], _ format: B, _ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, k: int, accumulate: bool = false) async throws {
    if k % B.Size() != 0 {
        fatalError("linalg.Gemv: rows of \(k) are not whole blocks of \(B.Size())")
    }
    let (m0, m1, m) = try _stack(parts.map { $0.count }, rows, k, k / B.Size() * B.Bytes(), x.count, y.count)
    if m == 0 { return }
    let w0 = parts[0], w1 = parts[parts.count > 1 ? 1 : 0], w2 = parts[parts.count > 2 ? 2 : 0]
    let wave = y.Device.WaveSize
    if wave == 32 && B.GGMLType() == 2 {
        // q4_0 on a GPU: 8 rows a group of two waves.
        try await _gemvQ4_0Kernel.Launch(w0, w1, w2, x, y, m0, m1, m, k, accumulate, over: (m + 7) / 8 * 64, workgroup: 64)
        return
    }
    if wave == 32 && B.GGMLType() == 12 {
        // q4_K on a GPU: 4 rows a group of two waves.
        try await _gemvQ4_KKernel.Launch(w0, w1, w2, x, y, m0, m1, m, k, accumulate, over: (m + 3) / 4 * 64, workgroup: 64)
        return
    }
    if wave == 32 && B.GGMLType() == 14 {
        // q6_K on a GPU: 4 rows a group of two waves.
        try await _gemvQ6_KKernel.Launch(w0, w1, w2, x, y, m0, m1, m, k, accumulate, over: (m + 3) / 4 * 64, workgroup: 64)
        return
    }
    if wave == 32 && B.GGMLType() == 8 && k >= 1024 {
        // q8_0 on a GPU: 2 rows a group of four waves, which a row of fewer
        // than 32 blocks cannot keep busy -- a model's small head is
        // better served by the kernel below.
        try await _gemvQ8_0Kernel.Launch(w0, w1, w2, x, y, m0, m1, m, k, accumulate, over: (m + 1) / 2 * 128, workgroup: 128)
        return
    }
    let per = _lanes(k / B.Size(), wave)
    let group = 128 / wave * (wave / per)
    try await _gemvBlockKernel.Launch(format, w0, w1, w2, x, y, m0, m1, m, k, per, accumulate, over: (m + group - 1) / group * 128, workgroup: 128)
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
