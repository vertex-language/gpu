// Fused building blocks of neural networks, float32 for now: row-wise
// normalizations and softmax, activations, rotary position embeddings,
// cross entropy. They are kernels over row-major buffers, not layers:
// layers, parameters and autodiff are nn and tensor, above.
//
// A row-wise op runs one workgroup per row, and every sum in it is taken
// in a fixed order (gpu/parallel's group functions), so a result is the
// same on every device that keeps subnormals.
import (
    "gpu"
    "gpu/parallel"
    "math"
)

/// RowGroup is how many work-items share one row of a row-wise op.
public let RowGroup = 256

// ---- softmax ----

func _softmaxRows(_ x: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>, _ cols: int, _ log: bool) kernel {
    let row = gpu.GroupIndex.x
    let me = parallel.GroupRank()
    let step = parallel.GroupCount()
    let base = row * cols
    var m = -float32.infinity
    var j = me
    while j < cols {
        m = math.Max(m, x[base + j])
        j += step
    }
    let most = parallel.GroupMax(m)
    var s: float32 = 0
    j = me
    while j < cols {
        s = s + math.Exp(x[base + j] - most)
        j += step
    }
    let total = parallel.GroupSum(s)
    let logTotal = math.Log(total)
    j = me
    while j < cols {
        let d = x[base + j] - most
        y[base + j] = log ? d - logTotal : math.Exp(d) / total
        j += step
    }
}

/// Softmax writes each row of x (rows x cols, row-major) as its softmax
/// into y: e^x / Σe^x, computed from x minus the row's greatest element so
/// that nothing overflows.
public func Softmax(_ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, rows: int, cols: int) async throws {
    if rows == 0 || cols == 0 { return }
    try await _softmaxRows.Launch(x, y, cols, false, over: rows * 256, workgroup: 256)
}

/// LogSoftmax writes each row's log-softmax: x - max - log Σe^(x - max).
public func LogSoftmax(_ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, rows: int, cols: int) async throws {
    if rows == 0 || cols == 0 { return }
    try await _softmaxRows.Launch(x, y, cols, true, over: rows * 256, workgroup: 256)
}

func _logSumExpRows(_ x: gpu.Span<float32>, _ out: gpu.MutableSpan<float32>, _ cols: int) kernel {
    let row = gpu.GroupIndex.x
    let me = parallel.GroupRank()
    let step = parallel.GroupCount()
    let base = row * cols
    var m = -float32.infinity
    var j = me
    while j < cols {
        m = math.Max(m, x[base + j])
        j += step
    }
    let most = parallel.GroupMax(m)
    var s: float32 = 0
    j = me
    while j < cols {
        s = s + math.Exp(x[base + j] - most)
        j += step
    }
    let total = parallel.GroupSum(s)
    if me == 0 {
        out[row] = most + math.Log(total)
    }
}

/// LogSumExp is log Σ e^x of each row, one value a row.
public func LogSumExp(_ x: gpu.Buffer<float32>, rows: int, cols: int) async throws -> gpu.Buffer<float32> {
    let out = try await x.Device.CreateBuffer(of: float32.self, count: rows)
    if rows > 0 && cols > 0 {
        try await _logSumExpRows.Launch(x, out, cols, over: rows * 256, workgroup: 256)
    }
    return out
}

// ---- argmax ----

/// _argmaxLanes is the greater of each lane's (v, at) and its partner's
/// across the wave, the lower index of equals.
@inlinable func _argmaxLanes(_ v: float32, _ at: int) -> (float32, int) {
    var bv = v, bi = at
    var m = gpu.Wave.Size >> 1
    while m > 0 {
        let ov = gpu.Wave.ShuffleXor(bv, mask: m)
        let oi = int(gpu.Wave.ShuffleXor(int32(bi), mask: m))
        if ov > bv || (ov == bv && oi < bi) {
            bv = ov
            bi = oi
        }
        m = m >> 1
    }
    return (bv, bi)
}

func _argmaxRows(_ x: gpu.Span<float32>, _ out: gpu.MutableSpan<int32>, _ cols: int) kernel {
    // A group a row: 32 waves of 32 on a GPU, one work-item on the CPU.
    // Each work-item keeps the first greatest of its strided share, the
    // lanes of a wave meet by shuffles, and the waves in shared storage.
    let bestV = gpu.Shared<float32>(count: 32)
    let bestI = gpu.Shared<int32>(count: 32)
    let row = gpu.GroupIndex.x
    let t = gpu.LocalIndex.x
    let n = gpu.GroupSize.x
    let p = x._base + row &* cols
    var v = -float32.infinity
    var at = cols
    var j = t
    while j < cols {
        let e = p[j]
        if e > v {
            v = e
            at = j
        }
        j = j &+ n
    }
    var best = _argmaxLanes(v, at)
    if n > 1 {
        if t & 31 == 0 {
            bestV[t >> 5] = best.0
            bestI[t >> 5] = int32(best.1)
        }
        gpu.Barrier()
        if t < 32 {
            best = _argmaxLanes(bestV[t], int(bestI[t]))
        }
    }
    if t == 0 {
        // Nothing greater than -inf: the first element, as on the host.
        out[row] = int32(best.1 == cols ? 0 : best.1)
    }
}

/// Argmax writes each row's index of its greatest element into out, the
/// first of equals; a row with nothing greater than -inf (or only NaNs)
/// gets 0. Greedy decoding's next token without reading the logits back.
public func Argmax(_ x: gpu.Buffer<float32>, into out: gpu.Buffer<int32>, rows: int, cols: int) async throws {
    if rows == 0 || cols == 0 { return }
    if x.count < rows * cols || out.count < rows || cols >= 1 << 31 {
        throw gpu.DeviceError.badCount
    }
    let group = out.Device.WaveSize == 32 ? 1024 : 1
    try await _argmaxRows.Launch(x, out, cols, over: rows * group, workgroup: group)
}

// ---- normalization ----

func _rmsNormRows(_ x: gpu.Span<float32>, _ w: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>, _ cols: int, _ eps: float32) kernel {
    let row = gpu.GroupIndex.x
    let me = parallel.GroupRank()
    let step = parallel.GroupCount()
    let base = row * cols
    var s: float32 = 0
    var j = me
    while j < cols {
        let v = x[base + j]
        s = s + v * v
        j += step
    }
    let scale = math.Rsqrt(parallel.GroupSum(s) / float32(cols) + eps)
    j = me
    while j < cols {
        y[base + j] = x[base + j] * scale * w[j]
        j += step
    }
}

func _rmsNormWaves(_ x: gpu.Span<float32>, _ w: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>, _ cols: int, _ eps: float32, _ waves: int) kernel {
    // A row takes `waves` waves of 32 lanes. Each lane squares a strided
    // share of it, read unchecked (RMSNorm checked the counts), and a wave
    // adds its lanes by shuffles; only with more than one wave do their
    // sums meet in shared storage, past one barrier. A decoded token's
    // row, a few hundred wide, is one wave and no barrier at all.
    let parts = gpu.Shared<float32>(count: 32)
    let row = gpu.GroupIndex.x
    let me = gpu.LocalIndex.x
    let step = waves &* 32
    let p = x._base + row &* cols
    var s: float32 = 0
    var j = me
    while j < cols {
        let v = p[j]
        s = s + v * v
        j = j &+ step
    }
    s = gpu.Wave.Sum(s)
    if waves > 1 {
        if gpu.Wave.Lane == 0 { parts[me &>> 5] = s }
        gpu.Barrier()
        s = 0
        var i = 0
        while i < waves {
            s = s + parts[i]
            i = i &+ 1
        }
    }
    let scale = math.Rsqrt(s / float32(cols) + eps)
    let q = y._base + row &* cols
    let g = w._base
    j = me
    while j < cols {
        q[j] = p[j] * scale * g[j]
        j = j &+ step
    }
}

/// RMSNorm writes each row of x scaled to unit root-mean-square and then
/// by weight (cols elements): y = x / sqrt(mean(x²) + eps) · weight.
public func RMSNorm(_ x: gpu.Buffer<float32>, weight: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, rows: int, cols: int, eps: float32 = 1e-6) async throws {
    if rows == 0 || cols == 0 { return }
    if y.Device.WaveSize == 32 {
        if x.count < rows * cols || y.count < rows * cols || weight.count < cols {
            throw gpu.DeviceError.badCount
        }
        // A wave for every 128 elements of a row, up to a group of 1024:
        // a lane takes about four, since a row this short is a chain of
        // dependent loads, and the longer each lane's the slower it runs.
        let waves = min(32, max(1, (cols + 127) / 128))
        try await _rmsNormWaves.Launch(x, weight, y, cols, eps, waves, over: rows * waves * 32, workgroup: waves * 32)
        return
    }
    try await _rmsNormRows.Launch(x, weight, y, cols, eps, over: rows * 256, workgroup: 256)
}

func _layerNormRows(_ x: gpu.Span<float32>, _ w: gpu.Span<float32>, _ b: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>, _ cols: int, _ eps: float32) kernel {
    let row = gpu.GroupIndex.x
    let me = parallel.GroupRank()
    let step = parallel.GroupCount()
    let base = row * cols
    var s: float32 = 0
    var j = me
    while j < cols {
        s = s + x[base + j]
        j += step
    }
    let mean = parallel.GroupSum(s) / float32(cols)
    // The variance about the mean, in a second pass: no cancellation.
    var q: float32 = 0
    j = me
    while j < cols {
        let d = x[base + j] - mean
        q = q + d * d
        j += step
    }
    let scale = math.Rsqrt(parallel.GroupSum(q) / float32(cols) + eps)
    j = me
    while j < cols {
        y[base + j] = (x[base + j] - mean) * scale * w[j] + b[j]
        j += step
    }
}

/// LayerNorm writes each row of x normalized to zero mean and unit
/// variance, then scaled and shifted: y = (x - mean) / sqrt(var + eps) ·
/// weight + bias.
public func LayerNorm(_ x: gpu.Buffer<float32>, weight: gpu.Buffer<float32>, bias: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, rows: int, cols: int, eps: float32 = 1e-5) async throws {
    if rows == 0 || cols == 0 { return }
    try await _layerNormRows.Launch(x, weight, bias, y, cols, eps, over: rows * 256, workgroup: 256)
}

// ---- activations ----

/// Activation is an elementwise function Activate applies.
public enum Activation {
    case ReLU
    case GELU
    case GELUTanh
    case SiLU
    case Sigmoid
    case Tanh
    case Exp
    case Sin

    public var _code: int32 {
        switch self {
        case .ReLU: return 0
        case .GELU: return 1
        case .GELUTanh: return 2
        case .SiLU: return 3
        case .Sigmoid: return 4
        case .Tanh: return 5
        case .Exp: return 6
        case .Sin: return 7
        }
    }
}

/// Apply is activation a of x, as a device function any kernel can call.
@inlinable public func Apply(_ a: int32, _ x: float32) -> float32 {
    if a == 0 { return x > 0 ? x : 0 }
    // 1 + erf(x/√2) is erfc(-x/√2), and 1 + tanh(z) is 2·sigmoid(2z):
    // the same functions, without the cancellation both sums have for
    // large negative x.
    if a == 1 { return 0.5 * x * math.Erfc(-x * 0.70710678) }
    if a == 2 { return x * math.Sigmoid(1.5957691216 * (x + 0.044715 * x * x * x)) }
    if a == 3 { return x * math.Sigmoid(x) }
    if a == 4 { return math.Sigmoid(x) }
    if a == 6 { return math.Exp(x) }
    if a == 7 { return math.Sin(x) }
    return math.Tanh(x)
}

func _activate(_ x: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>, _ a: int32) kernel {
    let i = gpu.Index.x
    if i < x.count {
        y[i] = Apply(a, x[i])
    }
}

func _gated(_ x: gpu.Span<float32>, _ gate: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>, _ a: int32) kernel {
    let i = gpu.Index.x
    if i < x.count {
        y[i] = x[i] * Apply(a, gate[i])
    }
}

/// Activate writes a(x) into y, element by element: .ReLU, .GELU (with
/// erf), .GELUTanh (the tanh approximation), .SiLU, .Sigmoid, .Tanh, and
/// .Exp and .Sin (a vocoder's magnitude and phase).
public func Activate(_ x: gpu.Buffer<float32>, _ a: Activation, into y: gpu.Buffer<float32>) async throws {
    if x.count == 0 { return }
    try await _activate.Launch(x, y, a._code, over: x.count)
}

/// Gated writes x · a(gate) into y: SwiGLU is Gated(up, gate, .SiLU),
/// GeGLU is Gated(up, gate, .GELU).
public func Gated(_ x: gpu.Buffer<float32>, gate: gpu.Buffer<float32>, _ a: Activation, into y: gpu.Buffer<float32>) async throws {
    if x.count == 0 { return }
    try await _gated.Launch(x, gate, y, a._code, over: x.count)
}

// ---- rotary position embeddings ----

/// RopeLayout is which elements of a head RoPE turns together. It is a
/// property of the weights: the rows of Q and K a checkpoint holds are laid
/// out for one or the other.
public enum RopeLayout: Equatable {
    /// (x[2p], x[2p+1]): GPT-J's, and llama.cpp's for the llama GGUFs it
    /// converts (it permutes Q and K for this).
    case interleaved
    /// (x[p], x[p + dim/2]): rotate_half, as Hugging Face's Llama, Qwen,
    /// Gemma and Phi have it (NeoX's; llama.cpp's ROPE_TYPE_NEOX).
    case halves

    var code: int { return self == .halves ? 1 : 0 }
}

// ropeTurn rotates pair p of the head starting at head by angle position ·
// base^(-2p/dim), in the layout halves says (0 interleaved, 1 halves).
func ropeTurn(_ x: gpu.MutableSpan<float32>, _ head: int, _ p: int, _ dim: int, _ position: int, _ base: float32, _ halves: int) {
    let freq = math.Exp2(-float32(2 * p) / float32(dim) * math.Log2(base))
    let angle = float32(position) * freq
    let c = math.Cos(angle)
    let s = math.Sin(angle)
    var ia = head + 2 * p
    var ib = ia + 1
    if halves != 0 {
        ia = head + p
        ib = ia + dim / 2
    }
    let a = x[ia]
    let b = x[ib]
    x[ia] = a * c - b * s
    x[ib] = a * s + b * c
}

func _rope(_ x: gpu.MutableSpan<float32>, _ positions: gpu.Span<int32>, _ heads: int, _ dim: int, _ base: float32, _ halves: int) kernel {
    // One work-item a pair: token t, head h, pair p.
    let i = gpu.Index.x
    let pairs = dim / 2
    let total = positions.count * heads * pairs
    if i >= total { return }
    let p = i % pairs
    let t = i / (heads * pairs)
    ropeTurn(x, (i / pairs) * dim, p, dim, int(positions[t]), base, halves)
}

func _ropeAt(_ x: gpu.MutableSpan<float32>, _ position: int, _ heads: int, _ dim: int, _ base: float32, _ halves: int) kernel {
    // _rope for one token, its position a scalar: what decoding a token
    // needs, with no buffer for the host to write.
    let i = gpu.Index.x
    let pairs = dim / 2
    if i >= heads * pairs { return }
    ropeTurn(x, (i / pairs) * dim, i % pairs, dim, position, base, halves)
}

/// RoPE rotates one token's heads in place (heads x dim), all at position:
/// RoPE(x, positions:) for a single token, the position passed as a value.
public func RoPE(_ x: gpu.Buffer<float32>, position: int, heads: int, dim: int, base: float32 = 10000,
                 layout: RopeLayout = .interleaved) async throws {
    let total = heads * (dim / 2)
    if total == 0 { return }
    try await _ropeAt.Launch(x, position, heads, dim, base, layout.code, over: total)
}

/// RoPE rotates x in place (tokens x heads x dim, row-major, one position
/// per token) by rotary position embeddings: each pair p of a head turned
/// by position · base^(-2p/dim), the pairs as layout says.
public func RoPE(_ x: gpu.Buffer<float32>, positions: gpu.Buffer<int32>, heads: int, dim: int, base: float32 = 10000,
                 layout: RopeLayout = .interleaved) async throws {
    let total = positions.count * heads * (dim / 2)
    if total == 0 { return }
    try await _rope.Launch(x, positions, heads, dim, base, layout.code, over: total)
}

/// RopeTable is the cosines and sines RoPE turns by: for each position
/// below Positions and each pair p of a head of Dim, of the angle
/// position · base^(-2p/Dim), computed once on the host in float64. With
/// it a decoded token's turn is two loads a pair, where computing it is an
/// exponential, a logarithm, a cosine and a sine. A model's layers share
/// one.
public final class RopeTable {
    /// Cos and Sin are [Positions, Dim/2], row-major.
    public let Cos: gpu.Buffer<float32>
    public let Sin: gpu.Buffer<float32>
    public let Dim: int
    public let Positions: int
    public let Base: float32

    init(cos: gpu.Buffer<float32>, sin: gpu.Buffer<float32>, dim: int, positions: int, base: float32) {
        self.Cos = cos
        self.Sin = sin
        self.Dim = dim
        self.Positions = positions
        self.Base = base
    }

    /// Make is the table for heads of dim (even) and positions below
    /// positions, on device d. factors, when given, divide each pair's
    /// frequency: Llama 3's rope_freqs, llama.cpp's freq_factors.
    public static func Make(on d: gpu.Device, dim: int, positions: int, base: float32, factors: [float32]? = nil) async throws -> RopeTable {
        let pairs = dim / 2
        var c = [float32](repeating: 0, count: max(1, positions * pairs))
        var s = c
        for p in 0..<pairs {
            var freq = math.Pow(float64(base), -2 * float64(p) / float64(dim))
            if let f = factors, p < f.count {
                freq = freq / float64(f[p])
            }
            for t in 0..<positions {
                let a = float64(t) * freq
                c[t * pairs + p] = float32(math.Cos(a))
                s[t * pairs + p] = float32(math.Sin(a))
            }
        }
        return RopeTable(cos: try await d.Upload(c), sin: try await d.Upload(s), dim: dim, positions: positions, base: base)
    }

    /// Fits is whether the table turns heads of dim at every position below
    /// positions, by base.
    public func Fits(dim: int, positions: int, base: float32) -> bool {
        return Dim == dim && Positions >= positions && Base == base
    }
}

// ---- losses ----

func _crossEntropyRows(_ x: gpu.Span<float32>, _ targets: gpu.Span<int32>, _ loss: gpu.MutableSpan<float32>, _ cols: int) kernel {
    let row = gpu.GroupIndex.x
    let me = parallel.GroupRank()
    let step = parallel.GroupCount()
    let base = row * cols
    var m = -float32.infinity
    var j = me
    while j < cols {
        m = math.Max(m, x[base + j])
        j += step
    }
    let most = parallel.GroupMax(m)
    var s: float32 = 0
    j = me
    while j < cols {
        s = s + math.Exp(x[base + j] - most)
        j += step
    }
    let total = parallel.GroupSum(s)
    if me == 0 {
        loss[row] = most + math.Log(total) - x[base + int(targets[row])]
    }
}

/// CrossEntropy is each row's loss against its target class: log Σe^x -
/// x[target], the softmax and the log fused. logits is rows x cols.
public func CrossEntropy(_ logits: gpu.Buffer<float32>, targets: gpu.Buffer<int32>, rows: int, cols: int) async throws -> gpu.Buffer<float32> {
    let loss = try await logits.Device.CreateBuffer(of: float32.self, count: rows)
    if rows > 0 && cols > 0 {
        try await _crossEntropyRows.Launch(logits, targets, loss, cols, over: rows * 256, workgroup: 256)
    }
    return loss
}
