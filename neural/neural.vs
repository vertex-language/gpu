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

/// RMSNorm writes each row of x scaled to unit root-mean-square and then
/// by weight (cols elements): y = x / sqrt(mean(x²) + eps) · weight.
public func RMSNorm(_ x: gpu.Buffer<float32>, weight: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, rows: int, cols: int, eps: float32 = 1e-6) async throws {
    if rows == 0 || cols == 0 { return }
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
