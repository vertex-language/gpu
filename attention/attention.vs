// Fused attention: O = softmax(scale · Q·Kᵀ + mask) · V, without ever
// writing the scores matrix. This is FlashAttention's online softmax:
// keys are taken a block at a time, and the running maximum, the running
// sum and the output accumulator are rescaled as each block raises the
// maximum. Memory is O(d) per query rather than O(sequence).
//
// One workgroup of 128 computes one query row: each work-item scores one
// key of the block (a dot product over d) and owns one dimension of the
// output. Every sum is taken in a fixed order, so the result is the same
// on every device that keeps subnormals. Head dimensions up to 128.
//
// Layouts are row-major: Q and O are [batch, heads, queries, d], K and V
// [batch, kvHeads, capacity, d], the first keys of each head's capacity
// rows used -- a KV cache as it fills. heads a multiple of kvHeads is grouped-query
// attention; kvHeads 1 is multi-query.
import "gpu"
import "gpu/parallel"
import "math"

/// MaxHeadDim is the largest head dimension Forward takes.
public let MaxHeadDim = 128

/// Mask is which keys a query may attend to.
public enum Mask {
    /// Every key.
    case None
    /// Keys at or before the query's position. With a KV cache, the
    /// queries are the last of the keys: query i is at position
    /// keys - queries + i.
    case Causal
    /// Causal, and at most window keys back, the query's own included.
    case SlidingWindow(int)

    public var _causal: bool {
        switch self {
        case .None: return false
        default: return true
        }
    }

    public var _window: int {
        switch self {
        case .SlidingWindow(let w): return w
        default: return 0
        }
    }
}

func _attend(_ q: gpu.Span<float32>, _ k: gpu.Span<float32>, _ v: gpu.Span<float32>, _ o: gpu.MutableSpan<float32>,
             _ heads: int, _ kvHeads: int, _ queries: int, _ keys: int, _ capacity: int, _ d: int,
             _ scale: float32, _ causal: bool, _ window: int) kernel {
    let qs = gpu.Shared<float32>(count: 128)
    let ps = gpu.Shared<float32>(count: 128)
    let t = parallel.GroupRank()
    let row = gpu.GroupIndex.x            // (batch · heads + head) · queries + i
    let i = row % queries
    let bh = row / queries
    let head = bh % heads
    let batch = bh / heads
    let kvHead = head / (heads / kvHeads)
    let qBase = row * d
    let kvBase = (batch * kvHeads + kvHead) * capacity * d
    let position = keys - queries + i     // the query's place among the keys
    if t < d {
        qs[t] = q[qBase + t]
    }
    gpu.Barrier()
    var m = -float32.infinity             // running maximum score
    var l: float32 = 0                    // running Σ e^(score - m)
    var acc: float32 = 0                  // this work-item's output dimension
    var block = 0
    while block < keys {
        // Score key block + t.
        let j = block + t
        var s = -float32.infinity
        var visible = j < keys
        if causal && j > position {
            visible = false
        }
        if window > 0 && j <= position - window {
            visible = false
        }
        if visible {
            var dot: float32 = 0
            var c = 0
            while c < d {
                dot = dot + qs[c] * k[kvBase + j * d + c]
                c += 1
            }
            s = dot * scale
        }
        let blockMax = parallel.GroupMax(s)
        let mNew = math.Max(m, blockMax)
        var p: float32 = 0
        if visible {
            p = math.Exp(s - mNew)
        }
        ps[t] = p
        let blockSum = parallel.GroupSum(p)
        // Rescale what came before to the new maximum, then add this block.
        let correction = m == -float32.infinity ? float32(0) : math.Exp(m - mNew)
        l = l * correction + blockSum
        if t < d {
            var a = acc * correction
            var jj = 0
            while jj < 128 && block + jj < keys {
                a = a + ps[jj] * v[kvBase + (block + jj) * d + t]
                jj += 1
            }
            acc = a
        }
        m = mNew
        gpu.Barrier()
        block += 128
    }
    if t < d {
        o[qBase + t] = l > 0 ? acc / l : 0
    }
}

/// DecodeKeys is the most keys the one-query kernel holds scores for.
public let DecodeKeys = 4096

func _attendOne(_ q: gpu.Span<float32>, _ k: gpu.Span<float32>, _ v: gpu.Span<float32>, _ o: gpu.MutableSpan<float32>,
                _ heads: int, _ kvHeads: int, _ keys: int, _ capacity: int, _ d: int, _ scale: float32, _ first: int) kernel {
    // One query a head -- decoding a token. A group of 128 scores every
    // visible key at once into shared storage, reduces twice (the most,
    // then the sum of exponentials), and each work-item then gathers one
    // dimension of the output over the keys: two group reductions in all,
    // where the blocked kernel pays several a block of 128 keys. Keys
    // before `first` are outside a sliding window.
    let scores = gpu.Shared<float32>(count: 4096)
    let qs = gpu.Shared<float32>(count: 128)
    let t = parallel.GroupRank()
    let n = parallel.GroupCount()
    let head = gpu.GroupIndex.x
    let kvHead = head / (heads / kvHeads)
    let qBase = head * d
    let kvBase = kvHead * capacity * d
    if t < d {
        qs[t] = q[qBase + t]
    }
    gpu.Barrier()
    var most = -float32.infinity
    var j = first + t
    while j < keys {
        let row = k._base + (kvBase &+ j &* d)
        var dot: float32 = 0
        var c = 0
        while c < d {
            dot = dot + qs[c] * row[c]
            c = c &+ 1
        }
        let s = dot * scale
        scores[j] = s
        most = math.Max(most, s)
        j += n
    }
    let m = parallel.GroupMax(most)
    var sum: float32 = 0
    j = first + t
    while j < keys {
        let e = math.Exp(scores[j] - m)
        scores[j] = e
        sum = sum + e
        j += n
    }
    let total = parallel.GroupSum(sum)
    if t < d {
        var acc: float32 = 0
        var jj = first
        let col = v._base + (kvBase &+ t)
        while jj < keys {
            acc = acc + scores[jj] * col[jj &* d]
            jj = jj &+ 1
        }
        o[qBase + t] = total > 0 ? acc / total : 0
    }
}

/// Forward is softmax(scale · Q·Kᵀ + mask) · V, written into o. scale is
/// usually 1/√d. A query that may attend to no key gets zeros.
public func Forward(q: gpu.Buffer<float32>, k: gpu.Buffer<float32>, v: gpu.Buffer<float32>, into o: gpu.Buffer<float32>,
                    _ shape: Shape, mask: Mask = .None) async throws {
    if shape.batch == 0 || shape.heads == 0 || shape.queries == 0 {
        return
    }
    let scale = shape.scale != 0 ? shape.scale : 1 / math.Sqrt(float32(shape.headDim))
    if shape.batch == 1 && shape.queries == 1 && shape.keys <= DecodeKeys {
        // Decoding: the one query is the last key, so a causal mask hides
        // nothing, and a window of w keys starts w - 1 before it.
        let w = mask._window
        let first = w > 0 && shape.keys > w ? shape.keys - w : 0
        try await _attendOne.Launch(q, k, v, o, shape.heads, shape.kvHeads, shape.keys, shape.keyCapacity, shape.headDim,
                                    scale, first, over: shape.heads * 128, workgroup: 128)
        return
    }
    try await _attend.Launch(q, k, v, o, shape.heads, shape.kvHeads, shape.queries, shape.keys, shape.keyCapacity, shape.headDim,
                             scale, mask._causal, mask._window,
                             over: shape.batch * shape.heads * shape.queries * 128, workgroup: 128)
}

/// Shape is an attention problem: batch, query heads, key/value heads
/// (fewer for grouped-query attention), queries and keys per sequence,
/// the head dimension, and the score scale (0 for 1/√headDim).
/// keyCapacity is how many keys each head's rows of K and V have room
/// for: a KV cache allocated for the whole context holds keys of them so
/// far. 0 means keys, rows packed.
public struct Shape {
    public var batch: int
    public var heads: int
    public var kvHeads: int
    public var queries: int
    public var keys: int
    public var headDim: int
    public var scale: float32
    public var keyCapacity: int

    public init(batch: int = 1, heads: int, kvHeads: int = 0, queries: int, keys: int, headDim: int, scale: float32 = 0, keyCapacity: int = 0) {
        self.batch = batch
        self.heads = heads
        self.kvHeads = kvHeads == 0 ? heads : kvHeads
        self.queries = queries
        self.keys = keys
        self.headDim = headDim
        self.scale = scale
        self.keyCapacity = keyCapacity == 0 ? keys : keyCapacity
    }
}
