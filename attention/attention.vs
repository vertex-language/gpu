// Fused attention: O = softmax(scale · Q·Kᵀ + mask) · V, without ever
// writing the scores matrix. This is FlashAttention's online softmax:
// keys are taken a block at a time, and the running maximum, the running
// sum and the output accumulator are rescaled as each block raises the
// maximum. Memory is O(d) per query rather than O(sequence).
//
// One workgroup of 128 computes one query row: each work-item scores one
// key of the block (a dot product over d) and owns one dimension of the
// output. Every sum is taken in a fixed order, so the result is the same
// on every device that keeps subnormals -- but for decoding (one query) on
// a GPU of 32-wide waves, which adds by waves (_attendDecode) and so
// differs in the last bits. Head dimensions up to 128.
//
// Layouts are row-major: Q and O are [batch, heads, queries, d], K and V
// [batch, kvHeads, capacity, d], the first keys of each head's capacity
// rows used -- a KV cache as it fills. heads a multiple of kvHeads is grouped-query
// attention; kvHeads 1 is multi-query.
import (
    "gpu"
    "gpu/parallel"
    "math"
)

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

func _attendDecode(_ q: gpu.Span<float32>, _ k: gpu.Span<float32>, _ v: gpu.Span<float32>, _ o: gpu.MutableSpan<float32>,
                   _ group: int, _ keys: int, _ capacity: int, _ d: int, _ scale: float32, _ first: int) kernel {
    // _attendOne on a GPU with waves of 32: one query a head, a group of
    // eight waves. The grid is (heads a KV head has, KV heads), so a head's
    // KV head is its group's y, not a division. Each work-item scores
    // every 256th visible key; the most and the sum of exponentials are
    // each a wave's shuffles and one barrier; and the output is gathered
    // by the eight waves over an eighth of the keys each -- a lane holding
    // up to four dimensions -- and their parts added in shared storage.
    let scores = gpu.Shared<float32>(count: 4096)
    let qs = gpu.Shared<float32>(count: 128)
    let red = gpu.Shared<float32>(count: 16)
    let parts = gpu.Shared<float32>(count: 1024)
    let t = gpu.LocalIndex.x
    let lane = t & 31
    let wave = t >> 5
    let kvHead = gpu.GroupIndex.y
    let head = kvHead &* group &+ gpu.GroupIndex.x
    let qp = q._base + head &* d
    let kvBase = kvHead &* capacity &* d
    if t < d {
        qs[t] = qp[t]
    }
    gpu.Barrier()
    // Scores: eight lanes a key, a lane every eighth dimension of it, so
    // that the eight read the key's row together; four keys a wave and
    // thirty-two a step. The step is the wave's, so every lane shuffles.
    var most = -float32.infinity
    let sub = lane & 7
    var jb = first &+ (wave &<< 2)
    while jb < keys {
        let j = jb &+ (lane &>> 3)
        var dot: float32 = 0
        if j < keys {
            let row = k._base + (kvBase &+ j &* d)
            var c = sub
            while c < d {
                dot = dot + qs[c] * row[c]
                c = c &+ 8
            }
        }
        dot = dot + gpu.Wave.ShuffleXor(dot, mask: 4)
        dot = dot + gpu.Wave.ShuffleXor(dot, mask: 2)
        dot = dot + gpu.Wave.ShuffleXor(dot, mask: 1)
        if j < keys && sub == 0 {
            let s = dot * scale
            scores[j] = s
            most = s > most ? s : most
        }
        jb = jb &+ 32
    }
    most = gpu.Wave.Max(most)
    if lane == 0 { red[wave] = most }
    gpu.Barrier()
    var m = red[0]
    var w = 1
    while w < 8 {
        m = math.Max(m, red[w])
        w = w &+ 1
    }
    var sum: float32 = 0
    var j = first &+ t
    while j < keys {
        let e = math.Exp(scores[j] - m)
        scores[j] = e
        sum = sum + e
        j = j &+ 256
    }
    sum = gpu.Wave.Sum(sum)
    if lane == 0 { red[8 &+ wave] = sum }
    gpu.Barrier()
    var total: float32 = 0
    w = 8
    while w < 16 {
        total = total + red[w]
        w = w &+ 1
    }
    var a0: float32 = 0, a1: float32 = 0, a2: float32 = 0, a3: float32 = 0
    let vb = v._base + kvBase
    j = first &+ wave
    while j < keys {
        let p = scores[j]
        let row = vb + j &* d
        if lane < d { a0 = a0 + p * row[lane] }
        if lane &+ 32 < d { a1 = a1 + p * row[lane &+ 32] }
        if lane &+ 64 < d { a2 = a2 + p * row[lane &+ 64] }
        if lane &+ 96 < d { a3 = a3 + p * row[lane &+ 96] }
        j = j &+ 8
    }
    let at = wave &* 128 &+ lane
    parts[at] = a0
    parts[at &+ 32] = a1
    parts[at &+ 64] = a2
    parts[at &+ 96] = a3
    gpu.Barrier()
    if t < d {
        var acc: float32 = 0
        w = 0
        while w < 8 {
            acc = acc + parts[w &* 128 &+ t]
            w = w &+ 1
        }
        o[head &* d &+ t] = total > 0 ? acc / total : 0
    }
}

func _attendTiled(_ q: gpu.Span<float32>, _ k: gpu.Span<float32>, _ v: gpu.Span<float32>, _ o: gpu.MutableSpan<float32>,
                  _ heads: int, _ kvHeads: int, _ queries: int, _ keys: int, _ capacity: int, _ d: int,
                  _ scale: float32, _ causal: bool, _ window: int) kernel {
    // Many queries at once -- a prompt -- a workgroup 16 of one head's,
    // with every tile of 16 keys and values loaded once for all 16:
    // flash attention's tiling, its online softmax kept a row at a time.
    // Q, the output and a K or V tile live in shared storage; 128
    // work-items score 2 query-key pairs each, one per row folds the
    // scores into its running maximum and sum, and 8 a row add the tile
    // of V into the row's output, each its own dimensions.
    let qs = gpu.Shared<float32>(count: 2048)   // 16 queries x d (d ≤ 128)
    let kv = gpu.Shared<float32>(count: 2048)   // 16 keys x d: K, then V
    let os = gpu.Shared<float32>(count: 2048)   // 16 rows of output x d
    let ss = gpu.Shared<float32>(count: 256)    // 16 x 16 scores, then weights
    let st = gpu.Shared<float32>(count: 48)     // per row: max, sum, correction
    let t = gpu.LocalIndex.x
    let groups = (queries &+ 15) / 16
    let g = gpu.GroupIndex.x
    let qb = (g % groups) &* 16                  // the tile's first query
    let bh = g / groups                          // batch · heads + head
    let head = bh % heads
    let batch = bh / heads
    let kvHead = head / (heads / kvHeads)
    let kvBase = (batch &* kvHeads &+ kvHead) &* capacity &* d
    let qBase = bh &* queries &* d
    let n = queries &- qb < 16 ? queries &- qb : 16
    var e = t
    while e < 16 &* d {
        let r = e / d
        qs[e] = r < n ? q[qBase &+ (qb &+ r) &* d &+ e % d] : 0
        os[e] = 0
        e = e &+ 128
    }
    if t < 16 {
        st[t] = -float32.infinity
        st[16 &+ t] = 0
    }
    // The keys any of the tile's queries may see: up to the last one's
    // position, and from the first one's window.
    let first = keys &- queries &+ qb
    var end = keys
    if causal { end = first &+ n }
    var start = 0
    if window > 0 && first &- window &+ 1 > 0 { start = first &- window &+ 1 }
    var kb = start / 16 &* 16
    gpu.Barrier()
    while kb < end {
        e = t
        while e < 16 &* d {
            let j = kb &+ e / d
            kv[e] = j < keys ? k[kvBase &+ j &* d &+ e % d] : 0
            e = e &+ 128
        }
        gpu.Barrier()
        var pr = t
        while pr < 256 {
            let r = pr >> 4
            let c = pr & 15
            let j = kb &+ c
            let position = first &+ r
            var visible = r < n && j < keys
            if causal && j > position { visible = false }
            if window > 0 && j <= position &- window { visible = false }
            var sc = -float32.infinity
            if visible {
                var dot: float32 = 0
                var x = 0
                while x < d {
                    dot = dot + qs[r &* d &+ x] * kv[c &* d &+ x]
                    x = x &+ 1
                }
                sc = dot * scale
            }
            ss[pr] = sc
            pr = pr &+ 128
        }
        gpu.Barrier()
        // V's tile over K's, while a thread a row turns its scores into
        // weights against the row's new maximum.
        e = t
        while e < 16 &* d {
            let j = kb &+ e / d
            kv[e] = j < keys ? v[kvBase &+ j &* d &+ e % d] : 0
            e = e &+ 128
        }
        if t < 16 {
            var m = st[t]
            var c = 0
            while c < 16 {
                m = math.Max(m, ss[t &* 16 &+ c])
                c = c &+ 1
            }
            let old = st[t]
            let corr = old == -float32.infinity ? float32(0) : math.Exp(old - m)
            var sum: float32 = 0
            c = 0
            while c < 16 {
                let sv = ss[t &* 16 &+ c]
                let p = sv == -float32.infinity ? float32(0) : math.Exp(sv - m)
                ss[t &* 16 &+ c] = p
                sum = sum + p
                c = c &+ 1
            }
            st[t] = m
            st[16 &+ t] = st[16 &+ t] * corr + sum
            st[32 &+ t] = corr
        }
        gpu.Barrier()
        let r = t >> 3
        let part = t & 7
        let corr = st[32 &+ r]
        var x = part
        while x < d {
            var acc = os[r &* d &+ x] * corr
            var c = 0
            while c < 16 {
                acc = acc + ss[r &* 16 &+ c] * kv[c &* d &+ x]
                c = c &+ 1
            }
            os[r &* d &+ x] = acc
            x = x &+ 8
        }
        gpu.Barrier()
        kb = kb &+ 16
    }
    e = t
    while e < n &* d {
        let r = e / d
        let l = st[16 &+ r]
        o[qBase &+ (qb &+ r) &* d &+ e % d] = l > 0 ? os[e] / l : 0
        e = e &+ 128
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
        if o.Device.WaveSize == 32 && shape.headDim <= MaxHeadDim {
            if q.count < shape.heads * shape.headDim || o.count < shape.heads * shape.headDim
                || k.count < shape.kvHeads * shape.keyCapacity * shape.headDim
                || v.count < shape.kvHeads * shape.keyCapacity * shape.headDim {
                throw gpu.DeviceError.badCount
            }
            let group = shape.heads / shape.kvHeads
            try await _attendDecode.Launch(q, k, v, o, group, shape.keys, shape.keyCapacity, shape.headDim, scale, first,
                                           over: (group * 256, shape.kvHeads, 1), workgroup: (256, 1, 1))
            return
        }
        try await _attendOne.Launch(q, k, v, o, shape.heads, shape.kvHeads, shape.keys, shape.keyCapacity, shape.headDim,
                                    scale, first, over: shape.heads * 128, workgroup: 128)
        return
    }
    if shape.headDim <= MaxHeadDim {
        if q.count < shape.batch * shape.heads * shape.queries * shape.headDim
            || o.count < shape.batch * shape.heads * shape.queries * shape.headDim
            || k.count < shape.batch * shape.kvHeads * shape.keyCapacity * shape.headDim
            || v.count < shape.batch * shape.kvHeads * shape.keyCapacity * shape.headDim {
            throw gpu.DeviceError.badCount
        }
        let groups = (shape.queries + 15) / 16
        try await _attendTiled.Launch(q, k, v, o, shape.heads, shape.kvHeads, shape.queries, shape.keys, shape.keyCapacity,
                                      shape.headDim, scale, mask._causal, mask._window,
                                      over: shape.batch * shape.heads * groups * 128, workgroup: 128)
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
