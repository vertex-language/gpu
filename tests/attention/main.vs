// gpu/attention against attention computed on the host in float64: the
// scores, the softmax and the weighted sum written out plainly.
import "gpu"
import "gpu/attention"
import "gpu/gputest"

@_silgen_name("exp") func cExp(_ x: float64) -> float64

var rng = gputest.Random(seed: 5)

func host(_ q: [float32], _ k: [float32], _ v: [float32], _ s: attention.Shape, _ causal: bool, _ window: int) -> [float64] {
    var o = [float64](repeating: 0, count: s.batch * s.heads * s.queries * s.headDim)
    let d = s.headDim
    let scale = s.scale != 0 ? float64(s.scale) : 1 / float64(d).squareRoot()
    for b in 0..<s.batch {
        for h in 0..<s.heads {
            let kvh = h / (s.heads / s.kvHeads)
            for i in 0..<s.queries {
                let row = (b * s.heads + h) * s.queries + i
                let pos = s.keys - s.queries + i
                var scores = [float64](repeating: -float64.infinity, count: s.keys)
                var m = -float64.infinity
                for j in 0..<s.keys {
                    if causal && j > pos { continue }
                    if window > 0 && j <= pos - window { continue }
                    var dot: float64 = 0
                    for c in 0..<d { dot += float64(q[row * d + c]) * float64(k[((b * s.kvHeads + kvh) * s.keys + j) * d + c]) }
                    scores[j] = dot * scale
                    m = max(m, scores[j])
                }
                if m == -float64.infinity { continue }
                var sum: float64 = 0
                for j in 0..<s.keys { sum += scores[j] == -float64.infinity ? 0 : cExp(scores[j] - m) }
                for c in 0..<d {
                    var acc: float64 = 0
                    for j in 0..<s.keys where scores[j] != -float64.infinity {
                        acc += cExp(scores[j] - m) / sum * float64(v[((b * s.kvHeads + kvh) * s.keys + j) * d + c])
                    }
                    o[row * d + c] = acc
                }
            }
        }
    }
    return o
}

let cases: [(attention.Shape, int)] = [
    (attention.Shape(heads: 1, queries: 1, keys: 1, headDim: 1), 0),
    (attention.Shape(heads: 2, queries: 5, keys: 5, headDim: 8), 0),
    (attention.Shape(batch: 2, heads: 4, kvHeads: 2, queries: 7, keys: 300, headDim: 64), 0),
    (attention.Shape(heads: 8, kvHeads: 1, queries: 1, keys: 1000, headDim: 128), 0),
    (attention.Shape(heads: 3, queries: 129, keys: 129, headDim: 32), 0),
    (attention.Shape(heads: 2, queries: 40, keys: 200, headDim: 16, scale: 0.5), 17),
]
for (shape, window) in cases {
    let n = shape.batch * shape.heads * shape.queries * shape.headDim
    let nk = shape.batch * shape.kvHeads * shape.keys * shape.headDim
    var q: [float32] = [], k: [float32] = [], v: [float32] = []
    for _ in 0..<n { q.append(rng.Float32() * 2) }
    for _ in 0..<nk { k.append(rng.Float32() * 2); v.append(rng.Float32()) }
    let tag = "b\(shape.batch) h\(shape.heads)/\(shape.kvHeads) q\(shape.queries) k\(shape.keys) d\(shape.headDim)"
    for (mask, causal, w) in [(attention.Mask.None, false, 0), (.Causal, true, 0), (.SlidingWindow(window == 0 ? 4 : window), true, window == 0 ? 4 : window)] {
        let want = host(q, k, v, shape, causal, w)
        var cpu: [float32] = []
        for d in gputest.Devices() {
            let o = try await d.CreateBuffer(of: float32.self, count: n)
            try await attention.Forward(q: try await d.Upload(q), k: try await d.Upload(k), v: try await d.Upload(v), into: o, shape, mask: mask)
            let got = try await o.Download()
            // Values are averages of numbers in [-1, 1): an absolute bound.
            gputest.Near("Forward \(tag) \(mask)", d, got, want, bound: want.map { _ in 2e-5 })
            if d.IsCPU { cpu = got } else { gputest.Close("Forward \(tag) \(mask) vs cpu", d, got, cpu, ulps: 1) }
        }
    }
}

// ---- a KV cache: keys in rows with room for more ----

// The same K and V laid in a cache of capacity rows a head, the rows past
// keys NaN: a read of one would show. Decoding a token is this, one query
// and the keys so far.
for (heads, kvHeads, keys, capacity, d) in [(6, 6, 1, 128, 48), (6, 6, 37, 128, 48), (8, 4, 20, 2048, 8), (8, 2, 64, 64, 16)] {
    let shape = attention.Shape(heads: heads, kvHeads: kvHeads, queries: 1, keys: keys, headDim: d)
    var q: [float32] = [], k: [float32] = [], v: [float32] = []
    for _ in 0..<(heads * d) { q.append(rng.Float32() * 2) }
    for _ in 0..<(kvHeads * keys * d) { k.append(rng.Float32() * 2); v.append(rng.Float32()) }
    var kc = [float32](repeating: float32.nan, count: kvHeads * capacity * d), vc = kc
    for h in 0..<kvHeads {
        for j in 0..<(keys * d) {
            kc[h * capacity * d + j] = k[h * keys * d + j]
            vc[h * capacity * d + j] = v[h * keys * d + j]
        }
    }
    var cached = shape
    cached.keyCapacity = capacity
    for d in gputest.Devices() {
        let packed = try await d.CreateBuffer(of: float32.self, count: heads * shape.headDim)
        let qb = try await d.Upload(q)
        try await attention.Forward(q: qb, k: try await d.Upload(k), v: try await d.Upload(v), into: packed, shape, mask: .Causal)
        let o = try await d.CreateBuffer(of: float32.self, count: heads * shape.headDim)
        try await attention.Forward(q: qb, k: try await d.Upload(kc), v: try await d.Upload(vc), into: o, cached, mask: .Causal)
        gputest.Equal("Forward from a cache h\(heads)/\(kvHeads) k\(keys) of \(capacity)", d, try await o.Download(), try await packed.Download())
    }
}

gputest.Done()
