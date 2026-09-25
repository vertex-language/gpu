// gpu/linalg on every device: int32 exactly against the host; float32
// against an f64 host product within ULPs, and against the CPU device.
import "gpu"
import "gpu/linalg"
import "gpu/dtype"
import "gpu/gputest"

var rng = gputest.Random(seed: 3)

func hostMatmul(_ a: [float64], _ b: [float64], _ m: int, _ n: int, _ k: int, _ ta: bool, _ tb: bool) -> [float64] {
    return hostProducts(a, b, m, n, k, ta, tb, false)
}

// hostProducts is A·B, or with abs, Σ|a·b| for each element: what bounds
// a float32 dot product's error.
func hostProducts(_ a: [float64], _ b: [float64], _ m: int, _ n: int, _ k: int, _ ta: bool, _ tb: bool, _ abs: bool) -> [float64] {
    var c = [float64](repeating: 0, count: m * n)
    for i in 0..<m {
        for j in 0..<n {
            var s: float64 = 0
            for p in 0..<k {
                let x = ta ? a[p * m + i] : a[i * k + p]
                let y = tb ? b[j * k + p] : b[p * n + j]
                s += abs ? (x * y).magnitude : x * y
            }
            c[i * n + j] = s
        }
    }
    return c
}

let shapes = [(1, 1, 1), (3, 5, 7), (16, 16, 16), (17, 33, 65), (64, 48, 80), (100, 1, 50), (1, 100, 3)]
for (m, n, k) in shapes {
    for (ta, tb) in [(false, false), (true, false), (false, true)] {
        let batch = m * n * k < 5000 ? 3 : 1
        var af: [float32] = [], bf: [float32] = [], ai: [int32] = [], bi: [int32] = []
        for _ in 0..<(batch * m * k) { af.append(rng.Float32()); ai.append(int32(rng.Uint32() % 11) - 5) }
        for _ in 0..<(batch * k * n) { bf.append(rng.Float32()); bi.append(int32(rng.Uint32() % 11) - 5) }
        var wantF: [float64] = [], boundF: [float64] = [], wantI: [int32] = []
        for bt in 0..<batch {
            let pa = Array(af[(bt * m * k)..<((bt + 1) * m * k)]).map { float64($0) }
            let pb = Array(bf[(bt * k * n)..<((bt + 1) * k * n)]).map { float64($0) }
            wantF += hostMatmul(pa, pb, m, n, k, ta, tb)
            // k roundings of a sum of k products, each at most half an ulp
            // of what it rounds: (k + 1)·2^-24·Σ|a·b|, and a little more.
            boundF += hostProducts(pa, pb, m, n, k, ta, tb, true).map { $0 * float64(k + 2) * 5.960464477539063e-08 + 1e-30 }
            let qa = Array(ai[(bt * m * k)..<((bt + 1) * m * k)]).map { float64($0) }
            let qb = Array(bi[(bt * k * n)..<((bt + 1) * k * n)]).map { float64($0) }
            wantI += hostMatmul(qa, qb, m, n, k, ta, tb).map { int32($0) }
        }
        let what = "\(m)x\(n)x\(k) b\(batch) t\(ta ? 1 : 0)\(tb ? 1 : 0)"
        var oracle: [float32] = []
        for d in gputest.Devices() {
            let a = try await d.Upload(af), b = try await d.Upload(bf)
            let c = try await d.CreateBuffer(of: float32.self, count: batch * m * n)
            let shape = linalg.Shape(m: m, n: n, k: k, batch: batch, transposeA: ta, transposeB: tb)
            try await linalg.Matmul(a, b, into: c, shape)
            let got = try await c.Download()
            gputest.Near("Matmul f32 \(what)", d, got, wantF, bound: boundF)
            if d.IsCPU { oracle = got } else { gputest.Close("Matmul f32 \(what) vs cpu", d, got, oracle, ulps: 4 * k + 4) }
            let x = try await d.Upload(ai), y = try await d.Upload(bi)
            let z = try await d.CreateBuffer(of: int32.self, count: batch * m * n)
            try await linalg.Matmul(x, y, into: z, shape)
            gputest.Equal("Matmul i32 \(what)", d, try await z.Download(), wantI)
        }
    }
}

// The epilogue: relu(2 * A·B + bias[col] + residual), in int32 exactly.
do {
    let m = 5, n = 7, k = 9
    var a: [int32] = [], b: [int32] = [], bias: [int32] = [], res: [int32] = []
    for _ in 0..<(m * k) { a.append(int32(rng.Uint32() % 7) - 3) }
    for _ in 0..<(k * n) { b.append(int32(rng.Uint32() % 7) - 3) }
    for _ in 0..<n { bias.append(int32(rng.Uint32() % 9) - 4) }
    for _ in 0..<(m * n) { res.append(int32(rng.Uint32() % 9) - 4) }
    let p = hostMatmul(a.map { float64($0) }, b.map { float64($0) }, m, n, k, false, false)
    var want: [int32] = []
    for i in 0..<m { for j in 0..<n { want.append(max(0, 2 * int32(p[i * n + j]) + bias[j] + res[i * n + j])) } }
    for d in gputest.Devices() {
        let c = try await d.CreateBuffer(of: int32.self, count: m * n)
        let e = linalg.Epilogue<int32>(scale: 2, bias: try await d.Upload(bias), residual: try await d.Upload(res), activation: .ReLU)
        try await linalg.Matmul(try await d.Upload(a), try await d.Upload(b), into: c, linalg.Shape(m: m, n: n, k: k), e)
        gputest.Equal("Matmul epilogue", d, try await c.Download(), want)
    }
}

// Gemv and Transpose.
for (m, k) in [(1, 1), (7, 3), (64, 256), (300, 1000), (5, 4097)] {
    var a: [int32] = [], x: [int32] = []
    for _ in 0..<(m * k) { a.append(int32(rng.Uint32() % 9) - 4) }
    for _ in 0..<k { x.append(int32(rng.Uint32() % 9) - 4) }
    var want: [int32] = []
    for i in 0..<m { var s: int32 = 0; for j in 0..<k { s += a[i * k + j] * x[j] }; want.append(s) }
    var wantT = [int32](repeating: 0, count: m * k)
    for i in 0..<m { for j in 0..<k { wantT[j * m + i] = a[i * k + j] } }
    for d in gputest.Devices() {
        let ab = try await d.Upload(a)
        let y = try await d.CreateBuffer(of: int32.self, count: m)
        try await linalg.Gemv(ab, try await d.Upload(x), into: y, m: m, k: k)
        gputest.Equal("Gemv \(m)x\(k)", d, try await y.Download(), want)
        let t = try await d.CreateBuffer(of: int32.self, count: m * k)
        try await linalg.Transpose(ab, into: t, rows: m, cols: k)
        gputest.Equal("Transpose \(m)x\(k)", d, try await t.Download(), wantT)
    }
}

// ---- float16 and bfloat16 ----

// Halves in, halves out, the sum taken in float32: within the float32
// bound of a sum of k products, plus one rounding to the half at the end
// (eps, relative) and its smallest subnormal step (tiny).
func halfMatmul<T: dtype.Number>(_ name: string, _ t: T.Type, eps: float64, tiny: float64) async throws {
    for (m, n, k) in shapes {
        for (ta, tb) in [(false, false), (true, false), (false, true)] {
            var ah: [T] = [], bh: [T] = []
            for _ in 0..<(m * k) { ah.append(T.FromFloat64(float64(rng.Float32()) * 2 - 1)) }
            for _ in 0..<(k * n) { bh.append(T.FromFloat64(float64(rng.Float32()) * 2 - 1)) }
            let pa = ah.map { T.ToFloat64($0) }, pb = bh.map { T.ToFloat64($0) }
            let want = hostMatmul(pa, pb, m, n, k, ta, tb)
            let sums = hostProducts(pa, pb, m, n, k, ta, tb, true)
            var bound: [float64] = []
            for i in 0..<want.count {
                bound.append(sums[i] * float64(k + 2) * 5.960464477539063e-08 + want[i].magnitude * eps + tiny)
            }
            let what = "\(name) \(m)x\(n)x\(k) t\(ta ? 1 : 0)\(tb ? 1 : 0)"
            for d in gputest.Devices() {
                let c = try await d.CreateBuffer(of: T.self, count: m * n)
                try await linalg.Matmul(try await d.Upload(ah), try await d.Upload(bh), into: c,
                                        linalg.Shape(m: m, n: n, k: k, transposeA: ta, transposeB: tb))
                gputest.Near("Matmul \(what)", d, (try await c.Download()).map { float32(T.ToFloat64($0)) }, want, bound: bound)
            }
        }
    }
    // A sum a half cannot hold on the way: 4096 ones, which a half sum
    // stops growing at 2048 (float16) or 256 (bfloat16).
    let ones = [T](repeating: T.FromFloat64(1), count: 4096)
    for d in gputest.Devices() {
        let y = try await d.CreateBuffer(of: T.self, count: 1)
        try await linalg.Gemv(try await d.Upload(ones), try await d.Upload(ones), into: y, m: 1, k: 4096)
        let c = try await d.CreateBuffer(of: T.self, count: 1)
        try await linalg.Matmul(try await d.Upload(ones), try await d.Upload(ones), into: c, linalg.Shape(m: 1, n: 1, k: 4096))
        gputest.Equal("Gemv \(name) accumulates in float32", d, try await y.Download(), [T.FromFloat64(4096)])
        gputest.Equal("Matmul \(name) accumulates in float32", d, try await c.Download(), [T.FromFloat64(4096)])
    }
}

try await halfMatmul("f16", float16.self, eps: 4.8828125e-04, tiny: 5.960464477539063e-08)
try await halfMatmul("bf16", bfloat16.self, eps: 3.90625e-03, tiny: 1e-38)

gputest.Done()
