// gpu/neural on every device, against float64 references from the host's
// libm, and against the CPU device.
import "gpu"
import "gpu/neural"
import "gpu/gputest"

@_silgen_name("exp") func cExp(_ x: float64) -> float64
@_silgen_name("log") func cLog(_ x: float64) -> float64
@_silgen_name("erf") func cErf(_ x: float64) -> float64
@_silgen_name("erfc") func cErfc(_ x: float64) -> float64
@_silgen_name("tanh") func cTanh(_ x: float64) -> float64
@_silgen_name("cos") func cCos(_ x: float64) -> float64
@_silgen_name("sin") func cSin(_ x: float64) -> float64
@_silgen_name("pow") func cPow(_ x: float64, _ y: float64) -> float64

var rng = gputest.Random(seed: 11)

func matrix(_ n: int, _ scale: float32) -> [float32] {
    var out: [float32] = []
    for _ in 0..<n { out.append(rng.Float32() * scale) }
    return out
}

// Each check: the host reference within ulps, and the CPU device's bits on
// every other device.
func compare(_ what: string, _ got: [(gpu.Device, [float32])], _ want: [float64], ulps: int) {
    for (d, g) in got {
        gputest.Close(what, d, g, want.map { float32($0) }, ulps: ulps)
        if !d.IsCPU { gputest.Close(what + " vs cpu", d, g, got[0].1, ulps: 1) }
    }
}

func compareNear(_ what: string, _ got: [(gpu.Device, [float32])], _ want: [float64], _ bound: [float64]) {
    for (d, g) in got {
        gputest.Near(what, d, g, want, bound: bound.map { $0 + 1e-38 })
        if !d.IsCPU { gputest.Close(what + " vs cpu", d, g, got[0].1, ulps: 1) }
    }
}

for (rows, cols) in [(1, 1), (3, 7), (4, 256), (5, 1000), (2, 5000)] {
    let x = matrix(rows * cols, 20)
    // Host references, row by row.
    var soft: [float64] = [], logSoft: [float64] = [], lse: [float64] = [], rms: [float64] = [], ln: [float64] = [], ce: [float64] = []
    // Error bounds from the inputs, as float32 arithmetic has them: e^(x -
    // max) inherits (x - max)'s rounding, |x - max|·2^-24 relatively;
    // LayerNorm's mean carries cols·ε·mean|x| into every output.
    var softB: [float64] = [], logSoftB: [float64] = [], lnB: [float64] = []
    var w: [float32] = [], b: [float32] = [], targets: [int32] = []
    for j in 0..<cols { w.append(1 + float32(j % 5) * 0.25); b.append(float32(j % 3) - 1) }
    for r in 0..<rows { targets.append(int32((r * 7) % cols)) }
    for r in 0..<rows {
        var m = -float64.infinity
        for j in 0..<cols { m = max(m, float64(x[r * cols + j])) }
        var s: float64 = 0, sq: float64 = 0, sum: float64 = 0
        for j in 0..<cols { let v = float64(x[r * cols + j]); s += cExp(v - m); sq += v * v; sum += v }
        for j in 0..<cols {
            let v = float64(x[r * cols + j])
            soft.append(cExp(v - m) / s)
            softB.append(cExp(v - m) / s * ((v - m).magnitude + 16) * 1.1920929e-07)
            logSoft.append(v - m - cLog(s))
            logSoftB.append((v.magnitude + m.magnitude + cLog(s).magnitude + 1) * 4.76837158e-07)
        }
        lse.append(m + cLog(s))
        ce.append(m + cLog(s) - float64(x[r * cols + int(targets[r])]))
        let scale = 1 / (sq / float64(cols) + 1e-6).squareRoot()
        let mean = sum / float64(cols)
        var vr: float64 = 0
        for j in 0..<cols { let d = float64(x[r * cols + j]) - mean; vr += d * d }
        let inv = 1 / (vr / float64(cols) + 1e-5).squareRoot()
        var absMean: float64 = 0
        for j in 0..<cols { absMean += float64(x[r * cols + j]).magnitude / float64(cols) }
        for j in 0..<cols {
            let v = float64(x[r * cols + j])
            rms.append(v * scale * float64(w[j]))
            ln.append((v - mean) * inv * float64(w[j]) + float64(b[j]))
            lnB.append(((v - mean).magnitude * inv * float64(w[j]) + float64(b[j]).magnitude) * 2e-6 +
                       float64(cols + 16) * 1.1920929e-07 * absMean * inv * float64(w[j]))
        }
    }
    var gs: [(gpu.Device, [float32])] = [], gl: [(gpu.Device, [float32])] = [], ge: [(gpu.Device, [float32])] = []
    var gr: [(gpu.Device, [float32])] = [], gn: [(gpu.Device, [float32])] = [], gc: [(gpu.Device, [float32])] = []
    for d in gputest.Devices() {
        let xb = try await d.Upload(x), wb = try await d.Upload(w), bb = try await d.Upload(b)
        let y = try await d.CreateBuffer(of: float32.self, count: rows * cols)
        try await neural.Softmax(xb, into: y, rows: rows, cols: cols)
        gs.append((d, try await y.Download()))
        try await neural.LogSoftmax(xb, into: y, rows: rows, cols: cols)
        gl.append((d, try await y.Download()))
        ge.append((d, try await neural.LogSumExp(xb, rows: rows, cols: cols).Download()))
        try await neural.RMSNorm(xb, weight: wb, into: y, rows: rows, cols: cols)
        gr.append((d, try await y.Download()))
        try await neural.LayerNorm(xb, weight: wb, bias: bb, into: y, rows: rows, cols: cols)
        gn.append((d, try await y.Download()))
        gc.append((d, try await neural.CrossEntropy(xb, targets: try await d.Upload(targets), rows: rows, cols: cols).Download()))
    }
    let tag = "\(rows)x\(cols)"
    compareNear("Softmax \(tag)", gs, soft, softB)
    compareNear("LogSoftmax \(tag)", gl, logSoft, logSoftB)
    compare("LogSumExp \(tag)", ge, lse, ulps: 16)
    compare("RMSNorm \(tag)", gr, rms, ulps: 16)
    compareNear("LayerNorm \(tag)", gn, ln, lnB)
    compare("CrossEntropy \(tag)", gc, ce, ulps: 64)
}

// Activations, elementwise, against libm.
func reference(_ a: neural.Activation, _ x: float64) -> float64 {
    switch a {
    case .ReLU: return x > 0 ? x : 0
    // In float64 too, 1 + erf and 1 + tanh cancel in the tail: the
    // references are written the way the kernels are.
    case .GELU: return 0.5 * x * cErfc(-x / 2.0.squareRoot())
    case .GELUTanh: return x / (1 + cExp(-1.5957691216057308 * (x + 0.044715 * x * x * x)))
    case .SiLU: return x / (1 + cExp(-x))
    case .Sigmoid: return 1 / (1 + cExp(-x))
    case .Tanh: return cTanh(x)
    }
}

do {
    let x = matrix(50000, 8)
    let gate = matrix(50000, 8)
    for (a, ulps) in [(neural.Activation.ReLU, 0), (.GELU, 8), (.GELUTanh, 8), (.SiLU, 8), (.Sigmoid, 4), (.Tanh, 4)] {
        let want = x.map { reference(a, float64($0)) }
        var gated: [float64] = []
        for i in 0..<x.count { gated.append(float64(x[i]) * reference(a, float64(gate[i]))) }
        var got: [(gpu.Device, [float32])] = [], gotGated: [(gpu.Device, [float32])] = []
        for d in gputest.Devices() {
            let xb = try await d.Upload(x)
            let y = try await d.CreateBuffer(of: float32.self, count: x.count)
            try await neural.Activate(xb, a, into: y)
            got.append((d, try await y.Download()))
            try await neural.Gated(xb, gate: try await d.Upload(gate), a, into: y)
            gotGated.append((d, try await y.Download()))
        }
        // An activation's error is its argument's rounding times its
        // condition number, which in the GELU tails grows as x²:
        // (x² + 16 + ulps)·2^-23 relatively, and a floor far below any
        // value a model keeps.
        var bound: [float64] = [], gatedBound: [float64] = []
        for i in 0..<x.count {
            let xv = float64(x[i]), gv = float64(gate[i])
            bound.append(want[i].magnitude * (xv * xv + 16 + float64(ulps)) * 1.1920929e-07 + 1e-30)
            gatedBound.append(gated[i].magnitude * (gv * gv + 18 + float64(ulps)) * 1.1920929e-07 + xv.magnitude * 1e-16)
        }
        compareNear("Activate \(a)", got, want, bound)
        compareNear("Gated \(a)", gotGated, gated, gatedBound)
    }
}

// RoPE, against the rotation done in float64.
do {
    let tokens = 9, heads = 3, dim = 16
    let x = matrix(tokens * heads * dim, 1)
    var positions: [int32] = []
    for t in 0..<tokens { positions.append(int32(t * 37)) }
    var want: [float64] = x.map { float64($0) }
    for t in 0..<tokens {
        for h in 0..<heads {
            for p in 0..<(dim / 2) {
                let at = (t * heads + h) * dim + 2 * p
                let angle = float64(positions[t]) * cPow(10000, -float64(2 * p) / float64(dim))
                let a = want[at], b = want[at + 1]
                want[at] = a * cCos(angle) - b * cSin(angle)
                want[at + 1] = a * cSin(angle) + b * cCos(angle)
            }
        }
    }
    var got: [(gpu.Device, [float32])] = []
    for d in gputest.Devices() {
        let xb = try await d.Upload(x)
        try await neural.RoPE(xb, positions: try await d.Upload(positions), heads: heads, dim: dim)
        got.append((d, try await xb.Download()))
    }
    // The angle is position · frequency in float32, so its error grows with
    // the position: an absolute bound, as a model's own float32 RoPE has.
    for (d, g) in got {
        gputest.Near("RoPE", d, g, want, bound: want.map { _ in 2e-4 })
        if !d.IsCPU { gputest.Close("RoPE vs cpu", d, g, got[0].1, ulps: 1) }
    }
}

gputest.Done()
