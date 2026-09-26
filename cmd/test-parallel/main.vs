// gpu/parallel on every device, against the host and the CPU device.
import (
    "gpu"
    "gpu/dtype"
    "gpu/gputest"
    "gpu/parallel"
)

var rng = gputest.Random(seed: 1)

func floats(_ n: int) -> [float32] {
    var out: [float32] = []
    for _ in 0..<n { out.append(rng.Float32()) }
    return out
}

func ints(_ n: int) -> [int32] {
    var out: [int32] = []
    for _ in 0..<n { out.append(rng.Int32() >> 4) }
    return out
}

func uints(_ n: int) -> [uint32] {
    var out: [uint32] = []
    for _ in 0..<n { out.append(rng.Uint32()) }
    return out
}

// ---- group-scope device functions, called from a kernel in this module ----

func groupOps(_ x: gpu.Span<int32>, _ sum: gpu.MutableSpan<int32>, _ inc: gpu.MutableSpan<int32>, _ ex: gpu.MutableSpan<int32>, _ mn: gpu.MutableSpan<int32>, _ mx: gpu.MutableSpan<int32>) kernel {
    let i = gpu.Index.x
    let v = x[i]
    sum[i] = parallel.GroupSum(v)
    inc[i] = parallel.GroupScan(v)
    ex[i] = parallel.GroupExclusiveScan(v).prefix
    mn[i] = parallel.GroupMin(v)
    mx[i] = parallel.GroupMax(v)
}

for group in [1, 3, 32, 64, 100, 256, 1024] {
    let n = group * 3
    let host = ints(n)
    var wantSum: [int32] = [], wantInc: [int32] = [], wantEx: [int32] = []
    var wantMin: [int32] = [], wantMax: [int32] = []
    var g = 0
    while g < n {
        var s: int32 = 0, lo = int32.max, hi = int32.min
        for j in g..<(g + group) { s = s &+ host[j]; lo = min(lo, host[j]); hi = max(hi, host[j]) }
        var run: int32 = 0
        for j in g..<(g + group) {
            wantEx.append(run)
            run = run &+ host[j]
            wantInc.append(run)
            wantSum.append(s)
            wantMin.append(lo)
            wantMax.append(hi)
        }
        g += group
    }
    for d in gputest.Devices() {
        let x = try await d.Upload(host)
        let bufs = [try await d.CreateBuffer(of: int32.self, count: n), try await d.CreateBuffer(of: int32.self, count: n),
                    try await d.CreateBuffer(of: int32.self, count: n), try await d.CreateBuffer(of: int32.self, count: n),
                    try await d.CreateBuffer(of: int32.self, count: n)]
        try await groupOps.Launch(x, bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], over: n, workgroup: group)
        gputest.Equal("GroupSum/\(group)", d, try await bufs[0].Download(), wantSum)
        gputest.Equal("GroupScan/\(group)", d, try await bufs[1].Download(), wantInc)
        gputest.Equal("GroupExclusiveScan/\(group)", d, try await bufs[2].Download(), wantEx)
        gputest.Equal("GroupMin/\(group)", d, try await bufs[3].Download(), wantMin)
        gputest.Equal("GroupMax/\(group)", d, try await bufs[4].Download(), wantMax)
    }
}

// ---- Reduce ----

for n in gputest.Sizes {
    let fs = floats(n), xs = ints(n), us = uints(n)
    var fsum: float64 = 0, fmin = float32.infinity, fmax = -float32.infinity
    for v in fs { fsum += float64(v); fmin = min(fmin, v); fmax = max(fmax, v) }
    var isum: int32 = 0, imin = int32.max, imax = int32.min
    for v in xs { isum = isum &+ v; imin = min(imin, v); imax = max(imax, v) }
    var usum: uint32 = 0, umin = uint32.max, umax: uint32 = 0
    for v in us { usum = usum &+ v; umin = min(umin, v); umax = max(umax, v) }
    var oracle: float32 = 0
    for d in gputest.Devices() {
        let fb = try await d.Upload(fs), ib = try await d.Upload(xs), ub = try await d.Upload(us)
        let s = try await parallel.Reduce(fb, .Sum)
        // The tree's order xs not the host's: close to the f64 sum, and
        // identical to the CPU device's.
        gputest.Close("Reduce f32 Sum/\(n)", d, [s], [float32(fsum)], ulps: 64 + n / 64)
        if d.IsCPU { oracle = s } else { gputest.Equal("Reduce f32 Sum/\(n) vs cpu", d, [s], [oracle]) }
        gputest.Equal("Reduce f32 Min/\(n)", d, [try await parallel.Reduce(fb, .Min)], [fmin])
        gputest.Equal("Reduce f32 Max/\(n)", d, [try await parallel.Reduce(fb, .Max)], [fmax])
        gputest.Equal("Reduce i32/\(n)", d, [try await parallel.Reduce(ib, .Sum), try await parallel.Reduce(ib, .Min), try await parallel.Reduce(ib, .Max)], [isum, imin, imax])
        gputest.Equal("Reduce u32/\(n)", d, [try await parallel.Reduce(ub, .Sum), try await parallel.Reduce(ub, .Min), try await parallel.Reduce(ub, .Max)], [usum, umin, umax])
    }
}

// ---- Scan ----

for n in gputest.Sizes {
    let xs = ints(n), fs = floats(n)
    var inc: [int32] = [], ex: [int32] = []
    var run: int32 = 0
    for v in xs { ex.append(run); run = run &+ v; inc.append(run) }
    var oracle: [float32] = []
    for d in gputest.Devices() {
        let a = try await d.Upload(xs)
        try await parallel.Scan(a)
        gputest.Equal("Scan i32/\(n)", d, try await a.Download(), inc)
        let b = try await d.Upload(xs)
        try await parallel.Scan(b, exclusive: true)
        gputest.Equal("Scan i32 exclusive/\(n)", d, try await b.Download(), ex)
        let f = try await d.Upload(fs)
        try await parallel.Scan(f)
        let got = try await f.Download()
        if d.IsCPU { oracle = got } else { gputest.Equal("Scan f32/\(n) vs cpu", d, got, oracle) }
    }
}

// ---- Sort ----

// What a stable sort promises, checked in one pass rather than against a
// host sort: the keys ascend; the values are the original positions,
// each once; each key is the one that started at its value's position;
// and equal keys keep the order they came in.
func sortProblem(_ order: [uint64], _ keys: [uint64], _ values: [uint32]) -> string {
    let n = order.count
    if keys.count != n || values.count != n { return "counts differ" }
    var seen = [bool](repeating: false, count: n)
    var i = 0
    while i < n {
        let v = Int(values[i])
        if v < 0 || v >= n || seen[v] { return "values are not a permutation at \(i)" }
        seen[v] = true
        if keys[i] != order[v] { return "key \(i) is not the key that started at \(v)" }
        if i > 0 {
            if keys[i] < keys[i - 1] { return "keys descend at \(i)" }
            if keys[i] == keys[i - 1] && values[i] < values[i - 1] { return "not stable at \(i)" }
        }
        i += 1
    }
    return ""
}

func floatOrder(_ f: float32) -> uint64 {
    let b = f.bitPattern
    return uint64((b & 0x80000000) != 0 ? ~b : (b | 0x80000000))
}

for n in gputest.Sizes {
    let us = uints(n)
    let xs = ints(n)
    var fs = floats(n)
    if n > 4 {
        // Signed zeros, and repeats for stability to show.
        fs[0] = -0.0
        fs[1] = 0.0
        fs[2] = fs[3]
    }
    // Few distinct keys, so that stability is tested hard.
    var few: [uint32] = []
    for i in 0..<n { few.append(us.count > 0 ? us[i] % 7 : 0) }
    var positions: [uint32] = []
    for i in 0..<n { positions.append(uint32(i)) }
    for d in gputest.Devices() {
        let k = try await d.Upload(us), v = try await d.Upload(positions)
        try await parallel.Sort(k, values: v)
        gputest._record("Sort u32/\(n)", d, sortProblem(us.map { uint64($0) }, (try await k.Download()).map { uint64($0) }, try await v.Download()))
        let k1 = try await d.Upload(few), v1 = try await d.Upload(positions)
        try await parallel.Sort(k1, values: v1)
        gputest._record("Sort u32 few keys/\(n)", d, sortProblem(few.map { uint64($0) }, (try await k1.Download()).map { uint64($0) }, try await v1.Download()))
        let k2 = try await d.Upload(xs), v2 = try await d.Upload(positions)
        try await parallel.Sort(k2, values: v2)
        gputest._record("Sort i32/\(n)", d, sortProblem(xs.map { uint64(uint32(bitPattern: $0) ^ 0x80000000) }, (try await k2.Download()).map { uint64(uint32(bitPattern: $0) ^ 0x80000000) }, try await v2.Download()))
        let k3 = try await d.Upload(fs), v3 = try await d.Upload(positions)
        try await parallel.Sort(k3, values: v3)
        gputest._record("Sort f32/\(n)", d, sortProblem(fs.map { floatOrder($0) }, (try await k3.Download()).map { floatOrder($0) }, try await v3.Download()))
        let k4 = try await d.Upload(us)
        try await parallel.Sort(k4)
        let got = try await k4.Download()
        var ok = got.count == n
        var j = 1
        while ok && j < n { if got[j] < got[j - 1] { ok = false }; j += 1 }
        gputest._record("Sort u32 keys only/\(n)", d, ok ? "" : "keys are not in order")
    }
}

// ---- Count, Select, SelectIndices ----

func keeps(_ x: float64, _ w: parallel.Where) -> bool {
    switch w {
    case .Less(let v): return x < v
    case .LessEqual(let v): return x <= v
    case .Greater(let v): return x > v
    case .GreaterEqual(let v): return x >= v
    case .Equal(let v): return x == v
    case .NotEqual(let v): return x != v
    }
}

for n in gputest.Sizes {
    let fs = floats(n)
    var xs: [int32] = []
    for _ in 0..<n { xs.append(int32(bitPattern: rng.Uint32() % 21) - 10) }
    let us = xs.map { uint32(bitPattern: $0 + 10) }
    for w in [parallel.Where.Less(0.25), .GreaterEqual(-0.5), .Equal(3), .NotEqual(0), .LessEqual(-11), .Greater(1e12), .Equal(2.5)] {
        var wantF: [float32] = [], wantI: [int32] = [], wantU: [uint32] = [], wantAt: [uint32] = []
        for i in 0..<n {
            if keeps(float64(fs[i]), w) { wantF.append(fs[i]) }
            if keeps(float64(xs[i]), w) { wantI.append(xs[i]); wantAt.append(uint32(i)) }
        }
        wantU = us.filter { keeps(float64($0), w) }
        for d in gputest.Devices() {
            let fb = try await d.Upload(fs), ib = try await d.Upload(xs), ub = try await d.Upload(us)
            gputest.Equal("Select f32 \(w)/\(n)", d, try await parallel.Select(fb, where: w).Download(), wantF)
            gputest.Equal("Select i32 \(w)/\(n)", d, try await parallel.Select(ib, where: w).Download(), wantI)
            gputest.Equal("Select u32 \(w)/\(n)", d, try await parallel.Select(ub, where: w).Download(), wantU)
            gputest.Equal("SelectIndices i32 \(w)/\(n)", d, try await parallel.SelectIndices(ib, where: w).Download(), wantAt)
            gputest.Equal("Count f32 \(w)/\(n)", d, [try await parallel.Count(fb, where: w)], [wantF.count])
        }
    }
}

// ---- Gather, Scatter, ScatterAdd ----

for n in gputest.Sizes {
    let fs = floats(n), xs = ints(n)
    var picks: [uint32] = [], reverse: [uint32] = [], buckets: [uint32] = []
    for i in 0..<n {
        picks.append(rng.Uint32() % uint32(n))
        reverse.append(uint32(n - 1 - i))
        buckets.append(rng.Uint32() % 17)
    }
    var wantGather: [int32] = [], wantScatter = [float32](repeating: 0, count: n)
    var wantAddI = [int32](repeating: 0, count: 17), wantAddF = [float64](repeating: 0, count: 17)
    for i in 0..<n {
        wantGather.append(xs[Int(picks[i])])
        wantScatter[n - 1 - i] = fs[i]
        wantAddI[Int(buckets[i])] = wantAddI[Int(buckets[i])] &+ xs[i]
        wantAddF[Int(buckets[i])] += float64(fs[i])
    }
    for d in gputest.Devices() {
        let fb = try await d.Upload(fs), ib = try await d.Upload(xs)
        let pb = try await d.Upload(picks), rb = try await d.Upload(reverse), bb = try await d.Upload(buckets)
        gputest.Equal("Gather i32/\(n)", d, try await parallel.Gather(ib, pb).Download(), wantGather)
        let dst = try await d.CreateBuffer(of: float32.self, count: n)
        try await parallel.Scatter(fb, rb, into: dst)
        gputest.Equal("Scatter f32/\(n)", d, try await dst.Download(), wantScatter)
        let sumsI = try await d.Upload([int32](repeating: 0, count: 17))
        try await parallel.ScatterAdd(ib, bb, into: sumsI)
        gputest.Equal("ScatterAdd i32/\(n)", d, try await sumsI.Download(), wantAddI)
        let sumsF = try await d.Upload([float32](repeating: 0, count: 17))
        try await parallel.ScatterAdd(fb, bb, into: sumsF)
        gputest.Close("ScatterAdd f32/\(n)", d, try await sumsF.Download(), wantAddF.map { float32($0) }, ulps: 64 + n)
    }
}

// ---- Histogram ----

for n in gputest.Sizes {
    for bins in [1, 10, 4096, 10000] {
        var xs: [uint32] = []
        var want = [uint32](repeating: 0, count: bins)
        for _ in 0..<n {
            let v = rng.Uint32() % uint32(bins + 3)
            xs.append(v)
            if v < uint32(bins) { want[Int(v)] += 1 }
        }
        for d in gputest.Devices() {
            let b = try await d.Upload(xs)
            gputest.Equal("Histogram \(bins)/\(n)", d, try await parallel.Histogram(b, bins: bins).Download(), want)
        }
    }
}

// ---- TopK ----

// The host's top k: stable, largest first, ties to the earlier position.
func hostTop(_ xs: [float32], _ k: int) -> [uint32] {
    var top: [uint32] = []
    for i in 0..<xs.count {
        var at = top.count
        while at > 0 && xs[Int(top[at - 1])] < xs[i] { at -= 1 }
        if at < k {
            top.insert(uint32(i), at: at)
            if top.count > k { top.removeLast() }
        }
    }
    return top
}

for n in gputest.Sizes {
    var fs = floats(n)
    for i in 0..<n { fs[i] = float32(int(fs[i] * 50)) }   // repeats, to see ties
    for k in [1, 5, 32] {
        let want = hostTop(fs, min(k, n))
        for d in gputest.Devices() {
            let b = try await d.Upload(fs)
            let top = try await parallel.TopK(b, k: k)
            gputest.Equal("TopK indices k=\(k)/\(n)", d, try await top.indices.Download(), want)
            gputest.Equal("TopK values k=\(k)/\(n)", d, try await top.values.Download(), want.map { fs[Int($0)] })
        }
    }
}

// ---- float16 and bfloat16 ----

// The half floats through the same generic functions. A half's sum rounds
// at every step, so a sum or scan is checked against the CPU device's,
// which adds in the same tree; min, max, select and sort are exact; and a
// scatter-add adds small whole numbers, exact in any order.
func halves<T: dtype.Number>(_ name: string, _ t: T.Type) async throws {
    for n in gputest.Sizes {
        let xs = floats(n).map { T.FromFloat64(float64($0) * 8 - 4) }
        var lo = T.Highest(), hi = T.Lowest()
        for v in xs { lo = min(lo, v); hi = max(hi, v) }
        var positions: [uint32] = []
        for i in 0..<n { positions.append(uint32(i)) }
        var sum: [T] = [], scan: [T] = []
        for d in gputest.Devices() {
            let b = try await d.Upload(xs)
            gputest.Equal("Reduce \(name) Min Max/\(n)", d, [try await parallel.Reduce(b, .Min), try await parallel.Reduce(b, .Max)], [lo, hi])
            let s = [try await parallel.Reduce(b, .Sum)]
            let c = try await d.Upload(xs)
            try await parallel.Scan(c)
            let got = try await c.Download()
            if d.IsCPU { sum = s; scan = got } else {
                gputest.Equal("Reduce \(name) Sum/\(n) vs cpu", d, s, sum)
                gputest.Equal("Scan \(name)/\(n) vs cpu", d, got, scan)
            }
            let w = parallel.Where.Less(0.5)
            gputest.Equal("Select \(name)/\(n)", d, try await parallel.Select(b, where: w).Download(), xs.filter { T.ToFloat64($0) < 0.5 })
            let k = try await d.Upload(xs), v = try await d.Upload(positions)
            try await parallel.Sort(k, values: v)
            gputest._record("Sort \(name)/\(n)", d, sortProblem(xs.map { uint64(T.OrderKey($0)) }, (try await k.Download()).map { uint64(T.OrderKey($0)) }, try await v.Download()))
        }
        if n > 1000 { continue }
        var small: [T] = [], buckets: [uint32] = []
        var want = [T](repeating: T.FromFloat64(0), count: 17)
        for _ in 0..<n {
            let v = T.FromFloat64(float64(rng.Uint32() % 4)), at = rng.Uint32() % 17
            small.append(v); buckets.append(at)
            want[Int(at)] = want[Int(at)] + v
        }
        for d in gputest.Devices() {
            let sums = try await d.Upload([T](repeating: T.FromFloat64(0), count: 17))
            try await parallel.ScatterAdd(try await d.Upload(small), try await d.Upload(buckets), into: sums)
            gputest.Equal("ScatterAdd \(name)/\(n)", d, try await sums.Download(), want)
        }
    }
}

try await halves("f16", float16.self)
try await halves("bf16", bfloat16.self)

gputest.Done()
