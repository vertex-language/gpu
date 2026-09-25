// gpu/random: Philox against its published answers, and every device
// against the host bit for bit.
import "gpu"
import "gpu/random"
import "gpu/gputest"

let host = gputest.Devices()[0]

// Random123's known-answer vectors for philox4x32_10.
func kat(_ what: string, _ key: random.Key, _ c: (uint32, uint32, uint32, uint32), _ want: [uint32]) {
    let b = random.Block(key, c.0, c.1, c.2, c.3)
    gputest.Equal("Philox KAT \(what)", host, [b.0, b.1, b.2, b.3], want)
}
kat("zero", random.Key(lo: 0, hi: 0), (0, 0, 0, 0), [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8])
kat("ones", random.Key(lo: 0xffffffff, hi: 0xffffffff), (0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff),
    [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd])
kat("pi", random.Key(lo: 0xa4093822, hi: 0x299f31d0), (0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344),
    [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1])

// A kernel of this module calling the device functions.
func draw(_ out: gpu.MutableSpan<uint32>, _ lo: uint32, _ hi: uint32) kernel {
    let i = gpu.Index.x
    let k = random.Fold(random.Key(lo: lo, hi: hi), 7)
    out[i] = random.Below(k, uint64(i), 1000) + (random.Bernoulli(k, uint64(i) + 1000000, 0.5) ? 1000 : 0)
}

let key = random.Key(seed: 42)
let (a, b) = random.Split(key)
gputest.Equal("Split differs", host, [a.lo == b.lo && a.hi == b.hi ? 1 : 0, a.lo == key.lo ? 1 : 0], [0, 0])

for n in [1, 1000, 100000] {
    var wantF: [float32] = [], wantU: [uint32] = [], wantD: [uint32] = []
    let k7 = random.Fold(key, 7)
    for i in 0..<n {
        wantF.append(random.Uniform(key, uint64(i) + 5))
        wantU.append(random.Uint32(key, uint64(i)))
        wantD.append(random.Below(k7, uint64(i), 1000) + (random.Bernoulli(k7, uint64(i) + 1000000, 0.5) ? 1000 : 0))
    }
    for d in gputest.Devices() {
        let f = try await d.CreateBuffer(of: float32.self, count: n)
        try await random.Fill(f, key, start: 5)
        gputest.Equal("Fill f32/\(n)", d, try await f.Download(), wantF)
        let u = try await d.CreateBuffer(of: uint32.self, count: n)
        try await random.Fill(u, key)
        gputest.Equal("Fill u32/\(n)", d, try await u.Download(), wantU)
        let o = try await d.CreateBuffer(of: uint32.self, count: n)
        try await draw.Launch(o, key.lo, key.hi, over: n)
        gputest.Equal("device functions/\(n)", d, try await o.Download(), wantD)
    }
    // Uniform's mean and range, roughly.
    var sum: float64 = 0, lo: float32 = 1, hi: float32 = 0
    for v in wantF { sum += float64(v); lo = min(lo, v); hi = max(hi, v) }
    if n == 100000 {
        let mean = sum / float64(n)
        gputest._record("Uniform mean", host, mean > 0.49 && mean < 0.51 && lo >= 0 && hi < 1 ? "" : "mean \(mean), range \(lo)...\(hi)")
    }
}

// Normal: every device gives the host's deviates; their mean and
// variance are those of a standard normal.
do {
    let n = 100000
    var want: [float32] = []
    for i in 0..<n { want.append(random.Normal(key, uint64(i))) }
    for d in gputest.Devices() {
        let b = try await d.CreateBuffer(of: float32.self, count: n)
        try await random.FillNormal(b, key)
        gputest.Equal("FillNormal/\(n)", d, try await b.Download(), want)
    }
    var sum: float64 = 0, sq: float64 = 0
    for v in want { sum += float64(v); sq += float64(v) * float64(v) }
    let mean = sum / float64(n), variance = sq / float64(n) - mean * mean
    gputest._record("Normal moments", host, mean.magnitude < 0.01 && (variance - 1).magnitude < 0.02 ? "" : "mean \(mean), variance \(variance)")
}

gputest.Done()
