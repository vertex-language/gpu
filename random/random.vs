// Counter-based random numbers: Philox4x32-10 (Salmon et al., "Parallel
// Random Numbers: As Easy as 1, 2, 3", SC 2011), the generator cuRAND and
// JAX use. A stream is a pure function of a key and a counter: element i
// of a key's stream is the same on every device, in every run, whatever
// order the work-items run in. That is what makes a training run
// reproducible, and what lets the CPU device check a GPU bit for bit.
//
// Keys follow JAX: a Key is made from a seed and split into independent
// keys rather than advanced, so that no two users of randomness ever
// share a stream by accident.
import (
    "gpu"
    "math"
)

/// Key names one stream of random numbers. It is two words, a plain
/// value, so a kernel can take one (as its two words) and make it again.
public struct Key {
    public let lo: uint32
    public let hi: uint32

    @inlinable public init(lo: uint32, hi: uint32) {
        self.lo = lo
        self.hi = hi
    }

    /// A key from a seed. Different seeds give unrelated streams.
    @inlinable public init(seed: uint64) {
        self.lo = uint32(truncatingIfNeeded: seed)
        self.hi = uint32(truncatingIfNeeded: seed >> 32)
    }
}

/// Split makes two keys whose streams are unrelated to each other and to
/// key's own. Splitting again gives the same two.
@inlinable public func Split(_ key: Key) -> (Key, Key) {
    let b = Block(key, 0, 0, 0, 0xFFFFFFFF)
    return (Key(lo: b.0, hi: b.1), Key(lo: b.2, hi: b.3))
}

/// Fold makes the key for one of many users of key: a layer, a step, a
/// device. Folding in the same data gives the same key.
@inlinable public func Fold(_ key: Key, _ data: uint32) -> Key {
    let b = Block(key, data, 0, 0, 0xFFFFFFFE)
    return Key(lo: b.0, hi: b.1)
}

/// _mulhilo is a 32-by-32-bit product's high and low words.
@inlinable public func _mulhilo(_ a: uint32, _ b: uint32) -> (uint32, uint32) {
    let p = uint64(a) &* uint64(b)
    return (uint32(truncatingIfNeeded: p >> 32), uint32(truncatingIfNeeded: p))
}

/// Block is Philox4x32-10 itself: four random words for a key and a
/// four-word counter. Stream elements use counters with c2 and c3 zero;
/// Split and Fold use c3 = 0xFFFFFFFF and 0xFFFFFFFE, which no element
/// reaches.
@inlinable public func Block(_ key: Key, _ c0: uint32, _ c1: uint32, _ c2: uint32, _ c3: uint32) -> (uint32, uint32, uint32, uint32) {
    var x0 = c0, x1 = c1, x2 = c2, x3 = c3
    var k0 = key.lo, k1 = key.hi
    var round = 0
    while round < 10 {
        let a = _mulhilo(0xD2511F53, x0)
        let b = _mulhilo(0xCD9E8D57, x2)
        let y0 = b.0 ^ x1 ^ k0
        let y2 = a.0 ^ x3 ^ k1
        x0 = y0
        x1 = b.1
        x2 = y2
        x3 = a.1
        k0 = k0 &+ 0x9E3779B9
        k1 = k1 &+ 0xBB67AE85
        round += 1
    }
    return (x0, x1, x2, x3)
}

/// Bits is element i of key's stream, as four random words.
@inlinable public func Bits(_ key: Key, _ i: uint64) -> (uint32, uint32, uint32, uint32) {
    return Block(key, uint32(truncatingIfNeeded: i), uint32(truncatingIfNeeded: i >> 32), 0, 0)
}

/// Uint32 is element i of key's stream as one random word.
@inlinable public func Uint32(_ key: Key, _ i: uint64) -> uint32 {
    return Bits(key, i).0
}

/// Uniform is element i of key's stream as a float32 in [0, 1): 24 random
/// bits, so every value it takes is equally likely.
@inlinable public func Uniform(_ key: Key, _ i: uint64) -> float32 {
    return float32(Uint32(key, i) >> 8) * 5.9604645e-08
}

/// Below is element i of key's stream as an integer in [0, bound), by
/// Lemire's multiply-and-shift. For a bound far below 2^32 the bias is
/// negligible (at most bound / 2^32).
@inlinable public func Below(_ key: Key, _ i: uint64, _ bound: uint32) -> uint32 {
    return _mulhilo(Uint32(key, i), bound).0
}

/// Bernoulli is element i of key's stream as true with probability p.
@inlinable public func Bernoulli(_ key: Key, _ i: uint64, _ p: float32) -> bool {
    return Uniform(key, i) < p
}

func _fillUniform(_ out: gpu.MutableSpan<float32>, _ lo: uint32, _ hi: uint32, _ start: uint64) kernel {
    let i = gpu.Index.x
    if i < out.count {
        out[i] = Uniform(Key(lo: lo, hi: hi), start + uint64(i))
    }
}

func _fillBits(_ out: gpu.MutableSpan<uint32>, _ lo: uint32, _ hi: uint32, _ start: uint64) kernel {
    let i = gpu.Index.x
    if i < out.count {
        out[i] = Uint32(Key(lo: lo, hi: hi), start + uint64(i))
    }
}

/// Fill writes elements start, start+1, ... of key's stream into b, as
/// uniform float32s in [0, 1).
public func Fill(_ b: gpu.Buffer<float32>, _ key: Key, start: uint64 = 0) async throws {
    if b.count == 0 { return }
    try await _fillUniform.Launch(b, key.lo, key.hi, start, over: b.count)
}

/// Fill writes elements start, start+1, ... of key's stream into b, as
/// random words.
public func Fill(_ b: gpu.Buffer<uint32>, _ key: Key, start: uint64 = 0) async throws {
    if b.count == 0 { return }
    try await _fillBits.Launch(b, key.lo, key.hi, start, over: b.count)
}

/// Normal is element i of key's stream as a standard normal deviate (mean
/// 0, variance 1), by the Box–Muller transform of the element's first two
/// words.
@inlinable public func Normal(_ key: Key, _ i: uint64) -> float32 {
    let b = Bits(key, i)
    // u in (0, 1]: never zero, whose logarithm is -infinity.
    let u = (float32(b.0 >> 8) + 1) * 5.9604645e-08
    let v = float32(b.1 >> 8) * 5.9604645e-08
    return math.Sqrt(-2 * math.Log(u)) * math.Cos(6.2831855 * v)
}

func _fillNormal(_ out: gpu.MutableSpan<float32>, _ lo: uint32, _ hi: uint32, _ start: uint64) kernel {
    let i = gpu.Index.x
    if i < out.count {
        out[i] = Normal(Key(lo: lo, hi: hi), start + uint64(i))
    }
}

/// FillNormal writes elements start, start+1, ... of key's stream into b,
/// as standard normal deviates.
public func FillNormal(_ b: gpu.Buffer<float32>, _ key: Key, start: uint64 = 0) async throws {
    if b.count == 0 { return }
    try await _fillNormal.Launch(b, key.lo, key.hi, start, over: b.count)
}
