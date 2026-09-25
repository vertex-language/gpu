// The oracle harness (README: gpu/gputest). A check runs on every device
// and compares each device's answer with a reference: the host's, or the
// CPU device's -- the same .vs source compiled for the host, which cannot
// drift from the kernel it checks. Where both paths round once, the
// comparison is exact; where the order of a floating-point sum differs, a
// check states its tolerance in ULPs.
import "gpu"

// The tally. Only non-generic functions touch it: Equal is generic, so it
// is specialized in the calling module, which cannot yet read or assign
// another module's variables (vsc_TODO.md). It calls _record instead.
var _checks = 0
var _failures = 0

/// _record counts one check, and prints why it failed if it did: why is
/// empty for a pass.
public func _record(_ what: string, _ d: gpu.Device, _ why: string) {
    _checks += 1
    if why != "" {
        _failures += 1
        print("FAIL \(what) on \(d.Name): \(why)")
    }
}

/// Devices is the CPU device first, then every other device there is: what
/// a check runs on. The CPU device's answer is the oracle for the rest.
public func Devices() -> [gpu.Device] {
    var out: [gpu.Device] = [gpu.CPU()]
    for d in gpu.Devices() {
        if !d.IsCPU {
            out.append(d)
        }
    }
    return out
}

/// Equal checks that got is want, element for element.
public func Equal<T: Equatable>(_ what: string, _ d: gpu.Device, _ got: [T], _ want: [T]) {
    _record(what, d, _mismatch(got, want))
}

/// _mismatch says how got differs from want, or is empty if it does not.
func _mismatch<T: Equatable>(_ got: [T], _ want: [T]) -> string {
    if got.count != want.count {
        return "got \(got.count) elements, want \(want.count)"
    }
    var i = 0
    while i < got.count {
        if got[i] != want[i] {
            return "element \(i) is \(got[i]), want \(want[i])"
        }
        i += 1
    }
    return ""
}

/// Close checks that each element of got is within ulps units in the last
/// place of want: for sums whose order differs from the reference's.
public func Close(_ what: string, _ d: gpu.Device, _ got: [float32], _ want: [float32], ulps: int) {
    if got.count != want.count {
        _record(what, d, "got \(got.count) elements, want \(want.count)")
        return
    }
    var i = 0
    while i < got.count {
        if _ulpDistance(got[i], want[i]) > ulps {
            _record(what, d, "element \(i) is \(got[i]), want \(want[i]) within \(ulps) ULPs")
            return
        }
        i += 1
    }
    _record(what, d, "")
}

/// Near checks that each element of got is within bound[i] of want[i]:
/// for a result whose error is bounded by what it was computed from rather
/// than by its own size, as a dot product's is by k·ε·Σ|a·b| however much
/// its terms cancel.
public func Near(_ what: string, _ d: gpu.Device, _ got: [float32], _ want: [float64], bound: [float64]) {
    if got.count != want.count || bound.count != want.count {
        _record(what, d, "got \(got.count) elements, want \(want.count)")
        return
    }
    var i = 0
    while i < got.count {
        let err = abs(float64(got[i]) - want[i])
        if !(err <= bound[i]) {
            _record(what, d, "element \(i) is \(got[i]), want \(want[i]) within \(bound[i])")
            return
        }
        i += 1
    }
    _record(what, d, "")
}

/// _ulpDistance is how many representable float32s apart a and b are; NaN
/// is equal only to NaN.
func _ulpDistance(_ a: float32, _ b: float32) -> int {
    if a.isNaN || b.isNaN {
        return a.isNaN && b.isNaN ? 0 : int.max
    }
    let x = _ordered(a)
    let y = _ordered(b)
    return x > y ? x - y : y - x
}

/// _ordered maps a float32's bits onto the integers in the float order.
func _ordered(_ f: float32) -> int {
    let b = int(f.bitPattern)
    return b >= 0x80000000 ? 0x80000000 - b : b
}

/// Done prints how many checks ran, and stops the program with a failure
/// if any of them failed.
public func Done() {
    print("\(_checks - _failures) of \(_checks) checks passed")
    if _failures > 0 {
        fatalError("\(_failures) checks failed")
    }
}

/// Random is a reproducible stream of test inputs: xorshift64*, the same
/// sequence on every run.
public struct Random {
    var _s: uint64

    public init(seed: uint64) {
        _s = seed == 0 ? 0x9E3779B97F4A7C15 : seed
    }

    public mutating func Next() -> uint64 {
        _s ^= _s >> 12
        _s ^= _s << 25
        _s ^= _s >> 27
        return _s &* 0x2545F4914F6CDD1D
    }

    public mutating func Uint32() -> uint32 { return uint32(truncatingIfNeeded: Next() >> 32) }

    public mutating func Int32() -> int32 { return int32(bitPattern: Uint32()) }

    /// Float32 is uniform in [-1, 1).
    public mutating func Float32() -> float32 {
        return float32(int32(bitPattern: Uint32()) >> 8) / 8388608
    }
}

/// Sizes are the element counts a check should try: the empty case, one,
/// either side of a group and of a group of groups, and a large one.
public let Sizes = [0, 1, 2, 255, 256, 257, 1000, 65535, 65536, 65537, 300000]
