// Reduce: a buffer to one value, a workgroup at a time, then the groups'
// results the same way, until one is left. The order is fixed by the
// count, so the result is the same on every device.
import "gpu"
import "gpu/dtype"

/// Reduction is what Reduce combines elements with.
public enum Reduction {
    case Sum
    case Min
    case Max

    public var _code: int32 {
        switch self {
        case .Sum: return 0
        case .Min: return 1
        case .Max: return 2
        }
    }
}

/// _identity is op's identity for T: the value that changes nothing.
@inlinable public func _identity<T: dtype.Number>(_ op: int32, _ of: T.Type) -> T {
    if op == 1 { return T.Highest() }
    if op == 2 { return T.Lowest() }
    return 0
}

@inlinable public func _reduceKernel<T: dtype.Number>(_ x: gpu.Span<T>, _ out: gpu.MutableSpan<T>, _ n: int, _ op: int32) kernel {
    let i = gpu.Index.x
    var v: T = 0
    if op == 1 {
        v = T.Highest()
    } else if op == 2 {
        v = T.Lowest()
    }
    if i < n {
        v = x[i]
    }
    let r = _groupReduce(v, op)
    if GroupRank() == 0 {
        out[gpu.GroupIndex.x] = r
    }
}

/// Reduce combines every element of b with op: its sum (wrapping, for an
/// integer type), least or greatest element. An empty buffer's is op's
/// identity.
@inlinable public func Reduce<T: dtype.Number>(_ b: gpu.Buffer<T>, _ op: Reduction) async throws -> T {
    var n = b.count
    if n == 0 {
        return _identity(op._code, T.self)
    }
    var cur = b
    while true {
        let groups = (n + 255) / 256
        let out = try await b.Device.CreateBuffer(of: T.self, count: groups)
        try await _reduceKernel.Launch(cur, out, n, op._code, over: groups * 256, workgroup: 256)
        if groups == 1 {
            return try await out.Download()[0]
        }
        cur = out
        n = groups
    }
}
