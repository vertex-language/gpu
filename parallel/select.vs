// Select: the elements of a buffer that satisfy a condition, in the order
// they came in -- stream compaction. Each element is tested into a flag,
// the flags are scanned into where each kept element goes, and the kept
// elements are written there.
import (
    "gpu"
    "gpu/dtype"
)

/// Where is the condition Select, SelectIndices and Count keep an element
/// by: how it compares with a value. For integer elements the comparison
/// is exact -- Less(0.5) keeps 0 and below, Less(-1) keeps no uint32 --
/// and for float ones the value is rounded to the element type first.
public enum Where {
    case Less(float64)
    case LessEqual(float64)
    case Greater(float64)
    case GreaterEqual(float64)
    case Equal(float64)
    case NotEqual(float64)

    public var _code: int32 {
        switch self {
        case .Less: return 0
        case .LessEqual: return 1
        case .Greater: return 2
        case .GreaterEqual: return 3
        case .Equal: return 4
        case .NotEqual: return 5
        }
    }

    public var _value: float64 {
        switch self {
        case .Less(let v): return v
        case .LessEqual(let v): return v
        case .Greater(let v): return v
        case .GreaterEqual(let v): return v
        case .Equal(let v): return v
        case .NotEqual(let v): return v
        }
    }

    /// _bounds is the comparison for an integer type whose values run
    /// from lo to hi: an operator and an integral value in that range, or
    /// 6 (always) or 7 (never) where the value is outside it.
    public func _bounds(_ lo: float64, _ hi: float64) -> (int32, float64) {
        let v = _value
        let up = v.rounded(.up)
        let down = v.rounded(.down)
        switch self {
        case .Less:
            return up <= lo ? (7, 0) : up > hi ? (6, 0) : (0, up)
        case .LessEqual:
            return down < lo ? (7, 0) : down >= hi ? (6, 0) : (1, down)
        case .Greater:
            return down < lo ? (6, 0) : down >= hi ? (7, 0) : (2, down)
        case .GreaterEqual:
            return up <= lo ? (6, 0) : up > hi ? (7, 0) : (3, up)
        case .Equal:
            return v != down || v < lo || v > hi ? (7, 0) : (4, v)
        case .NotEqual:
            return v != down || v < lo || v > hi ? (6, 0) : (5, v)
        }
    }
}

/// _op is w as an operator and a value of T.
@inlinable public func _op<T: dtype.Number>(_ w: Where, _ of: T.Type) -> (int32, T) {
    if T.IsInteger() {
        let (op, v) = w._bounds(T.ToFloat64(T.Lowest()), T.ToFloat64(T.Highest()))
        return (op, T.FromFloat64(v))
    }
    return (w._code, T.FromFloat64(w._value))
}

@inlinable public func _keeps<T: dtype.Number>(_ x: T, _ op: int32, _ v: T) -> bool {
    if op == 0 { return x < v }
    if op == 1 { return x <= v }
    if op == 2 { return x > v }
    if op == 3 { return x >= v }
    if op == 4 { return x == v }
    if op == 5 { return x != v }
    return op == 6
}

@inlinable public func _flag<T: dtype.Number>(_ x: gpu.Span<T>, _ flags: gpu.MutableSpan<uint32>, _ op: int32, _ v: T) kernel {
    let i = gpu.Index.x
    if i < x.count {
        flags[i] = _keeps(x[i], op, v) ? 1 : 0
    }
}

@inlinable public func _compact<T: dtype.Number>(_ x: gpu.Span<T>, _ at: gpu.Span<uint32>, _ out: gpu.MutableSpan<T>, _ indices: gpu.MutableSpan<uint32>, _ op: int32, _ v: T, _ values: bool) kernel {
    let i = gpu.Index.x
    if i < x.count && _keeps(x[i], op, v) {
        if values {
            out[int(at[i])] = x[i]
        } else {
            indices[int(at[i])] = uint32(i)
        }
    }
}

/// _where is where each kept element of b goes, and how many there are.
@inlinable public func _where<T: dtype.Number>(_ b: gpu.Buffer<T>, _ op: int32, _ v: T) async throws -> (gpu.Buffer<uint32>, int) {
    let flags = try await b.Device.CreateBuffer(of: uint32.self, count: b.count)
    try await _flag.Launch(b, flags, op, v, over: b.count)
    let kept = int(try await Reduce(flags, .Sum))
    try await Scan(flags, exclusive: true)
    return (flags, kept)
}

/// Count is how many elements of b satisfy w.
@inlinable public func Count<T: dtype.Number>(_ b: gpu.Buffer<T>, where w: Where) async throws -> int {
    if b.count == 0 {
        return 0
    }
    let (op, v) = _op(w, T.self)
    let flags = try await b.Device.CreateBuffer(of: uint32.self, count: b.count)
    try await _flag.Launch(b, flags, op, v, over: b.count)
    return int(try await Reduce(flags, .Sum))
}

/// Select is the elements of b that satisfy w, in their order in b.
@inlinable public func Select<T: dtype.Number>(_ b: gpu.Buffer<T>, where w: Where) async throws -> gpu.Buffer<T> {
    if b.count == 0 {
        return try await b.Device.CreateBuffer(of: T.self, count: 0)
    }
    let (op, v) = _op(w, T.self)
    let (at, kept) = try await _where(b, op, v)
    let out = try await b.Device.CreateBuffer(of: T.self, count: kept)
    if kept > 0 {
        try await _compact.Launch(b, at, out, at, op, v, true, over: b.count)
    }
    return out
}

/// SelectIndices is where in b the elements that satisfy w are, in
/// ascending order.
@inlinable public func SelectIndices<T: dtype.Number>(_ b: gpu.Buffer<T>, where w: Where) async throws -> gpu.Buffer<uint32> {
    if b.count == 0 {
        return try await b.Device.CreateBuffer(of: uint32.self, count: 0)
    }
    let (op, v) = _op(w, T.self)
    let (at, kept) = try await _where(b, op, v)
    let out = try await b.Device.CreateBuffer(of: uint32.self, count: kept)
    if kept > 0 {
        try await _compact.Launch(b, at, b, out, op, v, false, over: b.count)
    }
    return out
}
