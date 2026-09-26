// Scan: prefix sums of a buffer, in place. Each workgroup scans its own
// elements and writes its total; the totals are scanned (recursively, the
// same way) into each group's offset; the offsets are added in. The order
// of every addition is fixed by the count, so the result is the same on
// every device.
import (
    "gpu"
    "gpu/dtype"
)

@inlinable public func _scanBlocks<T: dtype.Number>(_ x: gpu.MutableSpan<T>, _ sums: gpu.MutableSpan<T>, _ n: int, _ exclusive: bool) kernel {
    let i = gpu.Index.x
    var v: T = 0
    if i < n {
        v = x[i]
    }
    let r = _groupScan(v)
    if i < n {
        if exclusive {
            x[i] = r.exclusive
        } else {
            x[i] = r.inclusive
        }
    }
    if GroupRank() == 0 {
        sums[gpu.GroupIndex.x] = r.total
    }
}

@inlinable public func _addOffsets<T: dtype.Number>(_ x: gpu.MutableSpan<T>, _ offsets: gpu.Span<T>, _ n: int) kernel {
    let i = gpu.Index.x
    if i < n {
        x[i] = T.Plus(offsets[gpu.GroupIndex.x], x[i])
    }
}

/// Scan replaces each element of b with the sum of the elements up to
/// it: including itself (inclusive, the default), or before it
/// (exclusive, whose first element is zero). An integer sum wraps.
@inlinable public func Scan<T: dtype.Number>(_ b: gpu.Buffer<T>, exclusive: bool = false) async throws {
    let n = b.count
    if n == 0 {
        return
    }
    let groups = (n + 255) / 256
    let sums = try await b.Device.CreateBuffer(of: T.self, count: groups)
    try await _scanBlocks.Launch(b, sums, n, exclusive, over: groups * 256, workgroup: 256)
    if groups == 1 {
        return
    }
    try await Scan(sums, exclusive: true)
    try await _addOffsets.Launch(b, sums, n, over: groups * 256, workgroup: 256)
}
