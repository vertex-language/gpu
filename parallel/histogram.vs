// Histogram: how many of a buffer's values fall in each bin. Each
// workgroup counts into shared storage with atomics, then adds its counts
// into the result; with more bins than shared storage holds, the counts go
// straight to the result. Counting is exact, so the order the atomics land
// in makes no difference.
import "gpu"

/// SharedBins is the most bins Histogram counts in shared storage.
public let SharedBins = 4096

func _histogramShared(_ x: gpu.Span<uint32>, _ counts: gpu.MutableSpan<uint32>, _ bins: int) kernel {
    let local = gpu.Shared<uint32>(count: 4096)
    let me = GroupRank()
    let group = GroupCount()
    var b = me
    while b < bins {
        local[b] = 0
        b += group
    }
    gpu.Barrier()
    let i = gpu.Index.x
    if i < x.count {
        let v = x[i]
        if v < uint32(bins) {
            _ = gpu.Atomic.Add(local.Address(int(v)), uint32(1))
        }
    }
    gpu.Barrier()
    b = me
    while b < bins {
        let c = local[b]
        if c != 0 {
            _ = gpu.Atomic.Add(counts.Address(b), c)
        }
        b += group
    }
}

func _histogramGlobal(_ x: gpu.Span<uint32>, _ counts: gpu.MutableSpan<uint32>, _ bins: int) kernel {
    let i = gpu.Index.x
    if i < x.count {
        let v = x[i]
        if v < uint32(bins) {
            _ = gpu.Atomic.Add(counts.Address(int(v)), uint32(1))
        }
    }
}

/// Histogram counts the values of b in bins 0..<bins: counts[v] is how many
/// elements equal v. A value of bins or more is not counted.
public func Histogram(_ b: gpu.Buffer<uint32>, bins: int) async throws -> gpu.Buffer<uint32> {
    let counts = try await b.Device.CreateBuffer(of: uint32.self, count: bins)
    try await counts.Fill(0)
    if b.count == 0 || bins == 0 {
        return counts
    }
    let groups = (b.count + 255) / 256
    if bins <= 4096 {
        try await _histogramShared.Launch(b, counts, bins, over: groups * 256, workgroup: 256)
    } else {
        try await _histogramGlobal.Launch(b, counts, bins, over: b.count)
    }
    return counts
}
