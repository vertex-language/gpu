// Sort: a stable least-significant-digit radix sort, four bits a pass.
//
// Each pass, every workgroup counts how many of its keys have each digit
// (sixteen GroupSums); the counts, laid out digit-major, are scanned into
// where each group's keys of each digit go; then each group places its
// keys: the digit's offset plus how many of the group's keys with that
// digit came before (a GroupExclusiveScan). Keys with equal digits keep
// their order, which is what makes the next pass's work stand on this
// one's, and the whole sort stable.
//
// Any Number is sorted as the uint32s whose order is its order
// (dtype.Number.OrderKey): for a float, IEEE 754's total order.
import (
    "gpu"
    "gpu/dtype"
)

func _radixCount(_ keys: gpu.Span<uint32>, _ counts: gpu.MutableSpan<uint32>, _ n: int, _ shift: uint32, _ groups: int) kernel {
    let hist = gpu.Shared<uint32>(count: 16)
    let me = GroupRank()
    if me < 16 {
        hist[me] = 0
    }
    gpu.Barrier()
    let i = gpu.Index.x
    if i < n {
        _ = gpu.Atomic.Add(hist.Address(int((keys[i] >> shift) & 15)), uint32(1))
    }
    gpu.Barrier()
    if me < 16 {
        counts[me * groups + gpu.GroupIndex.x] = hist[me]
    }
}

func _radixPlace(_ keys: gpu.Span<uint32>, _ values: gpu.Span<uint32>, _ outKeys: gpu.MutableSpan<uint32>, _ outValues: gpu.MutableSpan<uint32>, _ offsets: gpu.Span<uint32>, _ n: int, _ shift: uint32, _ groups: int, _ withValues: bool) kernel {
    // The group's keys, sorted among themselves by digit and stably, a
    // bit at a time: each split sends the keys whose bit is 0 ahead of
    // those whose bit is 1, keeping order within each. A work-item follows
    // its own key through the splits in pos. A key past the end has digit
    // 16, whose fifth bit sends it after every real one.
    let slot = gpu.Shared<uint32>(count: 1024)
    let moved = gpu.Shared<uint32>(count: 1024)
    let me = GroupRank()
    let count = GroupCount()
    let i = gpu.Index.x
    var digit: uint32 = 16
    if i < n {
        digit = (keys[i] >> shift) & 15
    }
    let bits: uint32 = (gpu.GroupIndex.x + 1) * count > n ? 5 : 4
    var pos = me
    var bit: uint32 = 0
    while bit < bits {
        // Which bit the key now in each slot has.
        slot[pos] = (digit >> bit) & 1
        gpu.Barrier()
        let b = slot[me]
        let zeros = GroupExclusiveScan(1 - b)
        moved[me] = b == 0 ? zeros.prefix : zeros.total + uint32(me) - zeros.prefix
        gpu.Barrier()
        pos = int(moved[pos])
        gpu.Barrier()
        bit += 1
    }
    // Where each digit starts among the group's sorted keys.
    let starts = gpu.Shared<uint32>(count: 17)
    slot[pos] = digit
    gpu.Barrier()
    let here = slot[me]
    if me == 0 || slot[me - 1] != here {
        starts[int(here)] = uint32(me)
    }
    gpu.Barrier()
    if i < n {
        let to = int(offsets[int(digit) * groups + gpu.GroupIndex.x] + uint32(pos) - starts[int(digit)])
        outKeys[to] = keys[i]
        if withValues {
            outValues[to] = values[i]
        }
    }
}

/// _radixSort sorts keys, and values with them where withValues holds, in
/// place. The generic Sort calls it for every key type.
public func _radixSort(_ keys: gpu.Buffer<uint32>, _ values: gpu.Buffer<uint32>, _ withValues: bool) async throws {
    let n = keys.count
    if n < 2 {
        return
    }
    let d = keys.Device
    let groups = (n + 255) / 256
    let counts = try await d.CreateBuffer(of: uint32.self, count: groups * 16)
    var inKeys = keys
    var inValues = values
    var outKeys = try await d.CreateBuffer(of: uint32.self, count: n)
    var outValues = withValues ? try await d.CreateBuffer(of: uint32.self, count: n) : values
    var shift: uint32 = 0
    while shift < 32 {
        try await _radixCount.Launch(inKeys, counts, n, shift, groups, over: groups * 256, workgroup: 256)
        try await Scan(counts, exclusive: true)
        try await _radixPlace.Launch(inKeys, inValues, outKeys, outValues, counts, n, shift, groups, withValues,
                                     over: groups * 256, workgroup: 256)
        let k = inKeys
        inKeys = outKeys
        outKeys = k
        if withValues {
            let v = inValues
            inValues = outValues
            outValues = v
        }
        shift += 4
    }
    // Eight passes: the sorted keys are back in the caller's buffers.
}

@inlinable public func _toOrderKeys<T: dtype.Number>(_ x: gpu.Span<T>, _ out: gpu.MutableSpan<uint32>, _ descending: bool) kernel {
    let i = gpu.Index.x
    if i < x.count {
        let k = T.OrderKey(x[i])
        out[i] = descending ? ~k : k
    }
}

@inlinable public func _fromOrderKeys<T: dtype.Number>(_ k: gpu.Span<uint32>, _ out: gpu.MutableSpan<T>) kernel {
    let i = gpu.Index.x
    if i < k.count {
        out[i] = T.FromOrderKey(k[i])
    }
}

/// _sortBy sorts keys as their order keys, moving values with them where
/// withValues holds.
@inlinable public func _sortBy<T: dtype.Number>(_ keys: gpu.Buffer<T>, _ values: gpu.Buffer<uint32>, _ withValues: bool) async throws {
    let n = keys.count
    if n < 2 {
        return
    }
    let bits = try await keys.Device.CreateBuffer(of: uint32.self, count: n)
    try await _toOrderKeys.Launch(keys, bits, false, over: n)
    try await _radixSort(bits, withValues ? values : bits, withValues)
    try await _fromOrderKeys.Launch(bits, keys, over: n)
}

/// Sort puts keys in ascending order, in place: for floats, IEEE 754's
/// total order, -0 before +0 and NaNs at the ends by sign. The sort is
/// stable.
@inlinable public func Sort<T: dtype.Number>(_ keys: gpu.Buffer<T>) async throws {
    let none = try await keys.Device.CreateBuffer(of: uint32.self, count: 1)
    try await _sortBy(keys, none, false)
}

/// Sort puts keys in ascending order and moves each value with its key:
/// values[i] ends up beside the key it started beside. Equal keys keep
/// their order.
@inlinable public func Sort<T: dtype.Number>(_ keys: gpu.Buffer<T>, values: gpu.Buffer<uint32>) async throws {
    try await _sortBy(keys, values, true)
}
