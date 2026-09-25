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
// int32 and float32 keys are sorted as the uint32s whose order is theirs:
// the sign bit flipped for an int32; for a float32, every bit flipped if
// it is negative and the sign bit if not, which is IEEE 754's total
// order (-NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN).
import "gpu"

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
/// place.
func _radixSort(_ keys: gpu.Buffer<uint32>, _ values: gpu.Buffer<uint32>, _ withValues: bool) async throws {
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

/// Sort puts keys in ascending order, in place. The sort is stable.
public func Sort(_ keys: gpu.Buffer<uint32>) async throws {
    try await _radixSort(keys, keys, false)
}

/// Sort puts keys in ascending order and moves each value with its key:
/// values[i] ends up beside the key it started beside. Equal keys keep
/// their order.
public func Sort(_ keys: gpu.Buffer<uint32>, values: gpu.Buffer<uint32>) async throws {
    try await _radixSort(keys, values, true)
}

func _orderInt32(_ x: gpu.Span<int32>, _ out: gpu.MutableSpan<uint32>, _ n: int) kernel {
    let i = gpu.Index.x
    if i < n {
        out[i] = uint32(bitPattern: x[i]) ^ 0x80000000
    }
}

func _unorderInt32(_ x: gpu.Span<uint32>, _ out: gpu.MutableSpan<int32>, _ n: int) kernel {
    let i = gpu.Index.x
    if i < n {
        out[i] = int32(bitPattern: x[i] ^ 0x80000000)
    }
}

func _orderFloat32(_ x: gpu.Span<float32>, _ out: gpu.MutableSpan<uint32>, _ n: int) kernel {
    let i = gpu.Index.x
    if i < n {
        let b = x[i].bitPattern
        out[i] = (b & 0x80000000) != 0 ? ~b : (b | 0x80000000)
    }
}

func _unorderFloat32(_ x: gpu.Span<uint32>, _ out: gpu.MutableSpan<float32>, _ n: int) kernel {
    let i = gpu.Index.x
    if i < n {
        let b = x[i]
        out[i] = float32(bitPattern: (b & 0x80000000) != 0 ? (b & 0x7FFFFFFF) : ~b)
    }
}

/// Sort puts int32 keys in ascending order, in place, stably.
public func Sort(_ keys: gpu.Buffer<int32>) async throws {
    try await _sortInt32(keys, keys.Device.CreateBuffer(of: uint32.self, count: 1), false)
}

/// Sort puts int32 keys in ascending order and moves each value with its
/// key.
public func Sort(_ keys: gpu.Buffer<int32>, values: gpu.Buffer<uint32>) async throws {
    try await _sortInt32(keys, values, true)
}

func _sortInt32(_ keys: gpu.Buffer<int32>, _ values: gpu.Buffer<uint32>, _ withValues: bool) async throws {
    let n = keys.count
    if n < 2 { return }
    let bits = try await keys.Device.CreateBuffer(of: uint32.self, count: n)
    try await _orderInt32.Launch(keys, bits, n, over: n)
    try await _radixSort(bits, withValues ? values : bits, withValues)
    try await _unorderInt32.Launch(bits, keys, n, over: n)
}

/// Sort puts float32 keys in IEEE 754 total order, in place, stably: -0
/// before +0, and NaNs at the ends by sign.
public func Sort(_ keys: gpu.Buffer<float32>) async throws {
    try await _sortFloat32(keys, keys.Device.CreateBuffer(of: uint32.self, count: 1), false)
}

/// Sort puts float32 keys in total order and moves each value with its
/// key.
public func Sort(_ keys: gpu.Buffer<float32>, values: gpu.Buffer<uint32>) async throws {
    try await _sortFloat32(keys, values, true)
}

func _sortFloat32(_ keys: gpu.Buffer<float32>, _ values: gpu.Buffer<uint32>, _ withValues: bool) async throws {
    let n = keys.count
    if n < 2 { return }
    let bits = try await keys.Device.CreateBuffer(of: uint32.self, count: n)
    try await _orderFloat32.Launch(keys, bits, n, over: n)
    try await _radixSort(bits, withValues ? values : bits, withValues)
    try await _unorderFloat32.Launch(bits, keys, n, over: n)
}
