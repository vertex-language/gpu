// TopK: the k largest elements and where they were. Keys are sorted
// descending, stably, with their positions, and the first k kept; ties go
// to the earlier position, as a stable sort leaves them.
import "gpu"

func _iota(_ out: gpu.MutableSpan<uint32>, _ start: uint32) kernel {
    let i = gpu.Index.x
    if i < out.count {
        out[i] = start &+ uint32(i)
    }
}

/// Iota is count consecutive integers from start, on d.
public func Iota(_ d: gpu.Device, count: int, from start: uint32 = 0) async throws -> gpu.Buffer<uint32> {
    let out = try await d.CreateBuffer(of: uint32.self, count: count)
    if count > 0 {
        try await _iota.Launch(out, start, over: count)
    }
    return out
}

/// _topIndices is the positions of the k greatest of b.
public func _topIndices(_ bits: gpu.Buffer<uint32>, _ k: int) async throws -> gpu.Buffer<uint32> {
    let order = try await Iota(bits.Device, count: bits.count)
    try await _radixSort(bits, order, true)
    let top = try await bits.Device.CreateBuffer(of: uint32.self, count: k)
    if k > 0 {
        try await top.Copy(from: order.Slice(from: 0, count: k))
    }
    return top
}

/// TopK is b's k greatest elements, greatest first, and their positions
/// in b. Equal elements come in the order they were in b; for floats, NaNs
/// rank above every number, as in the total order Sort uses. k is at most
/// b.count.
@inlinable public func TopK<T: dtype.Number>(_ b: gpu.Buffer<T>, k: int) async throws -> (values: gpu.Buffer<T>, indices: gpu.Buffer<uint32>) {
    let bits = try await b.Device.CreateBuffer(of: uint32.self, count: b.count)
    if b.count > 0 {
        try await _toOrderKeys.Launch(b, bits, true, over: b.count)
    }
    let at = try await _topIndices(bits, min(k, b.count))
    return (values: try await Gather(b, at), indices: at)
}
