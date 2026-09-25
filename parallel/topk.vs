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

func _descendingFloat32(_ x: gpu.Span<float32>, _ out: gpu.MutableSpan<uint32>) kernel {
    let i = gpu.Index.x
    if i < x.count {
        let b = x[i].bitPattern
        out[i] = ~((b & 0x80000000) != 0 ? ~b : (b | 0x80000000))
    }
}

func _descendingInt32(_ x: gpu.Span<int32>, _ out: gpu.MutableSpan<uint32>) kernel {
    let i = gpu.Index.x
    if i < x.count {
        out[i] = ~(uint32(bitPattern: x[i]) ^ 0x80000000)
    }
}

func _descendingUint32(_ x: gpu.Span<uint32>, _ out: gpu.MutableSpan<uint32>) kernel {
    let i = gpu.Index.x
    if i < x.count {
        out[i] = ~x[i]
    }
}

/// _topIndices is the positions of b's k largest, by descending bits.
func _topIndices(_ bits: gpu.Buffer<uint32>, _ k: int) async throws -> gpu.Buffer<uint32> {
    let order = try await Iota(bits.Device, count: bits.count)
    try await Sort(bits, values: order)
    let top = try await bits.Device.CreateBuffer(of: uint32.self, count: k)
    if k > 0 {
        try await top.Copy(from: order.Slice(from: 0, count: k))
    }
    return top
}

/// TopK is b's k largest elements, largest first, and their positions in
/// b. Equal elements come in the order they were in b; NaNs sort above
/// every number, as in the total order Sort uses. k is at most b.count.
public func TopK(_ b: gpu.Buffer<float32>, k: int) async throws -> (values: gpu.Buffer<float32>, indices: gpu.Buffer<uint32>) {
    let bits = try await b.Device.CreateBuffer(of: uint32.self, count: b.count)
    if b.count > 0 {
        try await _descendingFloat32.Launch(b, bits, over: b.count)
    }
    let at = try await _topIndices(bits, min(k, b.count))
    return (values: try await Gather(b, at), indices: at)
}

/// TopK is b's k largest elements, largest first, and their positions.
public func TopK(_ b: gpu.Buffer<int32>, k: int) async throws -> (values: gpu.Buffer<int32>, indices: gpu.Buffer<uint32>) {
    let bits = try await b.Device.CreateBuffer(of: uint32.self, count: b.count)
    if b.count > 0 {
        try await _descendingInt32.Launch(b, bits, over: b.count)
    }
    let at = try await _topIndices(bits, min(k, b.count))
    return (values: try await Gather(b, at), indices: at)
}

/// TopK is b's k largest elements, largest first, and their positions.
public func TopK(_ b: gpu.Buffer<uint32>, k: int) async throws -> (values: gpu.Buffer<uint32>, indices: gpu.Buffer<uint32>) {
    let bits = try await b.Device.CreateBuffer(of: uint32.self, count: b.count)
    if b.count > 0 {
        try await _descendingUint32.Launch(b, bits, over: b.count)
    }
    let at = try await _topIndices(bits, min(k, b.count))
    return (values: try await Gather(b, at), indices: at)
}
