// Gather, Scatter and ScatterAdd: moving elements by an index buffer.
// An index out of range traps, as every span access does.
import "gpu"
import "gpu/dtype"

@inlinable public func _gather<T: dtype.Number>(_ src: gpu.Span<T>, _ indices: gpu.Span<uint32>, _ out: gpu.MutableSpan<T>) kernel {
    let i = gpu.Index.x
    if i < indices.count {
        out[i] = src[int(indices[i])]
    }
}

@inlinable public func _scatter<T: dtype.Number>(_ src: gpu.Span<T>, _ indices: gpu.Span<uint32>, _ dst: gpu.MutableSpan<T>) kernel {
    let i = gpu.Index.x
    if i < indices.count {
        dst[int(indices[i])] = src[i]
    }
}

@inlinable public func _scatterAdd<T: dtype.Number>(_ src: gpu.Span<T>, _ indices: gpu.Span<uint32>, _ dst: gpu.MutableSpan<T>) kernel {
    let i = gpu.Index.x
    if i < indices.count {
        _ = T.AtomicAdd(dst.Address(int(indices[i])), src[i])
    }
}

/// Gather is src at each of indices: out[i] = src[indices[i]].
@inlinable public func Gather<T: dtype.Number>(_ src: gpu.Buffer<T>, _ indices: gpu.Buffer<uint32>) async throws -> gpu.Buffer<T> {
    let out = try await src.Device.CreateBuffer(of: T.self, count: indices.count)
    if indices.count > 0 {
        try await _gather.Launch(src, indices, out, over: indices.count)
    }
    return out
}

/// Scatter writes each element of src to dst at its index: dst[indices[i]]
/// = src[i]. Where two indices are the same, which write lands is not
/// defined.
@inlinable public func Scatter<T: dtype.Number>(_ src: gpu.Buffer<T>, _ indices: gpu.Buffer<uint32>, into dst: gpu.Buffer<T>) async throws {
    if indices.count > 0 {
        try await _scatter.Launch(src, indices, dst, over: indices.count)
    }
}

/// ScatterAdd adds each element of src into dst at its index, atomically:
/// dst[indices[i]] += src[i]. An integer sum is exact whatever the order; a
/// float one is not deterministic, as the order of the additions is not.
@inlinable public func ScatterAdd<T: dtype.Number>(_ src: gpu.Buffer<T>, _ indices: gpu.Buffer<uint32>, into dst: gpu.Buffer<T>) async throws {
    if indices.count > 0 {
        try await _scatterAdd.Launch(src, indices, dst, over: indices.count)
    }
}
