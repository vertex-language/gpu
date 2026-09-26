// Rearrangements a network makes between its layers: attention's heads
// split out of a projection and merged back, and columns gathered by
// index -- a duration model's frames, each a copy of its token's.
import (
    "gpu"
)

func _splitHeads(_ x: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>, _ rows: int, _ heads: int, _ dim: int) kernel {
    let i = gpu.Index.x
    if i >= rows * heads * dim {
        return
    }
    // y is [heads, rows, dim]; x [rows, heads · dim].
    let h = i / (rows * dim)
    let rest = i - h * rows * dim
    let r = rest / dim
    let e = rest - r * dim
    y[i] = x[r * heads * dim + h * dim + e]
}

func _mergeHeads(_ x: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>, _ rows: int, _ heads: int, _ dim: int) kernel {
    let i = gpu.Index.x
    if i >= rows * heads * dim {
        return
    }
    // y is [rows, heads · dim]; x [heads, rows, dim].
    let r = i / (heads * dim)
    let rest = i - r * heads * dim
    let h = rest / dim
    let e = rest - h * dim
    y[i] = x[(h * rows + r) * dim + e]
}

/// SplitHeads writes x, rows x (heads · dim) -- a projection's output,
/// each row's heads side by side -- into y as heads x rows x dim, the
/// layout attention takes.
public func SplitHeads(_ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, rows: int, heads: int, dim: int) async throws {
    if rows * heads * dim == 0 {
        return
    }
    try await _splitHeads.Launch(x, y, rows, heads, dim, over: rows * heads * dim)
}

/// MergeHeads is SplitHeads undone: heads x rows x dim into rows x
/// (heads · dim).
public func MergeHeads(_ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, rows: int, heads: int, dim: int) async throws {
    if rows * heads * dim == 0 {
        return
    }
    try await _mergeHeads.Launch(x, y, rows, heads, dim, over: rows * heads * dim)
}

func _gatherColumns(_ x: gpu.Span<float32>, _ index: gpu.Span<int32>, _ y: gpu.MutableSpan<float32>, _ rows: int, _ cols: int, _ out: int) kernel {
    let i = gpu.Index.x
    if i >= rows * out {
        return
    }
    let r = i / out
    let c = i - r * out
    y[i] = x[r * cols + int(index[c])]
}

/// GatherColumns writes y[r, c] = x[r, index[c]] for x of rows x cols and
/// index of out columns: x times a one-hot alignment matrix, taken
/// without one -- a duration model's token features, each repeated for
/// its frames.
public func GatherColumns(_ x: gpu.Buffer<float32>, index: gpu.Buffer<int32>, into y: gpu.Buffer<float32>, rows: int, cols: int) async throws {
    let out = index.count
    if rows * out == 0 {
        return
    }
    try await _gatherColumns.Launch(x, index, y, rows, cols, out, over: rows * out)
}
