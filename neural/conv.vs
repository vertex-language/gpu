// One-dimensional convolutions and what surrounds them in audio and
// speech networks, over [channels, length] row-major buffers (PyTorch's
// NCL at batch 1): Conv1d, ConvTranspose1d, weight normalization,
// padding, and nearest and linear resampling along the length.
//
// These are direct kernels, one work-item an output element, each sum
// taken over input channels, then taps, in order. PyTorch's own shapes
// and index rules are followed exactly, so a layer that loads its
// weights reproduces torch.nn's output.
import (
    "gpu"
    "gpu/parallel"
    "math"
)

// ---- Conv1d ----

/// Conv1dShape is a one-dimensional convolution's problem: channels in
/// and out, the input's length, the kernel's taps, and how they stride,
/// pad, dilate and group, as torch.nn.Conv1d and ConvTranspose1d take
/// them. OutputPadding is ConvTranspose1d's alone.
public struct Conv1dShape {
    public var In: int
    public var Out: int
    public var Length: int
    public var Kernel: int
    public var Stride: int
    public var Padding: int
    public var Dilation: int
    public var Groups: int
    public var OutputPadding: int

    public init(in cin: int, out cout: int, length: int, kernel: int, stride: int = 1, padding: int = 0,
                dilation: int = 1, groups: int = 1, outputPadding: int = 0) {
        self.In = cin
        self.Out = cout
        self.Length = length
        self.Kernel = kernel
        self.Stride = stride
        self.Padding = padding
        self.Dilation = dilation
        self.Groups = groups
        self.OutputPadding = outputPadding
    }

    /// OutLength is a Conv1d's output length.
    public var OutLength: int {
        return Conv1dLength(Length, kernel: Kernel, stride: Stride, padding: Padding, dilation: Dilation)
    }

    /// TransposedOutLength is a ConvTranspose1d's output length.
    public var TransposedOutLength: int {
        return ConvTranspose1dLength(Length, kernel: Kernel, stride: Stride, padding: Padding, outputPadding: OutputPadding, dilation: Dilation)
    }
}

func _conv1d(_ x: gpu.Span<float32>, _ w: gpu.Span<float32>, _ b: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>,
             _ cin: int, _ tin: int, _ cout: int, _ tout: int, _ k: int,
             _ stride: int, _ pad: int, _ dilation: int, _ groups: int, _ bias: bool) kernel {
    let i = gpu.Index.x
    if i >= cout * tout {
        return
    }
    let co = i / tout
    let t = i - co * tout
    let per = cin / groups
    let first = (co / (cout / groups)) * per
    let start = t * stride - pad
    var s: float32 = bias ? b[co] : 0
    var c = 0
    while c < per {
        let xrow = (first + c) * tin
        let wrow = (co * per + c) * k
        var j = 0
        while j < k {
            let p = start + j * dilation
            if p >= 0 && p < tin {
                s = s + w[wrow + j] * x[xrow + p]
            }
            j += 1
        }
        c += 1
    }
    y[i] = s
}

/// Conv1dLength is the output length of a Conv1d over tin samples.
public func Conv1dLength(_ tin: int, kernel k: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int {
    return (tin + 2 * padding - dilation * (k - 1) - 1) / stride + 1
}

/// Conv1d writes torch.nn.functional.conv1d(x, w, b, stride, padding,
/// dilation, groups) into y: x is [In, Length], w [Out, In/Groups,
/// Kernel], b [Out] or nil, y [Out, s.OutLength].
public func Conv1d(_ x: gpu.Buffer<float32>, weight w: gpu.Buffer<float32>, bias b: gpu.Buffer<float32>?, into y: gpu.Buffer<float32>, _ s: Conv1dShape) async throws {
    let tout = s.OutLength
    if s.Out * tout == 0 {
        return
    }
    precondition(s.In % s.Groups == 0 && s.Out % s.Groups == 0, "neural: Conv1d of \(s.In) → \(s.Out) channels in \(s.Groups) groups")
    precondition(w.count == s.Out * (s.In / s.Groups) * s.Kernel && x.count >= s.In * s.Length && y.count >= s.Out * tout, "neural: Conv1d buffers of the wrong size")
    try await _conv1d.Launch(x, w, b ?? w, y, s.In, s.Length, s.Out, tout, s.Kernel, s.Stride, s.Padding, s.Dilation, s.Groups, b != nil, over: s.Out * tout)
}

// ---- ConvTranspose1d ----

// Each output element gathers the inputs that scatter to it: input
// position q reaches output q·stride - pad + j·dilation through tap j.
func _convTranspose1d(_ x: gpu.Span<float32>, _ w: gpu.Span<float32>, _ b: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>,
                      _ cin: int, _ tin: int, _ cout: int, _ tout: int, _ k: int,
                      _ stride: int, _ pad: int, _ dilation: int, _ groups: int, _ bias: bool) kernel {
    let i = gpu.Index.x
    if i >= cout * tout {
        return
    }
    let co = i / tout
    let t = i - co * tout
    let outPer = cout / groups
    let inPer = cin / groups
    let g = co / outPer
    let o = co - g * outPer
    var s: float32 = bias ? b[co] : 0
    var c = 0
    while c < inPer {
        let ci = g * inPer + c
        let wrow = (ci * outPer + o) * k
        var j = 0
        while j < k {
            let at = t + pad - j * dilation
            if at >= 0 {
                let q = at / stride
                if q * stride == at && q < tin {
                    s = s + w[wrow + j] * x[ci * tin + q]
                }
            }
            j += 1
        }
        c += 1
    }
    y[i] = s
}

/// ConvTranspose1dLength is the output length of a ConvTranspose1d over
/// tin samples.
public func ConvTranspose1dLength(_ tin: int, kernel k: int, stride: int = 1, padding: int = 0, outputPadding: int = 0, dilation: int = 1) -> int {
    return (tin - 1) * stride - 2 * padding + dilation * (k - 1) + outputPadding + 1
}

/// ConvTranspose1d writes torch.nn.functional.conv_transpose1d(x, w, b,
/// stride, padding, output_padding, groups, dilation) into y: x is
/// [In, Length], w [In, Out/Groups, Kernel], y [Out, s.TransposedOutLength].
public func ConvTranspose1d(_ x: gpu.Buffer<float32>, weight w: gpu.Buffer<float32>, bias b: gpu.Buffer<float32>?, into y: gpu.Buffer<float32>, _ s: Conv1dShape) async throws {
    let tout = s.TransposedOutLength
    if s.Out * tout == 0 {
        return
    }
    precondition(s.In % s.Groups == 0 && s.Out % s.Groups == 0, "neural: ConvTranspose1d of \(s.In) → \(s.Out) channels in \(s.Groups) groups")
    precondition(w.count == s.In * (s.Out / s.Groups) * s.Kernel && x.count >= s.In * s.Length && y.count >= s.Out * tout, "neural: ConvTranspose1d buffers of the wrong size")
    try await _convTranspose1d.Launch(x, w, b ?? w, y, s.In, s.Length, s.Out, tout, s.Kernel, s.Stride, s.Padding, s.Dilation, s.Groups, b != nil, over: s.Out * tout)
}

// ---- weight normalization ----

func _weightNorm(_ g: gpu.Span<float32>, _ v: gpu.Span<float32>, _ w: gpu.MutableSpan<float32>, _ cols: int) kernel {
    let row = gpu.GroupIndex.x
    let me = parallel.GroupRank()
    let step = parallel.GroupCount()
    let base = row * cols
    var s: float32 = 0
    var j = me
    while j < cols {
        s = s + v[base + j] * v[base + j]
        j += step
    }
    let scale = g[row] / math.Sqrt(parallel.GroupSum(s))
    j = me
    while j < cols {
        w[base + j] = v[base + j] * scale
        j += step
    }
}

/// WeightNorm writes the weight torch.nn.utils.weight_norm (dim 0) makes
/// of its parameters into w: w = g · v / ‖v‖, each of v's rows (its first
/// dimension, rows of it) normalized over the rest. g has one element a
/// row.
public func WeightNorm(g: gpu.Buffer<float32>, v: gpu.Buffer<float32>, into w: gpu.Buffer<float32>, rows: int) async throws {
    if rows == 0 {
        return
    }
    precondition(g.count == rows && v.count % rows == 0 && w.count == v.count, "neural: WeightNorm of \(g.count) gains over \(v.count) weights in \(rows) rows")
    try await _weightNorm.Launch(g, v, w, v.count / rows, over: rows * RowGroup, workgroup: RowGroup)
}

// ---- padding ----

func _pad(_ x: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>, _ rows: int, _ cols: int, _ left: int, _ outCols: int, _ reflect: bool) kernel {
    let i = gpu.Index.x
    if i >= rows * outCols {
        return
    }
    let r = i / outCols
    var src = i - r * outCols - left
    if reflect {
        if src < 0 {
            src = -src
        }
        if src >= cols {
            src = 2 * (cols - 1) - src
        }
    }
    y[i] = src >= 0 && src < cols ? x[r * cols + src] : 0
}

/// Pad writes each of x's rows (rows x cols) padded along its length into
/// y (rows x (left + cols + right)): with zeros, or reflected about the
/// row's first and last samples as ReflectionPad1d does (left and right
/// less than cols).
public func Pad(_ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, rows: int, cols: int, left: int, right: int, reflect: bool = false) async throws {
    let out = left + cols + right
    if rows * out == 0 {
        return
    }
    precondition(!reflect || (left < cols && right < cols), "neural: reflecting \(left) and \(right) about a row of \(cols)")
    try await _pad.Launch(x, y, rows, cols, left, out, reflect, over: rows * out)
}

// ---- resampling along the length ----

/// Resize is how Upsample maps output positions to input ones: .nearest
/// or .linear, as torch.nn.functional.interpolate's modes of those names
/// (align_corners false).
public enum Resize: Equatable {
    case nearest
    case linear

    public var _code: int32 { return self == .nearest ? 0 : 1 }
}

func _upsample(_ x: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>, _ rows: int, _ tin: int, _ tout: int, _ scale: float32, _ mode: int32) kernel {
    let i = gpu.Index.x
    if i >= rows * tout {
        return
    }
    let r = i / tout
    let t = i - r * tout
    let base = r * tin
    if mode == 0 {
        // PyTorch's nearest_idx: the same length is the identity, and
        // twice it a halving, whatever the scale says.
        var s = t
        if tout == 2 * tin {
            s = t >> 1
        } else if tout != tin {
            s = int(math.Floor(float32(t) * scale))
            if s > tin - 1 {
                s = tin - 1
            }
        }
        y[i] = x[base + s]
        return
    }
    // area_pixel_compute_source_index, align_corners false, and the
    // blend, each fused as PyTorch's compiler fuses them: so the result
    // is torch's to the bit, which matters where the positions are a
    // phase of thousands of radians.
    var src = (-0.5 as float32).addingProduct(scale, float32(t) + 0.5)
    if src < 0 {
        src = 0
    }
    var i0 = int(src)
    if i0 > tin - 1 {
        i0 = tin - 1
    }
    let i1 = i0 < tin - 1 ? i0 + 1 : i0
    var l1 = src - float32(i0)
    l1 = l1 < 0 ? 0 : (l1 > 1 ? 1 : l1)
    let l0: float32 = 1 - l1
    y[i] = (l1 * x[base + i1]).addingProduct(l0, x[base + i0])
}

/// UpsampleLength is the length interpolate(scale_factor: factor) makes
/// of tin samples: floor(tin · factor), in double as PyTorch takes it.
public func UpsampleLength(_ tin: int, factor: float64) -> int {
    return int((float64(tin) * factor).rounded(.down))
}

/// Upsample writes each of x's rows (rows x tin) resampled along its
/// length by factor into y, as torch.nn.functional.interpolate(x,
/// scale_factor: factor, mode) does -- factor below 1 shrinks. The output
/// is UpsampleLength(tin, factor) long, and positions map through
/// 1/factor, not tin/tout, as PyTorch's do when given a scale factor.
public func Upsample(_ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, rows: int, length tin: int, factor: float64, mode: Resize) async throws {
    let tout = UpsampleLength(tin, factor: factor)
    if rows * tout == 0 {
        return
    }
    try await _upsample.Launch(x, y, rows, tin, tout, float32(1 / factor), mode._code, over: rows * tout)
}
