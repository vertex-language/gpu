# gpu/neural

`Softmax`, `LogSoftmax`, `LogSumExp`, `RMSNorm`, `LayerNorm` (a workgroup a row, fixed-order sums), `Activate` and `Gated` (`.ReLU .GELU .GELUTanh .SiLU .Sigmoid .Tanh .Exp .Sin`, cancellation-free in the tails), `RoPE` (a buffer of positions, or one token's position as a value), `CrossEntropy`.

```vertex
import "gpu/neural"
```

## Types

- **`Conv1dShape`** (struct): Conv1dShape is a one-dimensional convolution's problem: channels in and out, the input's length, the kernel's taps, and how they stride, pad, dilate and group, as torch.nn.Conv1d and ConvTranspose1d take them.
- **`Resize`** (enum): Resize is how Upsample maps output positions to input ones: .nearest or .linear, as torch.nn.functional.interpolate's modes of those names (align_corners false).
- **`Activation`** (enum): Activation is an elementwise function Activate applies.
- **`RopeLayout`** (enum): RopeLayout is which elements of a head RoPE turns together.

## Functions

- `func Conv1dLength(_ tin: int, kernel k: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int`: Conv1dLength is the output length of a Conv1d over tin samples.
- `func Conv1d(_ x: gpu.Buffer<float32>, weight w: gpu.Buffer<float32>, bias b: gpu.Buffer<float32>?, into y: gpu.Buffer<float32>, _ s: Conv1dShape) async throws`: Conv1d writes torch.nn.functional.conv1d(x, w, b, stride, padding, dilation, groups) into y: x is [In, Length], w [Out, In/Groups, Kernel], b [Out] or nil, y [Out, s.OutLength].
- `func ConvTranspose1dLength(_ tin: int, kernel k: int, stride: int = 1, padding: int = 0, outputPadding: int = 0, dilation: int = 1) -> int`: ConvTranspose1dLength is the output length of a ConvTranspose1d over tin samples.
- `func ConvTranspose1d(_ x: gpu.Buffer<float32>, weight w: gpu.Buffer<float32>, bias b: gpu.Buffer<float32>?, into y: gpu.Buffer<float32>, _ s: Conv1dShape) async throws`: ConvTranspose1d writes torch.nn.functional.conv_transpose1d(x, w, b, stride, padding, output_padding, groups, dilation) into y: x is [In, Length], w [In, Out/Groups, Kernel], y [Out, s.TransposedOutLength].
- `func WeightNorm(g: gpu.Buffer<float32>, v: gpu.Buffer<float32>, into w: gpu.Buffer<float32>, rows: int) async throws`: WeightNorm writes the weight torch.nn.utils.weight_norm (dim 0) makes of its parameters into w: w = g · v / ‖v‖, each of v's rows (its first dimension, rows of it) normalized over the rest.
- `func Pad(_ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, rows: int, cols: int, left: int, right: int, reflect: bool = false) async throws`: Pad writes each of x's rows (rows x cols) padded along its length into y (rows x (left + cols + right)): with zeros, or reflected about the row's first and last samples as ReflectionPad1d does (left and right less than cols).
- `func UpsampleLength(_ tin: int, factor: float64) -> int`: UpsampleLength is the length interpolate(scale_factor: factor) makes of tin samples: floor(tin · factor), in double as PyTorch takes it.
- `func Upsample(_ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, rows: int, length tin: int, factor: float64, mode: Resize) async throws`: Upsample writes each of x's rows (rows x tin) resampled along its length by factor into y, as torch.nn.functional.interpolate(x, scale_factor: factor, mode) does -- factor below 1 shrinks.
- `func SplitHeads(_ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, rows: int, heads: int, dim: int) async throws`: SplitHeads writes x, rows x (heads · dim) -- a projection's output, each row's heads side by side -- into y as heads x rows x dim, the layout attention takes.
- `func MergeHeads(_ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, rows: int, heads: int, dim: int) async throws`: MergeHeads is SplitHeads undone: heads x rows x dim into rows x (heads · dim).
- and 25 more

Part of the [`gpu`](https://github.com/vertex-language/gpu) repository.
