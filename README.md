# gpu

[![package: vs-package](https://img.shields.io/badge/package-vs--package-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language)
[![devices: metal | cpu](https://img.shields.io/badge/devices-metal%20%7C%20cpu-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language/gpu)
[![kernels: .vs](https://img.shields.io/badge/kernels-.vs-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language/gpu)
[![license: MIT](https://img.shields.io/badge/license-MIT-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](LICENSE)

High-performance accelerated compute library for Vertex: parallel algorithms, linear algebra, block-quantized matrix multiplication, FlashAttention, neural network activations and norms, audio and signal processing kernels, counter-based random number streams, and 2D/3D compute rasterization written natively in `.vs`.

---

## Architecture Overview

`import "gpu"` is built into the Vertex compiler and runtime: its device half exposes hardware intrinsics (`gpu.Index`, `gpu.Barrier()`, `gpu.Shared`, `gpu.Wave`, `gpu.Atomic`, `gpu.Span`), and its host half provides unified driver management (`gpu.Device`, `gpu.Buffer`, `Launch`, `Map`) across Apple Silicon (Metal) and CPU.

This repository (`github.com/vertex-language/gpu`) sits directly above the compiler primitives:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      Higher-Level Packages                             │
│       nn (layers, decoder, moe)  ·  llm (families, generate)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                       gpu Repository (This Library)                    │
│  gpu/dtype     · Data types (Number, Block: Q4_0, Q8_0, Q4_K, Q6_K)    │
│  gpu/parallel  · Reductions, scans, radix sort, top-k, gather/scatter  │
│  gpu/linalg    · Matmul, Gemv (dense & quantized), Gemm, GemvIds       │
│  gpu/neural    · RMSNorm, LayerNorm, activations, RoPE, audio/conv     │
│  gpu/attention · FlashAttention, tiled multi-query, causal & SWA masks │
│  gpu/random    · Philox4x32-10 CBRNG, distributions, buffer fills      │
│  gpu/raster    · Compute & hardware rasterizer, textures, blending     │
│  gpu/gputest   · Cross-device verification harness & tolerance checks  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                     Built-in Compiler Runtime                          │
│             import "gpu" (Intrinsics, Device, Buffers)                 │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

Run the test suites in `cmd/` with `vsc run`:

```bash
vsc run test-parallel    # 1,638 checks: scans, reductions, radix sorts across 5 dtypes
vsc run test-linalg      # 437 checks: matmul, quantized Gemv, tiled Gemm, GemvIds
vsc run test-neural      # 162 checks: activations, norms, RoPE, audio kernels vs float64
vsc run test-attention   # 62 checks: FlashAttention & tiled multi-query vs host float64
vsc run test-random      # Philox4x32-10 test vectors & moment validation
vsc run test-raster      # Software & Metal rasterizer rendering & pipeline checks
```

---

## Packages Summary

| Package | Capabilities | Hardware & Verification |
| --- | --- | --- |
| **`gpu/dtype`** | `Number` protocol (`float32`, `float16`, `bfloat16`, `int32`, `uint32`), wrapping sums, atomic operations, order keys, and `Accumulator` widening. `Block` tag types (`Q4_0`, `Q8_0`, `Q4_K`, `Q6_K`, `F16`, `BF16`, `F32`), exact in-place `Decode` and `Dot`, and zero-copy device row decoders (`Dequantize`, `DequantizeRow`, `DequantizeRows`). | Tested across all formats; bit-identical to ggml `dequantize_row` on all tensors. |
| **`gpu/parallel`** | Generic over `dtype.Number`: `Reduce` (`.Sum`, `.Min`, `.Max`), `Scan` (inclusive & exclusive), stable 32-bit `Sort` (keys and key-value pairs), `TopK`, `Count`, `Select`, `SelectIndices`, `Gather`, `Scatter`, `ScatterAdd`, `Histogram`, `Iota`. Workgroup primitives: `GroupSum`, `GroupMin`, `GroupMax`, `GroupScan`, `GroupExclusiveScan`, `GroupRank`, `GroupCount`. | 1,638 checks on sizes up to 300,000 across workgroup boundaries on CPU and Metal. |
| **`gpu/linalg`** | `Matmul` over `Shape` (m, n, k, batch, transposes) with fused `Epilogue` (scale, bias, residual, `.ReLU`). `Gemv` for dense float32 and stacked weights. Specialized block-quantized `Gemv` (`Q4_0`, `Q8_0`, `Q4_K`, `Q6_K`, `F16`, `BF16`, `F32`) with wave-level reductions and no barriers. `Gemm` for multi-token prompts ($32 \times 32$ shared-memory tiling). `GemvIds` for dynamic Mixture of Experts routing. `Transpose`. | 437 checks: odd shapes, batch strides, transpositions, and quantized products within strict theoretical bounds. |
| **`gpu/neural`** | `RMSNorm` (wave-shuffled on 32-wide GPUs), `LayerNorm`, `Softmax`, `LogSoftmax`, `LogSumExp`, `Argmax` (on-device greedy decode), `CrossEntropy`. `Activate` and `Gated` (`.ReLU`, `.GELU`, `.GELUTanh`, `.SiLU`, `.Sigmoid`, `.Tanh`, `.Exp`, `.Sin`, `.ReLUSquared`). `RoPE` and precomputed `RopeTable`. Audio & vocoder kernels: `Conv1d`, `ConvTranspose1d`, `WeightNorm`, `Pad`, `Upsample`, `InstanceNorm`, `ChannelNorm`, `Modulate`, `LeakyReLU`, `Snake`, `Axpby`, `CumSum`, `LSTMCell`, `STFT`, `ISTFT`, `SplitHeads`, `MergeHeads`. | 162 checks against float64 libm and PyTorch reference models. |
| **`gpu/attention`** | FlashAttention online softmax (`Forward`): single-query autoregressive decode (8 waves per head on 32-lane hardware) and tiled prompt kernel (16 queries per workgroup). Grouped-Query (GQA) and Multi-Query Attention (MQA) via `Shape`. `Mask` support (`.None`, `.Causal`, `.SlidingWindow`). Continuous KV-cache decoding. | 62 checks against host float64 references and packed cache parity. |
| **`gpu/random`** | Philox4x32-10 10-round counter-based RNG: `Key`, `Split`, `Fold`, `Block`, `Bits`, `Uint32`, `Uniform`, `Normal` (Box–Muller), `Below`, `Bernoulli`. Buffer generators `Fill` and `FillNormal`. | Verified against Random123 reference vectors and statistical moment tests. |
| **`gpu/raster`** | 2D/3D triangle, line, and point rasterizer on Metal or CPU via unified API: `Renderer`, `Program`, `Draw`, `Target`, `Texture`, `Sampler`, `State`. Subpixel coverage, clipping, culling, 2×2 derivative quad fragment shading, depth/stencil, blending, mipmap generation. | Verified by `cmd/test-raster` covering interpolation, filtering, depth, stencil, scissor, and viewport flips. |
| **`gpu/gputest`** | Testing oracle: `Devices()`, `Equal`, `Close` (ULPs), `Near` (input-bounded error tolerances), `Random` (deterministic xorshift64* stream), and `Done()`. | Standard testing framework across all Vertex GPU libraries. |

---

## Detailed Package Specifications

### 1. `gpu/dtype`

Provides numeric primitives and block-quantized formats for accelerated computing.

```vertex
import "gpu/dtype"
```

#### Numeric Types & `Number` Protocol
- `Number`: Protocol for scalar types compute kernels evaluate: `float32`, `float16`, `bfloat16`, `int32`, `uint32`.
  - `Lowest()` / `Highest()`: Identities for reduction operations.
  - `Plus(a, b)`: Wrapping arithmetic for integers and IEEE float additions.
  - `OrderKey(x)` / `FromOrderKey(k)`: Total order mapping for radix sorting (IEEE 754 total ordering: $-\text{NaN} < -\infty < \dots < -0 < +0 < \dots < +\infty < +\text{NaN}$).
  - `AtomicAdd(pointer, value)`: Hardware-accelerated atomic summation.
  - `Accumulator`: Wide accumulation type (`float32` for 16-bit floats, scalar type for others).

#### Quantized Formats & `Block` Protocol
Tag structs implementing `Block` represent GGUF/ggml quantized byte sequences with zero allocation:
- **`Q4_0`**: 32 elements in 18 bytes (float16 scale + 16 bytes 4-bit nibbles). GGMLType 2.
- **`Q8_0`**: 32 elements in 34 bytes (float16 scale + 32 int8 quants). GGMLType 8.
- **`Q4_K`**: 256 elements in 144 bytes (float16 scale, float16 minimum, 12 scale bytes for 8 sub-blocks of 32, 128 bytes 4-bit quants). GGMLType 12.
- **`Q6_K`**: 256 elements in 210 bytes (128 bytes low 4-bits, 64 bytes high 2-bits, 16 int8 scales, float16 scale). GGMLType 14.
- **`F16`**, **`BF16`**, **`F32`**: Unquantized primitives adapted to the `Block` interface.

#### Dequantization Functions
- `Dequantize<B: Block>(b, format, at: 0, count:, into: y)`: Decodes contiguous blocks into float32.
- `DequantizeRow<B: Block>(b, format, rows: buffer, index:, rowBytes:, count:, into: y)`: Reads row index from device buffer, decoding the table row directly into `y` without host round-trip (device-driven greedy token lookup).
- `DequantizeRows<B: Block>(b, format, rows: buffer, first: 0, count:, rowBytes:, n:, into: y)`: Batched embedding lookup for sequence decoding in a single dispatch.

---

### 2. `gpu/parallel`

Hardware-accelerated data-parallel primitives operating over any `dtype.Number`.

```vertex
import "gpu/parallel"
```

#### Parallel Algorithms
- `Reduce<T: Number>(_ b: gpu.Buffer<T>, _ op: Reduction) async throws -> T`: `.Sum`, `.Min`, `.Max`.
- `Scan<T: Number>(_ b: gpu.Buffer<T>, exclusive: bool = false) async throws`: In-place inclusive or exclusive prefix scan.
- `Sort<T: Number>(_ keys: gpu.Buffer<T>) async throws`: Stable radix sort ordering keys in total order.
- `Sort<T: Number>(_ keys: gpu.Buffer<T>, values: gpu.Buffer<uint32>) async throws`: Stable key-value radix sort.
- `TopK<T: Number>(_ b: gpu.Buffer<T>, k: int) async throws -> (values: gpu.Buffer<T>, indices: gpu.Buffer<uint32>)`: Selects $k$ greatest elements and their indices.
- `Select<T: Number>(_ b: gpu.Buffer<T>, where: Where) async throws -> gpu.Buffer<T>`: Compaction filter.
- `SelectIndices<T: Number>(_ b: gpu.Buffer<T>, where: Where) async throws -> gpu.Buffer<uint32>`: Matching index extraction.
- `Count<T: Number>(_ b: gpu.Buffer<T>, where: Where) async throws -> int`: Counts elements meeting `Where` condition.
- `Gather<T: Number>(_ src: gpu.Buffer<T>, _ indices: gpu.Buffer<uint32>) async throws -> gpu.Buffer<T>`: $y[i] = src[indices[i]]$.
- `Scatter<T: Number>(_ src: gpu.Buffer<T>, _ indices: gpu.Buffer<uint32>, into dst: gpu.Buffer<T>) async throws`: $dst[indices[i]] = src[i]$.
- `ScatterAdd<T: Number>(_ src: gpu.Buffer<T>, _ indices: gpu.Buffer<uint32>, into dst: gpu.Buffer<T>) async throws`: Atomic accumulation into indices.
- `Histogram(_ b: gpu.Buffer<uint32>, bins: int) async throws -> gpu.Buffer<uint32>`: Computes bin counts.
- `Iota(_ d: gpu.Device, count: int, from start: uint32 = 0) async throws -> gpu.Buffer<uint32>`: Generates sequential integer ranges.

#### Workgroup Primitives
In-kernel cooperative functions for threadgroup-wide communication:
- `GroupSum(x)`, `GroupMin(x)`, `GroupMax(x)`
- `GroupScan(x)`, `GroupExclusiveScan(x)`
- `GroupRank()`, `GroupCount()`

---

### 3. `gpu/linalg`

Matrix multiplications, linear projections, and mixture of experts kernels.

```vertex
import "gpu/linalg"
```

#### General Matrix Multiplication (`Matmul`)
- Computes $C = \text{epilogue}(A \cdot B)$ over arbitrary `Shape(m, n, k, batch, transposeA, transposeB)`.
- Fused `Epilogue<T>`: Scale factor, bias vector, residual matrix addition, and post-activation (`.ReLU`).
- Specialized kernel dispatches tiled in shared memory, accumulating internally in `T.Accumulator`.

#### Vector-Matrix Products (`Gemv`)
- `Gemv(a, x, into: y, m, k, accumulate:)`: Dense matrix-vector product $y = A \cdot x$ ($+ y$ when accumulating).
- `Gemv(parts: [Buffer<float32>], rows: [int], x, into: y, k, accumulate:)`: Stacked projections (e.g., fused Q, K, V) computed in a single kernel pass.
- `Gemv<B: Block>(w, format, x, into: y, m, k, accumulate:)`: Specialized block-quantized matrix-vector multiplication (`Q4_0`, `Q8_0`, `Q4_K`, `Q6_K`). Dequantizes in registers during wave-level reduction with zero barrier overhead.
- `Gemv<B: Block>(parts: [Buffer<uint8>], rows: [int], format, x, into: y, k, accumulate:)`: Stacked quantized projections.

#### Batched Multi-Token Multiplication (`Gemm`)
- `Gemm<B: Block>(w, format, x, into: y, n, k, accumulate:, stride:, column:)`: Tiled linear layer over $N$ tokens ($Y = X \cdot W^T$).
- Employs a $32 \times 32$ output tile per workgroup; loads weight blocks into shared memory once for 32 tokens, drastically cutting bandwidth compared to token-by-token decoding during prompt evaluation and prefill.
- Supports stacked inputs and arbitrary row strides.

#### Mixture of Experts Multiplication (`GemvIds`)
- `GemvIds<B: Block>(w, format, ids: Buffer<int32>, x, into: y, rows, k, ...)`: Evaluates selected expert matrices on-device without sorting or scattering tokens on the host.

#### Transposition
- `Transpose<T: Number>(x, into: y, rows, cols)`: Cache-efficient matrix transposition.

---

### 4. `gpu/neural`

Neural network building blocks and signal processing operations.

```vertex
import "gpu/neural"
```

#### Normalization & Classification
- `RMSNorm(x, weight, into: y, rows, cols, eps:)`: Optimized RMS normalization. Uses wave shuffles on 32-lane SIMD GPUs; workgroup reduction elsewhere.
- `LayerNorm(x, weight, bias, into: y, rows, cols, eps:)`: Mean and variance normalization with affine scaling.
- `Softmax(x, into: y, rows, cols)` / `LogSoftmax` / `LogSumExp`
- `Argmax(x, into: out, rows, cols)`: Computes the index of the maximum value per row directly on the GPU for zero-copy greedy generation.
- `CrossEntropy(logits, targets, rows, cols)`: Fused categorical cross-entropy loss calculation.

#### Activations
- `Activate(x, activation, into: y)`
- `Gated(x, gate:, activation, into: y)`: Gated elementwise multiplication $y = \text{gate} \odot \text{activation}(x)$.
- Supported activations (`Activation` enum):
  - `.ReLU`, `.GELU`, `.GELUTanh`, `.SiLU`, `.Sigmoid`, `.Tanh`, `.Exp`, `.Sin`, `.ReLUSquared`.

#### Rotary Embeddings (RoPE)
- `RoPE(x, position:, heads, dim, base:, layout:)`: In-place rotary position embedding application.
- `RoPE(x, positions: buffer, heads, dim, base:, layout:)`: Batched position-dependent rotation.
- `RopeTable`: Precomputed cosine/sine tables avoiding trigonometric evaluation during inference.
- `RopeLayout`: `.interleaved` (pairs of adjacent coordinates) or `.halves` (split head dimensions).

#### Audio & Sequence Processing Kernels
- `Conv1d(x, weight, bias, into: y, shape)` & `ConvTranspose1d`: 1D convolution with arbitrary stride, padding, dilation, and grouping.
- `WeightNorm(g, v, into: w, rows)`: Decouples magnitude and direction ($w = g \cdot v / \|v\|$).
- `Pad(x, into: y, rows, cols, left, right, reflect:)`: Zero or reflection padding.
- `Upsample(x, into: y, rows, length, factor, mode:)`: Nearest-neighbor or linear resampling matching PyTorch interpolation semantics.
- `InstanceNorm`, `ChannelNorm`, `Modulate`: Conditioning layers for vocoders and generative audio.
- `Snake`, `LeakyReLU`, `Axpby`, `CumSum`, `LSTMCell`
- `STFT` / `ISTFT`: Direct Discrete Fourier Transform for short windows with `HannWindow`.
- `SplitHeads` / `MergeHeads` / `GatherColumns`: Tensor layout transformations for multi-head attention.

---

### 5. `gpu/attention`

Fused FlashAttention implementation for autoregressive inference and prefill prompts.

```vertex
import "gpu/attention"
```

#### Capabilities & Algorithms
- `Forward(q, k, v, into: o, shape, mask:)`:
  - Online softmax accumulation avoiding materialization of the $N \times N$ attention matrix.
  - Supports Grouped-Query Attention (GQA) and Multi-Query Attention (MQA) where `heads % kvHeads == 0`.
  - Head dimensions up to 128.
- **Decoding Path (`_attendDecode`)**: Specialized for single-token autoregressive generation ($queries = 1, batch = 1$). Dispatches 8 waves per head on 32-lane GPUs (Metal), streaming keys and values directly from continuous or strided KV cache storage.
- **Tiled Batched Path (`_attendTiled`)**: Tiled FlashAttention for prompt processing. Groups 16 queries per workgroup, reusing loaded K and V tiles across all queries in the group.
- **Attention `Shape`**:
  - `batch`, `heads`, `kvHeads`, `queries`, `keys`, `headDim`, `scale`, `keyCapacity` (for preallocated KV cache buffers).
- **Mask Modes (`Mask`)**:
  - `.None`: Full bidirectional attention.
  - `.Causal`: Standard autoregressive masking.
  - `.SlidingWindow(w)`: Local window attention attending only to current token and preceding $w - 1$ keys.

---

### 6. `gpu/random`

Counter-based random number generation using the Philox4x32-10 algorithm.

```vertex
import "gpu/random"
```

- Stateless, reproducible RNG keys following JAX/Random123 principles.
- `Key(seed)` / `Split(key)` / `Fold(key, data)`: Generates independent streams without mutable global state.
- Distributions:
  - `Uniform(key, index)`: IEEE `float32` in $[0, 1)$ with 24 random bits.
  - `Normal(key, index)`: Standard normal distribution via Box–Muller transform.
  - `Below(key, index, bound)`: Unbiased integer in $[0, \text{bound})$ via Lemire's algorithm.
  - `Bernoulli(key, index, p)`: Boolean with probability $p$.
- Buffer Generation:
  - `Fill(buffer, key, start: 0)`: Fills buffer with uniform float32 or uint32 words.
  - `FillNormal(buffer, key, start: 0)`: Fills buffer with normally distributed float32 values.

---

### 7. `gpu/raster`

Hardware-accelerated (Metal) and software compute rasterization behind a single API.

```vertex
import "gpu/raster"
```

- Supports rendering triangles, lines, and points into color and depth-stencil targets.
- Features:
  - Subpixel fixed-point coverage with antisymmetric fill rules (no cracked seams).
  - Analytical derivatives and 2×2 quad fragment shading.
  - Textures: 2D, cube maps, mipmap generation, bilinear/trilinear filtering, texture swizzling.
  - Depth/stencil testing, polygon offset, blending equations, and scissoring.
  - Metal backend with pipeline state caching; seamless fallback to CPU compute interpreter when hardware is unavailable.

---

### 8. `gpu/gputest`

Verification framework ensuring numerical correctness across devices.

```vertex
import "gpu/gputest"
```

- `Devices()`: Returns available compute devices (CPU first, followed by GPU devices).
- `Equal(what, device, got, want)`: Strict element-for-element bit equality.
- `Close(what, device, got, want, ulps:)`: Checks floating-point error in Units in the Last Place for order-dependent parallel reductions.
- `Near(what, device, got, want, bound:)`: Per-element input-derived theoretical error bounds (e.g., $k \cdot \epsilon \cdot \sum |a \cdot b|$ for dot products).
- `Done()`: Validates that all checks passed; halts with diagnostic trace if any failed.

---

## License

[MIT](LICENSE)
