# gpu

[![package: core](https://img.shields.io/badge/package-core-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language)
[![devices: metal | cuda | hip | cpu](https://img.shields.io/badge/devices-metal%20%7C%20cuda%20%7C%20hip%20%7C%20cpu-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language/gpu)
[![kernels: .vs](https://img.shields.io/badge/kernels-.vs-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language/gpu)
[![status: early](https://img.shields.io/badge/status-early-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language/gpu)

The shared library of accelerated functions for Vertex. These are the
sorts, scans, matrix multiplies, FFTs, random streams, BVHs and attention
kernels that AI, 3D, media, simulation and data packages all need, written
once as `.vs` kernels. Everyone else calls them instead of shipping their own.

> **Status: early.** Seven packages are built and tested on Metal and the
> CPU device; the rest of this document is the blueprint they grow into.
> The `proposed_ai_packages.md` and `proposed_3d_packages.md` design docs
> were a sketch. Where this document differs from them, this one wins
> (see [From the sketch](#from-the-sketch)).

| Package | Built | Tested |
| --- | --- | --- |
| `gpu/dtype` | `Number`: the protocol kernels compute with (`float32`, `int32`, `uint32`), with the identities, wrapping sum, order key and atomic add the other packages need | through every package above it |
| `gpu/parallel` | Generic over `dtype.Number`, one definition each: `Reduce` (`.Sum .Min .Max`), `Scan` (inclusive, exclusive), `Sort` (stable radix in total order; optional `uint32` values), `TopK`, `Select`, `SelectIndices`, `Count` (with `Where`, exact for integers), `Gather`, `Scatter`, `ScatterAdd`, `Histogram`, `Iota`. Device functions `GroupSum`, `GroupMin`, `GroupMax`, `GroupScan`, `GroupExclusiveScan`, `GroupRank`, `GroupCount` | 1434 checks, at `float32`, `int32` and `uint32`: every size from 0 to 300,000 around the group boundaries, every group size to 1024; results bit-identical on Metal and the CPU device |
| `gpu/linalg` | `Matmul` over a `Shape` (m, n, k, batch, transposes) with an optional fused `Epilogue` (scale, bias, residual, `.ReLU`), tiled through shared storage; `Gemv`; `Transpose`. Generic over `dtype.Number` | 127 checks: odd shapes, batches, transposes and the epilogue; `int32` exactly, `float32` within k·ε·Σ\|a·b\| of an `f64` host product |
| `gpu/neural` | `Softmax`, `LogSoftmax`, `LogSumExp`, `RMSNorm`, `LayerNorm` (a workgroup a row, fixed-order sums), `Activate` and `Gated` (`.ReLU .GELU .GELUTanh .SiLU .Sigmoid .Tanh`, cancellation-free in the tails), `RoPE`, `CrossEntropy`; float32, over the `math` package | 129 checks against float64 libm references within bounds derived from the inputs; every device bit-identical to the CPU device |
| `gpu/attention` | `Forward`: FlashAttention's online softmax over key blocks (no scores matrix), a workgroup per query row; `Mask` `.None`, `.Causal` (with a KV-cache offset), `.SlidingWindow(n)`; grouped- and multi-query heads through `Shape`; head dims to 128; float32 | 54 checks against float64 attention on the host; every device bit-identical to the CPU device |
| `gpu/random` | Philox4x32-10: `Key`, `Split`, `Fold`, `Block`, `Bits`, `Uint32`, `Uniform`, `Normal` (Box–Muller), `Below`, `Bernoulli`; `Fill` for `float32` and `uint32` buffers, `FillNormal` | Random123's known answers; every device bit-identical to the host; `Normal`'s moments |
| `gpu/gputest` | `Devices`, `Equal`, `Close` (ULPs), `Near` (an absolute bound per element), `Random`, `Sizes`, `Done` | the harness the others are tested with |

Run the tests with `vsc run test-parallel`, `test-linalg`, `test-neural`, `test-attention` and `test-random` (with `-update` the first time, to fetch `math`).

---

## What this repository is, and isn't

`import "gpu"` is **not** in this repository. It is built into the
compiler. Its device half is the intrinsics a kernel uses (`gpu.Index`,
`gpu.Barrier()`, `gpu.Shared`, `gpu.Wave`, `gpu.Atomic`, `gpu.Span`). Its
host half is the one driver runtime (`gpu.Device`, `gpu.Buffer`, `Launch`,
`Map`) that speaks to Metal, CUDA, HIP and the CPU.

This repository is the layer directly above that, in the way `net/http`
sits above `net/tcp`:

```
 tensor · nn · model        render · scene · mesh        media · sim · db · ui ...
 ═══════════════════════════════ callers: no kernels of their own for common work ═══
 gpu/parallel  gpu/linalg  gpu/fft  gpu/random  gpu/spatial  gpu/image  gpu/attention ...
 ═══════════════════════════════ this repository: .vs kernels + device functions ═══
 gpu (built into vsc): intrinsics · Device · Buffer · Launch · Map · CPU device
 ═══════════════════════════════════════════════════════════════════════════════════
 vsc · ir → AIR · PTX · HSACO · host code
```

The rule it enforces: **a package outside `gpu/*` should not need a
`kernel` function for anything common.** If two unrelated packages would
each write the same kernel, that kernel belongs here. If a package's
kernel is unique to its domain, such as a splat blending pass or an MoE
router's scoring, it stays in that package, written in `.vs` and built
from the functions here.

---

## Two tiers in every package

The examples use today's kernel syntax (`kernel`, `gpu.Span`, `Launch`,
`Map`) and call functions this repository will provide. They are
illustrative until each package lands.

Each package offers its work in two forms, and the second is what makes
the reuse real.

**Operations** are host-side, `async`, and work on whole `gpu.Buffer`s. An
operation launches one or more kernels and returns when the result is
ready:

```vertex
import "gpu"
import "gpu/parallel"

let d = gpu.Default()
let keys = try await d.Upload(depths)          // [float32]
let ids  = try await d.Upload(triangleIDs)     // [uint32]
try await parallel.Sort(keys, values: ids)     // stable radix sort, in place
let n = try await parallel.Count(keys, where: .Less(0.5))
```

**Device functions** are ordinary `.vs` functions that are kernel-safe
(below), so they can be called *inside your own kernel*. vsc compiles
everything a kernel reaches into that kernel's device image. A device
function therefore costs no launch, allocates no buffer, and is inlined
where you call it. This is CUB's block and warp layer, and Metal's
simdgroup functions, made into a shared library:

```vertex
import "gpu"
import "gpu/parallel"

// One workgroup per row: scale each row to unit length.
func normalizeRows(_ x: gpu.MutableSpan<float32>, _ cols: int) kernel {
    let row = gpu.GroupIndex.x
    var sum: float32 = 0
    var c = gpu.LocalIndex.x
    while c < cols {
        let v = x[row * cols + c]
        sum += v * v
        c += gpu.GroupSize.x
    }
    let total = parallel.GroupSum(sum)      // waves, then shared memory, then broadcast
    c = gpu.LocalIndex.x
    while c < cols {
        x[row * cols + c] /= total.squareRoot()
        c += gpu.GroupSize.x
    }
}
```

### Scope is in the name

A device function's name says which threads cooperate in it, using the
scopes the built-in `gpu` already uses. No scope prefix means the function
runs within a single thread:

| Scope | Prefix | Who takes part | Example |
| --- | --- | --- | --- |
| thread | — | one work-item | `random.Uniform(key, i)`, `dtype.Decode(b)` |
| wave | `Wave` | one SIMD group (32 or 64 lanes; Apple's simdgroup, NVIDIA's warp, AMD's wavefront) | `parallel.WaveScan(x)` |
| group | `Group` | one workgroup, through shared memory | `parallel.GroupSum(x)`, `parallel.GroupSort(k)` |
| grid | *(an operation)* | the whole launch, across kernels | `parallel.Scan(buffer)` |

Every group function must be reached by every work-item in the group,
just as `gpu.Barrier()` must. Wave functions assume every lane is active
unless they take a `mask:`.

---

## The kernel-safe rule

A device function, and every type it takes, obeys the rules vsc already
enforces for kernels. It makes no runtime calls, allocates nothing on the
heap, uses no protocol witness or class vtable, and touches no global.
In practice this means:

- Parameters are numbers, `bool`, `gpu.Span` / `gpu.MutableSpan`, and plain
  structs of those.
- Behaviour is chosen by **enum values and specialization**, not by
  closures or existentials. A function over elements is generic over
  `dtype.Number` and `@inlinable`, with its kernels `@inlinable` too, so
  the module that calls it builds it for its element types.
- Constant tables (bit-reversal permutations, Sobol direction numbers,
  BRDF lookup tables) arrive as buffers or as literals that fold away once
  inlined. They are never globals.

A type that breaks the rule fails at the first kernel that uses it. It
should be designed in, not found out later.

---

## Packages

| Package | Provides | Main users |
| --- | --- | --- |
| [`gpu/dtype`](#gpudtype) | Element types, and block-scaled formats as a single idea | everyone |
| [`gpu/layout`](#gpulayout) | Shape, strides, views: one way to describe any n-d region of a buffer | everyone |
| [`gpu/parallel`](#gpuparallel) | Reduce, scan, sort, select, histogram, top-k, gather/scatter; wave and group forms | everyone |
| [`gpu/linalg`](#gpulinalg) | Matmul with epilogues, batched, grouped, quantized; GEMV; small fixed-size matrices | AI, 3D skinning, sim, science |
| [`gpu/sparse`](#gpusparse) | CSR/COO/blocked formats; SpMV, SpMM, SDDMM; segment ops | GNNs, MoE, sim, graph analytics |
| [`gpu/fft`](#gpufft) | FFT 1/2/3-D, real and complex, batched; FFT convolution | audio, imaging, ocean/bloom, science, long-conv AI |
| [`gpu/random`](#gpurandom) | Counter-based, splittable RNG keys; distributions; quasi-random sequences | training, sampling, path tracing, Monte Carlo |
| [`gpu/image`](#gpuimage) | Resampling, mip chains, color conversion, filters, BCn/ASTC encode and decode | media, 3D, vision preprocessing, UI |
| [`gpu/spatial`](#gpuspatial) | Morton/Hilbert codes, LBVH build, ray/box/k-NN/radius queries, hash grids | 3D, physics, point clouds, vector search |
| [`gpu/neural`](#gpuneural) | Norms, softmax, activations, RoPE, sampling, convolution, losses | AI, neural rendering, denoisers |
| [`gpu/attention`](#gpuattention) | Fused attention (flash, paged, variable-length); KV-cache ops; masks | AI, vision transformers |
| [`gpu/raster`](#gpuraster) | Compute rasterizer and hardware raster pipelines behind one API | 3D, 2D UI, visualization, differentiable rendering |
| [`gpu/profile`](#gpuprofile) | GPU timestamps, per-operation spans, Perfetto trace export | everyone |
| [`gpu/gputest`](#gpugputest) | The oracle harness: a device result against the CPU device, exact or within a stated tolerance | every package that writes kernels, in or out of this repo |

`internal/dispatch` is the one piece not for callers. It is the table of
device images keyed by `(function, dtype, device family, shape bucket)`,
together with the persistent tuning cache.

### `gpu/dtype`

The element vocabulary for every package here and above:
`float32 float16 bfloat16 float8e4m3 float8e5m2 float6e3m2 float4e2m1
int32 int16 int8 uint8 int4 bool`.

The idea that has converged across the field is the **block-scaled
format**. OCP Microscaling (MXFP8, MXFP6, MXFP4), NVFP4, GGUF's k-quants,
and AWQ/GPTQ group quantization are all the same shape: small elements
plus a shared scale per block. They differ only in element format, block
size, scale format and where the scales live. `dtype.Scaled` describes all
of them with those four fields, so there is one dequantize path rather
than one per format. `q4_k`, `mxfp4` and `nvfp4` are named `Scaled`
constants, not special cases.

Device functions: `dtype.Decode`, `dtype.Encode` (round-to-nearest-even,
or stochastic when given a `random` key), `dtype.DequantBlock`. Operations:
`dtype.Convert(from:, into:)`, `dtype.Quantize(_, as:)`.

`float16` and `bfloat16` compute today, because VIR has them. The 8-, 6-
and 4-bit formats are storage only: loaded, widened, then computed.

### `gpu/layout`

`layout.Strided` is a shape and strides up to rank 8, as a plain struct.
`layout.View<T>` is a `gpu.Buffer<T>` plus a `Strided`. This is the
DLPack/CuTe model in its simplest form. A tensor, an image's pitched rows,
a mesh's interleaved vertex stream and a matrix's leading dimension are
all views. That lets `linalg.Matmul` take an image channel, or
`parallel.Reduce` take one axis of a tensor, with no copy. `tensor` is
built on this package; it does not replace it.

### `gpu/parallel`

The primitives nearly every other package decomposes into.

| Operation | Device forms | Notes |
| --- | --- | --- |
| `Reduce`, `Count` | `WaveSum/Min/Max`, `GroupReduce`, `GroupSum` | `segments:` for segmented; `.Sum .Min .Max .And .Or .ArgMin .ArgMax` |
| `Scan` (inclusive, exclusive) | `WaveScan`, `GroupScan` | decoupled look-back, single pass |
| `Sort` (keys, key-value pairs) | `GroupSort` | stable radix sort on bit patterns; float keys sort in IEEE total order; `segments:` |
| `Select`, `Partition`, `Unique`, `RunLength` | `GroupCompact` | stream compaction, used for culling and filtering |
| `Histogram` | `GroupHistogram` | shared-memory privatized |
| `TopK` | `WaveTopK` | sampling, vector search, beam search, culling |
| `Gather`, `Scatter`, `ScatterReduce` | — | MoE dispatch, GNN message passing, splat binning |
| `Merge`, `SearchSorted` | — | sorted joins, bucketing |

An MoE token permutation, a transparency sort and a splat depth sort are
all `Sort(keys, values:)`, and that shared call is the point of this
package.

### `gpu/linalg`

`linalg.Matmul(a, b, into: c, shape, epilogue)`: the problem is a
`Shape` descriptor (m, n, k, batch, transposes), and the **fused
epilogue** -- the work done on each output tile before it is written --
is an `Epilogue` value: scale, bias, residual, activation, and later a
cast. It is data, in the style of cuBLASLt's matmul descriptors and
CUTLASS epilogues and Triton fusion. Each combination is specialized into its own image; no
epilogue interpreter runs on the device. Also here:

- `Matmul` with batch strides, and `GroupedMatmul` (differently sized
  problems in one launch, the MoE shape);
- quantized matmul, where `a` or `b` is a `dtype.Scaled` view,
  dequantized inside the tile loop;
- `Gemv` for decode-time matrix-vector products, and `Transpose`;
- `linalg.Mat3` and `Mat4` batch operations, and batched small solvers
  (Cholesky, 3×3 SVD) for skinning, physics and geometry.

Device form: `linalg.Tile` is a workgroup's register tile with `Load`,
`MultiplyAdd` and `Store`. It lowers to tensor cores, MFMA and simdgroup
matrices once VIR has wave-matrix instructions. Until then it lowers to
plain FMAs, which gives the same results more slowly.

### `gpu/sparse`

`sparse.CSR`, `COO` and `BlockedELL` views, plus `SpMV`, `SpMM` and
`SDDMM`, and `SegmentReduce` over CSR offsets. Graph neural networks,
block-sparse attention masks, finite-element solves and graph analytics
use it. Phase 3; `parallel.ScatterReduce` covers the early cases.

### `gpu/fft`

A plan-free API: `fft.Forward(x)`, `fft.Inverse(x)` and `fft.Real(x)`,
over 1-, 2- and 3-D views, batched along the remaining axes. Plans are
cached per shape and device inside the package. `fft.Convolve(signal,
filter)` picks direct or FFT convolution by size. Users include audio
and `media`, bloom and ocean simulation in 3D, spectral solvers, and
long-convolution models.

### `gpu/random`

JAX-style **keys**, with the counter-based generators CUDA's Philox and
JAX's Threefry use: `random.Key(seed)`, `key.Split()`, `key.Fold(i)`.
Every stream is a pure function of a key and a counter, so results are
the same on every device, every run, and every order of work-items. That
is the property training reproducibility and the oracle tests both need.

Device functions: `random.Uniform(key, i)`, `Normal`, `Bits`,
`Bernoulli`, and `Categorical` for token sampling. Operations fill buffers.
Quasi-random `random.Sobol` and `random.Halton` sequences (Owen-scrambled)
serve path tracing and quasi-Monte Carlo integration, which converge
faster on them than on pseudo-random numbers.

### `gpu/image`

Pixel work shared across workloads: `Resize` (box, bilinear, bicubic and
Lanczos, gamma-correct), `MipChain`, `ConvertColor` (sRGB, linear,
Display P3, BT.709/2020, YUV planar and semi-planar), `Blur`, `Convolve2D`,
`Normalize` (vision-model preprocessing), and BCn/ASTC `Encode` and
`Decode`. Operations take `layout.View`s of pixels. Sampler and texture
objects stay in the built-in `gpu`. Users: `media` decode pipelines, 3D
asset loading, `model` vision preprocessing, and a future GPU `ui/draw`.

### `gpu/spatial`

Space-partitioning and queries: `MortonCodes`, `HilbertCodes`,
`BuildBVH` (an LBVH from sorted Morton codes, refinable) and `HashGrid`.
Queries: `Raycast` / `Occluded`, `Overlap(boxes)`, `Nearest(k:)` and
`Within(radius:)`. Traversal device functions let a kernel walk a BVH
itself, for custom shading or collision. Hardware ray tracing (Metal RT,
later DXR) sits behind the same query API when present. Nearest-neighbour
search here is the same machinery a brute-force or IVF embedding index
needs, which is why vector search and point clouds share it with 3D.

### `gpu/neural`

Kernels that neural networks need and other workloads occasionally
reuse: `RMSNorm`, `LayerNorm`, `GroupNorm`, `Softmax` and `LogSumExp`
(online, single pass), activations and gated activations (`SiLU`, `GELU`,
`SwiGLU`, `GeGLU`), `RoPE` (with YaRN and NTK scaling), `CrossEntropy`
(fused with softmax), `Conv` (1-3-D, implicit GEMM through `linalg`),
and token `Sample` (temperature, top-k/p, min-p). Each op has a backward
form beside it. These are fused building blocks, not layers. Layers,
parameters and autodiff are `nn` and `tensor`, above.

### `gpu/attention`

The fastest-moving family in the field, so it gets its own package.
`attention.Forward` is fused, tiled, online-softmax attention (the
FlashAttention algorithm) and takes masks as data: `.Causal`,
`.SlidingWindow(n)`, `.BlockSparse(layout)`, `.Document(ids)`. Also here:
grouped-query and multi-latent attention head layouts, **paged** KV caches
(`attention.PagedCache`, block tables in the vLLM style), variable-length
batches, and a backward pass. Hopper- and Blackwell-specific versions
come in as vendor sources behind the same signatures (below).

### `gpu/raster`

Triangles into pixels. There is one API and two paths: hardware raster
where the platform owns it (Metal, later D3D12), and a `.vs` compute
rasterizer everywhere else, including headless machines and Linux on CUDA
and HIP. Both write a **visibility buffer** of 64-bit depth and triangle
ids, merged with `gpu.Atomic.Min`, so everything after "which triangle
covers this pixel" is shared compute. The compute path doubles as the
oracle for the hardware path. `ui/draw`'s GPU backend and differentiable
rendering reuse it.

### `gpu/profile`

`profile.Span("name") { … }` records GPU timestamps around operations.
Every operation in this repository opens a span automatically when
profiling is on. `profile.Trace` exports in Perfetto / Chrome trace
format, so a timeline opens in any browser.

### `gpu/gputest`

The harness the whole repository is tested with, public so packages
outside it can test their own kernels the same way:

```vertex
import "gpu/gputest"

gputest.Compare(normalizeRows, inputs: gputest.Random(rows: 1...4096, cols: 1...4096),
                exact: false, ulps: 4)
```

It runs a kernel or operation on a device and on the CPU device, which
runs the same `.vs` source compiled for the host, and compares the two.
Shapes are generated ragged, and odd sizes are deliberate. The contract
it checks is stated per function:

- **Exact.** Elementwise ops, `float16`/`bfloat16` included: both paths
  round once, so the bits must match.
- **Deterministic.** Sort, scan, histogram, and every op with
  `deterministic: true`: the same result on every run and every device.
- **Tolerance.** Reductions whose order differs by device, matmul,
  attention, FFT, and the approximate intrinsics, with the bound stated
  in the function's documentation, never a blanket epsilon.

---

## Conventions

- **Named for what it provides.** `random`, not `philox`; `linalg`, not
  `blas`; `spatial`, not `bvh`. A package name reads well at the call
  site: `parallel.Sort`, `fft.Forward`, `attention.Forward`.
- **Operations are verbs on buffers or views.** Output parameters use the
  label `into:`, as `Map(…, into:)` does. Variants are argument labels,
  not new names: `segments:`, `values:`, `where:`, `deterministic:`.
- **Scope prefixes** (`Wave`, `Group`) on device functions only.
- **Determinism is stated.** Every operation documents whether it is
  deterministic. Where the fast algorithm is not (atomic float
  accumulation, split-K matmul), `deterministic: true` selects an
  order-fixed one.
- **Scratch memory is the operation's.** An operation allocates any
  temporary storage it needs from the device's pool. `WorkspaceSize(…)`
  exists for callers that plan their memory, such as graph capture.
- **No silent fallback.** If a device lacks an image for an operation,
  such as a `float64` kernel on an Apple GPU, the operation throws
  `gpu.DeviceError`. Running on the CPU is something the caller asks for,
  never something that happens silently.

## Vendor sources

`.vs` comes first, and for most functions `.vs` is all there is. A vendor
source (`.cu`, `.hip`, `.metal`) is allowed **only** in this repository,
**only** as a faster version of an existing `.vs` function with the same
signature, and **only** with that `.vs` twin kept in place. The twin is
the fallback on every device the vendor source doesn't cover, and it is
the oracle the vendor source is tested against. vcx compiles the vendor
sources in-process: no toolkit is installed and nothing is prebuilt.
Vendor libraries (cuBLAS, MPS) are test oracles for correctness and speed,
and are never linked into anything that ships.

```
gpu/
├── README.md
├── package.vs
├── internal/dispatch/     image tables, shape buckets, tuning cache
├── dtype/  layout/
├── parallel/              *.vs            (+ cuda/, hip/, metal/ only where profiling earns it)
├── linalg/                *.vs  cuda/ hip/ metal/
├── sparse/  fft/  random/  image/  spatial/
├── neural/  attention/    *.vs  cuda/ hip/ metal/
├── raster/  profile/  gputest/
└── tests/                 one ladder per package, run on every device and the CPU
```

## What does not belong here

- **Devices, buffers, launches, queues:** the built-in `gpu`.
- **Tensors, autodiff, layers, models:** `tensor`, `nn`, `model`.
- **Meshes, scenes, materials, the render graph:** `mesh`, `scene`,
  `render`.
- **Collective communication** (all-reduce across GPUs and nodes):
  `dist/collective`. It is transport, built on `gpu` peer memory, and its
  reductions call `parallel`.
- **A kernel only one domain will ever run.** It lives in that domain's
  package, written in `.vs` from the functions here.

## What it waits on

| Needs | For | Where it stands |
| --- | --- | --- |
| `.vs` kernels, `import "gpu"`, `Launch`, `Map` | everything | built for Metal and the CPU |
| `@inlinable` across modules | device functions, called from other modules' kernels | built: functions, methods and initializers |
| Fast barriers on the CPU device | the CPU device as oracle and fallback | built: a group's work-items run as fibers (arm64) |
| CUDA and HIP images for `.vs` | NVIDIA and AMD | not started |
| Generic kernels | one source per op across dtypes | built: each launch's specialization is its own image |
| vsc `Float16` / `BFloat16` | half-precision anything | VIR has both, and every backend runs them; the language types are next |
| fp8/fp6/fp4 storage types in VIR | `dtype.Scaled` on the device | reserved |
| Wave-matrix instructions in VIR | `linalg.Tile` at tensor-core speed | reserved; plain FMAs until then |
| `math` in kernels (K9) | `neural`, `random.Normal`, GELU/SiLU epilogues | built: float32 elementary functions in pure Vertex, the `math` repository |
| Queues and events | overlapping operations; interactive `raster` | a launch waits today |
| `i64` atomic min | the visibility buffer | in VIR and all three GPU backends |

## Build order

1. **`dtype`, `layout`, `parallel`, `random`, `gputest`.** Every later
   package builds on these, and each works on Metal and the CPU today.
2. **`linalg`, `neural`, `attention`, `image`.** These are enough to run
   a model, and to decode, resize and present images.
3. **`spatial`, `raster`, `fft`.** These are enough to render a scene
   headless and simulate a field.
4. **`sparse`, `profile` export**, then vendor sources wherever a
   profile says the `.vs` version is the bottleneck.

## From the sketch

| Sketch | Here | Why |
| --- | --- | --- |
| `gpu/blas` | `gpu/linalg` | The API is matmul plus epilogues, batched, grouped and quantized, not the BLAS level-1/2/3 catalogue. `linalg` names the subject |
| `gpu/dnn` | `gpu/neural` + `gpu/attention` | Attention moves fastest and is the largest, so it is split out (the AI sketch's open question 3). `nn` is taken by the layer library |
| `gpu/philox` | `gpu/random` | Philox is one algorithm; the package provides keys, distributions and quasi-random sequences |
| `gpu/texture` | `gpu/image` | Its users are media, vision preprocessing and UI as much as 3D. Texture objects stay in the built-in `gpu` |
| `gpu/raytrace` | `gpu/spatial` | BVHs and nearest-neighbour queries serve physics, point clouds and vector search, not only rays |
| `gpu/prof` | `gpu/profile` | Spelled out |
| `internal/select` | `internal/dispatch` + public `gpu/gputest` | Choosing an image is internal; the oracle harness is useful to everyone |
| — | `gpu/layout`, `gpu/sparse` | Strided views are the common language; sparse is the fourth shape of data, after dense, image and spatial |
| `gpu/dtype` `Block` | `dtype.Scaled` | MX, NVFP4, GGUF k-quants and group quantization are one idea |
