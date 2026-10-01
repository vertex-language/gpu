# gpu/attention

`Forward`: FlashAttention's online softmax over key blocks (no scores matrix), a workgroup per query row; `Mask` `.None`, `.Causal` (with a KV-cache offset), `.SlidingWindow(n)`; grouped- and multi-query heads through `Shape`; head dims to 128; float32.

```vertex
import "gpu/attention"
```

## Types

- **`Mask`** (enum): Mask is which keys a query may attend to.
- **`Shape`** (struct): Shape is an attention problem: batch, query heads, key/value heads (fewer for grouped-query attention), queries and keys per sequence, the head dimension, and the score scale (0 for 1/√headDim).

## Functions

- `func Forward(q: gpu.Buffer<float32>, k: gpu.Buffer<float32>, v: gpu.Buffer<float32>, into o: gpu.Buffer<float32>, _ shape: Shape, mask: Mask = .None) async throws`: Forward is softmax(scale · Q·Kᵀ + mask) · V, written into o. scale is usually 1/√d. A query that may attend to no key gets zeros.

Part of the [`gpu`](https://github.com/vertex-language/gpu) repository.
