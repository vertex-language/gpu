# gpu/dtype

`Number`: the protocol kernels compute with (`float32`, `float16`, `bfloat16`, `int32`, `uint32`), with the identities, wrapping sum, order key, atomic add and `Accumulator` (`float32` for a half; `Widen`, `Narrow`) the other packages need.

```vertex
import "gpu/dtype"
```

## Types

- **`Block`** (protocol): Block is a block-quantized format. A row of n elements (n a multiple of Size) is n / Size blocks, one after another.
- **`Q4_0`** (struct): Q4_0 is ggml's q4_0: a float16 scale d, then 16 bytes of 4-bit quants.
- **`Q8_0`** (struct): Q8_0 is ggml's q8_0: a float16 scale d, then 32 int8 quants. Element j is q[j]·d.
- **`F16`** (struct): F16 is IEEE half floats as a Block: 16 of them, 32 bytes, no scale -- so that a float16 weight (safetensors' F16, GGUF's type 1) runs through the same Gemv and Dequantize as the quantized ones.
- **`BF16`** (struct): BF16 is bfloat16 as a Block: the high half of a float32, 16 of them in 32 bytes -- the dtype most checkpoints on the Hugging Face Hub hold.
- **`Q4_K`** (struct): Q4_K is ggml's q4_K: 256 elements in 144 bytes -- a float16 scale d and min dmin, 12 bytes of 6-bit scales and mins for 8 sub-blocks of 32, then 128 bytes of 4-bit quants.
- **`Q6_K`** (struct): Q6_K is ggml's q6_K: 256 elements in 210 bytes -- 128 bytes of each quant's low 4 bits, 64 of its high 2, 16 int8 scales (one a sub-block of 16), and a float16 scale d last.
- **`Number`** (protocol): Number is a type a kernel computes with: float32, float16, bfloat16, int32 or uint32.

## Functions

- `func Dequantize<B: Block>(_ b: gpu.Buffer<uint8>, _ format: B, at: int = 0, count: int, into y: gpu.Buffer<float32>) async throws`: Dequantize decodes count elements of the format, starting at byte `at` of b (a block boundary), into y.

Part of the [`gpu`](https://github.com/vertex-language/gpu) repository.
