# gpu/linalg

`Matmul` over a `Shape` (m, n, k, batch, transposes) with an optional fused `Epilogue` (scale, bias, residual, `.ReLU`), tiled through shared storage; `Gemv`, and `Gemv` of block-quantized weights (`dtype.Q4_0`, `dtype.Q8_0`: a kernel specialized per format.

```vertex
import "gpu/linalg"
```

## Types

- **`Activation`** (enum): Activation is applied to each element of a Matmul's result, after scale, bias and residual. GELU and SiLU wait on math inside kernels.
- **`Shape`** (struct): Shape is a matrix product's problem: C (m x n) = A (m x k) · B (k x n), batch of them laid end to end, with A or B read transposed -- stored k x m, or n x k.
- **`Epilogue`** (struct): Epilogue is what Matmul does to each element of A · B before it writes it: C = activation(scale · (A · B) + bias[column] + residual).

## Functions

- `func Gemv(_ parts: [gpu.Buffer<float32>], rows: [int], _ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, k: int, accumulate: bool = false) async throws`: Gemv is y = [A0; A1; A2] · x for float32 weights stacked by rows: parts holds one to three weights of rows[i] rows of k each.
- `func Matmul<T: dtype.Number>(_ a: gpu.Buffer<T>, _ b: gpu.Buffer<T>, into c: gpu.Buffer<T>, _ shape: Shape) async throws`: Matmul is C = A · B for the problem shape: all row-major.
- `func Matmul<T: dtype.Number>(_ a: gpu.Buffer<T>, _ b: gpu.Buffer<T>, into c: gpu.Buffer<T>, _ s: Shape, _ e: Epilogue<T>) async throws`: Matmul is C = epilogue(A · B) for the problem shape.
- `func Gemv<T: dtype.Number>(_ a: gpu.Buffer<T>, _ x: gpu.Buffer<T>, into y: gpu.Buffer<T>, m: int, k: int, accumulate: bool = false) async throws`: Gemv is y = A · x: A is m x k, row-major; x has k elements and y m. With accumulate, y = A · x + y: a residual added in the same pass.
- `func Gemv<B: dtype.Block>(_ w: gpu.Buffer<uint8>, _ format: B, _ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, m: int, k: int, accumulate: bool = false) async throws`: Gemv is y = W · x with W block-quantized: m rows of k elements, each row k / format.Size() blocks of the format, one after another -- a GGUF tensor's bytes as they are.
- `func Gemv<B: dtype.Block>(_ parts: [gpu.Buffer<uint8>], rows: [int], _ format: B, _ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, k: int, accumulate: bool = false) async throws`: Gemv is y = [W0; W1; W2] · x for block-quantized weights of one format stacked by rows: parts holds one to three weights of rows[i] rows of k.
- `func Transpose<T: dtype.Number>(_ x: gpu.Buffer<T>, into y: gpu.Buffer<T>, rows: int, cols: int) async throws`: Transpose writes the rows x cols row-major matrix x into y as its cols x rows transpose.

Part of the [`gpu`](https://github.com/vertex-language/gpu) repository.
