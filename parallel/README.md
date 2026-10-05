# gpu/parallel

Generic over `dtype.Number`, one definition each: `Reduce` (`.Sum .Min .Max`), `Scan` (inclusive, exclusive), `Sort` (stable radix in total order; optional `uint32` values), `TopK`, `Select`, `SelectIndices`, `Count` (with `Where`, exact for integers), `Gather`, `Scatter`, `ScatterAdd`, `Histogram`, `Iota`.

```vertex
import "gpu/parallel"
```

## Types

- **`Reduction`** (enum): Reduction is what Reduce combines elements with.
- **`Where`** (enum): Where is the condition Select, SelectIndices and Count keep an element by: how it compares with a value.

## Functions

- `func GroupRank() -> int`: GroupRank is this work-item's index within its workgroup, counting x fastest: the order the group-scope functions reduce and scan in.
- `func GroupCount() -> int`: GroupCount is how many work-items the workgroup has.
- `func Histogram(_ b: gpu.Buffer<uint32>, bins: int) async throws -> gpu.Buffer<uint32>`: Histogram counts the values of b in bins 0..<bins: counts[v] is how many elements equal v. A value of bins or more is not counted.
- `func Iota(_ d: gpu.Device, count: int, from start: uint32 = 0) async throws -> gpu.Buffer<uint32>`: Iota is count consecutive integers from start, on d.
- `func Gather<T: dtype.Number>(_ src: gpu.Buffer<T>, _ indices: gpu.Buffer<uint32>) async throws -> gpu.Buffer<T>`: Gather is src at each of indices: out[i] = src[indices[i]].
- `func Scatter<T: dtype.Number>(_ src: gpu.Buffer<T>, _ indices: gpu.Buffer<uint32>, into dst: gpu.Buffer<T>) async throws`: Scatter writes each element of src to dst at its index: dst[indices[i]] = src[i]. Where two indices are the same, which write lands is not defined.
- `func ScatterAdd<T: dtype.Number>(_ src: gpu.Buffer<T>, _ indices: gpu.Buffer<uint32>, into dst: gpu.Buffer<T>) async throws`: ScatterAdd adds each element of src into dst at its index, atomically: dst[indices[i]] += src[i].
- `func GroupSum<T: dtype.Number>(_ x: T) -> T`: GroupSum is the sum of x over the workgroup, added in a fixed tree order; an integer sum wraps.
- `func GroupMin<T: dtype.Number>(_ x: T) -> T`: GroupMin is the least x in the workgroup.
- `func GroupMax<T: dtype.Number>(_ x: T) -> T`: GroupMax is the greatest x in the workgroup.
- and 10 more

Part of the [`gpu`](https://github.com/vertex-language/gpu) repository.
