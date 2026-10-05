# gpu/gputest

The test harness the other gpu packages use: devices to run on, exact and ULP comparisons, per-element error bounds, and random inputs and sizes.

```vertex
import "gpu/gputest"
```

## Types

- **`Random`** (struct): Random is a reproducible stream of test inputs: xorshift64*, the same sequence on every run.

## Functions

- `func Devices() -> [gpu.Device]`: Devices is the CPU device first, then every other device there is: what a check runs on. The CPU device's answer is the oracle for the rest.
- `func Equal<T: Equatable>(_ what: string, _ d: gpu.Device, _ got: [T], _ want: [T])`: Equal checks that got is want, element for element.
- `func Close(_ what: string, _ d: gpu.Device, _ got: [float32], _ want: [float32], ulps: int)`: Close checks that each element of got is within ulps units in the last place of want: for sums whose order differs from the reference's.
- `func Near(_ what: string, _ d: gpu.Device, _ got: [float32], _ want: [float64], bound: [float64])`: Near checks that each element of got is within bound[i] of want[i]: for a result whose error is bounded by what it was computed from rather than by its own size, as a dot product's is by k·ε·Σ|a·b| however much its terms cancel.
- `func Done()`: Done prints how many checks ran, and stops the program with a failure if any of them failed.

Part of the [`gpu`](https://github.com/vertex-language/gpu) repository.
