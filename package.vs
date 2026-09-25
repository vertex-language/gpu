// The 'gpu' repository: the shared accelerated functions over the
// built-in gpu module. See README.md for what each package provides.
import PackageDescription

let package = Package(
    name: "gpu",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .library(name: "gpu/dtype", targets: ["dtype"]),
        .library(name: "gpu/parallel", targets: ["parallel"]),
        .library(name: "gpu/linalg", targets: ["linalg"]),
        .library(name: "gpu/neural", targets: ["neural"]),
        .library(name: "gpu/random", targets: ["random"]),
        .library(name: "gpu/gputest", targets: ["gputest"]),
        .executable(name: "test-parallel", targets: ["test_parallel"]),
        .executable(name: "test-random", targets: ["test_random"]),
        .executable(name: "test-linalg", targets: ["test_linalg"]),
        .executable(name: "test-neural", targets: ["test_neural"]),
    ],
    targets: [
        // Reduce, scan and sort, as host operations and as group-scope
        // device functions any kernel can call.
        // The element types kernels compute with.
        .target(
            name: "dtype",
            path: "dtype"
        ),
        .target(
            name: "parallel",
            dependencies: ["dtype"],
            path: "parallel"
        ),
        // Dense linear algebra: Matmul with epilogues, Gemv, Transpose.
        .target(
            name: "linalg",
            dependencies: ["dtype", "parallel"],
            path: "linalg"
        ),
        // Neural-network building blocks: softmax, norms, activations,
        // RoPE, cross entropy.
        .target(
            name: "neural",
            dependencies: ["parallel"],
            path: "neural"
        ),
        // Counter-based random numbers: keys, streams, fills.
        .target(
            name: "random",
            path: "random"
        ),
        // The oracle harness: every device against the CPU device and a
        // host reference.
        .target(
            name: "gputest",
            path: "gputest"
        ),
        .executableTarget(
            name: "test_parallel",
            dependencies: ["parallel", "dtype", "gputest"],
            path: "tests/parallel"
        ),
        .executableTarget(
            name: "test_random",
            dependencies: ["random", "gputest"],
            path: "tests/random"
        ),
        .executableTarget(
            name: "test_linalg",
            dependencies: ["linalg", "dtype", "gputest"],
            path: "tests/linalg"
        ),
        .executableTarget(
            name: "test_neural",
            dependencies: ["neural", "gputest"],
            path: "tests/neural"
        ),
    ]
)
