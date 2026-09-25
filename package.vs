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
        .library(name: "gpu/random", targets: ["random"]),
        .library(name: "gpu/gputest", targets: ["gputest"]),
        .executable(name: "test-parallel", targets: ["test_parallel"]),
        .executable(name: "test-random", targets: ["test_random"]),
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
    ]
)
