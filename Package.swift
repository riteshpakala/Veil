// swift-tools-version: 6.3
// Veil — protective adapters for foundation models. Given a model link and photos of a
// person who asked to be protected, Veil finds the prompts (routes) that pull the model
// toward that person, fits a LoRA that closes them, and verifies the shipped file:
// suppression on held-out photos, attack cost, and drift on everyone else.
//
// MLX and FluxKit come from Frigate (JIT Metal kernels, so plain `swift build` works).
// Frigate is a sibling checkout for now: branch scorpion-fluxkit-hooks (FluxKit linear
// hook, in-memory stores, VAE encoder, text encoder from a store, input-embedding path),
// on top of https://github.com/rao-studios/Frigate/commit/1d3455e9b55e9ce01b093f7546c802cc88e7a040.
// Once that branch is pushed, replace the path with:
//   .package(url: "https://github.com/rao-studios/Frigate", revision: "<sha>")

import PackageDescription

let package = Package(
    name: "Veil",

    platforms: [
        .macOS(.v14)
    ],

    products: [
        .library(name: "VeilKit", targets: ["VeilKit"]),
        .library(name: "VeilFlux2", targets: ["VeilFlux2"]),
        .executable(name: "veil", targets: ["veil"]),
        .executable(name: "VeilApp", targets: ["VeilApp"]),
    ],

    dependencies: [
        .package(path: "../Frigate"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],

    targets: [
        // Model-agnostic: links, photos, routes, pull, guard fitting, verification, export.
        .target(
            name: "VeilKit",
            dependencies: [
                .product(name: "MLX", package: "Frigate"),
                .product(name: "MLXRandom", package: "Frigate"),
                .product(name: "MLXNN", package: "Frigate"),
                .product(name: "MLXLinalg", package: "Frigate"),
                .product(name: "MLXOptimizers", package: "Frigate"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // FLUX.2 Klein executor (Frigate FluxKit). A separate target keeps VeilKit
        // model-agnostic; the executables register it with the ExecutorRegistry.
        .target(
            name: "VeilFlux2",
            dependencies: [
                "VeilKit",
                .product(name: "FluxKit", package: "Frigate"),
                .product(name: "MLX", package: "Frigate"),
                .product(name: "MLXNN", package: "Frigate"),
                .product(name: "Hub", package: "Frigate"),
                .product(name: "Tokenizers", package: "Frigate"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "veil",
            dependencies: [
                "VeilKit",
                "VeilFlux2",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "VeilApp",
            dependencies: ["VeilKit", "VeilFlux2"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "VeilKitTests",
            dependencies: [
                "VeilKit",
                .product(name: "MLX", package: "Frigate"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "VeilFlux2Tests",
            dependencies: [
                "VeilFlux2",
                "VeilKit",
                .product(name: "FluxKit", package: "Frigate"),
                .product(name: "MLX", package: "Frigate"),
                .product(name: "MLXNN", package: "Frigate"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
