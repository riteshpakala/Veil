//
//  HookedLinear.swift
//  VeilKit
//  Generalizes Scorpion's DeltaLinear (github.com/riteshpakala/Scorpion, ScorpionFlux2), GPL-3.0.
//
//  Every editable linear of a model is wrapped once at load time. Its output is
//
//      y = base(x) [+ fixed(x)] [+ guard(x)]        then optionally recorded or restored
//
//  - fixed: an adapter that is part of the deployment being protected (base + community LoRA);
//    always on.
//  - guard: the protective delta — frozen (a fitted or exported guard) or live (the parameters
//    being trained). Applied only while `HookController.guardEnabled`; off, the model is
//    *exactly* the deployment (the branch is skipped, not multiplied by zero).
//  - record / restore: capture this output on one pass and substitute it on another (the
//    locator's activation restoration).
//
//  One model therefore serves as both "base" and "guarded", sharing every weight bit for bit.
//

import Foundation
import MLX
import MLXNN

/// A factored weight delta y += x·downᵀ·upᵀ, scale folded into `up`. Float32.
public struct LowRankDelta: @unchecked Sendable {
    /// (out, r)
    public let up: MLXArray
    /// (r, in)
    public let down: MLXArray

    public init(up: MLXArray, down: MLXArray) {
        self.up = up
        self.down = down
    }

    public var rank: Int { down.dim(0) }
    public var outFeatures: Int { up.dim(0) }
    public var inFeatures: Int { down.dim(1) }
    public var dense: MLXArray { matmul(up.asType(.float32), down.asType(.float32)) }

    public func apply(_ x: MLXArray) -> MLXArray {
        matmul(matmul(x.asType(.float32), down.asType(.float32).transposed()), up.asType(.float32).transposed())
    }

    public static func zero(out: Int, in inFeatures: Int, rank: Int) -> LowRankDelta {
        LowRankDelta(up: MLXArray.zeros([out, rank]), down: MLXArray.zeros([rank, inFeatures]))
    }
}

/// A delta that may be dense (full diffs, LoHa/LoKr) or factored.
public enum LinearDelta: @unchecked Sendable {
    case lowRank(LowRankDelta)
    case dense(MLXArray)

    public init(_ update: WeightUpdate) {
        switch update {
        case .lowRank(let up, let down, let scale):
            self = .lowRank(LowRankDelta(up: (up * scale).asType(.float32), down: down.asType(.float32)))
        case .dense(let w):
            self = .dense(w.asType(.float32))
        }
    }

    public var shape: (out: Int, in: Int) {
        switch self {
        case .lowRank(let d): return (d.outFeatures, d.inFeatures)
        case .dense(let w): return (w.dim(0), w.dim(1))
        }
    }

    public var materialized: MLXArray {
        switch self {
        case .lowRank(let d): return d.dense
        case .dense(let w): return w
        }
    }

    public func apply(_ x: MLXArray) -> MLXArray {
        switch self {
        case .lowRank(let d): return d.apply(x)
        case .dense(let w): return matmul(x.asType(.float32), w.transposed())
        }
    }
}

/// Guard parameters being trained: [down₀, up₀, down₁, up₁, …] in `keys` order. The trainer
/// swaps `params` for tracer arrays while it differentiates.
public final class TrainableGuard: @unchecked Sendable {
    public let keys: [String]
    public var params: [MLXArray]
    private let index: [String: Int]

    public init(deltas: [(key: String, delta: LowRankDelta)]) {
        keys = deltas.map(\.key)
        params = deltas.flatMap { [$0.delta.down, $0.delta.up] }
        index = Dictionary(uniqueKeysWithValues: keys.enumerated().map { ($1, $0) })
    }

    public func delta(for key: String) -> LowRankDelta? {
        guard let i = index[key] else { return nil }
        return LowRankDelta(up: params[2 * i + 1], down: params[2 * i])
    }

    /// Current values as frozen deltas.
    public func frozen() -> [String: LowRankDelta] {
        Dictionary(uniqueKeysWithValues: keys.map { ($0, delta(for: $0)!) })
    }
}

/// Shared by every hooked linear of one model; read while a forward graph is built.
public final class HookController: @unchecked Sendable {
    /// Guard deltas apply only while true. Off = the model as deployed.
    public var guardEnabled = false
    /// Live parameters during training (take precedence over frozen guard deltas).
    public var trainable: TrainableGuard?
    /// Outputs of these keys are recorded on each pass.
    public var recordKeys: Set<String> = []
    public private(set) var recorded: [String: MLXArray] = [:]
    /// Outputs substituted for these keys.
    public var restore: [String: MLXArray] = [:]
    private let lock = NSLock()

    public init() {}

    func record(_ key: String, _ value: MLXArray) { lock.withLock { recorded[key] = value } }

    public func clearRecordings() { lock.withLock { recorded.removeAll() } }

    /// Run `body` with the guard switched on or off, restoring the previous state.
    public func with<T>(guard enabled: Bool, _ body: () throws -> T) rethrows -> T {
        let previous = guardEnabled
        guardEnabled = enabled
        defer { guardEnabled = previous }
        return try body()
    }
}

public final class HookedLinear: Linear {
    public let key: String
    public let base: Linear
    let controller: HookController
    /// Part of the deployment (e.g. a community LoRA); always applied.
    public var fixed: LinearDelta?
    /// Frozen guard; applied while the controller's guard is enabled.
    public var guardDelta: LowRankDelta?

    public init(key: String, base: Linear, controller: HookController) {
        self.key = key
        self.base = base
        self.controller = controller
        super.init(weight: base.weight, bias: base.bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = base(x)
        if let fixed { y = y + fixed.apply(x).asType(y.dtype) }
        if controller.guardEnabled, let g = controller.trainable?.delta(for: key) ?? guardDelta {
            y = y + g.apply(x).asType(y.dtype)
        }
        if let substitute = controller.restore[key] { y = substitute }
        if controller.recordKeys.contains(key) { controller.record(key, y) }
        return y
    }

    /// The base weight (out, in) in float32, dequantized when stored quantized; the fixed
    /// deployment delta included, since a guard edits the deployed layer.
    public func deployedWeight() -> MLXArray {
        var w = base.denseWeight.asType(.float32)
        if let fixed { w = w + fixed.materialized }
        return w
    }
}

extension Linear {
    /// (out, in) of the layer as used (quantized layers store packed weights).
    public var logicalShape: (out: Int, in: Int) {
        if let h = self as? HookedLinear { return h.base.logicalShape }
        if let q = self as? QuantizedLinear { return (q.weight.dim(0), q.scales.dim(1) * q.groupSize) }
        return (weight.dim(0), weight.dim(1))
    }

    /// The weight as a dense (out, in) array.
    public var denseWeight: MLXArray {
        if let h = self as? HookedLinear { return h.base.denseWeight }
        if let q = self as? QuantizedLinear {
            return dequantized(q.weight, scales: q.scales, biases: q.biases, groupSize: q.groupSize, bits: q.bits)
        }
        return weight
    }
}

extension GuardSlots {
    /// Install frozen guard deltas (nil clears them).
    public func installGuard(_ deltas: [String: LowRankDelta]?) {
        for slot in slots { linear(slot.key)?.guardDelta = deltas?[slot.key] }
    }
}
