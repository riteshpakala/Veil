//
//  Executor.swift
//  VeilKit
//
//  What Veil needs from a model family, split by responsibility (interface segregation):
//
//    PhotoEncoder          photo → latent (the last two axes tile the whole photo)
//    PromptEmbedder        prompt → the conditioning the denoiser reads
//    VelocityModel         v̂(x_σ, σ, conditioning), differentiable, in the flow convention
//    GuardSlots            the linears a guard may edit, wrapped in `HookedLinear`
//    AdapterNaming         guard deltas ↔ the tensor names hosts load
//    TokenSearchInterface  (optional) token-level access for discrete prompt search
//
//  Conventions: flow matching, x_σ = (1 − σ)·x₀ + σ·ε, v = ε − x₀. A family with another
//  parametrization converts inside its executor. Nothing here decodes a latent to pixels.
//

import Foundation
import MLX

public protocol PhotoEncoder: AnyObject {
    /// (C, H, W) float32 latent of the whole photo; the H×W cells tile the photo.
    func encodeLatent(_ photo: SubjectPhoto) throws -> MLXArray
    /// Identifies the encoder and working resolution (cache key).
    var encodingKey: String { get }
}

public struct PromptEmbedding: @unchecked Sendable {
    public let text: String
    /// The conditioning the denoiser reads, (1, T, E).
    public let value: MLXArray
    /// Real (non-padding) token ids, template tokens included; nil when the executor has none.
    public let tokenIDs: [Int]?

    public init(text: String, value: MLXArray, tokenIDs: [Int]?) {
        self.text = text
        self.value = value
        self.tokenIDs = tokenIDs
    }
}

public protocol PromptEmbedder: AnyObject {
    /// Embeddings in the order of `prompts` (cached by the executor).
    func embed(_ prompts: [String]) throws -> [PromptEmbedding]
}

public protocol VelocityModel: AnyObject {
    var identifier: String { get }
    /// Largest batch one call should take.
    var maxBatch: Int { get }
    /// σ values the deployed sampler visits for this latent shape, high to low (no terminal 0).
    func samplerSigmas(latentShape: [Int]) -> [Float]
    /// v̂ for x (B, C, H, W) at one σ; `conditioning` is (1 or B, T, E). Differentiable in x,
    /// in the conditioning and in any live guard parameters.
    func velocity(_ x: MLXArray, sigma: Float, conditioning: MLXArray) -> MLXArray
}

public struct GuardSlot: Codable, Sendable, Hashable {
    public enum Role: String, Codable, Sendable {
        /// Reads the text encoder's output directly — the closed-form edit is exact here.
        case textInput = "text-input"
        /// Carries text-derived activations only (never image tokens).
        case textPath = "text-path"
    }

    public let key: String
    public let role: Role
    public let inFeatures: Int
    public let outFeatures: Int
    /// Whether the image reads this output (a locator site: where a route can live).
    public let locatorSite: Bool

    public init(key: String, role: Role, inFeatures: Int, outFeatures: Int, locatorSite: Bool) {
        self.key = key
        self.role = role
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.locatorSite = locatorSite
    }
}

public protocol GuardSlots: AnyObject {
    var hooks: HookController { get }
    /// Every linear a guard may edit, in model order.
    var slots: [GuardSlot] { get }
    func linear(_ key: String) -> HookedLinear?
}

public protocol AdapterNaming: AnyObject {
    /// Export formats this family writes (the first is the default), e.g. ["diffusers", "bfl"].
    var exportFormats: [String] { get }
    /// Tensors (host naming) for guard deltas keyed by slot key.
    func exportTensors(_ deltas: [String: LowRankDelta], format: String) throws -> [String: MLXArray]
    /// Slot deltas from an adapter file (this family's names in any supported convention).
    func importDeltas(_ file: AdapterFile) -> (deltas: [String: LinearDelta], unmapped: [String])
}

public protocol TokenSearchInterface: AnyObject {
    /// (V, E) token-embedding table.
    func tokenTable() throws -> MLXArray
    /// Real token ids for a user text: the template prefix, the text itself, the template suffix.
    func tokenize(_ text: String) throws -> (prefix: [Int], text: [Int], suffix: [Int])
    /// Conditioning (1, T, E_cond) from the embeddings (1, n, E) of a real-token sequence
    /// (prefix + text + suffix); padding is added inside. Differentiable in the embeddings.
    func conditioning(fromTokenEmbeddings embeddings: MLXArray) -> MLXArray
    func decode(_ ids: [Int]) -> String
    /// Tokens search never proposes (specials, padding).
    var excludedTokenIDs: Set<Int> { get }
}

public protocol GuardableModel: PhotoEncoder, PromptEmbedder, VelocityModel, GuardSlots, AdapterNaming {
    var descriptor: ModelDescriptor { get }
    /// Token-level access for discrete prompt search, when the family supports it.
    var tokenSearch: TokenSearchInterface? { get }
    /// Load-time findings that limit what a run can conclude.
    var issues: [String] { get }
    /// Findings that make a verification invalid (e.g. the deployment couldn't be reproduced).
    var blockingIssues: [String] { get }
}

extension GuardableModel {
    public var blockingIssues: [String] { [] }
}

// MARK: - Sampling (decode-free)

public enum LatentSampler {
    /// Euler flow sampling over the model's own schedule: base-model latents used as fallback
    /// controls. The result is a latent; it is never decoded. With `unconditional` and a
    /// guidance g > 1, classifier-free guidance: v = v_u + g·(v_c − v_u) — how undistilled base
    /// models are deployed.
    public static func sample(_ model: VelocityModel, conditioning: MLXArray, shape: [Int], noise: MLXArray,
                              steps: Int? = nil, unconditional: MLXArray? = nil, guidance: Float = 1) -> MLXArray {
        var sigmas = model.samplerSigmas(latentShape: shape)
        if let steps, steps < sigmas.count {
            sigmas = (0..<steps).map { sigmas[Int((Double($0) * Double(sigmas.count - 1) / Double(max(steps - 1, 1))).rounded())] }
        }
        sigmas.append(0)
        var x = noise.reshaped([1] + shape)
        for i in 0..<(sigmas.count - 1) {
            var v = model.velocity(x, sigma: sigmas[i], conditioning: conditioning)
            if let unconditional, guidance != 1 {
                let vu = model.velocity(x, sigma: sigmas[i], conditioning: unconditional)
                v = vu + guidance * (v - vu)
            }
            x = x + (sigmas[i + 1] - sigmas[i]) * v
            eval(x)
        }
        return x.squeezed(axis: 0)
    }
}
