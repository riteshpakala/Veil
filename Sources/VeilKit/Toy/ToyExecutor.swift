//
//  ToyExecutor.swift
//  VeilKit
//
//  The toy world as a `GuardableModel`: two hooked slots, an exact velocity, token search, and
//  its own adapter naming ("toy.<slot>.lora_A/B.weight"). Selected with the link "toy".
//

import Foundation
import MLX
import MLXNN

public final class ToyModel: GuardableModel, @unchecked Sendable {
    public let world: ToyIdentityWorld
    public let hooks = HookController()
    let contextEmbedder: HookedLinear
    let reader: HookedLinear
    public let descriptor: ModelDescriptor
    public var issues: [String] = []
    public var blockingIssues: [String] = []
    public var maxBatch = 64

    public init(world: ToyIdentityWorld = .shared) {
        self.world = world
        contextEmbedder = HookedLinear(key: "context_embedder", base: Linear(weight: world.contextWeight), controller: hooks)
        reader = HookedLinear(key: "reader", base: Linear(weight: world.readerWeight), controller: hooks)
        descriptor = ToyExecutor.descriptor(for: "toy")!
    }

    public var identifier: String { "toy" }
    public var encodingKey: String { "toy.v1" }

    public func encodeLatent(_ photo: SubjectPhoto) throws -> MLXArray { world.latent(of: photo) }

    public func embed(_ prompts: [String]) throws -> [PromptEmbedding] { prompts.map(world.text.encode) }

    public func samplerSigmas(latentShape: [Int]) -> [Float] { [0.95, 0.85, 0.72, 0.6, 0.48, 0.36, 0.24, 0.12] }

    /// Conditioning mean m(c), (B, C, H, W).
    func conditionalMean(_ conditioning: MLXArray, batch: Int) -> MLXArray {
        let c = contextEmbedder(conditioning.asType(.float32))                 // (b, T, H)
        let m = reader(c.mean(axis: 1))                                        // (b, D)
        let shaped = m.reshaped(m.dim(0), world.channels, world.side, world.side)
        return m.dim(0) == batch ? shaped : broadcast(shaped, to: [batch, world.channels, world.side, world.side])
    }

    public func velocity(_ x: MLXArray, sigma: Float, conditioning: MLXArray) -> MLXArray {
        let m = conditionalMean(conditioning, batch: x.dim(0))
        let a = 1 - sigma, s2 = world.photoStd * world.photoStd
        let den = a * a * s2 + sigma * sigma
        let kappa = a * s2 / den, beta = sigma / den
        return (beta - kappa) * (x.asType(.float32) - a * m) - m
    }

    public var slots: [GuardSlot] {
        [GuardSlot(key: "context_embedder", role: .textInput, inFeatures: world.text.tokenDim, outFeatures: world.contextDim,
                   locatorSite: false),
         GuardSlot(key: "reader", role: .textPath, inFeatures: world.contextDim, outFeatures: world.latentDim, locatorSite: true)]
    }

    public func linear(_ key: String) -> HookedLinear? {
        switch key {
        case "context_embedder": return contextEmbedder
        case "reader": return reader
        default: return nil
        }
    }

    // MARK: Adapter naming

    public var exportFormats: [String] { ["diffusers"] }

    public func exportTensors(_ deltas: [String: LowRankDelta], format: String) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, d) in deltas {
            out["toy.\(key).lora_A.weight"] = d.down
            out["toy.\(key).lora_B.weight"] = d.up
        }
        return out
    }

    public func importDeltas(_ file: AdapterFile) -> (deltas: [String: LinearDelta], unmapped: [String]) {
        var out: [String: LinearDelta] = [:], unmapped: [String] = []
        for (name, update) in file.updates {
            let key = name.hasPrefix("toy.") ? String(name.dropFirst(4)) : name
            if linear(key) != nil { out[key] = LinearDelta(update) } else { unmapped.append(name) }
        }
        return (out, unmapped.sorted())
    }

    public var tokenSearch: TokenSearchInterface? { ToyTokenSearch(text: world.text) }
}

final class ToyTokenSearch: TokenSearchInterface {
    let text: ToyText

    init(text: ToyText) { self.text = text }

    func tokenTable() throws -> MLXArray { text.table }

    func tokenize(_ s: String) throws -> (prefix: [Int], text: [Int], suffix: [Int]) {
        let ids = text.tokenize(s)
        return ([ids[0]], Array(ids.dropFirst().dropLast()), [ids[ids.count - 1]])
    }

    func conditioning(fromTokenEmbeddings embeddings: MLXArray) -> MLXArray { text.contextualize(embeddings) }

    func decode(_ ids: [Int]) -> String { ids.map(text.word).joined(separator: " ") }

    var excludedTokenIDs: Set<Int> { [text.beginID, text.endID, text.padID] }
}

public enum ToyExecutor {
    /// Descriptor for the link "toy" (nil for anything else).
    public static func descriptor(for link: String) -> ModelDescriptor? {
        let l = link.trimmingCharacters(in: .whitespaces).lowercased()
        guard l == "toy" || l.hasPrefix("toy:") else { return nil }
        return ModelDescriptor(link: link, host: .local, repo: "veil/toy-identity-world", family: "toy",
                               detail: "analytic conditional-Gaussian world (exact velocity)", variant: "base",
                               components: ["text_encoder", "transformer"], supported: true,
                               notes: ["Known answer: \(ToyIdentityWorld.subjectName) is reachable by name and by one description."])
    }

    @Sendable public static func make(_ request: ExecutorRequest) async throws -> GuardableModel {
        let model = ToyModel()
        if let adapter = request.withAdapter {
            let file = try AdapterFile.load(adapter)
            let (deltas, unmapped) = model.importDeltas(file)
            for (key, delta) in deltas { model.linear(key)?.fixed = delta }
            if !unmapped.isEmpty { model.blockingIssues.append("with-adapter: \(unmapped.count) modules not placed: \(unmapped.prefix(4).joined(separator: ", "))") }
        }
        return model
    }
}
