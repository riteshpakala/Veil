//
//  Flux2GuardModel.swift
//  VeilFlux2
//
//  FLUX.2 Klein as a `GuardableModel`. Every linear of the transformer is wrapped in a
//  `HookedLinear` at load time (FluxKit's linear hook), so one transformer serves as the
//  deployment (guard off) and the guarded model (guard on), sharing every weight.
//
//  - Latents: the VAE's normalized latents unpacked to (32, H/8, W/8); the transformer sees 2×2-
//    packed 128-channel tokens.
//  - Velocity: the transformer's own output (flow matching, timestep = 1000·σ), float32.
//  - σ grid: the deployed sampler's (Flux2Scheduler at the working resolution): 4 steps for
//    distilled Klein, 50 for the base.
//  - Guard slots: the text path only — `context_embedder` (reads Qwen3's output: the closed-form
//    slot) and each double block's text-stream projections and feed-forward. Image-path and
//    single-stream weights are never edited.
//

import CoreGraphics
import FluxKit
import Foundation
import MLX
import MLXNN
import Tokenizers
import VeilKit

public final class Flux2GuardModel: GuardableModel, @unchecked Sendable {
    public let descriptor: ModelDescriptor
    public var issues: [String] = []
    public var blockingIssues: [String] = []
    /// The controller every hooked linear was built with.
    public let hooks: HookController
    public var maxBatch: Int
    public let transformer: Flux2Transformer
    public let transformerConfig: Flux2Transformer.Config
    let hooked: [String: HookedLinear]
    let pipeline: Flux2Pipeline
    let vaeDirectory: URL
    let textEncoderSource: TextEncoderSource
    let longSide: Int
    let samplerSteps: Int
    private var textEncoder: Qwen3TextEncoder?
    private var embeddingCache: [String: PromptEmbedding] = [:]
    private var ids: [String: MLXArray] = [:]
    private let lock = NSRecursiveLock()

    public enum TextEncoderSource: Sendable {
        /// A diffusers/HF Qwen3 folder (bf16), with its config.
        case repo(URL)
        /// An mflux export's text encoder (4-bit).
        case export(URL)

        var label: String {
            switch self {
            case .repo(let u): return "repo:\(u.lastPathComponent)"
            case .export: return "mflux 4-bit export"
            }
        }
    }

    init(descriptor: ModelDescriptor, transformer: Flux2Transformer, config: Flux2Transformer.Config,
         hooked: [String: HookedLinear], hooks: HookController, tokenizerDirectory: URL, vaeDirectory: URL,
         textEncoder: TextEncoderSource, longSide: Int, maxBatch: Int, samplerSteps: Int) {
        self.descriptor = descriptor
        self.transformer = transformer
        self.transformerConfig = config
        self.hooked = hooked
        self.pipeline = Flux2Pipeline(modelDirectory: tokenizerDirectory)
        self.vaeDirectory = vaeDirectory
        self.textEncoderSource = textEncoder
        self.longSide = max(16, longSide / 16 * 16)
        self.maxBatch = maxBatch
        self.samplerSteps = samplerSteps
        self.hooks = hooks
    }

    public var identifier: String { "flux2-klein:\(descriptor.pinnedName)" }
    public var encodingKey: String { "flux2-vae|\(longSide)" }

    // MARK: Photos

    /// Pixel size a photo is resized to: aspect preserved, long side `longSide`, multiples of 16.
    public func workingSize(width: Int, height: Int) -> (width: Int, height: Int) {
        let s = Double(longSide) / Double(max(width, height))
        func snap(_ v: Double) -> Int { max(16, Int((v / 16).rounded()) * 16) }
        return (snap(Double(width) * s), snap(Double(height) * s))
    }

    public func encodeLatent(_ photo: SubjectPhoto) throws -> MLXArray {
        let (w, h) = workingSize(width: photo.width, height: photo.height)
        let px = photo.rgbPlanar(width: w, height: h)
        let nhwc = (MLXArray(px, [3, h, w]).transposed(1, 2, 0) * 2 - 1).expandedDimensions(axis: 0)
        let encoder = try Flux2VAEEncoder(componentDir: vaeDirectory)
        let packed = encoder.encodePackedNormalized(nhwc).asType(.float32)
        let latent = Flux2Packing.unpack(packed, packedH: h / 16, packedW: w / 16).squeezed(axis: 0)
        eval(latent)
        Memory.clearCache()
        return latent
    }

    // MARK: Text

    func encoder() throws -> Qwen3TextEncoder {
        try lock.withLock {
            if let textEncoder { return textEncoder }
            let e: Qwen3TextEncoder
            switch textEncoderSource {
            case .repo(let dir):
                e = try Qwen3TextEncoder(store: Flux2Components.textEncoderStore(dir), config: Flux2Components.textEncoderConfig(dir))
            case .export(let dir):
                e = try Qwen3TextEncoder(componentDir: dir)
            }
            textEncoder = e
            return e
        }
    }

    /// Frees the text encoder (embeddings stay cached).
    public func releaseTextEncoder() {
        lock.withLock { textEncoder = nil }
        Memory.clearCache()
    }

    public func embed(_ prompts: [String]) throws -> [PromptEmbedding] {
        try lock.withLock {
            let missing = Array(Set(prompts.filter { embeddingCache[$0] == nil }))
            if !missing.isEmpty {
                let e = try encoder()
                for p in missing {
                    let (idArray, mask) = try pipeline.tokenize(prompt: p)
                    let value = e.promptEmbeddings(inputIDs: idArray, attentionMask: mask)
                    eval(value)
                    let real = mask.sum().item(Int32.self)
                    let tokens = idArray[0, ..<Int(real)].asArray(Int32.self).map(Int.init)
                    embeddingCache[p] = PromptEmbedding(text: p, value: value, tokenIDs: tokens)
                }
            }
            return prompts.map { embeddingCache[$0]! }
        }
    }

    // MARK: Velocity

    func positionIDs(packedH: Int, packedW: Int, text: Int) -> (MLXArray, MLXArray) {
        lock.withLock {
            let key = "\(packedH)x\(packedW)|\(text)"
            if let img = ids[key + "|img"], let txt = ids[key + "|txt"] { return (img, txt) }
            let img = Flux2Pipeline.latentIDs(packedH: packedH, packedW: packedW)
            let txt = Flux2Pipeline.textIDs(count: text)
            ids[key + "|img"] = img
            ids[key + "|txt"] = txt
            return (img, txt)
        }
    }

    public func samplerSigmas(latentShape: [Int]) -> [Float] {
        guard latentShape.count == 3 else { return [1, 0.75, 0.5, 0.25] }
        let seq = (latentShape[1] / 2) * (latentShape[2] / 2)
        return Array(Flux2Scheduler(imageSequenceLength: seq, steps: samplerSteps).sigmas.dropLast())
    }

    public func velocity(_ x: MLXArray, sigma: Float, conditioning: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let encoder = conditioning.dim(0) == b ? conditioning
            : broadcast(conditioning, to: [b] + Array(conditioning.shape.dropFirst()))
        let (hp, wp) = (x.dim(2) / 2, x.dim(3) / 2)
        let (imgIDs, txtIDs) = positionIDs(packedH: hp, packedW: wp, text: encoder.dim(1))
        let v = transformer(hidden: Flux2Packing.pack(x), encoder: encoder, timestep: sigma * 1000, imgIDs: imgIDs, txtIDs: txtIDs)
        return Flux2Packing.unpack(v.asType(.float32), packedH: hp, packedW: wp)
    }

    // MARK: Slots

    public var slots: [GuardSlot] {
        var keys = ["context_embedder"]
        for i in 0..<transformerConfig.numLayers {
            keys += ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"].map { "transformer_blocks.\(i).attn.\($0)" }
            keys += ["linear_in", "linear_out"].map { "transformer_blocks.\(i).ff_context.\($0)" }
        }
        let last = transformerConfig.numLayers - 1
        return keys.compactMap { key in
            guard let l = hooked[key] else { return nil }
            let (out, inF) = l.logicalShape
            let site = key.hasSuffix("add_k_proj") || key.hasSuffix("add_v_proj") || key == "transformer_blocks.\(last).ff_context.linear_out"
            return GuardSlot(key: key, role: key == "context_embedder" ? .textInput : .textPath, inFeatures: inF, outFeatures: out,
                             locatorSite: site)
        }
    }

    public func linear(_ key: String) -> HookedLinear? { hooked[key] }

    public var linearKeys: Set<String> { Set(hooked.keys) }

    // MARK: Adapter naming

    public var exportFormats: [String] { ["diffusers", "bfl"] }

    public func exportTensors(_ deltas: [String: LowRankDelta], format: String) throws -> [String: MLXArray] {
        format == "bfl" ? try Flux2KeyMap.bflTensors(deltas) : Flux2KeyMap.diffusers(deltas)
    }

    public func importDeltas(_ file: AdapterFile) -> (deltas: [String: LinearDelta], unmapped: [String]) {
        let mapping = Flux2KeyMap.map(file.updates, linearKeys: linearKeys)
        var out: [String: LinearDelta] = [:], bad = mapping.unmapped + mapping.unsupported
        for (key, update) in mapping.updates {
            let delta = LinearDelta(update)
            guard let l = hooked[key], l.logicalShape == delta.shape else { bad.append(key); continue }
            out[key] = delta
        }
        return (out, bad.sorted())
    }

    public var tokenSearch: TokenSearchInterface? { Flux2TokenSearch(model: self) }
}

/// Token-level access for discrete prompt search: Qwen3's embedding table, the chat template's
/// fixed prefix and suffix, and differentiable conditioning from token embeddings.
final class Flux2TokenSearch: TokenSearchInterface {
    unowned let model: Flux2GuardModel
    private var template: (prefix: Int, suffix: Int)?

    init(model: Flux2GuardModel) { self.model = model }

    func tokenTable() throws -> MLXArray { try model.encoder().embeddingTable() }

    func realIDs(_ text: String) throws -> [Int] {
        let (ids, mask) = try model.pipeline.tokenize(prompt: text)
        let n = Int(mask.sum().item(Int32.self))
        return ids[0, ..<n].asArray(Int32.self).map(Int.init)
    }

    func tokenize(_ text: String) throws -> (prefix: [Int], text: [Int], suffix: [Int]) {
        if template == nil {
            let a = try realIDs("x"), b = try realIDs("y")
            var p = 0
            while p < min(a.count, b.count), a[p] == b[p] { p += 1 }
            var s = 0
            while s < min(a.count, b.count) - p, a[a.count - 1 - s] == b[b.count - 1 - s] { s += 1 }
            template = (p, s)
        }
        let ids = try realIDs(text)
        let (p, s) = template!
        return (Array(ids.prefix(p)), Array(ids.dropFirst(p).dropLast(s)), Array(ids.suffix(s)))
    }

    func conditioning(fromTokenEmbeddings embeddings: MLXArray) -> MLXArray {
        let n = embeddings.dim(1), total = 512
        guard let encoder = try? model.encoder() else { return embeddings }
        let padID = 151_643
        let pad = encoder.tokenEmbeddings(MLXArray([Int32(padID)]).reshaped(1, 1))
        let padded = n < total ? concatenated([embeddings.asType(pad.dtype), broadcast(pad, to: [1, total - n, pad.dim(2)])], axis: 1)
            : embeddings[0..., ..<total, 0...]
        let mask = MLXArray((0..<total).map { Int32($0 < n ? 1 : 0) }).reshaped(1, total)
        return encoder.promptEmbeddings(inputEmbeddings: padded, attentionMask: mask)
    }

    func decode(_ ids: [Int]) -> String {
        (try? model.pipeline.tokenizer().decode(tokens: ids)) ?? ids.map(String.init).joined(separator: " ")
    }

    var excludedTokenIDs: Set<Int> { Set(151_643..<152_000) }
}
