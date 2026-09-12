//
//  Flux2Components.swift
//  VeilFlux2
//  Weight-layout helpers adapted from Scorpion (github.com/riteshpakala/Scorpion, ScorpionFlux2), GPL-3.0.
//
//  Where a linked FLUX.2 Klein model's parts come from, and how they become FluxKit layers:
//
//    transformer   the linked repo: diffusers (`Flux2Transformer2DModel`, renamed to FluxKit's
//                  keys) or an mflux export; config from its config.json (so 4B and 9B both load)
//    text encoder  the linked repo's Qwen3 in bf16 (`model.` prefix stripped) — what hosts run —
//                  or, on request, the shared mflux export's 4-bit copy
//    tokenizer     the linked repo's, else the shared export's
//    VAE encoder   the shared mflux export (the same FLUX.2 VAE for every Klein variant; only
//                  used to encode the person's photos)
//

import FluxKit
import Foundation
import Hub
import MLX
import VeilKit

public enum Flux2Components {
    public static let sharedExportRepo = "mlx-community/flux2-klein-4b-4bit"

    /// The mflux export holding the VAE (and, on request, a 4-bit text encoder): `VEIL_FLUX2_DIR`,
    /// else the Swift Hub default location.
    public static var sharedExportDirectory: URL {
        if let env = ProcessInfo.processInfo.environment["VEIL_FLUX2_DIR"], !env.isEmpty {
            return URL(fileURLWithPath: (env as NSString).expandingTildeInPath)
        }
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return documents.appendingPathComponent("huggingface/models/\(sharedExportRepo)")
    }

    public enum ComponentError: Error, LocalizedError {
        case missing(String)
        case layout(String)

        public var errorDescription: String? {
            switch self {
            case .missing(let s): return "FLUX.2 Klein component not found: \(s)"
            case .layout(let s): return s
            }
        }
    }

    // MARK: Transformer

    /// FluxKit (mflux) key for a diffusers Flux2Transformer2DModel key.
    public static func fluxKitKey(_ diffusersKey: String) -> String {
        diffusersKey.replacingOccurrences(of: ".attn.to_out.0.", with: ".attn.to_out.")
            .replacingOccurrences(of: "time_guidance_embed.timestep_embedder.", with: "time_guidance_embed.")
    }

    /// Diffusers module name for a FluxKit linear key (inverse of `fluxKitKey` on module names).
    public static func diffusersModule(_ key: String) -> String {
        var k = key
        if k.hasPrefix("transformer_blocks."), k.hasSuffix(".attn.to_out") { k += ".0" }
        if k.hasPrefix("time_guidance_embed.") { k = k.replacingOccurrences(of: "time_guidance_embed.", with: "time_guidance_embed.timestep_embedder.") }
        return k
    }

    /// A diffusers transformer folder (or one .safetensors file) renamed to FluxKit's keys.
    /// `quantizeBits` re-quantizes 2-D weights (4 or 8); nil keeps bf16.
    public static func diffusersTransformer(at url: URL, quantizeBits: Int? = nil) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for file in try safetensorsFiles(url) {
            for (key, value) in try loadArrays(url: file) {
                let k = fluxKitKey(key)
                if let bits = quantizeBits, k.hasSuffix(".weight"), value.ndim == 2, value.dim(1) % 64 == 0 {
                    let prefix = String(k.dropLast(".weight".count))
                    let q = quantized(value, groupSize: 64, bits: bits)
                    out[prefix + ".weight"] = q.wq
                    out[prefix + ".scales"] = q.scales
                    if let b = q.biases { out[prefix + ".biases"] = b }
                } else {
                    out[k] = value
                }
            }
        }
        guard out["x_embedder.weight"] != nil || out["x_embedder.scales"] != nil else {
            throw ComponentError.layout("\(url.path) is not a FLUX.2 transformer (no x_embedder)")
        }
        return out
    }

    static func safetensorsFiles(_ url: URL) throws -> [URL] {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else { throw ComponentError.missing(url.path) }
        guard isDir.boolValue else { return [url] }
        let files = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw ComponentError.missing("*.safetensors in \(url.path)") }
        return files
    }

    static func json(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    public static func transformerConfig(_ dir: URL) -> Flux2Transformer.Config {
        var c = Flux2Transformer.Config()
        guard let j = json(dir.appendingPathComponent("config.json")) else { return c }
        if let v = j["in_channels"] as? Int { c.inChannels = v }
        if let v = j["num_layers"] as? Int { c.numLayers = v }
        if let v = j["num_single_layers"] as? Int { c.numSingleLayers = v }
        if let v = j["attention_head_dim"] as? Int { c.headDim = v }
        if let v = j["num_attention_heads"] as? Int { c.numHeads = v }
        if let v = j["joint_attention_dim"] as? Int { c.jointDim = v }
        if let v = j["timestep_guidance_channels"] as? Int { c.timestepChannels = v }
        if let v = (j["mlp_ratio"] as? NSNumber)?.floatValue { c.mlpRatio = v }
        if let v = (j["rope_theta"] as? NSNumber)?.floatValue { c.ropeTheta = v }
        if let v = j["axes_dims_rope"] as? [Int] { c.ropeAxes = v }
        if let v = (j["eps"] as? NSNumber)?.floatValue { c.eps = v }
        return c
    }

    // MARK: Text encoder

    public static func textEncoderConfig(_ dir: URL) -> Qwen3TextEncoder.Config {
        var c = Qwen3TextEncoder.Config()
        guard let j = json(dir.appendingPathComponent("config.json")) else { return c }
        if let v = j["hidden_size"] as? Int { c.hiddenSize = v }
        if let v = j["num_hidden_layers"] as? Int { c.numLayers = v }
        if let v = j["num_attention_heads"] as? Int { c.numHeads = v }
        if let v = j["num_key_value_heads"] as? Int { c.numKVHeads = v }
        if let v = j["head_dim"] as? Int { c.headDim = v }
        if let v = j["intermediate_size"] as? Int { c.intermediateSize = v }
        if let v = (j["rope_theta"] as? NSNumber)?.floatValue { c.ropeTheta = v }
        if let v = (j["rms_norm_eps"] as? NSNumber)?.floatValue { c.rmsEps = v }
        return c
    }

    /// A diffusers/HF Qwen3 text encoder in its stored precision, keyed like mflux (`model.` dropped,
    /// no LM head).
    public static func textEncoderStore(_ dir: URL) throws -> TensorStore {
        var out: [String: MLXArray] = [:]
        for file in try safetensorsFiles(dir) {
            for (key, value) in try loadArrays(url: file) where !key.hasPrefix("lm_head") {
                out[key.hasPrefix("model.") ? String(key.dropFirst("model.".count)) : key] = value
            }
        }
        guard out["embed_tokens.weight"] != nil || out["embed_tokens.scales"] != nil else {
            throw ComponentError.layout("\(dir.path) is not a Qwen3 text encoder (no embed_tokens)")
        }
        return TensorStore(tensors: out)
    }

    // MARK: Download

    /// Snapshot the linked repo's components (pinned to its commit when known).
    public static func snapshot(repo: String, revision: String?, components: [String],
                                progress: (@Sendable (String) -> Void)?) async throws -> URL {
        let globs = components.map { "\($0)/*" } + ["model_index.json", "README.md"]
        return try await HubApi().snapshot(from: repo, revision: revision ?? "main", matching: globs) { p in
            progress?(String(format: "downloading %@ %.0f%%", repo, 100 * p.fractionCompleted))
        }
    }
}

/// Rectified-flow helpers.
public enum Flux2Packing {
    /// (B, 32, 2Hp, 2Wp) → (B, Hp·Wp, 128); channel = c·4 + dy·2 + dx (FLUX.2's 2×2 patchify).
    public static func pack(_ x: MLXArray) -> MLXArray {
        let (b, c, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return x.reshaped(b, c, h / 2, 2, w / 2, 2).transposed(0, 1, 3, 5, 2, 4)
            .reshaped(b, c * 4, (h / 2) * (w / 2)).transposed(0, 2, 1)
    }

    /// (B, Hp·Wp, 128) → (B, 32, 2Hp, 2Wp).
    public static func unpack(_ t: MLXArray, packedH: Int, packedW: Int) -> MLXArray {
        let b = t.dim(0), c = t.dim(2) / 4
        return t.transposed(0, 2, 1).reshaped(b, c, 2, 2, packedH, packedW).transposed(0, 1, 4, 2, 5, 3)
            .reshaped(b, c, 2 * packedH, 2 * packedW)
    }
}
