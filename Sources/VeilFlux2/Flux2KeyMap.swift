//
//  Flux2KeyMap.swift
//  VeilFlux2
//  Import mapping adapted from Scorpion (github.com/riteshpakala/Scorpion, ScorpionFlux2: Flux2LoRAMapper), GPL-3.0.
//
//  Adapter module names ↔ FluxKit linear keys, both ways.
//
//  Import (a community LoRA in the deployment, or a guard read back): PEFT/kohya prefixes,
//  kohya's underscore flattening, diffusers' `to_out.0`, and BFL's native names (fused qkv split
//  row-wise with the down projection shared; the final adaLN halves swapped). Nothing is placed
//  silently: every module must land on a linear of the right shape, or it is reported.
//
//  Export (the guard): diffusers/PEFT names (`transformer.<module>.lora_A/B.weight`) by
//  default, or BFL/ComfyUI names (`diffusion_model.<bfl>.lora_A/B.weight`), where the text
//  stream's q/k/v are one fused projection: the parts are fused exactly by concatenating ranks,
//  A = [A_q; A_k; A_v], B = blockdiag(B_q, B_k, B_v).
//

import Foundation
import MLX
import VeilKit

public struct Flux2Mapping {
    public var updates: [String: WeightUpdate]
    public var unmapped: [String]
    /// Modules for parts the executor doesn't run (the text encoder).
    public var unsupported: [String]
    /// Modules that needed a BFL-native name mapping.
    public var translated: Int
}

public enum Flux2KeyMap {
    static let prefixes = ["base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer.", "unet.",
                           "lora_unet_", "lora_transformer_", "lycoris_", "lora_"]
    static let textEncoderMarkers = ["lora_te", "text_encoder", "te1_", "te2_", "qwen", "text_model"]

    // MARK: Import

    /// - Parameter linearKeys: every linear key the transformer loads (FluxKit names).
    public static func map(_ updates: [String: WeightUpdate], linearKeys: Set<String>) -> Flux2Mapping {
        let underscored = Dictionary(linearKeys.map { ($0.replacingOccurrences(of: ".", with: "_"), $0) }) { a, _ in a }
        var out = Flux2Mapping(updates: [:], unmapped: [], unsupported: [], translated: 0)
        for (name, update) in updates.sorted(by: { $0.key < $1.key }) {
            let lower = name.lowercased()
            if textEncoderMarkers.contains(where: { lower.hasPrefix($0) || lower.contains(".\($0)") }) {
                out.unsupported.append(name)
                continue
            }
            var key = name
            for p in prefixes where key.hasPrefix(p) { key.removeFirst(p.count) }
            key = key.replacingOccurrences(of: ".attn.to_out.0", with: ".attn.to_out")
                .replacingOccurrences(of: "time_guidance_embed.timestep_embedder.", with: "time_guidance_embed.")
            if linearKeys.contains(key) {
                out.updates[key] = merge(out.updates[key], update)
                continue
            }
            if let dotted = underscored[key.replacingOccurrences(of: ".", with: "_")] {
                out.updates[dotted] = merge(out.updates[dotted], update)
                continue
            }
            if let parts = bfl(key, update), parts.allSatisfy({ linearKeys.contains($0.0) }) {
                for (k, u) in parts { out.updates[k] = merge(out.updates[k], u) }
                out.translated += 1
                continue
            }
            out.unmapped.append(name)
        }
        return out
    }

    /// Two modules landing on one linear (e.g. fused and unfused parts) sum.
    static func merge(_ a: WeightUpdate?, _ b: WeightUpdate) -> WeightUpdate {
        guard let a else { return b }
        return .dense(a.materialized + b.materialized)
    }

    /// BFL-native module names → (FluxKit key, ΔW) parts.
    static func bfl(_ key: String, _ update: WeightUpdate) -> [(String, WeightUpdate)]? {
        func match(_ pattern: String) -> [String]? {
            guard let re = try? NSRegularExpression(pattern: "^" + pattern + "$"),
                  let m = re.firstMatch(in: key, range: NSRange(key.startIndex..., in: key)) else { return nil }
            return (1..<m.numberOfRanges).compactMap { Range(m.range(at: $0), in: key).map { String(key[$0]) } }
        }
        if let g = match(#"double_blocks\.(\d+)\.img_attn\.qkv"#) {
            return zip(["to_q", "to_k", "to_v"], split(update, 3)).map { ("transformer_blocks.\(g[0]).attn.\($0)", $1) }
        }
        if let g = match(#"double_blocks\.(\d+)\.txt_attn\.qkv"#) {
            return zip(["add_q_proj", "add_k_proj", "add_v_proj"], split(update, 3)).map { ("transformer_blocks.\(g[0]).attn.\($0)", $1) }
        }
        for (pattern, target) in bflSimple {
            if let g = match(pattern) { return [(target(g[0]), update)] }
        }
        if let t = bflFixed[key] { return [(t, update)] }
        if key == "final_layer.adaLN_modulation.1" { return [("norm_out.linear", swapHalves(update))] }
        return nil
    }

    static let bflSimple: [(String, (String) -> String)] = [
        (#"double_blocks\.(\d+)\.img_attn\.proj"#, { "transformer_blocks.\($0).attn.to_out" }),
        (#"double_blocks\.(\d+)\.txt_attn\.proj"#, { "transformer_blocks.\($0).attn.to_add_out" }),
        (#"double_blocks\.(\d+)\.img_mlp\.0"#, { "transformer_blocks.\($0).ff.linear_in" }),
        (#"double_blocks\.(\d+)\.img_mlp\.2"#, { "transformer_blocks.\($0).ff.linear_out" }),
        (#"double_blocks\.(\d+)\.txt_mlp\.0"#, { "transformer_blocks.\($0).ff_context.linear_in" }),
        (#"double_blocks\.(\d+)\.txt_mlp\.2"#, { "transformer_blocks.\($0).ff_context.linear_out" }),
        (#"single_blocks\.(\d+)\.linear1"#, { "single_transformer_blocks.\($0).attn.to_qkv_mlp_proj" }),
        (#"single_blocks\.(\d+)\.linear2"#, { "single_transformer_blocks.\($0).attn.to_out" }),
    ]

    static let bflFixed: [String: String] = [
        "img_in": "x_embedder", "txt_in": "context_embedder",
        "time_in.in_layer": "time_guidance_embed.linear_1", "time_in.out_layer": "time_guidance_embed.linear_2",
        "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
        "single_stream_modulation.lin": "single_stream_modulation.linear",
        "final_layer.linear": "proj_out",
    ]

    /// Split ΔW row-wise into `n` equal parts (a fused projection); low rank keeps `down` shared.
    static func split(_ update: WeightUpdate, _ n: Int) -> [WeightUpdate] {
        switch update {
        case .lowRank(let up, let down, let scale):
            let rows = up.dim(0) / n
            return (0..<n).map { .lowRank(up: up[($0 * rows) ..< (($0 + 1) * rows)], down: down, scale: scale) }
        case .dense(let w):
            let rows = w.dim(0) / n
            return (0..<n).map { .dense(w[($0 * rows) ..< (($0 + 1) * rows)]) }
        }
    }

    /// BFL's final modulation emits (shift, scale); diffusers/mflux expect (scale, shift).
    static func swapHalves(_ update: WeightUpdate) -> WeightUpdate {
        func swap(_ a: MLXArray) -> MLXArray {
            let h = a.dim(0) / 2
            return concatenated([a[h ..< (2 * h)], a[0 ..< h]], axis: 0)
        }
        switch update {
        case .lowRank(let up, let down, let scale): return .lowRank(up: swap(up), down: down, scale: scale)
        case .dense(let w): return .dense(swap(w))
        }
    }

    // MARK: Export

    public enum ExportError: Error, LocalizedError {
        case noBFLName(String)

        public var errorDescription: String? {
            switch self {
            case .noBFLName(let k): return "No BFL name for \(k); export this guard in diffusers format."
            }
        }
    }

    /// PEFT/diffusers tensors: `transformer.<module>.lora_A.weight` (r, in), `.lora_B.weight` (out, r).
    public static func diffusers(_ deltas: [String: LowRankDelta]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, d) in deltas {
            let module = "transformer." + Flux2Components.diffusersModule(key)
            out[module + ".lora_A.weight"] = d.down
            out[module + ".lora_B.weight"] = d.up
        }
        return out
    }

    /// BFL/ComfyUI tensors: `diffusion_model.<bfl module>.lora_A/B.weight`, text q/k/v fused exactly.
    public static func bflTensors(_ deltas: [String: LowRankDelta]) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        var fused: [String: [Int: LowRankDelta]] = [:]   // block → part index (0 q, 1 k, 2 v)
        let fusedParts = ["add_q_proj": 0, "add_k_proj": 1, "add_v_proj": 2]
        for (key, d) in deltas.sorted(by: { $0.key < $1.key }) {
            if let m = key.firstMatch(of: #/^transformer_blocks\.(\d+)\.attn\.(add_[qkv]_proj)$/#) {
                fused[String(m.1), default: [:]][fusedParts[String(m.2)]!] = d
                continue
            }
            guard let name = bflName(key) else { throw ExportError.noBFLName(key) }
            out["diffusion_model.\(name).lora_A.weight"] = d.down
            out["diffusion_model.\(name).lora_B.weight"] = d.up
        }
        for (block, parts) in fused {
            let inner = parts.values.first!.outFeatures, inF = parts.values.first!.inFeatures
            var downs: [MLXArray] = [], upBlocks: [[MLXArray]] = [[], [], []]
            for i in 0..<3 {
                guard let d = parts[i] else { continue }
                downs.append(d.down)
                for j in 0..<3 { upBlocks[j].append(j == i ? d.up : MLXArray.zeros([inner, d.rank])) }
            }
            let up = concatenated(upBlocks.map { concatenated($0, axis: 1) }, axis: 0)     // (3·inner, Σr)
            let down = concatenated(downs, axis: 0)                                          // (Σr, in)
            precondition(down.dim(1) == inF)
            out["diffusion_model.double_blocks.\(block).txt_attn.qkv.lora_A.weight"] = down
            out["diffusion_model.double_blocks.\(block).txt_attn.qkv.lora_B.weight"] = up
        }
        return out
    }

    static func bflName(_ key: String) -> String? {
        if let t = bflFixed.first(where: { $0.value == key })?.key { return t }
        let patterns: [(Regex<(Substring, Substring)>, (Substring) -> String)] = [
            (#/^transformer_blocks\.(\d+)\.attn\.to_add_out$/#, { "double_blocks.\($0).txt_attn.proj" }),
            (#/^transformer_blocks\.(\d+)\.ff_context\.linear_in$/#, { "double_blocks.\($0).txt_mlp.0" }),
            (#/^transformer_blocks\.(\d+)\.ff_context\.linear_out$/#, { "double_blocks.\($0).txt_mlp.2" }),
            (#/^transformer_blocks\.(\d+)\.attn\.to_out$/#, { "double_blocks.\($0).img_attn.proj" }),
            (#/^transformer_blocks\.(\d+)\.ff\.linear_in$/#, { "double_blocks.\($0).img_mlp.0" }),
            (#/^transformer_blocks\.(\d+)\.ff\.linear_out$/#, { "double_blocks.\($0).img_mlp.2" }),
            (#/^single_transformer_blocks\.(\d+)\.attn\.to_qkv_mlp_proj$/#, { "single_blocks.\($0).linear1" }),
            (#/^single_transformer_blocks\.(\d+)\.attn\.to_out$/#, { "single_blocks.\($0).linear2" }),
        ]
        for (re, name) in patterns {
            if let m = key.firstMatch(of: re) { return name(m.1) }
        }
        return nil
    }
}
