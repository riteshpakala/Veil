//
//  Flux2Registration.swift
//  VeilFlux2
//
//  The composition roots call `Flux2Registration.register()`; VeilKit then loads any linked
//  FLUX.2 Klein repo through `Flux2Executor.make`.
//
//  Options (`--option key=value`):
//    text-encoder  repo (default: the linked repo's bf16 Qwen3) | mflux (the shared 4-bit export)
//    export-dir    the shared mflux export (VAE; 4-bit text encoder) — default VEIL_FLUX2_DIR or
//                  ~/Documents/huggingface/models/mlx-community/flux2-klein-4b-4bit
//    quantize      4 | 8 (re-quantize a bf16 transformer on load)
//    long-side     photo working size (default 512)
//    max-batch     rows per transformer call (default 2)
//    steps         sampler steps defining the σ grid (default 4 distilled, 50 base)
//

import FluxKit
import Foundation
import Hub
import MLX
import VeilKit

public enum Flux2Registration {
    public static func register(_ registry: ExecutorRegistry = .shared) {
        registry.register(family: "flux2-klein", factory: Flux2Executor.make)
    }
}

public enum Flux2Executor {
    static func expand(_ path: String) -> URL { URL(fileURLWithPath: (path as NSString).expandingTildeInPath) }

    static func exists(_ url: URL) -> Bool { FileManager.default.fileExists(atPath: url.path) }

    @Sendable public static func make(_ request: ExecutorRequest) async throws -> GuardableModel {
        let d = request.descriptor
        let say = request.progress
        let isExport = d.detail == "mflux export"
        let textEncoderChoice = request.option("text-encoder") ?? "repo"
        var exportDir = request.option("export-dir").map(expand) ?? Flux2Components.sharedExportDirectory
        var issues: [String] = []

        // The linked repo's components.
        let root: URL
        if d.host == .local {
            root = URL(fileURLWithPath: d.repo)
        } else {
            var comps = ["transformer", "tokenizer"]
            if isExport { comps += ["text_encoder", "vae"] } else if textEncoderChoice == "repo" { comps.append("text_encoder") }
            say?("fetching \(d.pinnedName) (\(comps.joined(separator: ", ")))")
            root = try await Flux2Components.snapshot(repo: d.repo, revision: d.commit, components: comps, progress: say)
        }
        if isExport { exportDir = root }

        // The shared export supplies the VAE encoder (and the 4-bit text encoder on request).
        if !exists(exportDir.appendingPathComponent("vae")) {
            say?("fetching the FLUX.2 VAE from \(Flux2Components.sharedExportRepo)")
            exportDir = try await Flux2Components.snapshot(repo: Flux2Components.sharedExportRepo, revision: nil,
                                                          components: ["vae", "tokenizer"] + (textEncoderChoice == "mflux" ? ["text_encoder"] : []),
                                                          progress: say)
        }
        let vae = exportDir.appendingPathComponent("vae")
        guard exists(vae) else { throw Flux2Components.ComponentError.missing(vae.path) }
        let tokenizerRoot = exists(root.appendingPathComponent("tokenizer/tokenizer.json")) ? root : exportDir
        guard exists(tokenizerRoot.appendingPathComponent("tokenizer/tokenizer.json")) else {
            throw Flux2Components.ComponentError.missing("tokenizer/tokenizer.json in \(root.path) or \(exportDir.path)")
        }

        let textEncoder: Flux2GuardModel.TextEncoderSource
        if textEncoderChoice == "mflux" || isExport {
            if !exists(exportDir.appendingPathComponent("text_encoder")) {
                exportDir = try await Flux2Components.snapshot(repo: Flux2Components.sharedExportRepo, revision: nil,
                                                              components: ["text_encoder"], progress: say)
            }
            textEncoder = .export(exportDir.appendingPathComponent("text_encoder"))
            if !isExport {
                issues.append("Text encoder: the 4-bit mflux export, not the bf16 encoder hosts run; embeddings (and the closed-form edit) differ slightly from a host's.")
            }
        } else {
            let dir = root.appendingPathComponent("text_encoder")
            guard exists(dir) else {
                throw Flux2Components.ComponentError.missing("\(dir.path) — the linked repo has no text_encoder; pass --option text-encoder=mflux")
            }
            textEncoder = .repo(dir)
        }

        // Transformer, every linear hooked.
        say?("loading the transformer")
        let hooks = HookController()
        var hooked: [String: HookedLinear] = [:]
        let transformerDir = root.appendingPathComponent("transformer")
        let store: TensorStore
        let config: Flux2Transformer.Config
        if exists(transformerDir.appendingPathComponent("config.json")) {
            let bits = request.option("quantize").flatMap(Int.init)
            store = TensorStore(tensors: try Flux2Components.diffusersTransformer(at: transformerDir, quantizeBits: bits))
            config = Flux2Components.transformerConfig(transformerDir)
            if let bits { issues.append("Transformer re-quantized to \(bits) bits on load; the host's precision may differ.") }
        } else {
            store = try TensorStore(componentDir: transformerDir)
            config = Flux2Transformer.Config()
        }
        store.linearTransform = { key, linear in
            let h = HookedLinear(key: key, base: linear, controller: hooks)
            hooked[key] = h
            return h
        }
        let transformer = try Flux2Transformer(store: store, config: config)
        Memory.clearCache()

        let distilled = d.variant == "distilled"
        let steps = request.option("steps").flatMap(Int.init) ?? (distilled ? 4 : 50)
        let model = Flux2GuardModel(descriptor: d, transformer: transformer, config: config, hooked: hooked, hooks: hooks,
                                    tokenizerDirectory: tokenizerRoot, vaeDirectory: vae, textEncoder: textEncoder,
                                    longSide: request.option("long-side").flatMap(Int.init) ?? 512,
                                    maxBatch: request.option("max-batch").flatMap(Int.init) ?? 2, samplerSteps: steps)
        if d.variant == nil {
            issues.append("Could not tell whether this Klein is the base or the step-distilled model; the σ grid assumes \(steps) sampler steps (set --option steps=…).")
        }
        if d.size == "9B" { issues.append("FLUX.2 Klein 9B loads from its configs but has not been validated with Veil yet.") }

        // A deployment adapter (base + community LoRA), applied to every pass.
        if let url = request.withAdapter {
            let file = try AdapterFile.load(url)
            let (deltas, bad) = model.importDeltas(file)
            for (key, delta) in deltas { model.linear(key)?.fixed = delta }
            if !bad.isEmpty {
                model.blockingIssues.append("with-adapter: \(bad.count) module(s) not placed (\(bad.prefix(4).joined(separator: ", "))); the deployment isn't fully reproduced, so a guard fitted here isn't verified for it.")
            }
            if !file.dora.isEmpty { issues.append("with-adapter: DoRA magnitudes ignored (direction-only).") }
        }
        model.issues = issues
        return model
    }
}
