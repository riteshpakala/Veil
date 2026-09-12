//
//  Flux2Tests.swift
//  VeilFlux2Tests
//
//  Key mapping (always), and the executor on the real FLUX.2 Klein weights (when on disk):
//  the klein-base transformer (diffusers) with the shared mflux export's text encoder, VAE
//  and tokenizer. Nothing downloads.
//

import Foundation
import FluxKit
import MLX
import Testing
@testable import VeilFlux2
@testable import VeilKit

let exportDir = Flux2Components.sharedExportDirectory
let baseDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    .appendingPathComponent("huggingface/models/black-forest-labs/FLUX.2-klein-base-4B")
let haveExport = FileManager.default.fileExists(atPath: exportDir.appendingPathComponent("vae").path)
    && FileManager.default.fileExists(atPath: exportDir.appendingPathComponent("text_encoder").path)
let haveBase = haveExport && FileManager.default.fileExists(atPath: baseDir.appendingPathComponent("transformer/config.json").path)

/// Gap energy E‖v̂(prompt) − v̂₀(anchor)‖² over a fixed keyed set of noise draws and σ levels.
///
/// One draw is not enough to judge this edit: measured at a single noise seed the same closed
/// form reads as anywhere from 8% to 88% of the gap remaining, because the base gap itself varies
/// eightfold across draws. Every claim about how much of the gap an edit closes is averaged.
func gapEnergy(_ m: Flux2GuardModel, prompt: String, anchor: String, guarded: Bool,
               sigmas: [Float] = [0.9, 0.6], draws: Int = 4) throws -> Double {
    var total = 0.0, n = 0
    for (s, sigma) in sigmas.enumerated() {
        let a = try m.embed([anchor])[0].value
        let p = try m.embed([prompt])[0].value
        for d in 0..<draws {
            let x = SeedSchedule.research.normal(.toy, 900 + 10 * s + d, shape: [1, 32, 16, 16]) * 0.9
            let va = m.hooks.with(guard: false) { m.velocity(x, sigma: sigma, conditioning: a) }
            let vp = m.hooks.with(guard: guarded) { m.velocity(x, sigma: sigma, conditioning: p) }
            let e = (vp - va).square().mean()
            eval(e)
            total += Double(e.item(Float.self))
            n += 1
        }
    }
    return total / Double(max(n, 1))
}

/// E‖v̂_G − v̂₀‖² / E‖v̂₀‖² on one prompt, over the same draws: what the guard does to a prompt it
/// should leave alone.
func driftEnergy(_ m: Flux2GuardModel, prompt: String, sigmas: [Float] = [0.9, 0.6], draws: Int = 4) throws -> Double {
    var num = 0.0, den = 0.0
    for (s, sigma) in sigmas.enumerated() {
        let c = try m.embed([prompt])[0].value
        for d in 0..<draws {
            let x = SeedSchedule.research.normal(.toy, 900 + 10 * s + d, shape: [1, 32, 16, 16]) * 0.9
            let v0 = m.hooks.with(guard: false) { m.velocity(x, sigma: sigma, conditioning: c) }
            let vg = m.hooks.with(guard: true) { m.velocity(x, sigma: sigma, conditioning: c) }
            let dv = (vg - v0).square().mean(), base = v0.square().mean()
            eval(dv, base)
            num += Double(dv.item(Float.self))
            den += Double(base.item(Float.self))
        }
    }
    return num / max(den, 1e-12)
}

@Suite struct Flux2KeyMapTests {
    let s = SeedSchedule.research

    func delta(_ out: Int, _ inF: Int, _ r: Int, _ i: Int) -> LowRankDelta {
        LowRankDelta(up: s.normal(.toy, 700 + i, shape: [out, r]), down: s.normal(.toy, 800 + i, shape: [r, inF]))
    }

    @Test func diffusersAndBFLExportsReadBackToTheSameDeltas() throws {
        let keys = ["context_embedder", "transformer_blocks.0.attn.add_q_proj", "transformer_blocks.0.attn.add_k_proj",
                    "transformer_blocks.0.attn.add_v_proj", "transformer_blocks.0.attn.to_add_out",
                    "transformer_blocks.0.ff_context.linear_in", "transformer_blocks.0.ff_context.linear_out"]
        let shapes = [(24, 40), (16, 16), (16, 16), (16, 16), (16, 16), (48, 16), (16, 24)]
        var deltas: [String: LowRankDelta] = [:]
        for (i, (k, sh)) in zip(keys, shapes).enumerated() { deltas[k] = delta(sh.0, sh.1, 2 + i % 3, i) }
        let linearKeys = Set(keys)
        for tensors in [Flux2KeyMap.diffusers(deltas), try Flux2KeyMap.bflTensors(deltas)] {
            var arrays: [String: MLXArray] = [:]
            for (k, v) in tensors { arrays[k] = v }
            let file = try AdapterFile(url: URL(fileURLWithPath: "/dev/null"), arrays: arrays, metadata: [:])
            let mapping = Flux2KeyMap.map(file.updates, linearKeys: linearKeys)
            #expect(mapping.unmapped.isEmpty && mapping.unsupported.isEmpty)
            for (k, d) in deltas {
                let back = try #require(mapping.updates[k]).materialized
                #expect(abs(back - d.dense).max().item(Float.self) < 1e-4, "\(k)")
            }
        }
        // Diffusers names match what hosts load.
        #expect(Flux2KeyMap.diffusers(deltas).keys.contains("transformer.context_embedder.lora_A.weight"))
        #expect(try Flux2KeyMap.bflTensors(deltas).keys.contains("diffusion_model.double_blocks.0.txt_attn.qkv.lora_B.weight"))
        #expect(try Flux2KeyMap.bflTensors(deltas).keys.contains("diffusion_model.txt_in.lora_A.weight"))
    }

    @Test func familyDetectionReadsKleinConfigs() {
        let base = FamilyDetector.classify(modelIndex: nil, transformer: ["_class_name": "Flux2Transformer2DModel", "joint_attention_dim": 7680],
                                           unet: nil, readme: nil, repo: "black-forest-labs/FLUX.2-klein-base-4B", paths: [])
        #expect(base.family == "flux2-klein" && base.variant == "base" && base.size == "4B")
        let distilled = FamilyDetector.classify(modelIndex: nil, transformer: ["_class_name": "Flux2Transformer2DModel", "joint_attention_dim": 12288],
                                                unet: nil, readme: nil, repo: "black-forest-labs/FLUX.2-klein-9B", paths: [])
        #expect(distilled.variant == "distilled" && distilled.size == "9B")
        let sdxl = FamilyDetector.classify(modelIndex: nil, transformer: nil, unet: ["_class_name": "UNet2DConditionModel", "cross_attention_dim": 2048],
                                           readme: nil, repo: "stabilityai/sdxl", paths: [])
        #expect(sdxl.family == "sdxl")
    }
}

@Suite(.enabled(if: haveExport)) struct Flux2LocalDescriptorTests {
    @Test func localFoldersDescribeThemselves() {
        let export = FamilyDetector.describeLocal(exportDir)
        #expect(export.family == "flux2-klein" && export.detail == "mflux export" && export.variant == "distilled")
        #expect(export.host == .local && export.components.contains("vae"))
        if haveBase {
            let base = FamilyDetector.describeLocal(baseDir)
            #expect(base.family == "flux2-klein" && base.variant == "base" && base.size == "4B")
        }
    }
}

@Suite(.serialized, .enabled(if: haveBase, "FLUX.2 klein-base-4B transformer and the mflux export are not on disk"))
struct Flux2ExecutorTests {
    static let model: Flux2GuardModel? = {
        Flux2Registration.register()
        let semaphore = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var loaded: Flux2GuardModel?
        Task.detached {
            loaded = try? await ExecutorRegistry.shared.load(baseDir.path, options: ["text-encoder": "mflux", "long-side": "256"]) as? Flux2GuardModel
            semaphore.signal()
        }
        semaphore.wait()
        return loaded
    }()

    @Test func loadsAsTheBaseWithTheTextPathAsSlots() throws {
        let m = try #require(Self.model)
        #expect(m.descriptor.family == "flux2-klein" && m.descriptor.variant == "base")
        #expect(m.slots.count == 1 + 6 * m.transformerConfig.numLayers)
        #expect(m.slots.first?.key == "context_embedder" && m.slots.first?.role == .textInput)
        #expect(m.samplerSigmas(latentShape: [32, 32, 32]).count == 50)
        #expect(m.issues.contains { $0.contains("4-bit mflux export") })
    }

    @Test func guardOffIsBitwiseTheDeploymentAndOnChangesIt() throws {
        let m = try #require(Self.model)
        let x = SeedSchedule.research.normal(.toy, 5, shape: [1, 32, 16, 16])
        let c = try m.embed(["a photo of a person"])[0].value
        m.installGuard(nil)
        let base = m.hooks.with(guard: false) { m.velocity(x, sigma: 0.6, conditioning: c) }
        var guardDeltas: [String: LowRankDelta] = [:]
        for slot in m.slots.prefix(3) {
            guardDeltas[slot.key] = LowRankDelta(up: SeedSchedule.research.normal(.toy, 11, shape: [slot.outFeatures, 2]) * 0.01,
                                                 down: SeedSchedule.research.normal(.toy, 12, shape: [2, slot.inFeatures]) * 0.01)
        }
        m.installGuard(guardDeltas)
        let off = m.hooks.with(guard: false) { m.velocity(x, sigma: 0.6, conditioning: c) }
        let on = m.hooks.with(guard: true) { m.velocity(x, sigma: 0.6, conditioning: c) }
        m.installGuard(nil)
        #expect(abs(base - off).max().item(Float.self) == 0, "guard off must be exactly the deployment")
        #expect(abs(base - on).max().item(Float.self) > 0)
    }

    @Test func textEncoderEmbeddingPathMatchesTheIDPath() throws {
        let m = try #require(Self.model)
        let search = try #require(m.tokenSearch)
        let (p, t, s) = try search.tokenize("a photo of a person")
        #expect(!p.isEmpty && !t.isEmpty && !s.isEmpty)
        #expect(search.decode(t).contains("photo"))
        let encoder = try m.encoder()
        let ids = p + t + s
        let viaEmbeddings = search.conditioning(fromTokenEmbeddings: encoder.tokenEmbeddings(MLXArray(ids.map(Int32.init)).reshaped(1, ids.count)))
        let viaIDs = try m.embed(["a photo of a person"])[0].value
        let err = abs(viaEmbeddings.asType(.float32) - viaIDs.asType(.float32)).max().item(Float.self)
        let scale = abs(viaIDs.asType(.float32)).max().item(Float.self)
        #expect(err / scale < 1e-2, "relative \(err / scale)")
    }

    /// The closed-form edit on the real context_embedder, over 8 noise draws (decode-free): the
    /// fitted name's gap to its anchor closes by about 70% of its energy and everyday prompts
    /// barely move.
    ///
    /// A paraphrase outside the erase set does *not* improve — on Klein it measures worse than
    /// before the edit. Stage A closes the templates it was given and nothing else, which is why
    /// the trained stage exists; that number is printed and bounded here rather than asserted as
    /// an improvement, so a future stage A that does generalize won't be held back by this test.
    @Test func closedFormEditMovesTheNameToTheAnchorAndLeavesOthers() throws {
        let m = try #require(Self.model)
        let erase = Templates.nameRoute("Abraham Lincoln", count: 8, anchor: "a person")
        let preserve = Templates.generic.prefix(10) + Templates.people.prefix(6)
        let e = try zip(m.embed(erase.prompts), m.embed(erase.anchors)).map { ($0, $1) }
        let p = try m.embed(Array(preserve))
        let slot = try #require(m.linear("context_embedder"))
        let started = Date()
        let edit = ClosedFormEditor.fit(key: "context_embedder", weight: slot.deployedWeight(), erase: e, preserve: p,
                                        preserveWeight: 1, ridge: 0.1, maxRank: 64, energy: 0.99, schedule: .research)
        print("closed form: rank \(edit.delta.rank), energy \(edit.energy), truncation \(edit.truncationError), size \(edit.relativeSize), \(Date().timeIntervalSince(started)) s")
        m.installGuard(["context_embedder": edit.delta])
        defer { m.installGuard(nil) }
        let name = "a photo of Abraham Lincoln", anchor = "a photo of a person"
        let before = try gapEnergy(m, prompt: name, anchor: anchor, guarded: false)
        let after = try gapEnergy(m, prompt: name, anchor: anchor, guarded: true)
        let unseen = "a portrait of Abraham Lincoln in a library"   // not in the erase set
        let unseenAnchor = "a portrait of a person in a library"
        let unseenBefore = try gapEnergy(m, prompt: unseen, anchor: unseenAnchor, guarded: false)
        let unseenAfter = try gapEnergy(m, prompt: unseen, anchor: unseenAnchor, guarded: true)
        let drift = try driftEnergy(m, prompt: "a bowl of ramen")
        print(String(format: "gap energy left: name %.0f%%, unseen paraphrase %.0f%%; preserve drift %.3f",
                     100 * after / before, 100 * unseenAfter / unseenBefore, drift))
        #expect(after < 0.6 * before, "the edit closes most of the name→anchor gap")
        #expect(unseenAfter < 3 * unseenBefore, "an unseen paraphrase need not improve, but must not blow up")
        #expect(drift < 0.05, "while a preserved prompt barely moves")
    }

    /// A group in one guard (`veil protect`): the same closed form covering two people at once.
    /// Both names must move toward their anchors, a name the guard doesn't cover must keep its
    /// own direction, and everyday prompts must stay put. Stage A is text-only, so no photos are
    /// involved here.
    @Test func aGroupEditMovesEveryNameItCoversAndLeavesTheRestAlone() throws {
        let m = try #require(Self.model)
        var profile = VeilProfile.quick
        profile.closedFormMaxRank = 64
        let covered = ["Abraham Lincoln", "Frederick Douglass"]
        let people = covered.map {
            GuardPlan.Person(id: "test-\($0)", names: [$0], descriptions: [], anchor: "a person", fit: [])
        }
        let plan = GuardPlan.make(people: people, profile: profile, schedule: .research, templatesOverride: 6)
        #expect(plan.erase.count == 12, "six templates for each of two people")
        #expect(plan.erase.contains { $0.person == 1 }, "the second person's prompts carry their own index")
        #expect(!plan.preserve.contains { p in covered.contains { p.contains($0) } }, "no covered name is also preserved")

        let started = Date()
        let guardFit = try GuardFitter.fit(model: m, plan: plan, preservePhotos: [], method: .closedForm, profile: profile,
                                           schedule: .research, embed: m.embed)
        defer { m.installGuard(nil) }
        print("group closed form: rank \(guardFit.closedForm?.rank ?? 0), size \(guardFit.closedForm?.relativeSize ?? 0), \(Date().timeIntervalSince(started)) s")

        let anchor = "a photo of a person"
        for name in covered {
            let before = try gapEnergy(m, prompt: "a photo of \(name)", anchor: anchor, guarded: false)
            let after = try gapEnergy(m, prompt: "a photo of \(name)", anchor: anchor, guarded: true)
            print(String(format: "%@: %.0f%% of the gap energy left", name, 100 * after / before))
            #expect(after < 0.7 * before, "\(name) should move toward the anchor")
        }
        let outsideBefore = try gapEnergy(m, prompt: "a photo of Ulysses S. Grant", anchor: anchor, guarded: false)
        let outsideAfter = try gapEnergy(m, prompt: "a photo of Ulysses S. Grant", anchor: anchor, guarded: true)
        let drift = try driftEnergy(m, prompt: "a bowl of ramen")
        print(String(format: "Ulysses S. Grant (not covered): %.0f%% of his gap energy left; everyday drift %.3f",
                     100 * outsideAfter / outsideBefore, drift))
        #expect(outsideAfter > 0.4 * outsideBefore, "a name the guard doesn't cover keeps most of its own direction")
        #expect(drift < 0.05, "everyday prompts barely move")
    }
}
