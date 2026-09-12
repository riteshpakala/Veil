//
//  GuardPipelineTests.swift
//  VeilKitTests
//
//  find → block → verify on the toy world, where the right answer is known: Ada Quill is
//  reachable by name and by one description; Cleo Marsh shares her red hair; the attribute
//  prompt must keep working for everyone red-haired.
//

import Foundation
import MLX
import Testing
@testable import VeilKit

func toyRun(profile: VeilProfile = .toy, method: GuardMethod = .both, controls: Bool = true) throws -> VeilRun {
    let world = ToyIdentityWorld.shared
    let subject = ProtectedSubject(photos: world.subjectPhotos, names: [ToyIdentityWorld.subjectName],
                                   descriptions: [ToyIdentityWorld.subjectDescription], anchor: "a person",
                                   consent: Consent(basis: .selfAttested))
    let request = VeilRequest(subject: subject, controls: controls ? world.controls : .empty, profile: profile, faceWeighting: .off,
                              method: method, preservePrompts: [ToyIdentityWorld.attributePrompt])
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("veil-tests-\(UUID().uuidString)")
    let run = VeilRun(model: ToyModel(), request: request, schedule: .research, store: SubjectStore(root: root))
    try run.prepare()
    return run
}

func scratchURL(_ name: String) -> URL {
    FileManager.default.temporaryDirectory.appendingPathComponent("veil-tests-\(UUID().uuidString)").appendingPathComponent(name)
}

@Suite(.serialized) struct GuardPipelineTests {
    @Test func fullToyRunGuardsTheSubject() throws {
        let run = try toyRun()
        let a = try run.assess()
        #expect(a.capability == .nameBound)
        try run.fitGuard()
        let url = scratchURL("guard.safetensors")
        let file = try run.export(to: url)
        let v = try run.verify(guardAt: url)
        for s in v.suppression {
            print(String(format: "%@  base spec %.3f p %.2f → guarded spec %.3f p %.2f  residual %.2f  %@", s.routeID,
                         s.base.specificity, s.base.pValue ?? -1, s.guarded.specificity, s.guarded.pValue ?? -1, s.residual,
                         s.pass ? "pass" : "FAIL"))
        }
        for r in v.robustness {
            print("attack from \(r.base.start): base reached at \(r.base.reachedAt.map(String.init) ?? "never"), guarded at \(r.guarded.reachedAt.map(String.init) ?? "never")",
                  "spec base", r.base.specificity.map { String(format: "%.3f", $0) }, "guarded", r.guarded.specificity.map { String(format: "%.3f", $0) })
        }
        for d in v.preservation.controls + v.preservation.generic {
            print(String(format: "drift %.4f (loss %+.4f)  %@", d.velocityDrift, d.lossDrift, d.label))
        }
        for n in v.preservation.namedControls {
            print(String(format: "named %@: base %.3f guarded %.3f retained %.2f judged %@", n.name, n.basePull.mean, n.guardedPull.mean,
                         n.retained, n.judged ? "yes" : "no"))
        }
        print("verdict", v.verdict.rawValue, v.reasons)
        if let t = run.guardFit?.training {
            print("training erase \(t.eraseLossStart) → \(t.eraseLossEnd), preserve \(t.preserveLossStart) → \(t.preserveLossEnd), hardenings \(t.hardenings)")
        }
        #expect(v.verdict == .guarded)
        #expect(v.suppression.allSatisfy { $0.pass })
        #expect(v.guardSHA256 == file.sha256)

        // The name attack costs more on the guarded model than on the base (which is reached at once).
        if let nameAttack = v.robustness.first(where: { $0.base.start == "name" }) {
            #expect(nameAttack.base.reachedAt == 0)
            #expect((nameAttack.guarded.reachedAt ?? Int.max) > 0)
        }

        // The report is complete and shareable (no photo pixels, hashes only).
        let report = run.report()
        let reportURL = scratchURL("report.json")
        try report.write(to: reportURL)
        let back = try VeilReport.read(from: reportURL)
        #expect(back.verdict == "guarded" && back.schema == VeilInfo.reportSchema && back.mode == "guard")
        #expect(back.subjects.count == 1 && back.subjects[0].photos.allSatisfy { $0.sha256.count == 64 })
        #expect(run.figure() != nil)
    }

    @Test func closedFormAloneSuppressesTheName() throws {
        let run = try toyRun(method: .closedForm)
        try run.assess()
        let fit = try run.fitGuard()
        #expect(fit.training == nil && fit.closedForm != nil)
        print("closed form", fit.closedForm!)
        let url = scratchURL("closed.safetensors")
        try run.export(to: url)
        let v = try run.verify(guardAt: url)
        let name = v.suppression.first { $0.kind == .name }!
        print("name: base spec \(name.base.specificity) → guarded \(name.guarded.specificity); residual \(name.residual)")
        #expect(name.pass)
        #expect(name.residual < 0.5)
        #expect(v.preservation.generic.allSatisfy { $0.velocityDrift < 0.02 })
    }

    @Test func exportReadsBackToTheFittedDeltas() throws {
        let model = ToyModel()
        let s = SeedSchedule.research
        let (e, h, d) = (model.world.text.tokenDim, model.world.contextDim, model.world.latentDim)
        let deltas = [
            "context_embedder": LowRankDelta(up: s.normal(.toy, 500, shape: [h, 4]), down: s.normal(.toy, 501, shape: [4, e])),
            "reader": LowRankDelta(up: s.normal(.toy, 502, shape: [d, 4]), down: s.normal(.toy, 503, shape: [4, h])),
        ]
        let url = scratchURL("roundtrip.safetensors")
        let file = try AdapterWriter.write(deltas, model: model, format: nil, metadata: ["veil.subject": "x"], to: url)
        #expect(file.format == "diffusers" && file.rank == 4)
        let (back, adapter) = try AdapterWriter.readBack(url, model: model)
        #expect(adapter.metadata["veil.subject"] == "x" && adapter.metadata["format"] == "pt")
        for (key, d) in deltas {
            let err = abs(back[key]!.dense - d.dense).max().item(Float.self)
            let scale = abs(d.dense).max().item(Float.self)
            #expect(err / scale < 0.02, "\(key): bf16 round trip error \(err / scale)")
        }
    }

    @Test func withoutControlsTheModelsOwnSamplesStandIn() throws {
        let run = try toyRun(profile: .quick, controls: false)
        #expect(run.fallbackControls)
        #expect(run.controlPhotos.count == 4)
        #expect(run.issues.contains { $0.contains("No control photos") })
        let a = try run.assess()
        #expect(a.measurements.first { $0.kind == .name }?.subject.mean ?? 0 > 0)
    }

    @Test func anUnplacedDeploymentAdapterMakesVerificationInconclusive() async throws {
        // A "community LoRA" with one module the toy can place and one it can't.
        let url = scratchURL("community.safetensors")
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let s = SeedSchedule.research
        try save(arrays: ["toy.reader.lora_A.weight": s.normal(.toy, 950, shape: [2, 64]) * 0.01,
                          "toy.reader.lora_B.weight": s.normal(.toy, 951, shape: [192, 2]) * 0.01,
                          "unet.mid_block.attn1.to_q.lora_A.weight": s.normal(.toy, 952, shape: [2, 8]),
                          "unet.mid_block.attn1.to_q.lora_B.weight": s.normal(.toy, 953, shape: [8, 2])], url: url)
        let model = try await ToyExecutor.make(ExecutorRequest(descriptor: ToyExecutor.descriptor(for: "toy")!, source: nil, withAdapter: url))
        #expect(model.blockingIssues.count == 1)
        #expect(model.linear("reader")?.fixed != nil, "the placeable module is part of the deployment")
        let world = ToyIdentityWorld.shared
        let subject = ProtectedSubject(photos: world.subjectPhotos, names: [ToyIdentityWorld.subjectName], consent: Consent(basis: .selfAttested))
        var profile = VeilProfile.toy
        profile.trainSteps = 0
        let run = VeilRun(model: model, request: VeilRequest(subject: subject, controls: world.controls, profile: profile,
                                                              faceWeighting: .off, method: .closedForm),
                          schedule: .research, store: SubjectStore(root: scratchURL("store")))
        try run.prepare()
        try run.assess()
        try run.fitGuard()
        let guardURL = scratchURL("g.safetensors")
        try run.export(to: guardURL)
        let v = try run.verify(guardAt: guardURL)
        #expect(v.verdict == .inconclusive)
        #expect(v.reasons.contains { $0.contains("isn't fully reproduced") || $0.contains("not placed") })
    }

    @Test func forgettingRemovesEverythingHeldForTheSubject() throws {
        let run = try toyRun(profile: .quick)
        let dir = run.store.directory(for: run.subjectID)
        #expect(FileManager.default.fileExists(atPath: dir.path), "latents were cached")
        let forgot = try run.store.forget(run.subjectID)
        #expect(forgot)
        #expect(!FileManager.default.fileExists(atPath: dir.path))
    }
}
