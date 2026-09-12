//
//  ToyWorldTests.swift
//  VeilKitTests
//
//  The analytic world: exact answers for pull, reachability and the building blocks.
//

import Foundation
import MLX
import Testing
@testable import VeilKit

@Suite(.serialized) struct ToyWorldTests {
    let world = ToyIdentityWorld.shared

    @Test func photosRoundTripToTheirLatents() throws {
        let photo = world.subjectPhotos[0]
        let decoded = world.latent(of: photo)
        // Re-encode the decoded latent: quantization is the only loss (8 bits over [-4, 4]).
        let again = world.latent(of: SubjectPhoto(image: ToyIdentityWorld.image(decoded)))
        #expect(abs(decoded - again).max().item(Float.self) < 1e-5)
        let mean = world.means[ToyIdentityWorld.subjectName]!.reshaped(3, 8, 8)
        let spread = (decoded - mean).square().mean().sqrt().item(Float.self)
        #expect(spread > 0.25 && spread < 0.6, "photo noise ≈ s = 0.4, got \(spread)")
    }

    @Test func readerLandsTheDesignPrompts() {
        #expect(world.readerFit > 0.8, "reader fit \(world.readerFit)")
    }

    @Test func anchorAgainstItselfHasExactlyZeroPull() throws {
        let model = ToyModel()
        let photos = try PhotoEncoding.encode(world.subjectPhotos, model: model, faceWeighting: .off)
        let set = DrawSet(label: "s", photos: photos, sigmas: [0.7, 0.4], noisePerSigma: 2, stream: .pull, schedule: .research)
        let bench = PullBench(model: model, subject: set, controls: nil)
        let route = Route(kind: .name, label: "anchor", prompts: ["a photo of a person"], anchors: ["a photo of a person"])
        let pull = try bench.pull(route, on: set)
        #expect(pull.mean == 0 && pull.standardError == 0)
    }

    @Test func zeroGuardIsBitwiseTheBase() throws {
        let model = ToyModel()
        let x = SeedSchedule.research.normal(.toy, 77, shape: [3, 3, 8, 8])
        let c = try model.embed(["a photo of Ada Quill"])[0].value
        let base = model.hooks.with(guard: false) { model.velocity(x, sigma: 0.5, conditioning: c) }
        model.installGuard(Dictionary(uniqueKeysWithValues: model.slots.map {
            ($0.key, LowRankDelta.zero(out: $0.outFeatures, in: $0.inFeatures, rank: 4))
        }))
        let off = model.hooks.with(guard: false) { model.velocity(x, sigma: 0.5, conditioning: c) }
        let zero = model.hooks.with(guard: true) { model.velocity(x, sigma: 0.5, conditioning: c) }
        #expect(abs(base - off).max().item(Float.self) == 0)
        #expect(abs(base - zero).max().item(Float.self) == 0)
    }

    @Test func knownNameReachesAndInventedNamesDoNot() throws {
        let model = ToyModel()
        let schedule = SeedSchedule.research
        let photos = try PhotoEncoding.encode(Array(world.subjectPhotos.prefix(3)), model: model, faceWeighting: .off)
        let controls = try PhotoEncoding.encode(world.controls.photos, model: model, faceWeighting: .off)
        let sigmas = SigmaGrid.evenly(model.samplerSigmas(latentShape: [3, 8, 8]), count: 4)
        let bench = PullBench(model: model,
                              subject: DrawSet(label: "s", photos: photos, sigmas: sigmas, noisePerSigma: 4, stream: .pull, schedule: schedule),
                              controls: DrawSet(label: "c", photos: controls, sigmas: sigmas, noisePerSigma: 1, stream: .pull, schedule: schedule))
        let routes = [Templates.nameRoute(ToyIdentityWorld.subjectName, count: 6, anchor: "a person"),
                      Templates.textRoute(ToyIdentityWorld.subjectDescription, kind: .description, anchor: "a person"),
                      Templates.nameRoute("Bram Oake", count: 6, anchor: "a person")]
        let nulls = NullNames.make(19, schedule: schedule, avoiding: [ToyIdentityWorld.subjectName]).map {
            Templates.nameRoute($0, count: 6, anchor: "a person", kind: .null)
        }
        let (ms, null) = try Reachability.assess(bench: bench, routes: routes, nulls: nulls, alpha: 0.05)
        for m in ms {
            print(String(format: "%@  pull %.3f ± %.3f (rel %.2f)  controls %.3f  spec %.3f  p %.2f  reaches %@", m.id,
                         m.subject.mean, m.subject.standardError, m.subject.relative, m.controls?.mean ?? 0, m.specificity,
                         m.pValue ?? -1, m.reaches ? "yes" : "no"))
        }
        print("null q95", null.q95, "max", null.specificities.max() ?? 0)
        #expect(ms[0].reaches && ms[0].pValue! <= 0.05, "the known name reaches")
        #expect(ms[1].reaches, "the subject's description reaches")
        #expect(!ms[2].reaches, "another person's name does not reach the subject")
        #expect(ms[0].specificity > (null.specificities.max() ?? 0))
        #expect(Reachability.capability(ms) == .nameBound)
    }

    @Test func splitIsKeyedDisjointAndReproducible() {
        let ids = (0..<9).map { "photo-\($0)" }
        let a = PhotoSplit.make(photoIDs: ids, schedule: .research)
        let b = PhotoSplit.make(photoIDs: ids.reversed(), schedule: .research)
        #expect(a == b)
        #expect(Set(a.fit).isDisjoint(with: a.heldOut))
        #expect(a.fit.count + a.heldOut.count == 9 && a.heldOut.count == 3)
        let other = PhotoSplit.make(photoIDs: ids, schedule: SeedSchedule(keyData: Data("another key".utf8)))
        #expect(other.fit.count == a.fit.count)
        #expect(PhotoSplit.make(photoIDs: ["only"], schedule: .research).augmented)
    }

    @Test func tokenAlignmentPairsSpanSuffixAndPadding() {
        // <u> a photo of NAME1 NAME2 <e> | <u> a photo of a person <e>
        let prompt = [0, 5, 6, 7, 40, 41, 1], anchor = [0, 5, 6, 7, 8, 9, 1]
        let pairs = TokenAlignment.pairs(prompt: prompt, anchor: anchor, sequence: 10)
        #expect(pairs.first! == (4, 4) && pairs[1] == (5, 5))
        #expect(pairs.contains { $0 == (6, 6) } && pairs.contains { $0 == (9, 9) })
        #expect(!pairs.contains { $0.0 < 4 }, "the shared prefix is skipped")
        // Different span lengths: three name tokens onto one anchor token.
        let longer = TokenAlignment.pairs(prompt: [0, 5, 40, 41, 42, 1], anchor: [0, 5, 9, 1], sequence: 8)
        #expect(longer.prefix(3).allSatisfy { $0.1 == 2 })
        #expect(longer.contains { $0 == (5, 3) } && longer.contains { $0 == (6, 4) })
    }

    @Test func lowRankFactorRecoversALowRankMatrix() {
        let s = SeedSchedule.research
        let m = matmul(s.normal(.toy, 900, shape: [40, 5]), s.normal(.toy, 901, shape: [5, 60]))
        let (d, energy, truncation) = LowRank.factor(m, maxRank: 16, energy: 0.9999, schedule: s)
        #expect(d.rank == 5, "rank \(d.rank)")
        #expect(energy > 0.9999 && truncation < 1e-3, "energy \(energy) truncation \(truncation)")
    }

    @Test func nullNamesAvoidRealNamesAndRepeat() {
        let a = NullNames.make(19, schedule: .research, avoiding: ["Ada Quill"])
        #expect(a.count == 19 && Set(a).count == 19)
        #expect(a == NullNames.make(19, schedule: .research, avoiding: ["Ada Quill"]))
    }
}

@Suite(.serialized) struct SearchAndLocatorTests {
    @Test func discreteSearchFindsTokensThatPullTowardTheSubject() throws {
        let model = ToyModel()
        let world = ToyIdentityWorld.shared
        let photos = try PhotoEncoding.encode(Array(world.subjectPhotos.prefix(4)), model: model, faceWeighting: .off)
        let fit = DrawSet(label: "fit", photos: photos, sigmas: [0.72, 0.48], noisePerSigma: 1, stream: .search, schedule: .research)
        let found = try PromptSearch.run(model: model, search: try #require(model.tokenSearch), fit: fit, tokens: 3, steps: 60,
                                         schedule: .research)
        let seedLoss = LossEvaluator.evaluate(model, fit, conditioning: try model.embed(["a photo of"])[0].value).losses
        let best = try #require(found.first)
        print("search: best “\(best.text)” loss \(best.fitLoss) vs seed \(Stats.mean(seedLoss))")
        #expect(best.fitLoss < Stats.mean(seedLoss), "search lowers the loss on the person's photos")
        let words = Set(best.text.split(separator: " ").map(String.init))
        #expect(!words.isDisjoint(with: ["ada", "quill", "painter", "lisbon", "red-haired"]), "and does it with her tokens: \(best.text)")
    }

    @Test func locatorPutsTheToyRouteInTheReader() throws {
        let model = ToyModel()
        let photos = try PhotoEncoding.encode(Array(ToyIdentityWorld.shared.subjectPhotos.prefix(2)), model: model, faceWeighting: .off)
        let set = DrawSet(label: "s", photos: photos, sigmas: [0.6], noisePerSigma: 1, stream: .pull, schedule: .research)
        let bench = PullBench(model: model, subject: set, controls: nil)
        let sites = try RouteLocator.locate(model: model, route: Templates.nameRoute("Ada Quill", count: 1, anchor: "a person"),
                                            set: set, embeddings: bench)
        #expect(sites.count == 1 && sites[0].key == "reader")
        #expect(abs(sites[0].restoredShare - 1) < 1e-3)
    }
}

@Suite struct ShareableTests {
    @Test func localPathsNeverReachSharedFiles() {
        let d = ModelDescriptor(link: "/Users/someone/models/FLUX.2-klein-base-4B", host: .local,
                                repo: "/Users/someone/models/FLUX.2-klein-base-4B", family: "flux2-klein", detail: "x")
        #expect(d.pinnedName == "FLUX.2-klein-base-4B (local)")
        #expect(d.shareable.repo == "FLUX.2-klein-base-4B" && d.shareable.link == "FLUX.2-klein-base-4B")
        let hosted = ModelDescriptor(link: "https://huggingface.co/a/b", host: .huggingFace, repo: "a/b", commit: "0123456789abcdef",
                                     family: "flux2-klein", detail: "x")
        #expect(hosted.pinnedName == "a/b@0123456789ab" && hosted.shareable.repo == "a/b")
        let file = GuardFile(path: "/Users/someone/out/guard.safetensors", sha256: "s", format: "diffusers", slots: [], rank: 1,
                             parameters: 1, bytes: 1)
        #expect(file.shareable.path == "guard.safetensors")
    }
}
