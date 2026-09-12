//
//  GroupEditDiagnostics.swift
//  VeilFlux2Tests
//
//  How much of the name→anchor velocity gap the closed form actually closes on the real slot,
//  measured over several noise draws and σ levels rather than one.
//
//  A single draw is not enough: the same edit measured at two different noise seeds read as
//  "40% of the gap closed" and "nothing happened". So the statistic here is the gap energy
//  E‖v̂_G(name) − v̂₀(anchor)‖² over a fixed keyed draw set, as a fraction of the same quantity
//  before the edit, with the per-draw spread printed beside it. It asserts nothing — this is
//  measurement, not a gate.
//

import Foundation
import FluxKit
import MLX
import Testing
@testable import VeilFlux2
@testable import VeilKit

@Suite(.serialized, .enabled(if: haveBase && ProcessInfo.processInfo.environment["VEIL_DIAGNOSTICS"] != nil,
                             "set VEIL_DIAGNOSTICS=1 to run this sweep (~3 minutes on the real weights)"))
struct GroupEditDiagnostics {
    static func pad(_ s: String, _ n: Int) -> String { s.count >= n ? s : s + String(repeating: " ", count: n - s.count) }

    @Test func howMuchOfTheGapTheClosedFormClosesOnTheRealSlot() throws {
        let m = try #require(Flux2ExecutorTests.model)
        var profile = VeilProfile.quick
        profile.closedFormMaxRank = 64
        let names = ["Abraham Lincoln", "Frederick Douglass"]
        let sigmas: [Float] = [0.9, 0.6]
        let draws = 4
        // One keyed noise latent per (draw, σ): every condition sees the same ones.
        let x: [[MLXArray]] = sigmas.indices.map { s in
            (0..<draws).map { d in SeedSchedule.research.normal(.toy, 900 + 10 * s + d, shape: [1, 32, 16, 16]) * 0.9 }
        }

        func velocity(_ prompt: String, _ guarded: Bool, _ s: Int, _ d: Int) throws -> MLXArray {
            let c = try m.embed([prompt])[0].value
            return m.hooks.with(guard: guarded) { m.velocity(x[s][d], sigma: sigmas[s], conditioning: c) }
        }
        /// Squared gap to the base model under the anchor, one value per (σ, draw).
        func gaps(_ name: String, guarded: Bool) throws -> [Double] {
            var out: [Double] = []
            for s in sigmas.indices {
                for d in 0..<draws {
                    let anchor = try velocity("a photo of a person", false, s, d)
                    let v = try velocity("a photo of \(name)", guarded, s, d)
                    let e = (v - anchor).square().mean()
                    eval(e)
                    out.append(Double(e.item(Float.self)))
                }
            }
            return out
        }

        var before: [String: [Double]] = [:]
        for name in names { before[name] = try gaps(name, guarded: false) }
        for name in names {
            let g = before[name]!
            print(String(format: "base gap %@: mean %.4f, per draw %@", name, Stats.mean(g),
                         g.map { String(format: "%.3f", $0) }.joined(separator: " ")))
        }
        let ramenBase = try velocity("a bowl of ramen", false, 0, 0)

        func run(_ label: String, people: [String], templates: Int, preserve: [String]) throws {
            var pairs: [(prompt: PromptEmbedding, anchor: PromptEmbedding)] = []
            for name in people {
                let route = Templates.nameRoute(name, count: templates, anchor: "a person")
                pairs += try zip(m.embed(route.prompts), m.embed(route.anchors)).map { (prompt: $0, anchor: $1) }
            }
            let preserveEmbeddings = try m.embed(preserve)
            let slot = try #require(m.linear("context_embedder"))
            let edit = ClosedFormEditor.fit(key: "context_embedder", weight: slot.deployedWeight(), erase: pairs,
                                            preserve: preserveEmbeddings, preserveWeight: profile.closedFormPreserve,
                                            ridge: profile.closedFormRidge, maxRank: 64, energy: profile.closedFormEnergy,
                                            schedule: .research)
            m.installGuard(["context_embedder": edit.delta])
            defer { m.installGuard(nil) }
            var line = Self.pad(label, 28)
            line += String(format: "erase %3d preserve %3d size %.2f | ", pairs.count, preserve.count, edit.relativeSize)
            for name in people {
                let after = try gaps(name, guarded: true)
                let b = before[name]!
                // Per-draw ratio, so the spread across draws is visible next to the mean.
                let ratios = zip(after, b).map { $0 / max($1, 1e-12) }
                let surname = name.split(separator: " ").last.map(String.init) ?? name
                line += String(format: "%@ %3.0f%% of the gap left (per draw %.0f–%.0f%%)  ", surname,
                               100 * Stats.mean(after) / Stats.mean(b), 100 * (ratios.min() ?? 0), 100 * (ratios.max() ?? 0))
            }
            let drift = try (velocity("a bowl of ramen", true, 0, 0) - ramenBase).square().mean().item(Float.self)
                / ramenBase.square().mean().item(Float.self)
            print(line + String(format: "drift %.3f", drift))
        }

        let minimal = Array(Templates.generic.prefix(10)) + Array(Templates.people.prefix(6))
        func planPreserve(_ people: [String], _ templates: Int) -> [String] {
            GuardPlan.make(people: people.map {
                GuardPlan.Person(id: $0, names: [$0], descriptions: [], anchor: "a person", fit: [])
            }, profile: profile, schedule: .research, templatesOverride: templates).preserve
        }

        print("— one person —")
        try run("minimal, 6 templates", people: [names[0]], templates: 6, preserve: minimal)
        try run("minimal, 8 templates", people: [names[0]], templates: 8, preserve: minimal)
        try run("plan preserve, 6", people: [names[0]], templates: 6, preserve: planPreserve([names[0]], 6))
        try run("plan preserve, 24", people: [names[0]], templates: 24, preserve: planPreserve([names[0]], 24))
        print("— two people —")
        try run("minimal, 6 each", people: names, templates: 6, preserve: minimal)
        try run("plan preserve, 6 each", people: names, templates: 6, preserve: planPreserve(names, 6))
        try run("plan preserve, 12 each", people: names, templates: 12, preserve: planPreserve(names, 12))
    }
}
