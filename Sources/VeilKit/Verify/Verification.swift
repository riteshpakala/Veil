//
//  Verification.swift
//  VeilKit
//
//  Verify: always on the exported guard file as read back, on held-out photos, with draws from
//  an independent stream. What ships is what's measured.
//
//    Suppression   every route that reached is back inside the null (guarded p > α)
//    Robustness    soft attacks on base vs guarded: the budget at which each crosses the null
//    Preservation  control identities and generic prompts barely move (velocity drift), and a
//                  named control's own name still pulls their photos
//
//  Verdict: guarded · partial · not-needed · inconclusive. Tolerances are uncalibrated until the
//  validation experiment runs; they are reported with every result.
//

import Foundation
import MLX

public struct SuppressionResult: Codable, Sendable, Hashable {
    public let routeID: String
    public let kind: Route.Kind
    public let label: String?
    public let labelHash: String
    public let base: RouteMeasurement
    public let guarded: RouteMeasurement
    /// Guarded pull on the person ÷ base pull (1 = untouched, ≤ 0 = gone).
    public let residual: Double
    public let pass: Bool
}

public struct DriftMeasurement: Codable, Sendable, Hashable {
    public let label: String
    /// E‖v̂_G − v̂₀‖² / E‖v̂₀ − u‖² over the draws.
    public let velocityDrift: Double
    /// (ℓ_G − ℓ₀) / ℓ₀, signed.
    public let lossDrift: Double
}

public struct NamedControlCheck: Codable, Sendable, Hashable {
    public let name: String
    public let basePull: PullEstimate
    public let guardedPull: PullEstimate
    /// guarded ÷ base pull.
    public let retained: Double
    /// Only identities the base model itself recognizes (base pull > 0 at 95%) are judged.
    public let judged: Bool
    public let pass: Bool
}

public struct PreservationResult: Codable, Sendable, Hashable {
    /// Other people under people prompts (and the user's descriptions).
    public let controls: [DriftMeasurement]
    /// Generic prompts with no people.
    public let generic: [DriftMeasurement]
    public let namedControls: [NamedControlCheck]
    public let maxDrift: Double
    public let tolerance: Double
    public let pass: Bool
}

public struct RobustnessResult: Codable, Sendable, Hashable {
    public let base: AttackCurve
    public let guarded: AttackCurve
}

public enum GuardVerdict: String, Codable, Sendable {
    /// Every reached route is back inside the null and preservation holds.
    case guarded
    /// Some route survives or preservation fails (see reasons).
    case partial
    /// Nothing reached the person, so there was nothing to block.
    case notNeeded = "not-needed"
    /// The test could not be run validly (see reasons).
    case inconclusive
}

public struct Verification: Codable, Sendable {
    public let guardSHA256: String?
    public let baseNull: NullDistribution
    public let guardedNull: NullDistribution
    public let suppression: [SuppressionResult]
    public let robustness: [RobustnessResult]
    public let preservation: PreservationResult
    public let verdict: GuardVerdict
    public let reasons: [String]
}

public enum Verifier {
    public struct Inputs {
        public let routes: [Route]
        /// Which of `routes` reached in the assessment (by route id).
        public let reached: Set<String>
        public let nulls: [Route]
        public let fit: [EncodedPhoto]
        public let heldOut: [EncodedPhoto]
        public let controls: [EncodedPhoto]
        public let controlIdentities: [String?]
        /// Descriptions that must keep working (those that didn't reach the person).
        public let descriptions: [String]
        public let anchor: String

        public init(routes: [Route], reached: Set<String>, nulls: [Route], fit: [EncodedPhoto], heldOut: [EncodedPhoto],
                    controls: [EncodedPhoto], controlIdentities: [String?], descriptions: [String], anchor: String) {
            self.routes = routes
            self.reached = reached
            self.nulls = nulls
            self.fit = fit
            self.heldOut = heldOut
            self.controls = controls
            self.controlIdentities = controlIdentities
            self.descriptions = descriptions
            self.anchor = anchor
        }
    }

    /// `model` must have the guard (as read back from the file) installed.
    public static func verify(model: GuardableModel, inputs: Inputs, guardSHA256: String?, profile: VeilProfile,
                              schedule: SeedSchedule, progress: ((String) -> Void)? = nil) throws -> (Verification, PullBench) {
        let sigmas = SigmaGrid.evenly(model.samplerSigmas(latentShape: inputs.heldOut.first?.shape ?? []), count: profile.sigmaCount)
        let subject = DrawSet(label: "verify-subject", photos: inputs.heldOut, sigmas: sigmas, noisePerSigma: profile.noisePerSigma,
                              stream: .verify, schedule: schedule)
        let controls = inputs.controls.isEmpty ? nil
            : DrawSet(label: "verify-controls", photos: inputs.controls, sigmas: sigmas, noisePerSigma: 1, stream: .verify, schedule: schedule)
        let bench = PullBench(model: model, subject: subject, controls: controls, controlIdentities: inputs.controlIdentities)
        let tag = "guard:" + (guardSHA256.map { String($0.prefix(12)) } ?? "memory")
        bench.setGuardTag(tag)
        // Names and descriptions always; discovered prompts when they reached.
        let measured = inputs.routes.filter { $0.kind != .discovered || inputs.reached.contains($0.id) }

        progress?("verify: base")
        let (baseMs, baseNull) = try model.hooks.with(guard: false) {
            try Reachability.assess(bench: bench, routes: measured, nulls: inputs.nulls, alpha: profile.alpha)
        }
        progress?("verify: guarded")
        let (guardMs, guardNull) = try model.hooks.with(guard: true) {
            try Reachability.assess(bench: bench, routes: measured, nulls: inputs.nulls, alpha: profile.alpha)
        }
        var suppression: [SuppressionResult] = []
        for (b, g) in zip(baseMs, guardMs) {
            let residual = g.subject.mean / (abs(b.subject.mean) > 1e-12 ? b.subject.mean : 1e-12)
            // Guarded, no route may reach the person (including routes that didn't reach before).
            let pass = !g.reaches
            suppression.append(SuppressionResult(routeID: b.id, kind: b.kind, label: b.label, labelHash: b.labelHash, base: b,
                                                 guarded: g, residual: residual, pass: pass))
        }

        // Robustness: soft attacks from the first name prompt (recovering the erased name) and
        // from the anchor (reaching the person from scratch), on base and guarded.
        try Task.checkCancellation()
        progress?("verify: attacks")
        var robustness: [RobustnessResult] = []
        let attackFit = DrawSet(label: "attack-fit", photos: Array(inputs.fit.prefix(2)),
                                sigmas: [sigmas[sigmas.count / 2]], noisePerSigma: 1, stream: .attack, schedule: schedule)
        var starts: [(String, String)] = []
        if let name = measured.first(where: { $0.kind == .name }) { starts.append((name.prompts[0], name.anchors[0])) }
        starts.append((Templates.fill(Templates.name[0], inputs.anchor), Templates.fill(Templates.name[0], inputs.anchor)))
        for (startPrompt, anchor) in starts where !inputs.fit.isEmpty {
            let startEmbedding = try bench.embeddings([startPrompt])[0].value
            let label = startPrompt == anchor ? "anchor" : "name"
            let base = try model.hooks.with(guard: false) {
                try SoftPromptAttack.curve(bench: bench, startLabel: label, start: startEmbedding, anchor: anchor, fit: attackFit,
                                           budgets: profile.attackBudgets, learningRate: profile.attackLearningRate,
                                           radius: profile.attackRadius, threshold: baseNull.q95Single)
            }
            let guarded = try model.hooks.with(guard: true) {
                try SoftPromptAttack.curve(bench: bench, startLabel: label, start: startEmbedding, anchor: anchor, fit: attackFit,
                                           budgets: profile.attackBudgets, learningRate: profile.attackLearningRate,
                                           radius: profile.attackRadius, threshold: guardNull.q95Single)
            }
            robustness.append(RobustnessResult(base: base, guarded: guarded))
        }

        try Task.checkCancellation()
        progress?("verify: preservation")
        let preservation = try preserve(model: model, bench: bench, inputs: inputs, sigmas: sigmas, profile: profile,
                                        schedule: schedule)

        var reasons: [String] = []
        let failed = suppression.filter { !$0.pass }
        for f in failed {
            reasons.append("route \(f.label.map { "“\($0)”" } ?? f.labelHash) still reaches the person (guarded p = \(f.guarded.pValue.map { String(format: "%.2f", $0) } ?? "–"))")
        }
        if !preservation.pass {
            reasons.append(String(format: "preservation: largest drift %.3f exceeds %.3f", preservation.maxDrift, preservation.tolerance))
            for n in preservation.namedControls where n.judged && !n.pass {
                reasons.append(String(format: "“%@” keeps only %.0f%% of its own pull", n.name, 100 * n.retained))
            }
        }
        let anyReached = !inputs.reached.isEmpty || baseMs.contains(where: \.reaches)
        var verdict: GuardVerdict = !anyReached ? .notNeeded : (failed.isEmpty && preservation.pass ? .guarded : .partial)
        if Set(baseNull.specificities).count <= 1 {
            verdict = .inconclusive
            reasons.append("the null is degenerate (every invented name measured the same), so nothing can be calibrated")
        }
        if !model.blockingIssues.isEmpty {
            verdict = .inconclusive
            reasons += model.blockingIssues
        }
        if verdict == .notNeeded { reasons.append("no route reached the person in the assessment; the guard is optional") }
        if verdict == .guarded {
            let costs = robustness.map { r in
                "\(r.guarded.start): \(r.guarded.reachedAt.map { "\($0) steps" } ?? "> \(profile.attackBudgets.max() ?? 0) steps") (base: \(r.base.reachedAt.map { "\($0)" } ?? "> \(profile.attackBudgets.max() ?? 0)"))"
            }
            reasons.append("blocked against the routes searched; soft-attack cost to cross the null — " + costs.joined(separator: "; "))
        }
        let v = Verification(guardSHA256: guardSHA256, baseNull: baseNull, guardedNull: guardNull, suppression: suppression,
                             robustness: robustness, preservation: preservation, verdict: verdict, reasons: reasons)
        return (v, bench)
    }

    static func preserve(model: GuardableModel, bench: PullBench, inputs: Inputs, sigmas: [Float], profile: VeilProfile,
                         schedule: SeedSchedule) throws -> PreservationResult {
        // Where "everyone else" lives: the controls, or the person's own photos when there are none.
        let others = inputs.controls.isEmpty ? inputs.fit + inputs.heldOut : inputs.controls
        let drawSet = DrawSet(label: "preserve", photos: others, sigmas: sigmas, noisePerSigma: 1, stream: .verify, schedule: schedule)
        func drift(_ prompt: String) throws -> DriftMeasurement {
            Drift.measure(model: model, set: drawSet, prompt: prompt, conditioning: try bench.embeddings([prompt])[0].value)
        }
        let people = Array(Templates.people.prefix(4)) + inputs.descriptions
        let controls = try people.map(drift)
        let generic = try Array(Templates.generic.prefix(profile.name == "quick" ? 3 : 6)).map(drift)

        var named: [NamedControlCheck] = []
        let byName = Dictionary(grouping: inputs.controls.indices.filter { inputs.controlIdentities[$0] != nil }) { inputs.controlIdentities[$0]! }
        for (name, indices) in byName.sorted(by: { $0.key < $1.key }) {
            let set = DrawSet(label: "named-\(name)", photos: indices.map { inputs.controls[$0] }, sigmas: sigmas,
                              noisePerSigma: profile.noisePerSigma, stream: .verify, schedule: schedule)
            let route = Templates.nameRoute(name, count: min(3, profile.templatesPerName), anchor: inputs.anchor)
            let base = try model.hooks.with(guard: false) { try bench.pull(route, on: set) }
            let guarded = try model.hooks.with(guard: true) { try bench.pull(route, on: set) }
            let judged = base.lowerBound > 0
            let retained = guarded.mean / max(base.mean, 1e-12)
            named.append(NamedControlCheck(name: name, basePull: base, guardedPull: guarded, retained: retained, judged: judged,
                                           pass: !judged || retained >= profile.retainedNamePull))
        }
        let maxDrift = (controls + generic).map(\.velocityDrift).max() ?? 0
        let pass = maxDrift <= profile.preservationTolerance && named.allSatisfy(\.pass)
        return PreservationResult(controls: controls, generic: generic, namedControls: named, maxDrift: maxDrift,
                                  tolerance: profile.preservationTolerance, pass: pass)
    }
}

/// How far the guard moves the model on a prompt it is supposed to leave alone. Base and guarded
/// run on the same draws, so the difference is the guard and nothing else.
public enum Drift {
    public static func measure(model: GuardableModel, set: DrawSet, prompt: String, conditioning: MLXArray) -> DriftMeasurement {
        var num = 0.0, den = 0.0, lossBase = 0.0, lossGuard = 0.0
        for g in set.groups {
            var start = 0
            while start < g.rows.count {
                let end = min(g.rows.count, start + max(1, model.maxBatch))
                let x = g.x[start..<end], u = g.u[start..<end]
                let v0 = model.hooks.with(guard: false) { model.velocity(x, sigma: g.sigma, conditioning: conditioning) }.asType(.float32)
                let vg = model.hooks.with(guard: true) { model.velocity(x, sigma: g.sigma, conditioning: conditioning) }.asType(.float32)
                let e0 = (v0 - u).square().mean(), eg = (vg - u).square().mean(), dv = (vg - v0).square().mean()
                eval(e0, eg, dv)
                let n = Double(end - start)
                num += Double(dv.item(Float.self)) * n
                den += Double(e0.item(Float.self)) * n
                lossBase += Double(e0.item(Float.self)) * n
                lossGuard += Double(eg.item(Float.self)) * n
                start = end
            }
        }
        return DriftMeasurement(label: prompt, velocityDrift: num / max(den, 1e-12),
                                lossDrift: (lossGuard - lossBase) / max(lossBase, 1e-12))
    }
}

public enum SigmaGrid {
    /// `count` σ values spread evenly (by index) over a schedule, keeping the ends.
    public static func evenly(_ sigmas: [Float], count: Int) -> [Float] {
        guard count < sigmas.count, count > 0 else { return sigmas }
        if count == 1 { return [sigmas[sigmas.count / 2]] }
        return (0..<count).map { sigmas[Int((Double($0) * Double(sigmas.count - 1) / Double(count - 1)).rounded())] }
    }
}
