//
//  SoftPromptAttack.swift
//  VeilKit
//
//  The strongest white-box attack at the model's text interface: optimize a perturbation δ of a
//  prompt's conditioning so the model denoises the person's photos better, within a norm bound
//  (‖δ‖_rms ≤ radius · ‖c‖_rms). Textual-inversion-style attacks recover erased concepts on
//  every published erasure method (Pham et al., ICLR 2024), so this is the yardstick for
//  robustness — "how many steps until the guard gives way" — not a capability claim: with
//  enough freedom it reaches almost any face.
//
//  The attacker optimizes on the fit photos; success is judged on held-out photos.
//

import Foundation
import MLX

public struct AttackCurve: Codable, Sendable, Hashable {
    /// Where the attack started (a route's first prompt, or the anchor).
    public let start: String
    public let budgets: [Int]
    /// Specificity on held-out photos after each budget.
    public let specificity: [Double]
    /// Person's held-out pull after each budget.
    public let subjectPull: [Double]
    /// The null threshold the attack had to cross (95th percentile of the null specificities).
    public let threshold: Double
    /// First budget at which the attack crossed the threshold; nil when it never did.
    public let reachedAt: Int?
}

public enum SoftPromptAttack {
    /// Snapshots of start + δ after each budget (in steps).
    public static func run(model: VelocityModel, start: MLXArray, fit: DrawSet, budgets: [Int], learningRate: Float,
                           radius: Float) -> [(budget: Int, conditioning: MLXArray)] {
        let startF = start.asType(.float32)
        let bound = radius * startF.square().mean().sqrt().item(Float.self)
        var delta = MLXArray.zeros(like: startF)
        var adam = AdamState(like: [delta], learningRate: learningRate * max(bound, 1e-6))
        let lossAndGrad = valueAndGrad({ (arrays: [MLXArray]) -> [MLXArray] in
            let conditioning = startF + arrays[0]
            var total = MLXArray(Float(0))
            for group in fit.groups {
                let v = model.velocity(group.x, sigma: group.sigma, conditioning: conditioning)
                total = total + Losses.weighted(v, group.u, group.w)
            }
            return [total / Float(max(fit.groups.count, 1))]
        }, argumentNumbers: [0])
        var snapshots: [(Int, MLXArray)] = []
        let maxBudget = budgets.max() ?? 0
        var step = 0
        if budgets.contains(0) { snapshots.append((0, startF)) }
        while step < maxBudget, !Task.isCancelled {
            let (_, grads) = lossAndGrad([delta])
            delta = adam.step([delta], grads)[0]
            let rms = delta.square().mean().sqrt().item(Float.self)
            if rms > bound { delta = delta * (bound / rms) }
            step += 1
            if budgets.contains(step) {
                let snap = startF + delta
                eval(snap)
                snapshots.append((step, snap))
            }
        }
        return snapshots
    }

    /// Run the attack from `start` and judge every snapshot on the bench's held-out sets.
    public static func curve(bench: PullBench, startLabel: String, start: MLXArray, anchor: String, fit: DrawSet,
                             budgets: [Int], learningRate: Float, radius: Float, threshold: Double) throws -> AttackCurve {
        let snaps = run(model: bench.model, start: start, fit: fit, budgets: budgets, learningRate: learningRate, radius: radius)
        var specs: [Double] = [], pulls: [Double] = []
        var reached: Int?
        for (budget, conditioning) in snaps {
            let (s, spec) = try bench.specificity(conditioning: conditioning, anchor: anchor)
            specs.append(spec)
            pulls.append(s.mean)
            if reached == nil, spec > threshold, s.lowerBound > 0 { reached = budget }
        }
        return AttackCurve(start: startLabel, budgets: snaps.map(\.budget), specificity: specs, subjectPull: pulls,
                           threshold: threshold, reachedAt: reached)
    }
}
