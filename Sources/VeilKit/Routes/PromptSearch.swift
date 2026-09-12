//
//  PromptSearch.swift
//  VeilKit
//
//  Discrete prompt search (PEZ — Wen et al., "Hard Prompts Made Easy", 2023): look for real
//  token sequences that pull the model toward the person without naming them. Continuous
//  token embeddings are optimized, projected to their nearest vocabulary tokens every step, and
//  the gradient at the projection updates the continuous copy (straight-through).
//
//  What it finds are attack recipes: they go into the sealed routes file, and shareable
//  reports list them by hash only.
//

import Foundation
import MLX

public struct SearchCandidate: Codable, Sendable, Hashable {
    public let text: String
    /// Mean weighted loss on the fit draws (lower = stronger pull).
    public let fitLoss: Double
    public let step: Int
}

public enum PromptSearch {
    /// Search `tokens` free tokens after `seedText`; returns distinct candidates, strongest first.
    public static func run(model: VelocityModel, search: TokenSearchInterface, fit: DrawSet, seedText: String = "a photo of",
                           tokens: Int, steps: Int, schedule: SeedSchedule, keep: Int = 3,
                           progress: ((String) -> Void)? = nil) throws -> [SearchCandidate] {
        guard steps > 0, tokens > 0, !fit.groups.isEmpty else { return [] }
        let table = try search.tokenTable()
        let tableF = table.asType(.float32)
        let norms = tableF.square().sum(axis: 1, keepDims: true).sqrt()
        let unit = tableF / maximum(norms, 1e-6)
        let vocab = table.dim(0)
        let excluded = search.excludedTokenIDs
        let mask = MLXArray((0..<vocab).map { excluded.contains($0) ? Float(-1e9) : 0 })
        let rowScale = norms.mean().item(Float.self)

        let (prefix, seed, suffix) = try search.tokenize(seedText)
        let head = tableF[MLXArray((prefix + seed).map(Int32.init))]
        let tail = tableF[MLXArray(suffix.map(Int32.init))]

        var rng = schedule.rng(.search, "init|\(seedText)")
        let allowed = (0..<vocab).filter { !excluded.contains($0) }
        var ids = (0..<tokens).map { _ in allowed[Int(rng.next() % UInt64(allowed.count))] }
        var soft = tableF[MLXArray(ids.map(Int32.init))]
        var adam = AdamState(like: [soft], learningRate: 0.1 * rowScale)

        let lossAndGrad = valueAndGrad({ (arrays: [MLXArray]) -> [MLXArray] in
            let embeddings = concatenated([head, arrays[0], tail], axis: 0).expandedDimensions(axis: 0)
            let conditioning = search.conditioning(fromTokenEmbeddings: embeddings)
            var total = MLXArray(Float(0))
            for g in fit.groups {
                total = total + Losses.weighted(model.velocity(g.x, sigma: g.sigma, conditioning: conditioning), g.u, g.w)
            }
            return [total / Float(fit.groups.count)]
        }, argumentNumbers: [0])

        var best: [String: SearchCandidate] = [:]
        for step in 0..<steps where !Task.isCancelled {
            // Project to the nearest real tokens (cosine), evaluate there, update the soft copy.
            let sims = matmul(soft / maximum(soft.square().sum(axis: 1, keepDims: true).sqrt(), 1e-6), unit.transposed()) + mask
            ids = argMax(sims, axis: 1).asArray(Int32.self).map(Int.init)
            let hard = tableF[MLXArray(ids.map(Int32.init))]
            let (values, grads) = lossAndGrad([hard])
            let loss = Double(values[0].item(Float.self))
            soft = adam.step([soft], grads)[0]
            let text = search.decode(seed + ids).trimmingCharacters(in: .whitespacesAndNewlines)
            if best[text].map({ loss < $0.fitLoss }) ?? true { best[text] = SearchCandidate(text: text, fitLoss: loss, step: step) }
            if step % max(1, steps / 5) == 0 { progress?("search step \(step + 1)/\(steps)") }
        }
        return Array(best.values.sorted { $0.fitLoss < $1.fitLoss }.prefix(keep))
    }
}

public struct LocatorSite: Codable, Sendable, Hashable {
    public let key: String
    /// Share of the route's pull restored when only this site carries the route's activations
    /// (the rest of the pass runs on the anchor).
    public let restoredShare: Double
}

/// Where a route lives: activation restoration (causal tracing, Meng et al. 2022; Basu et al.
/// 2024 for text-to-image). Run the anchor prompt, restore one site's output from the route's
/// pass, and see how much of the route's pull comes back.
public enum RouteLocator {
    public static func locate(model: GuardableModel, route: Route, set: DrawSet, embeddings: PullBench) throws -> [LocatorSite] {
        let sites = model.slots.filter(\.locatorSite).map(\.key)
        guard !sites.isEmpty, let group = set.groups.first else { return [] }
        let (routeC, anchorC) = try (embeddings.embeddings([route.prompts[0]])[0].value, embeddings.embeddings([route.anchors[0]])[0].value)
        let rows = min(group.rows.count, max(1, model.maxBatch))
        let x = group.x[0..<rows], u = group.u[0..<rows], w = group.w[0..<rows]
        let hooks = model.hooks
        func loss(_ c: MLXArray) -> Double {
            Double(Losses.weighted(model.velocity(x, sigma: group.sigma, conditioning: c), u, w).item(Float.self))
        }
        hooks.recordKeys = Set(sites)
        defer { hooks.recordKeys = []; hooks.restore = [:]; hooks.clearRecordings() }
        hooks.clearRecordings()
        let lRoute = loss(routeC)
        let recorded = hooks.recorded
        eval(Array(recorded.values))
        hooks.recordKeys = []
        let lAnchor = loss(anchorC)
        let total = lAnchor - lRoute
        guard abs(total) > 1e-12 else { return sites.map { LocatorSite(key: $0, restoredShare: 0) } }
        var out: [LocatorSite] = []
        for site in sites {
            guard let value = recorded[site] else { continue }
            hooks.restore = [site: value]
            out.append(LocatorSite(key: site, restoredShare: (lAnchor - loss(anchorC)) / total))
            hooks.restore = [:]
        }
        return out.sorted { $0.restoredShare > $1.restoredShare }
    }
}
