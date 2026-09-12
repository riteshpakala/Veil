//
//  Pull.swift
//  VeilKit
//
//  Pull: how much a prompt helps the model denoise a person's photos, compared with the same
//  prompt with the name replaced by a neutral anchor. Decode-free: it is the conditional-
//  likelihood gain the prompt buys on the photos themselves (a diffusion-classifier style
//  statistic), never a generated image.
//
//      x_σ = (1 − σ)·x₀ + σ·ε,   u = ε − x₀
//      ℓ(c) = Σ_p w_p · mean_ch (v̂(x_σ, σ, c) − u)² / Σ_p w_p      (w: likeness weight)
//      Pull(c) = E[ ℓ(anchor) − ℓ(c) ]                               (paired on x_σ, σ, ε)
//
//  Draws are keyed by (photo, σ, n) so every prompt, and base vs guarded, sees the same noise.
//

import Foundation
import MLX

/// A photo in the model's latent space with its likeness weights.
public struct EncodedPhoto: @unchecked Sendable {
    public let photoID: String
    public let name: String
    /// (C, H, W) float32.
    public let latent: MLXArray
    public let weight: LikenessWeight
    /// The source photo (for figures; never written into reports). Nil for model samples.
    public let photo: SubjectPhoto?

    public init(photoID: String, name: String, latent: MLXArray, weight: LikenessWeight, photo: SubjectPhoto?) {
        self.photoID = photoID
        self.name = name
        self.latent = latent
        self.weight = weight
        self.photo = photo
    }

    public var shape: [Int] { latent.shape }
}

public enum PhotoEncoding {
    /// Encode photos (cached per subject when a store is given) with likeness weights.
    public static func encode(_ photos: [SubjectPhoto], model: GuardableModel, faceWeighting: FaceWeighting,
                              store: SubjectStore? = nil, subjectID: String? = nil) throws -> [EncodedPhoto] {
        try photos.map { photo in
            let latent: MLXArray
            if let store, let subjectID,
               let cached = store.cachedLatent(subjectID: subjectID, encodingKey: model.encodingKey, photoID: photo.id) {
                latent = cached
            } else {
                latent = try model.encodeLatent(photo).asType(.float32)
                eval(latent)
                if let store, let subjectID {
                    store.storeLatent(latent, subjectID: subjectID, encodingKey: model.encodingKey, photoID: photo.id)
                }
            }
            let weight = LikenessWeighter.weight(for: photo, rows: latent.dim(1), cols: latent.dim(2), policy: faceWeighting)
            return EncodedPhoto(photoID: photo.id, name: photo.name, latent: latent, weight: weight, photo: photo)
        }
    }
}

/// A fixed set of keyed draws over photos and σ levels, pre-noised and grouped by (σ, shape) so
/// each group is one batch dimension.
public final class DrawSet: @unchecked Sendable {
    public struct Group {
        public let sigma: Float
        /// Rows of this group, as (photo index, noise index).
        public let rows: [(photo: Int, noise: Int)]
        let x: MLXArray        // (R, C, H, W) noised
        let u: MLXArray        // (R, C, H, W) target ε − x₀
        let w: MLXArray        // (R, 1, H, W) likeness weights
    }

    public let label: String
    public let photos: [EncodedPhoto]
    public let sigmas: [Float]
    public let noisePerSigma: Int
    public let groups: [Group]
    /// Row order of `evaluate` results: (group, row) flattened.
    public var count: Int { groups.reduce(0) { $0 + $1.rows.count } }
    /// Photo index of every draw, in result order.
    public var drawPhotos: [Int] { groups.flatMap { $0.rows.map(\.photo) } }

    public init(label: String, photos: [EncodedPhoto], sigmas: [Float], noisePerSigma: Int, stream: SeedSchedule.Stream,
                schedule: SeedSchedule) {
        self.label = label
        self.photos = photos
        self.sigmas = sigmas
        self.noisePerSigma = noisePerSigma
        var groups: [Group] = []
        let shapes = Array(Set(photos.map { $0.shape.map(String.init).joined(separator: "x") })).sorted()
        for sigma in sigmas {
            for shapeKey in shapes {
                var rows: [(Int, Int)] = [], xs: [MLXArray] = [], us: [MLXArray] = [], ws: [MLXArray] = []
                for (i, p) in photos.enumerated() where p.shape.map(String.init).joined(separator: "x") == shapeKey {
                    for n in 0..<noisePerSigma {
                        let eps = schedule.normal(stream, "\(p.photoID)|\(SeedSchedule.levelKey(sigma))|\(n)", shape: p.shape)
                        xs.append((1 - sigma) * p.latent + sigma * eps)
                        us.append(eps - p.latent)
                        ws.append(MLXArray(p.weight.values, [1, p.weight.rows, p.weight.cols]))
                        rows.append((i, n))
                    }
                }
                guard !rows.isEmpty else { continue }
                let g = Group(sigma: sigma, rows: rows, x: stacked(xs, axis: 0), u: stacked(us, axis: 0), w: stacked(ws, axis: 0))
                eval(g.x, g.u, g.w)
                groups.append(g)
            }
        }
        self.groups = groups
    }
}

/// Per-draw weighted losses for one conditioning, plus per-photo loss maps.
public struct LossResult: @unchecked Sendable {
    /// One per draw, in DrawSet order.
    public let losses: [Double]
    /// Per photo: mean over its draws of the per-position loss (channel mean), row-major.
    public let maps: [[Float]]
}

public enum LossEvaluator {
    public static func evaluate(_ model: VelocityModel, _ set: DrawSet, conditioning: MLXArray, maps wantMaps: Bool = false) -> LossResult {
        var losses: [Double] = []
        var mapSums: [[Float]] = set.photos.map { [Float](repeating: 0, count: $0.weight.rows * $0.weight.cols) }
        var mapCounts = [Int](repeating: 0, count: set.photos.count)
        for group in set.groups {
            let r = group.rows.count
            var start = 0
            while start < r {
                let end = min(r, start + max(1, model.maxBatch))
                let x = group.x[start..<end], u = group.u[start..<end], w = group.w[start..<end]
                let v = model.velocity(x, sigma: group.sigma, conditioning: conditioning)
                let perPosition = (v.asType(.float32) - u).square().mean(axis: 1, keepDims: true)   // (b, 1, H, W)
                let weighted = (perPosition * w).sum(axes: [1, 2, 3]) / w.sum(axes: [1, 2, 3])
                eval(weighted)
                losses += weighted.asArray(Float.self).map(Double.init)
                if wantMaps {
                    let m = perPosition.squeezed(axis: 1)
                    eval(m)
                    for (k, row) in group.rows[start..<end].enumerated() {
                        let values = m[k].asArray(Float.self)
                        for j in values.indices { mapSums[row.photo][j] += values[j] }
                        mapCounts[row.photo] += 1
                    }
                }
                start = end
            }
        }
        let maps = wantMaps ? zip(mapSums, mapCounts).map { sums, n in sums.map { $0 / Float(max(n, 1)) } } : []
        return LossResult(losses: losses, maps: maps)
    }
}

public struct PullEstimate: Codable, Sendable, Hashable {
    /// Mean of ℓ(anchor) − ℓ(route) over draws.
    public let mean: Double
    public let standardError: Double
    /// Mean pull relative to the mean anchor loss.
    public let relative: Double
    public let draws: Int

    public init(mean: Double, standardError: Double, relative: Double, draws: Int) {
        self.mean = mean
        self.standardError = standardError
        self.relative = relative
        self.draws = draws
    }

    /// One-sided 95% lower bound.
    public var lowerBound: Double { mean - Stats.t95(max(draws - 1, 1)) * standardError }

    /// Paired per-draw differences; `anchor[t][d]`, `route[t][d]` for template t and draw d
    /// (templates averaged per draw first). `draws` restricts to a subset of draw indices.
    public static func paired(anchor: [[Double]], route: [[Double]], draws subset: [Int]? = nil) -> PullEstimate {
        let indices = subset ?? Array(0..<(anchor.first?.count ?? 0))
        let draws = indices.count
        var diffs = [Double](repeating: 0, count: draws), anchorMean = 0.0
        for (a, r) in zip(anchor, route) {
            for (k, d) in indices.enumerated() {
                diffs[k] += (a[d] - r[d]) / Double(anchor.count)
                anchorMean += a[d] / Double(anchor.count * max(draws, 1))
            }
        }
        let m = Stats.mean(diffs)
        return PullEstimate(mean: m, standardError: Stats.standardError(diffs), relative: m / max(anchorMean, 1e-12), draws: draws)
    }
}
