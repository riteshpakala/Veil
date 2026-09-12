//
//  GuardTrainer.swift
//  VeilKit
//
//  Guard, stage B: a LoRA on the text path, trained against the person's own photos so that
//  descriptions and paraphrases lose their pull too — not just the literal name (Concept
//  Ablation / MACE-style anchoring, with explicit preservation):
//
//      L = mean_erase  w·‖v̂_G(x_σ, σ, c_r) − sg v̂₀(x_σ, σ, anchor(c_r))‖²     x₀ ∈ the person's fit photos
//        + λ · mean_pres ‖v̂_G(x'_σ, σ, c_j) − sg v̂₀(x'_σ, σ, c_j)‖²           x₀' ∈ controls / base samples
//
//  v̂₀ is the same model with the guard switched off — exactly the deployment — computed once
//  over a keyed pool. Image-path weights are never touched: the slots are the text path only.
//  Hardening (optional, experimental): every M steps a K-step soft attack against the current
//  guard joins the erase pool, so the guard learns to hold against it (AdvUnlearn-lite). Off by
//  default — on the toy world it traded away a description route's suppression.
//

import Foundation
import MLX

public struct TrainingReport: Codable, Sendable, Hashable {
    public let steps: Int
    public let slots: [String]
    public let rank: Int
    /// Protected people the guard covers; erase samples are spread evenly over them.
    public let people: Int
    /// Erase / preserve loss at the start and at the end (pool means).
    public let eraseLossStart: Double
    public let eraseLossEnd: Double
    public let preserveLossStart: Double
    public let preserveLossEnd: Double
    /// Loss every 10% of training: (step, erase, preserve).
    public let history: [[Double]]
    public let hardenings: Int
    public let seconds: Double
}

public struct GuardTrainingInputs {
    public let slots: [String]
    /// Initial deltas per slot (stage A's edit for the text-input slot).
    public let initial: [String: LowRankDelta]
    /// Fit photos per protected person: an erase prompt trains against its own person's photos.
    public let fit: [[EncodedPhoto]]
    public let erase: [(prompt: PromptEmbedding, anchor: PromptEmbedding, person: Int)]
    public let preservePhotos: [EncodedPhoto]
    public let preserve: [PromptEmbedding]
    public let sigmas: [Float]

    public init(slots: [String], initial: [String: LowRankDelta], fit: [[EncodedPhoto]],
                erase: [(prompt: PromptEmbedding, anchor: PromptEmbedding, person: Int)],
                preservePhotos: [EncodedPhoto], preserve: [PromptEmbedding], sigmas: [Float]) {
        self.slots = slots
        self.initial = initial
        self.fit = fit
        self.erase = erase
        self.preservePhotos = preservePhotos
        self.preserve = preserve
        self.sigmas = sigmas
    }

    /// One protected person.
    public init(slots: [String], initial: [String: LowRankDelta], fit: [EncodedPhoto],
                erase: [(prompt: PromptEmbedding, anchor: PromptEmbedding)], preservePhotos: [EncodedPhoto],
                preserve: [PromptEmbedding], sigmas: [Float]) {
        self.init(slots: slots, initial: initial, fit: [fit], erase: erase.map { ($0.prompt, $0.anchor, 0) },
                  preservePhotos: preservePhotos, preserve: preserve, sigmas: sigmas)
    }

    public var people: Int { max(fit.count, 1) }
    var allFit: [EncodedPhoto] { fit.flatMap { $0 } }
}

public enum GuardTrainer {
    struct Sample {
        let x: MLXArray       // (C, H, W)
        let w: MLXArray       // (1, H, W)
        var target: MLXArray  // (C, H, W)
        let conditioning: Int
    }

    /// Samples of one (σ, latent shape): one batch dimension.
    struct Pool {
        var groups: [String: (sigma: Float, samples: [Sample])] = [:]
        var keys: [String] { groups.keys.sorted() }
    }

    public static func train(model: GuardableModel, inputs: GuardTrainingInputs, profile: VeilProfile, schedule: SeedSchedule,
                             progress: ((String) -> Void)? = nil) -> (deltas: [String: LowRankDelta], report: TrainingReport) {
        let started = Date()
        // Parameters: stage A's factors where given, else LoRA init (down ~ N(0, 1/in), up = 0).
        var deltas: [(key: String, delta: LowRankDelta)] = []
        for key in inputs.slots {
            if let d = inputs.initial[key] { deltas.append((key, d)); continue }
            guard let linear = model.linear(key) else { continue }
            let (out, inF) = linear.logicalShape
            let down = schedule.normal(.train, "init|\(key)", shape: [profile.trainRank, inF]) / Float(inF).squareRoot()
            deltas.append((key, LowRankDelta(up: MLXArray.zeros([out, profile.trainRank]), down: down)))
        }
        let trainable = TrainableGuard(deltas: deltas)
        let rank = deltas.map(\.delta.rank).max() ?? 0
        guard profile.trainSteps > 0, !inputs.erase.isEmpty, !inputs.allFit.isEmpty else {
            return (trainable.frozen(), TrainingReport(steps: 0, slots: inputs.slots, rank: rank, people: inputs.people,
                                                       eraseLossStart: 0, eraseLossEnd: 0, preserveLossStart: 0,
                                                       preserveLossEnd: 0, history: [], hardenings: 0, seconds: 0))
        }

        // Pools, with targets from the deployment (guard off). The erase pool grows with the group
        // so each person keeps a real share of it.
        progress?("training pool")
        var erase = makeErasePool(count: max(profile.erasePool, 32 * inputs.people), inputs: inputs, sigmas: inputs.sigmas,
                                  schedule: schedule)
        var preserve = makePool(count: inputs.preserve.isEmpty ? 0 : profile.preservePool, photos: inputs.preservePhotos,
                                conditionings: inputs.preserve.count, sigmas: inputs.sigmas, label: "preserve", schedule: schedule)
        let eraseConds = inputs.erase.map { $0.prompt.value }
        let anchorConds = inputs.erase.map { $0.anchor.value }
        let preserveConds = inputs.preserve.map(\.value)
        model.hooks.with(guard: false) {
            fillTargets(&erase, model: model) { anchorConds[$0] }
            fillTargets(&preserve, model: model) { preserveConds[$0] }
        }

        var adversarial: [Int: [MLXArray]] = [:]
        var rng = schedule.rng(.train, "minibatch")
        func batch(_ pool: Pool, key: String, size: Int, conds: (Int) -> MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray)? {
            guard let g = pool.groups[key], !g.samples.isEmpty else { return nil }
            let picks = (0..<min(size, g.samples.count)).map { _ in g.samples[Int(rng.next() % UInt64(g.samples.count))] }
            let c = picks.map { conds($0.conditioning) }
            let width = c.map { $0.dim(1) }.max() ?? 0
            guard c.allSatisfy({ $0.dim(1) == width }) else { return nil }
            return (stacked(picks.map(\.x), axis: 0), stacked(picks.map(\.w), axis: 0),
                    stacked(picks.map(\.target), axis: 0), concatenated(c, axis: 0))
        }
        func eraseConditioning(_ i: Int) -> MLXArray {
            if let adv = adversarial[i], !adv.isEmpty, rng.next() % 2 == 0 { return adv[Int(rng.next() % UInt64(adv.count))] }
            return eraseConds[i]
        }

        model.hooks.trainable = trainable
        defer { model.hooks.trainable = nil }
        var adam = AdamState(like: trainable.params, learningRate: profile.learningRate)
        var history: [[Double]] = []
        var hardenings = 0
        let (eStart, pStart) = model.hooks.with(guard: true) { poolLoss(erase, preserve, model: model, eraseConds: eraseConds, preserveConds: preserveConds) }
        let eraseKeys = erase.keys
        let preserveBySigma = Dictionary(grouping: preserve.keys) { preserve.groups[$0]!.sigma }

        for step in 0..<profile.trainSteps {
            if Task.isCancelled { break }
            let key = eraseKeys[step % eraseKeys.count]
            let sigma = erase.groups[key]!.sigma
            guard let (xe, we, te, ce) = batch(erase, key: key, size: profile.eraseBatch, conds: eraseConditioning) else { continue }
            let pKey = preserveBySigma[sigma]?[Int(rng.next() % UInt64(max(preserveBySigma[sigma]?.count ?? 1, 1)))]
            let p = pKey.flatMap { batch(preserve, key: $0, size: profile.preserveBatch) { preserveConds[$0] } }
            let lambda = profile.preserveWeight
            let lossAndGrad = valueAndGrad({ (params: [MLXArray]) -> [MLXArray] in
                trainable.params = params
                let le = Losses.weighted(model.velocity(xe, sigma: sigma, conditioning: ce), te, we)
                guard let (xp, wp, tp, cp) = p else { return [le, le, MLXArray(Float(0))] }
                let lp = Losses.weighted(model.velocity(xp, sigma: sigma, conditioning: cp), tp, wp)
                return [le + lambda * lp, le, lp]
            }, argumentNumbers: Array(0..<trainable.params.count))
            let current = trainable.params
            let (values, grads) = model.hooks.with(guard: true) { lossAndGrad(current) }
            let (clipped, _) = Losses.clip(grads, maxNorm: 1)
            trainable.params = adam.step(current, clipped)
            if step % max(1, profile.trainSteps / 10) == 0 || step == profile.trainSteps - 1 {
                history.append([Double(step), Double(values[1].item(Float.self)), Double(values[2].item(Float.self))])
                progress?("training step \(step + 1)/\(profile.trainSteps)")
            }

            // Hardening: attack the current guard and add what the attack finds to the erase pool.
            if profile.hardenEvery > 0, profile.hardenSteps > 0, (step + 1) % profile.hardenEvery == 0 {
                let i = Int(rng.next() % UInt64(eraseConds.count))
                let fitSet = DrawSet(label: "harden", photos: Array(inputs.allFit.prefix(2)), sigmas: [sigma], noisePerSigma: 1,
                                     stream: .attack, schedule: schedule)
                let snaps = model.hooks.with(guard: true) {
                    SoftPromptAttack.run(model: model, start: eraseConds[i], fit: fitSet, budgets: [profile.hardenSteps],
                                         learningRate: profile.attackLearningRate, radius: profile.attackRadius)
                }
                if let found = snaps.last?.conditioning {
                    adversarial[i, default: []].append(found.asType(eraseConds[i].dtype))
                    hardenings += 1
                }
            }
        }
        let (eEnd, pEnd) = model.hooks.with(guard: true) { poolLoss(erase, preserve, model: model, eraseConds: eraseConds, preserveConds: preserveConds) }
        return (trainable.frozen(), TrainingReport(steps: profile.trainSteps, slots: trainable.keys, rank: rank,
                                                   people: inputs.people, eraseLossStart: eStart, eraseLossEnd: eEnd,
                                                   preserveLossStart: pStart, preserveLossEnd: pEnd, history: history,
                                                   hardenings: hardenings, seconds: Date().timeIntervalSince(started)))
    }

    /// Erase samples, balanced over the people the guard covers: each sample pairs one of that
    /// person's photos with one of their own prompts. Pairing a name with someone else's latents
    /// would spend the guard where that name never had any pull. With one person this is exactly
    /// the old draw order.
    static func makeErasePool(count: Int, inputs: GuardTrainingInputs, sigmas: [Float], schedule: SeedSchedule) -> Pool {
        var pool = Pool()
        var byPerson = [[Int]](repeating: [], count: inputs.fit.count)
        for (i, e) in inputs.erase.enumerated() where e.person >= 0 && e.person < byPerson.count {
            byPerson[e.person].append(i)
        }
        let usable = byPerson.indices.filter { !byPerson[$0].isEmpty && !inputs.fit[$0].isEmpty }
        guard count > 0, !usable.isEmpty, !sigmas.isEmpty else { return pool }
        var rng = schedule.rng(.train, "pool|erase")
        for i in 0..<count {
            let person = usable[i % usable.count]
            let photos = inputs.fit[person], prompts = byPerson[person]
            let photo = photos[Int(rng.next() % UInt64(photos.count))]
            let sigma = sigmas[i % sigmas.count]
            let c = prompts[Int(rng.next() % UInt64(prompts.count))]
            let eps = schedule.normal(.train, "erase|\(i)|\(photo.photoID)", shape: photo.shape)
            let x = (1 - sigma) * photo.latent + sigma * eps
            let w = MLXArray(photo.weight.values, [1, photo.weight.rows, photo.weight.cols])
            let key = "\(SeedSchedule.levelKey(sigma))|\(photo.shape.map(String.init).joined(separator: "x"))"
            pool.groups[key, default: (sigma, [])].samples.append(Sample(x: x, w: w, target: x, conditioning: c))
        }
        return pool
    }

    static func makePool(count: Int, photos: [EncodedPhoto], conditionings: Int, sigmas: [Float], label: String,
                         schedule: SeedSchedule) -> Pool {
        var pool = Pool()
        guard count > 0, !photos.isEmpty, conditionings > 0, !sigmas.isEmpty else { return pool }
        var rng = schedule.rng(.train, "pool|\(label)")
        for i in 0..<count {
            let photo = photos[Int(rng.next() % UInt64(photos.count))]
            let sigma = sigmas[i % sigmas.count]
            let c = Int(rng.next() % UInt64(conditionings))
            let eps = schedule.normal(.train, "\(label)|\(i)|\(photo.photoID)", shape: photo.shape)
            let x = (1 - sigma) * photo.latent + sigma * eps
            let w = MLXArray(photo.weight.values, [1, photo.weight.rows, photo.weight.cols])
            let key = "\(SeedSchedule.levelKey(sigma))|\(photo.shape.map(String.init).joined(separator: "x"))"
            pool.groups[key, default: (sigma, [])].samples.append(Sample(x: x, w: w, target: x, conditioning: c))
        }
        return pool
    }

    /// Targets v̂(x_σ, σ, conditioning) for every sample (the caller sets the guard state).
    static func fillTargets(_ pool: inout Pool, model: VelocityModel, conditioning: (Int) -> MLXArray) {
        for key in pool.keys {
            var g = pool.groups[key]!
            let byConditioning = Dictionary(grouping: g.samples.indices) { g.samples[$0].conditioning }
            for (c, indices) in byConditioning {
                var start = 0
                while start < indices.count {
                    let chunk = Array(indices[start..<min(indices.count, start + max(1, model.maxBatch))])
                    let x = stacked(chunk.map { g.samples[$0].x }, axis: 0)
                    let v = model.velocity(x, sigma: g.sigma, conditioning: conditioning(c)).asType(.float32)
                    eval(v)
                    for (k, i) in chunk.enumerated() { g.samples[i].target = v[k] }
                    start += chunk.count
                }
            }
            pool.groups[key] = g
        }
    }

    /// Mean erase and preserve loss over the whole pools (current guard state).
    static func poolLoss(_ erase: Pool, _ preserve: Pool, model: VelocityModel, eraseConds: [MLXArray],
                         preserveConds: [MLXArray]) -> (Double, Double) {
        func mean(_ pool: Pool, _ conds: [MLXArray]) -> Double {
            var total = 0.0, n = 0
            for key in pool.keys {
                let g = pool.groups[key]!
                for s in g.samples.prefix(8) {
                    let v = model.velocity(s.x.expandedDimensions(axis: 0), sigma: g.sigma, conditioning: conds[s.conditioning])
                    total += Double(Losses.weighted(v, s.target.expandedDimensions(axis: 0), s.w.expandedDimensions(axis: 0)).item(Float.self))
                    n += 1
                }
            }
            return n == 0 ? 0 : total / Double(n)
        }
        return (mean(erase, eraseConds), mean(preserve, preserveConds))
    }
}
