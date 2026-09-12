//
//  BaseSamples.swift
//  VeilKit
//
//  "Everyone else", when the user gave no control photos: latents of generic people sampled from
//  the deployment itself. Weaker than real controls — the model's idea of a person, not a person —
//  and flagged wherever it stands in. Never decoded.
//

import Foundation
import MLX

extension LatentSampler {
    /// Latents of the first `count` generic-people prompts, sampled from the model with the guard
    /// off (so they are the deployment's own).
    public static func genericPeople(_ model: GuardableModel, shape: [Int], count: Int, steps: Int = 12,
                                     schedule: SeedSchedule, label: String = "sample") throws -> [EncodedPhoto] {
        guard count > 0, shape.count >= 3 else { return [] }
        let prompts = try model.embed(Array(Templates.people.prefix(count)))
        let guidance: Float = model.descriptor.variant == "base" ? 4 : 1
        let unconditional = guidance != 1 ? try model.embed([""])[0].value : nil
        return model.hooks.with(guard: false) {
            prompts.enumerated().map { i, p in
                let noise = schedule.normal(.control, "\(label)|\(i)", shape: shape)
                let latent = sample(model, conditioning: p.value, shape: shape, noise: noise, steps: steps,
                                    unconditional: unconditional, guidance: guidance)
                return EncodedPhoto(photoID: "sample-\(i)", name: p.text, latent: latent,
                                    weight: .uniform(rows: shape[1], cols: shape[2], method: "sample"), photo: nil)
            }
        }
    }

    /// Denoiser evaluations one `genericPeople` sample costs (CFG doubles them).
    public static func genericPeopleCost(_ model: GuardableModel, steps: Int = 12) -> Int {
        steps * (model.descriptor.variant == "base" ? 2 : 1)
    }
}
