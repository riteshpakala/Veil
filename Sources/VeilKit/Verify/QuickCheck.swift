//
//  QuickCheck.swift
//  VeilKit
//
//  The cheap check that follows a guard fitted without an assessment: on the exported file read
//  back, on each person's held-out photos, does their own name still pull the model toward them,
//  and did everyday prompts move?
//
//  What it is not: a verdict. There is no invented-name null here, so "residual 8%" means the
//  name's pull dropped to 8% of what it was — not that the person is out of reach. `veil verify`
//  is the measurement that calibrates, attacks and judges. The caveat travels with the result.
//

import Foundation
import MLX

public struct QuickCheckPerson: Codable, Sendable, Hashable {
    public let subjectID: String
    /// Nil when names are redacted.
    public let name: String?
    public let nameHash: String
    public let basePull: PullEstimate
    public let guardedPull: PullEstimate
    /// Guarded ÷ base pull: 0 = gone, 1 = untouched.
    public let residual: Double
    /// The base model's own pull on their held-out photos had a 95% lower bound above zero. Not
    /// "reaches" — nothing was calibrated against invented names.
    public let baseKnew: Bool
    public let pass: Bool
}

public struct QuickCheck: Codable, Sendable {
    public let guardSHA256: String
    public let people: [QuickCheckPerson]
    /// Everyday prompts the guard must leave alone.
    public let drift: [DriftMeasurement]
    public let maxDrift: Double
    public let tolerance: Double
    public let residualTolerance: Double
    public let pass: Bool
    public let caveat: String
}

public enum QuickChecker {
    public struct Person: Sendable {
        public let subjectID: String
        public let name: String
        public let anchor: String
        public let heldOut: [EncodedPhoto]

        public init(subjectID: String, name: String, anchor: String, heldOut: [EncodedPhoto]) {
            self.subjectID = subjectID
            self.name = name
            self.anchor = anchor
            self.heldOut = heldOut
        }
    }

    public static let caveat = """
        Quick check only: the residual is this name's pull with the guard on, over its pull with the guard off, on \
        held-out photos. It is not calibrated against invented names and no attack was run — `veil verify` gives a verdict.
        """

    /// `model` must already have the guard (as read back from the file) installed.
    public static func check(model: GuardableModel, people: [Person], preservePhotos: [EncodedPhoto], guardSHA256: String,
                             profile: VeilProfile, schedule: SeedSchedule, redactNames: Bool = false,
                             progress: ((String) -> Void)? = nil) throws -> QuickCheck {
        var results: [QuickCheckPerson] = []
        for (i, person) in people.enumerated() where !person.heldOut.isEmpty {
            try Task.checkCancellation()
            progress?("checking \(i + 1)/\(people.count)")
            let sigmas = SigmaGrid.evenly(model.samplerSigmas(latentShape: person.heldOut[0].shape), count: profile.sigmaCount)
            let set = DrawSet(label: "check-\(person.subjectID)", photos: person.heldOut, sigmas: sigmas,
                              noisePerSigma: profile.noisePerSigma, stream: .verify, schedule: schedule)
            let bench = PullBench(model: model, subject: set, controls: nil)
            bench.setGuardTag("guard:" + String(guardSHA256.prefix(12)))
            let route = Templates.nameRoute(person.name, count: min(3, profile.templatesPerName), anchor: person.anchor)
            let base = try model.hooks.with(guard: false) { try bench.pull(route, on: set) }
            let guarded = try model.hooks.with(guard: true) { try bench.pull(route, on: set) }
            let residual = guarded.mean / (abs(base.mean) > 1e-12 ? base.mean : 1e-12)
            // A name the base model never pulled toward can't fail: there was nothing to remove.
            let knew = base.lowerBound > 0
            results.append(QuickCheckPerson(subjectID: person.subjectID, name: redactNames ? nil : person.name,
                                            nameHash: String(Hashing.sha256Hex("veil.name|" + person.name).prefix(12)),
                                            basePull: base, guardedPull: guarded, residual: residual, baseKnew: knew,
                                            pass: !knew || residual <= profile.residualTolerance))
        }

        // Drift: the controls when there are any, else the protected people's own photos.
        var drift: [DriftMeasurement] = []
        let others = preservePhotos.isEmpty ? people.flatMap(\.heldOut) : preservePhotos
        if let shape = others.first?.shape {
            let sigmas = SigmaGrid.evenly(model.samplerSigmas(latentShape: shape), count: profile.sigmaCount)
            let set = DrawSet(label: "check-preserve", photos: others, sigmas: sigmas, noisePerSigma: 1, stream: .verify,
                              schedule: schedule)
            for prompt in Array(Templates.people.prefix(2)) + Array(Templates.generic.prefix(3)) {
                try Task.checkCancellation()
                progress?("drift: \(prompt)")
                let c = try model.embed([prompt])[0].value
                drift.append(Drift.measure(model: model, set: set, prompt: prompt, conditioning: c))
            }
        }
        let maxDrift = drift.map(\.velocityDrift).max() ?? 0
        return QuickCheck(guardSHA256: guardSHA256, people: results, drift: drift, maxDrift: maxDrift,
                          tolerance: profile.preservationTolerance, residualTolerance: profile.residualTolerance,
                          pass: results.allSatisfy(\.pass) && maxDrift <= profile.preservationTolerance, caveat: caveat)
    }
}
