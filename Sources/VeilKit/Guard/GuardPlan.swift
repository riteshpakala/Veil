//
//  GuardPlan.swift
//  VeilKit
//
//  What a guard erases, what it must leave alone, and the fitting of both stages. One
//  construction serves both callers:
//
//    - an assessed run for one person (`VeilRun`): erase their names, plus whatever else reached;
//    - a group fitted without assessment (`ProtectRun`): erase every name and description given.
//
//  Erase pairs carry the person they belong to, so stage B trains her name against *her* photos.
//  Pairing a name with someone else's latents would spend the guard's capacity where that name
//  never had any pull.
//

import Foundation
import MLX

public struct GuardPlan: @unchecked Sendable {
    /// One protected person: who to erase, and the photos their erase loss is anchored on.
    public struct Person: @unchecked Sendable {
        /// Keyed subject id — reported, and never reveals which photos it covers.
        public let id: String
        public let names: [String]
        public let descriptions: [String]
        /// The neutral phrase this person's names are replaced with.
        public let anchor: String
        public let fit: [EncodedPhoto]

        public init(id: String, names: [String], descriptions: [String] = [], anchor: String = "a person",
                    fit: [EncodedPhoto]) {
            self.id = id
            self.names = names
            self.descriptions = descriptions
            self.anchor = anchor
            self.fit = fit
        }
    }

    /// A prompt, the anchor prompt it should behave like, and whose photos it trains against.
    public struct ErasePair: Sendable {
        public let prompt: String
        public let anchor: String
        public let person: Int
    }

    public let people: [Person]
    public let erase: [ErasePair]
    public let preserve: [String]
    /// Name templates per person (scaled down for groups, so cost and memory don't grow with them).
    public let templatesPerPerson: Int

    /// The distinct prompts erased — what the report records.
    public var erasedPrompts: [String] { Array(Set(erase.map(\.prompt))).sorted() }

    /// Templates per person: the profile's count for one or two people, fewer for a group, so the
    /// erase set stays around 48 prompts however many people share the guard.
    public static func templates(for people: Int, profile: VeilProfile, override: Int? = nil) -> Int {
        if let override { return max(1, override) }
        return max(4, min(profile.closedFormTemplates, 48 / max(people, 1)))
    }

    /// - Parameters:
    ///   - extraRoutes: routes beyond the names to erase, per person index (descriptions that
    ///     reached, discovered prompts, or — without an assessment — every description given).
    ///   - controlIdentities: named controls whose own names must keep working.
    ///   - extraPreserve: the user's `--preserve` prompts.
    public static func make(people: [Person], extraRoutes: [Int: [Route]] = [:], profile: VeilProfile,
                            schedule: SeedSchedule, controlIdentities: [String] = [], extraPreserve: [String] = [],
                            templatesOverride: Int? = nil) -> GuardPlan {
        let count = templates(for: people.count, profile: profile, override: templatesOverride)
        var erase: [ErasePair] = []
        for (i, person) in people.enumerated() {
            for name in person.names {
                let r = Templates.nameRoute(name, count: count, anchor: person.anchor)
                erase += zip(r.prompts, r.anchors).map { ErasePair(prompt: $0, anchor: $1, person: i) }
            }
            for route in extraRoutes[i] ?? [] {
                erase += zip(route.prompts, route.anchors).map { ErasePair(prompt: $0, anchor: $1, person: i) }
            }
        }
        let eraseSet = Set(erase.map(\.prompt))

        // Everyday prompts, people, the anchors themselves, invented names, and the controls' own
        // names: what the edit is held to leave where it is.
        let primaryAnchor = people.first?.anchor ?? "a person"
        var preserve = Templates.generic + Templates.people
        for anchor in Set(people.map(\.anchor)).sorted() {
            preserve += Templates.name.prefix(count).map { Templates.fill($0, anchor) }
        }
        let invented = NullNames.make(6, schedule: schedule, avoiding: people.flatMap(\.names) + controlIdentities)
        for name in invented {
            preserve += Templates.nameRoute(name, count: 2, anchor: primaryAnchor).prompts
        }
        for name in Set(controlIdentities) {
            preserve += Templates.nameRoute(name, count: 3, anchor: primaryAnchor).prompts
        }
        preserve += extraPreserve
        preserve = Array(Set(preserve.filter { !eraseSet.contains($0) })).sorted()
        return GuardPlan(people: people, erase: erase, preserve: preserve, templatesPerPerson: count)
    }
}

/// Stage A then stage B, for whatever the plan covers.
public enum GuardFitter {
    /// - Parameters:
    ///   - preservePhotos: the latents preservation is measured on (controls, or the model's own
    ///     samples of generic people).
    ///   - embed: the caller's prompt-embedding cache.
    /// Installs the fitted guard on the model before returning.
    public static func fit(model: GuardableModel, plan: GuardPlan, preservePhotos: [EncodedPhoto],
                           method: GuardMethod, profile: VeilProfile, schedule: SeedSchedule,
                           embed: ([String]) throws -> [PromptEmbedding],
                           progress: ((VeilPhase, String) -> Void)? = nil) throws -> GuardFit {
        guard !plan.erase.isEmpty else { throw VeilError.nothingToGuard }
        let eraseEmbeddings = try zip(embed(plan.erase.map(\.prompt)), embed(plan.erase.map(\.anchor))).map { ($0, $1) }
        let preserveEmbeddings = try embed(plan.preserve)

        // Stage A: the closed-form edit of the slot that reads the text encoder's output.
        var initial: [String: LowRankDelta] = [:]
        var summary: ClosedFormSummary?
        let textInput = model.slots.first { $0.role == .textInput }
        if method != .trained, let slot = textInput, let linear = model.linear(slot.key) {
            progress?(.closedForm, "closed-form edit of \(slot.key) (\(eraseEmbeddings.count) prompts, \(preserveEmbeddings.count) preserved)")
            let edit = ClosedFormEditor.fit(key: slot.key, weight: linear.deployedWeight(), erase: eraseEmbeddings,
                                            preserve: preserveEmbeddings, preserveWeight: profile.closedFormPreserve,
                                            ridge: profile.closedFormRidge, maxRank: profile.closedFormMaxRank,
                                            energy: profile.closedFormEnergy, schedule: schedule)
            initial[slot.key] = edit.delta
            summary = ClosedFormSummary(slot: slot.key, rank: edit.delta.rank, energy: edit.energy,
                                        truncationError: edit.truncationError, erasePairs: edit.erasePairs,
                                        preserveVectors: edit.preserveVectors, relativeSize: edit.relativeSize)
        }

        // Stage B: the trained text-path LoRA, anchored on each person's own photos.
        var deltas = initial
        var training: TrainingReport?
        if method != .closedForm, profile.trainSteps > 0 {
            progress?(.training, "training the text-path guard (\(profile.trainSteps) steps, \(plan.people.count) person(s))")
            let shape = plan.people.first(where: { !$0.fit.isEmpty })?.fit.first?.shape ?? []
            let trainSigmas = SigmaGrid.evenly(model.samplerSigmas(latentShape: shape), count: profile.trainSigmaCount)
            let inputs = GuardTrainingInputs(slots: model.slots.map(\.key), initial: initial, fit: plan.people.map(\.fit),
                                             erase: zip(eraseEmbeddings, plan.erase).map { (prompt: $0.0, anchor: $0.1, person: $1.person) },
                                             preservePhotos: preservePhotos, preserve: preserveEmbeddings, sigmas: trainSigmas)
            let (trained, report) = GuardTrainer.train(model: model, inputs: inputs, profile: profile, schedule: schedule) {
                progress?(.training, $0)
            }
            try Task.checkCancellation()
            deltas = trained
            training = report
        }
        let fit = GuardFit(deltas: deltas, method: method, erased: plan.erasedPrompts, closedForm: summary, training: training)
        model.installGuard(deltas)
        return fit
    }
}
