//
//  Protect.swift
//  VeilKit
//
//  Block, without find: fit one guard for one person or a group, for a deployer who has already
//  decided these people are not to be generated. The assessment is the expensive stage and it
//  answers a different question — *can* this model reach them — so this path skips it.
//
//    prepare   photos → latents per person, keyed fit / held-out split
//    fitGuard  one GuardPlan over everyone: their names and descriptions erased, the rest kept
//    export    a standard LoRA .safetensors
//    check     the exported file read back: the quick check, or the full verifier per person
//
//  What this never claims: that the model could reach these people, or that routes nobody named
//  are closed. Nothing measured either. Every report from here says so, and the verdict is
//  `unverified` / `quick-checked` / `quick-partial` unless the full verifier ran.
//

import Foundation
import MLX

/// What runs after the guard is written.
public enum ProtectCheck: String, Codable, Sendable {
    /// Write the file and stop.
    case none
    /// Each person's own name, before and after, on held-out photos, plus everyday drift.
    case quick
    /// The whole verifier, per person: null calibration, suppression, attacks, preservation.
    case full
}

public struct ProtectRequest: Sendable {
    /// Everyone this guard protects; each needs photos and at least one name or description.
    public var people: [ProtectedSubject]
    public var controls: ControlSet
    public var profile: VeilProfile
    public var faceWeighting: FaceWeighting
    public var method: GuardMethod
    /// Export format (nil = the executor's default).
    public var format: String?
    public var check: ProtectCheck
    /// Replace names with hashes in the shareable report.
    public var redactNames: Bool
    /// Extra prompts the guard must leave alone.
    public var preservePrompts: [String]
    /// Name templates per person (nil = scaled to the size of the group).
    public var templatesPerPerson: Int?
    /// The deployment's own adapter, for the report.
    public var withAdapter: String?

    public init(people: [ProtectedSubject], controls: ControlSet = .empty, profile: VeilProfile = .standard,
                faceWeighting: FaceWeighting = .auto, method: GuardMethod = .both, format: String? = nil,
                check: ProtectCheck = .quick, redactNames: Bool = false, preservePrompts: [String] = [],
                templatesPerPerson: Int? = nil, withAdapter: String? = nil) {
        self.people = people
        self.controls = controls
        self.profile = profile
        self.faceWeighting = faceWeighting
        self.method = method
        self.format = format
        self.check = check
        self.redactNames = redactNames
        self.preservePrompts = preservePrompts
        self.templatesPerPerson = templatesPerPerson
        self.withAdapter = withAdapter
    }
}

public enum ProtectError: Error, LocalizedError {
    case noPeople
    case noPhotos(String)
    case nothingToErase(String)
    case consentCoversOnePersonOnly

    public var errorDescription: String? {
        switch self {
        case .noPeople:
            return "No one to protect: give photos of at least one person."
        case .noPhotos(let who):
            return "No photos for \(who). A guard is fitted against the person's own photos, so every person needs at least one."
        case .nothingToErase(let who):
            return "\(who) has no name and no description. A guard closes ways of asking for someone, so give at least one."
        case .consentCoversOnePersonOnly:
            return """
                --consent self attests that you are the person in the photos, which cannot cover a group. \
                Use --consent representative when you are authorized to act for everyone listed.
                """
        }
    }
}

public final class ProtectRun: @unchecked Sendable {
    /// One protected person, prepared.
    public struct Member: @unchecked Sendable {
        public let subject: ProtectedSubject
        /// Keyed subject id: what `veil forget` takes.
        public let id: String
        public let split: PhotoSplit
        public let fit: [EncodedPhoto]
        public let heldOut: [EncodedPhoto]

        /// The name the person is reported under.
        public var label: String { subject.names.first ?? subject.descriptions.first ?? "this person" }
    }

    public let model: GuardableModel
    public let request: ProtectRequest
    public let schedule: SeedSchedule
    public let store: SubjectStore
    public let runID = UUID().uuidString
    public var onProgress: ((VeilProgress) -> Void)?
    private let started = Date()

    public private(set) var members: [Member] = []
    public private(set) var controlPhotos: [EncodedPhoto] = []
    public private(set) var controlIdentities: [String?] = []
    public private(set) var fallbackControls = false
    public private(set) var plan: GuardPlan?
    public private(set) var guardFit: GuardFit?
    public private(set) var guardFile: GuardFile?
    public private(set) var quickCheck: QuickCheck?
    public private(set) var verifications: [PersonVerification] = []
    public private(set) var issues: [String] = []
    private var embeddings: [String: PromptEmbedding] = [:]
    private var evaluationCount = 0

    public init(model: GuardableModel, request: ProtectRequest, schedule: SeedSchedule, store: SubjectStore = SubjectStore()) {
        self.model = model
        self.request = request
        self.schedule = schedule
        self.store = store
        issues = model.issues + model.blockingIssues
    }

    private func say(_ phase: VeilPhase, _ message: String) { onProgress?(VeilProgress(phase: phase, message: message)) }

    private var profile: VeilProfile { request.profile }

    /// Prompt embeddings, cached: the fitter asks for the same anchors many times over.
    private func embed(_ prompts: [String]) throws -> [PromptEmbedding] {
        let missing = Array(Set(prompts.filter { embeddings[$0] == nil })).sorted()
        if !missing.isEmpty {
            for (p, e) in zip(missing, try model.embed(missing)) { embeddings[p] = e }
        }
        return prompts.map { embeddings[$0]! }
    }

    // MARK: - Prepare

    public func prepare() throws {
        let people = request.people
        guard !people.isEmpty else { throw ProtectError.noPeople }
        if people.count > 1, people.contains(where: { $0.consent.basis == .selfAttested }) {
            throw ProtectError.consentCoversOnePersonOnly
        }
        for person in people {
            let who = person.names.first ?? person.descriptions.first ?? "one of the people given"
            guard !person.photos.isEmpty else { throw ProtectError.noPhotos(who) }
            guard !person.names.isEmpty || !person.descriptions.isEmpty else { throw ProtectError.nothingToErase(who) }
        }

        for person in people {
            try Task.checkCancellation()
            let id = person.id(schedule: schedule)
            let who = person.names.first ?? person.descriptions.first ?? id
            say(.preparing, "encoding \(person.photos.count) photo(s) for \(who)")
            let split = PhotoSplit.make(photoIDs: person.photos.map(\.id), schedule: schedule, maxHeldOut: profile.maxHeldOut)
            let encoded = try PhotoEncoding.encode(person.photos, model: model, faceWeighting: request.faceWeighting,
                                                   store: store, subjectID: id)
            let byID = Dictionary(encoded.map { ($0.photoID, $0) }, uniquingKeysWith: { a, _ in a })
            let fit = split.fit.compactMap { byID[$0] }
            var heldOut = split.heldOut.compactMap { byID[$0] }
            if split.augmented {
                let variants = PhotoAugment.variants(of: person.photos[0], count: 3, schedule: schedule)
                heldOut = try PhotoEncoding.encode(variants, model: model, faceWeighting: request.faceWeighting)
                issues.append("\(who): only one photo, so the held-out set is augmentations of it (same pose, light and background). Give 5–20 photos.")
            }
            if request.faceWeighting == .auto, !encoded.contains(where: \.weight.faceFound) {
                issues.append("\(who): no face was found, so the loss is weighted uniformly — backgrounds and clothing count as much as the likeness.")
            }
            members.append(Member(subject: person, id: id, split: split, fit: fit.isEmpty ? heldOut : fit, heldOut: heldOut))
        }

        let limited = request.controls.limited(profile.maxControls, schedule: schedule)
        controlIdentities = limited.identities
        if limited.photos.isEmpty {
            say(.preparing, "no controls given: sampling generic people from the model (decode-free)")
            let shape = members.first?.fit.first?.shape ?? []
            controlPhotos = try LatentSampler.genericPeople(model, shape: shape, count: min(4, profile.maxControls),
                                                            schedule: schedule)
            evaluationCount += controlPhotos.count * LatentSampler.genericPeopleCost(model)
            controlIdentities = Array(repeating: nil, count: controlPhotos.count)
            fallbackControls = true
            issues.append("No control photos: the model's own samples of generic people stand in for everyone else. Preservation is weaker without real controls (look-alikes especially).")
        } else {
            say(.preparing, "encoding \(limited.photos.count) control photo(s)")
            controlPhotos = try PhotoEncoding.encode(limited.photos, model: model, faceWeighting: request.faceWeighting)
        }
        issues.append("Fitted without assessing the model: nothing here says \(members.count == 1 ? "this person is" : "these people are") reachable on it, or that routes no one named are closed. `veil verify` measures that.")
    }

    // MARK: - Guard

    @discardableResult
    public func fitGuard() throws -> GuardFit {
        guard !members.isEmpty else { throw VeilError.notPrepared }
        try Task.checkCancellation()
        let people = members.map {
            GuardPlan.Person(id: $0.id, names: $0.subject.names, descriptions: $0.subject.descriptions,
                             anchor: $0.subject.anchor, fit: $0.fit)
        }
        // Nothing measured which descriptions reach, and the user gave them as ways of reaching
        // the person, so all of them are erased.
        var extra: [Int: [Route]] = [:]
        for (i, member) in members.enumerated() where !member.subject.descriptions.isEmpty {
            extra[i] = member.subject.descriptions.map {
                Templates.textRoute($0, kind: .description, anchor: member.subject.anchor)
            }
        }
        let plan = GuardPlan.make(people: people, extraRoutes: extra, profile: profile, schedule: schedule,
                                  controlIdentities: Array(Set(controlIdentities.compactMap { $0 })),
                                  extraPreserve: request.preservePrompts, templatesOverride: request.templatesPerPerson)
        self.plan = plan
        let fit = try GuardFitter.fit(model: model, plan: plan, preservePhotos: controlPhotos, method: request.method,
                                      profile: profile, schedule: schedule, embed: embed) { self.say($0, $1) }
        guardFit = fit
        return fit
    }

    // MARK: - Export

    /// Keyed id for the group as a whole (the members keep their own ids).
    public var groupID: String {
        members.count == 1 ? (members.first?.id ?? "") : schedule.keyedID("group|" + members.map(\.id).sorted().joined(separator: ","))
    }

    public var adapterMetadata: [String: String] {
        let d = model.descriptor
        let consent = request.people.first?.consent
        let iso = ISO8601DateFormatter().string(from: consent?.attestedAt ?? Date())
        let who = members.count == 1 ? "one protected person" : "\(members.count) protected people"
        return [
            "veil.schema": VeilInfo.adapterSchema,
            "veil.version": VeilInfo.version,
            "veil.run_id": runID,
            "veil.base_model": d.pinnedName,
            "veil.family": d.family,
            "veil.variant": d.variant ?? "unknown",
            "veil.subject": groupID,
            "veil.subject_ids": members.map(\.id).joined(separator: ","),
            "veil.people": String(members.count),
            "veil.consent": "\(consent?.basis.rawValue ?? "unknown") \(iso)",
            "veil.method": (guardFit?.method ?? request.method).rawValue,
            "veil.scale": "1.0",
            "veil.assessed": "no",
            "veil.intended_use": "Guard: suppresses the likeness of \(who) on \(d.pinnedName). Apply at scale 1.0. Fitted without assessing the model — check it with `veil verify` before relying on it.",
        ]
    }

    @discardableResult
    public func export(to url: URL) throws -> GuardFile {
        guard let guardFit else { throw VeilError.noGuard }
        say(.exporting, "writing \(url.lastPathComponent)")
        let file = try AdapterWriter.write(guardFit.deltas, model: model, format: request.format,
                                           metadata: adapterMetadata, to: url)
        guardFile = file
        return file
    }

    // MARK: - Check

    /// Read the guard file back, install exactly what it holds, and measure it.
    public func check(guardAt url: URL) throws {
        guard request.check != .none else { return }
        try Task.checkCancellation()
        say(.verifying, "reading back \(url.lastPathComponent)")
        let (deltas, file) = try AdapterWriter.readBack(url, model: model)
        model.installGuard(deltas)
        if guardFile == nil {
            let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
            guardFile = GuardFile(path: url.path, sha256: file.sha256, format: file.metadata["veil.format"] ?? "unknown",
                                  slots: deltas.keys.sorted(), rank: deltas.values.map(\.rank).max() ?? 0,
                                  parameters: deltas.values.reduce(0) { $0 + $1.rank * ($1.inFeatures + $1.outFeatures) },
                                  bytes: size)
        }
        if let base = file.metadata["veil.base_model"], base != model.descriptor.pinnedName {
            issues.append("The guard was fitted on \(base); this model is \(model.descriptor.pinnedName).")
        }
        switch request.check {
        case .none:
            break
        case .quick:
            let people = members.compactMap { m -> QuickChecker.Person? in
                guard let name = m.subject.names.first, !m.heldOut.isEmpty else { return nil }
                return QuickChecker.Person(subjectID: m.id, name: name, anchor: m.subject.anchor, heldOut: m.heldOut)
            }
            quickCheck = try QuickChecker.check(model: model, people: people, preservePhotos: controlPhotos,
                                                guardSHA256: file.sha256, profile: profile, schedule: schedule,
                                                redactNames: request.redactNames) { self.say(.verifying, $0) }
        case .full:
            for member in members {
                try Task.checkCancellation()
                verifications.append(try verify(member, guardSHA256: file.sha256))
            }
        }
    }

    /// The full verifier for one member. The other members are erased too, so they are never named
    /// controls here — only the user's controls are.
    private func verify(_ member: Member, guardSHA256: String) throws -> PersonVerification {
        let subject = member.subject
        say(.verifying, "verifying \(member.label)")
        let routes = subject.names.map { Templates.nameRoute($0, count: profile.templatesPerName, anchor: subject.anchor) }
            + subject.descriptions.map { Templates.textRoute($0, kind: .description, anchor: subject.anchor) }
        let nulls = NullNames.make(profile.nullCount, schedule: schedule,
                                   avoiding: request.people.flatMap(\.names) + controlIdentities.compactMap { $0 })
            .map { Templates.nameRoute($0, count: profile.templatesPerName, anchor: subject.anchor, kind: .null) }
        // Every description given was erased, so none is held to "must keep working".
        let inputs = Verifier.Inputs(routes: routes, reached: [], nulls: nulls, fit: member.fit, heldOut: member.heldOut,
                                     controls: controlPhotos, controlIdentities: controlIdentities, descriptions: [],
                                     anchor: subject.anchor)
        let (verification, bench) = try Verifier.verify(model: model, inputs: inputs, guardSHA256: guardSHA256,
                                                        profile: profile, schedule: schedule) {
            self.say(.verifying, "\(member.label): \($0)")
        }
        evaluationCount += bench.evaluations
        return PersonVerification(subjectID: member.id, name: request.redactNames ? nil : subject.names.first,
                                  nameHash: String(Hashing.sha256Hex("veil.name|" + member.label).prefix(12)),
                                  verification: verification)
    }

    // MARK: - Report

    public var evaluations: Int { evaluationCount }

    /// The verdict and why. Never `guarded` unless the full verifier ran.
    public var outcome: (verdict: String, reasons: [String]) {
        if !verifications.isEmpty {
            let verdicts = verifications.map(\.verification.verdict)
            let verdict: GuardVerdict = verdicts.contains(.inconclusive) ? .inconclusive
                : verdicts.contains(.partial) ? .partial
                : verdicts.allSatisfy { $0 == .notNeeded } ? .notNeeded : .guarded
            var reasons = verifications.map { v in
                "\(v.name.map { "“\($0)”" } ?? "#\(v.nameHash)"): \(v.verification.verdict.rawValue)"
            }
            reasons += verifications.flatMap(\.verification.reasons)
            return (verdict.rawValue, reasons)
        }
        if let quick = quickCheck {
            var reasons = quick.people.map { p in
                p.baseKnew
                    ? String(format: "%@: own-name pull %.3f → %.3f (%.0f%% left)", p.name.map { "“\($0)”" } ?? "#\(p.nameHash)",
                             p.basePull.mean, p.guardedPull.mean, 100 * max(p.residual, 0))
                    : "\(p.name.map { "“\($0)”" } ?? "#\(p.nameHash)"): the base model's pull on their photos was already within noise, so there was little to remove"
            }
            reasons.append(String(format: "largest drift on everyday prompts %.2f%% (tolerance %.0f%%)",
                                  100 * quick.maxDrift, 100 * quick.tolerance))
            reasons.append(quick.caveat)
            return (quick.pass ? "quick-checked" : "quick-partial", reasons)
        }
        return ("unverified", ["the guard was fitted to the names and photos given; nothing has measured it — run `veil verify` to find out what it does"])
    }

    public func report() -> VeilReport {
        func record(_ p: EncodedPhoto, role: String, identity: String? = nil) -> PhotoRecord {
            PhotoRecord(sha256: p.photoID, role: role, identity: identity, faceFound: p.weight.faceFound,
                        likeness: p.weight.method, faceArea: p.weight.faceArea)
        }
        let subjects = members.map { member -> SubjectRecord in
            let s = member.subject
            var photos = member.fit.map { record($0, role: "fit") }
            photos += member.heldOut.map { record($0, role: member.split.augmented ? "held-out (augmented)" : "held-out") }
            return SubjectRecord(id: member.id, names: request.redactNames ? nil : s.names,
                                 nameHashes: s.names.map { String(Hashing.sha256Hex("veil.name|" + $0).prefix(12)) },
                                 descriptions: s.descriptions.count, anchor: s.anchor, consent: s.consent,
                                 split: member.split, photos: photos)
        }
        let controls = ControlsRecord(count: controlPhotos.count,
                                      identities: Array(Set(controlIdentities.compactMap { $0 })).sorted(),
                                      fallbackSamples: fallbackControls,
                                      photos: zip(controlPhotos, controlIdentities).map { record($0, role: "control", identity: $1) })
        let guardRecord = guardFit.map {
            GuardRecord(method: $0.method, erased: request.redactNames ? [] : $0.erased, closedForm: $0.closedForm,
                        training: $0.training, file: guardFile?.shareable)
        } ?? guardFile.map { GuardRecord(method: request.method, erased: [], closedForm: nil, training: nil, file: $0.shareable) }
        let checks = quickCheck != nil || !verifications.isEmpty
            ? ProtectChecks(quick: quickCheck, verifications: verifications.isEmpty ? nil : verifications) : nil
        let (verdict, reasons) = outcome
        return VeilReport(schema: VeilInfo.reportSchema, version: VeilInfo.version, runID: runID, createdAt: Date(),
                          environment: .current(), mode: "protect", model: model.descriptor.shareable,
                          withAdapter: request.withAdapter.map { $0.contains("://") ? $0 : ($0 as NSString).lastPathComponent },
                          subjects: subjects, controls: controls, profile: profile, schedule: schedule.info,
                          assessment: nil, guardFit: guardRecord, verification: nil, checks: checks, verdict: verdict,
                          reasons: reasons, issues: issues, limits: VeilReport.limits, evaluations: evaluations,
                          seconds: Date().timeIntervalSince(started))
    }
}
