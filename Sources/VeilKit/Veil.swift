//
//  Veil.swift
//  VeilKit
//
//  The process, top to bottom: find → block → verify.
//
//    prepare   photos → latents + likeness weights; keyed fit / held-out split; routes and nulls
//    assess    which routes pull the model toward the person, specifically (Reachability)
//              + optional discrete search and a locator diagnostic
//    fitGuard  stage A closed form on the text-input slot, stage B trained text-path LoRA
//    export    a standard LoRA .safetensors with Veil's metadata
//    verify    the file read back: suppression, attack cost, preservation → verdict
//    report    veil.guard.v1 JSON (+ a local figure)
//
//  Each stage is its own type; this class only sequences them and keeps their results.
//

import CoreGraphics
import Foundation
import MLX

public enum GuardMethod: String, Codable, Sendable, CaseIterable {
    /// Closed-form edit, then training initialized from it (default).
    case both
    case closedForm = "closed-form"
    case trained
}

public struct VeilRequest: Sendable {
    public var subject: ProtectedSubject
    public var controls: ControlSet
    public var profile: VeilProfile
    public var faceWeighting: FaceWeighting
    /// Run discrete prompt search for routes that don't name the person.
    public var search: Bool
    public var method: GuardMethod
    /// Export format (nil = the executor's default).
    public var format: String?
    /// Put discovered prompts' texts into the shareable report.
    public var revealRoutes: Bool
    /// Replace names with hashes in the shareable report.
    public var redactNames: Bool
    /// Fit a guard even when nothing reached the person.
    public var forceGuard: Bool
    /// Extra prompts the guard must leave alone.
    public var preservePrompts: [String]
    /// The deployment's own adapter, for the report.
    public var withAdapter: String?

    public init(subject: ProtectedSubject, controls: ControlSet = .empty, profile: VeilProfile = .standard,
                faceWeighting: FaceWeighting = .auto, search: Bool = false, method: GuardMethod = .both, format: String? = nil,
                revealRoutes: Bool = false, redactNames: Bool = false, forceGuard: Bool = false, preservePrompts: [String] = [],
                withAdapter: String? = nil) {
        self.subject = subject
        self.controls = controls
        self.profile = profile
        self.faceWeighting = faceWeighting
        self.search = search
        self.method = method
        self.format = format
        self.revealRoutes = revealRoutes
        self.redactNames = redactNames
        self.forceGuard = forceGuard
        self.preservePrompts = preservePrompts
        self.withAdapter = withAdapter
    }
}

public enum VeilPhase: String, Sendable {
    case preparing, assessing, searching, locating
    case closedForm = "closed-form"
    case training, exporting, verifying, reporting
}

public struct VeilProgress: Sendable {
    public let phase: VeilPhase
    public let message: String

    public init(phase: VeilPhase, message: String) {
        self.phase = phase
        self.message = message
    }
}

public enum VeilError: Error, LocalizedError {
    case noPhotos
    case notPrepared
    case nothingToGuard
    case noGuard

    public var errorDescription: String? {
        switch self {
        case .noPhotos: return "No photos of the person were given."
        case .notPrepared: return "Run prepare() first."
        case .nothingToGuard: return "No route reached the person, so there is nothing to block (use --force-guard to fit one anyway)."
        case .noGuard: return "No guard has been fitted or loaded."
        }
    }
}

public struct Assessment: Sendable {
    public let measurements: [RouteMeasurement]
    public let null: NullDistribution
    public let capability: CapabilityClass
    public let search: [SearchCandidate]
    public let locator: [LocatorSite]

    public var reached: [RouteMeasurement] { measurements.filter(\.reaches) }
}

public struct GuardFit: @unchecked Sendable {
    public let deltas: [String: LowRankDelta]
    public let method: GuardMethod
    public let erased: [String]
    public let closedForm: ClosedFormSummary?
    public let training: TrainingReport?
}

public final class VeilRun: @unchecked Sendable {
    public let model: GuardableModel
    public let request: VeilRequest
    public let schedule: SeedSchedule
    public let store: SubjectStore
    public let runID = UUID().uuidString
    public var onProgress: ((VeilProgress) -> Void)?
    private let started = Date()

    public private(set) var subjectID = ""
    public private(set) var split: PhotoSplit?
    public private(set) var fit: [EncodedPhoto] = []
    public private(set) var heldOut: [EncodedPhoto] = []
    public private(set) var controlPhotos: [EncodedPhoto] = []
    public private(set) var controlIdentities: [String?] = []
    public private(set) var fallbackControls = false
    public private(set) var routes: [Route] = []
    public private(set) var nulls: [Route] = []
    public private(set) var assessment: Assessment?
    public private(set) var guardFit: GuardFit?
    public private(set) var guardFile: GuardFile?
    public private(set) var verification: Verification?
    public private(set) var issues: [String] = []
    private var bench: PullBench?
    private var verifyBench: PullBench?
    private var extraEvaluations = 0

    public init(model: GuardableModel, request: VeilRequest, schedule: SeedSchedule, store: SubjectStore = SubjectStore()) {
        self.model = model
        self.request = request
        self.schedule = schedule
        self.store = store
        issues = model.issues + model.blockingIssues
    }

    private func say(_ phase: VeilPhase, _ message: String) { onProgress?(VeilProgress(phase: phase, message: message)) }

    private var profile: VeilProfile { request.profile }

    var sigmas: [Float] {
        SigmaGrid.evenly(model.samplerSigmas(latentShape: (fit.first ?? heldOut.first)?.shape ?? []), count: profile.sigmaCount)
    }

    // MARK: - Prepare

    public func prepare() throws {
        let subject = request.subject
        guard !subject.photos.isEmpty else { throw VeilError.noPhotos }
        say(.preparing, "encoding \(subject.photos.count) photo(s)")
        subjectID = subject.id(schedule: schedule)
        let split = PhotoSplit.make(photoIDs: subject.photos.map(\.id), schedule: schedule, maxHeldOut: profile.maxHeldOut)
        self.split = split
        let encoded = try PhotoEncoding.encode(subject.photos, model: model, faceWeighting: request.faceWeighting, store: store,
                                               subjectID: subjectID)
        let byID = Dictionary(encoded.map { ($0.photoID, $0) }, uniquingKeysWith: { a, _ in a })
        fit = split.fit.compactMap { byID[$0] }
        if split.augmented {
            let variants = PhotoAugment.variants(of: subject.photos[0], count: 3, schedule: schedule)
            heldOut = try PhotoEncoding.encode(variants, model: model, faceWeighting: request.faceWeighting)
            issues.append("Only one photo: the held-out set is augmentations of it (same pose, light and background), which overstates how well results generalize. Give 5–20 photos.")
        } else {
            heldOut = split.heldOut.compactMap { byID[$0] }
        }
        if request.faceWeighting == .auto, !encoded.contains(where: \.weight.faceFound) {
            issues.append("No face was found in the person's photos; the loss is weighted uniformly, so backgrounds and clothing count as much as the likeness.")
        }

        let limited = request.controls.limited(profile.maxControls, schedule: schedule)
        controlIdentities = limited.identities
        if limited.photos.isEmpty {
            say(.preparing, "no controls given: sampling generic people from the model (decode-free)")
            controlPhotos = try baseSamples(count: min(4, profile.maxControls))
            controlIdentities = Array(repeating: nil, count: controlPhotos.count)
            fallbackControls = true
            issues.append("No control photos: the model's own samples of generic people stand in for other people. Specificity and preservation are weaker without real controls (ideally including look-alikes).")
        } else {
            say(.preparing, "encoding \(limited.photos.count) control photo(s)")
            controlPhotos = try PhotoEncoding.encode(limited.photos, model: model, faceWeighting: request.faceWeighting)
        }

        buildRoutes(avoiding: request.controls.identities.compactMap { $0 })
    }

    private func buildRoutes(avoiding others: [String]) {
        let subject = request.subject
        routes = subject.names.map { Templates.nameRoute($0, count: profile.templatesPerName, anchor: subject.anchor) }
            + subject.descriptions.map { Templates.textRoute($0, kind: .description, anchor: subject.anchor) }
        nulls = NullNames.make(profile.nullCount, schedule: schedule, avoiding: subject.names + others).map {
            Templates.nameRoute($0, count: profile.templatesPerName, anchor: subject.anchor, kind: .null)
        }
    }

    // MARK: - Research: a subject sampled from the model itself

    /// Research only. The deployment's own latent samples of `subjectPrompt` stand in for the
    /// person's photos, and samples of each control prompt for the controls — never decoded. A
    /// positive control on a real model without real photos: the name's pull on the model's own
    /// samples of that name is high by construction, so a guard must visibly suppress it.
    public func prepareSampled(subjectPrompt: String, controls: [(identity: String, prompt: String)], count: Int,
                               heldOut heldOutCount: Int? = nil, perControl: Int = 2, steps: Int = 28,
                               guidance: Float? = nil) throws {
        let probe = SubjectPhoto(image: ImageWriter.image(rgba: [UInt8](repeating: 128, count: 256 * 256 * 4), width: 256, height: 256)!)
        let shape = try model.encodeLatent(probe).shape
        // Undistilled base models are sampled with CFG (as hosts deploy them); distilled ones without.
        let g = guidance ?? (model.descriptor.variant == "base" ? 4 : 1)
        let unconditional = g != 1 ? try model.embed([""])[0].value : nil
        func samples(_ prompt: String, _ n: Int, _ label: String) throws -> [EncodedPhoto] {
            let c = try model.embed([prompt])[0].value
            return try (0..<n).map { i in
                try Task.checkCancellation()
                say(.preparing, "sampling \(label) \(i + 1)/\(n) (latent only, never decoded)")
                let noise = schedule.normal(.control, "sampled|\(prompt)|\(i)", shape: shape)
                let latent = model.hooks.with(guard: false) {
                    LatentSampler.sample(model, conditioning: c, shape: shape, noise: noise, steps: steps, unconditional: unconditional, guidance: g)
                }
                extraEvaluations += steps * (g != 1 ? 2 : 1)
                return EncodedPhoto(photoID: "sample-\(Hashing.sha256Hex("\(prompt)|\(i)").prefix(16))", name: prompt, latent: latent,
                                    weight: .uniform(rows: shape[1], cols: shape[2], method: "sample"), photo: nil)
            }
        }
        subjectID = schedule.keyedID("sampled|" + subjectPrompt)
        let subject = try samples(subjectPrompt, count, "the subject")
        let k = min(count - 1, heldOutCount ?? max(1, Int((Double(count) * 0.35).rounded())))
        heldOut = Array(subject.prefix(k))
        fit = Array(subject.dropFirst(k))
        split = PhotoSplit(fit: fit.map(\.photoID), heldOut: heldOut.map(\.photoID), augmented: false)
        controlPhotos = []
        controlIdentities = []
        for (identity, prompt) in controls {
            let s = try samples(prompt, perControl, identity)
            controlPhotos += s
            controlIdentities += Array(repeating: identity, count: s.count)
        }
        issues.append("Research run: the subject is the model's own samples of “\(subjectPrompt)” (CFG \(g)) and the controls its samples of \(controls.map(\.identity).joined(separator: ", ")) — latents, never decoded. Not photos of anyone.")
        buildRoutes(avoiding: controls.map(\.identity))
    }

    /// Latents of generic people sampled from the deployment itself; never decoded.
    private func baseSamples(count: Int) throws -> [EncodedPhoto] {
        guard let shape = (fit.first ?? heldOut.first)?.shape else { return [] }
        let samples = try LatentSampler.genericPeople(model, shape: shape, count: count, schedule: schedule)
        extraEvaluations += samples.count * LatentSampler.genericPeopleCost(model)
        return samples
    }

    // MARK: - Assess

    @discardableResult
    public func assess() throws -> Assessment {
        guard !fit.isEmpty || !heldOut.isEmpty else { throw VeilError.notPrepared }
        try Task.checkCancellation()
        let grid = sigmas
        let subjectSet = DrawSet(label: "subject", photos: heldOut, sigmas: grid, noisePerSigma: profile.noisePerSigma,
                                 stream: .pull, schedule: schedule)
        let controlSet = controlPhotos.isEmpty ? nil
            : DrawSet(label: "controls", photos: controlPhotos, sigmas: grid, noisePerSigma: 1, stream: .pull, schedule: schedule)
        let bench = PullBench(model: model, subject: subjectSet, controls: controlSet, controlIdentities: controlIdentities)
        self.bench = bench

        var candidates: [SearchCandidate] = []
        if request.search, let search = model.tokenSearch, !fit.isEmpty {
            say(.searching, "discrete prompt search (\(profile.searchSteps) steps)")
            let fitSet = DrawSet(label: "search-fit", photos: Array(fit.prefix(3)), sigmas: [grid[grid.count / 2]],
                                 noisePerSigma: 1, stream: .search, schedule: schedule)
            candidates = try model.hooks.with(guard: false) {
                try PromptSearch.run(model: model, search: search, fit: fitSet, tokens: profile.searchTokens,
                                     steps: profile.searchSteps, schedule: schedule) { self.say(.searching, $0) }
            }
            extraEvaluations += profile.searchSteps * fitSet.count * 3
            let existing = Set(routes.flatMap(\.prompts))
            for c in candidates where !existing.contains(c.text) {
                routes.append(Templates.textRoute(c.text, kind: .discovered, anchor: request.subject.anchor))
            }
        }

        say(.assessing, "measuring \(routes.count) route(s) against \(nulls.count) invented names")
        let (measurements, null) = try model.hooks.with(guard: false) {
            try Reachability.assess(bench: bench, routes: routes, nulls: nulls, alpha: profile.alpha) { self.say(.assessing, $0) }
        }
        if !candidates.isEmpty {
            let sealed = candidates.map { c in
                SealedRoute(hash: Route(kind: .discovered, label: c.text, prompts: [], anchors: []).labelHash, text: c.text,
                            kind: "discovered", pull: measurements.first { $0.id == "discovered:\(c.text)" }?.subject.mean ?? 0)
            }
            try? store.writeSealedRoutes(sealed, subjectID: subjectID)
        }

        var locator: [LocatorSite] = []
        let primary = measurements.first(where: \.reaches) ?? measurements.first { $0.kind == .name }
        if let primary, let route = routes.first(where: { $0.id == primary.id }) {
            say(.locating, "locating the route in the model")
            locator = (try? model.hooks.with(guard: false) { try RouteLocator.locate(model: model, route: route, set: subjectSet, embeddings: bench) }) ?? []
        }
        let result = Assessment(measurements: measurements, null: null, capability: Reachability.capability(measurements),
                                search: candidates, locator: locator)
        assessment = result
        return result
    }

    // MARK: - Guard

    @discardableResult
    public func fitGuard() throws -> GuardFit {
        guard let bench else { throw VeilError.notPrepared }
        let reached = Set(assessment?.reached.map(\.id) ?? [])
        if reached.isEmpty && !request.forceGuard { throw VeilError.nothingToGuard }
        let subject = request.subject

        // Erase every name, plus whatever else reached (descriptions, discovered prompts); preserve
        // the everyday world around them. `GuardPlan` builds both, and `GuardFitter` fits the two
        // stages — the same code a group guard goes through.
        let extra = routes.filter { $0.kind != .name && $0.kind != .null && reached.contains($0.id) }
        let person = GuardPlan.Person(id: subjectID, names: subject.names, descriptions: subject.descriptions,
                                      anchor: subject.anchor, fit: fit.isEmpty ? heldOut : fit)
        let plan = GuardPlan.make(people: [person], extraRoutes: extra.isEmpty ? [:] : [0: extra], profile: profile,
                                  schedule: schedule, controlIdentities: Array(Set(controlIdentities.compactMap { $0 })),
                                  extraPreserve: request.preservePrompts)
        let result = try GuardFitter.fit(model: model, plan: plan, preservePhotos: controlPhotos, method: request.method,
                                         profile: profile, schedule: schedule, embed: bench.embeddings) { self.say($0, $1) }
        guardFit = result
        return result
    }

    // MARK: - Export

    public var adapterMetadata: [String: String] {
        let d = model.descriptor
        let iso = ISO8601DateFormatter().string(from: request.subject.consent.attestedAt)
        return [
            "veil.schema": VeilInfo.adapterSchema,
            "veil.version": VeilInfo.version,
            "veil.run_id": runID,
            "veil.base_model": d.pinnedName,
            "veil.family": d.family,
            "veil.variant": d.variant ?? "unknown",
            "veil.subject": subjectID,
            "veil.consent": "\(request.subject.consent.basis.rawValue) \(iso)",
            "veil.method": guardFit?.method.rawValue ?? request.method.rawValue,
            "veil.scale": "1.0",
            "veil.intended_use": "Guard: suppresses one protected person's likeness on \(d.pinnedName). Apply at scale 1.0; check with `veil verify`.",
        ]
    }

    @discardableResult
    public func export(to url: URL) throws -> GuardFile {
        guard let guardFit else { throw VeilError.noGuard }
        say(.exporting, "writing \(url.lastPathComponent)")
        let file = try AdapterWriter.write(guardFit.deltas, model: model, format: request.format, metadata: adapterMetadata, to: url)
        guardFile = file
        return file
    }

    // MARK: - Verify

    /// Read the guard file back, install exactly what it holds, and verify it.
    @discardableResult
    public func verify(guardAt url: URL) throws -> Verification {
        guard !heldOut.isEmpty else { throw VeilError.notPrepared }
        try Task.checkCancellation()
        say(.verifying, "reading back \(url.lastPathComponent)")
        let (deltas, file) = try AdapterWriter.readBack(url, model: model)
        model.installGuard(deltas)
        if guardFile == nil {
            let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
            guardFile = GuardFile(path: url.path, sha256: file.sha256, format: file.metadata["veil.format"] ?? "unknown",
                                  slots: deltas.keys.sorted(), rank: deltas.values.map(\.rank).max() ?? 0,
                                  parameters: deltas.values.reduce(0) { $0 + $1.rank * ($1.inFeatures + $1.outFeatures) }, bytes: size)
        }
        if let base = file.metadata["veil.base_model"], base != model.descriptor.pinnedName {
            issues.append("The guard was fitted on \(base); this model is \(model.descriptor.pinnedName).")
        }
        let reached = Set(assessment?.reached.map(\.id) ?? [])
        // Descriptions that reached the person are erased; the others must keep working.
        let kept = request.subject.descriptions.filter { !reached.contains("description:\($0)") }
        let inputs = Verifier.Inputs(routes: routes, reached: reached, nulls: nulls, fit: fit.isEmpty ? heldOut : fit,
                                     heldOut: heldOut, controls: controlPhotos, controlIdentities: controlIdentities,
                                     descriptions: kept, anchor: request.subject.anchor)
        let (v, vb) = try Verifier.verify(model: model, inputs: inputs, guardSHA256: file.sha256, profile: profile,
                                          schedule: schedule) { self.say(.verifying, $0) }
        verifyBench = vb
        verification = v
        return v
    }

    // MARK: - Diagnostics

    /// A route's pull on every control identity (the assessment's draws), strongest first.
    public func controlBreakdown(routeID: String) throws -> [(identity: String, pull: PullEstimate)] {
        guard let bench, let controls = bench.controls, let route = routes.first(where: { $0.id == routeID }) else { return [] }
        let a = try route.anchors.map { try bench.losses(controls, prompt: $0) }
        let r = try route.prompts.map { try bench.losses(controls, prompt: $0) }
        return model.hooks.with(guard: false) {
            bench.controlGroups.map { ($0.identity, PullEstimate.paired(anchor: a, route: r, draws: $0.draws)) }
                .sorted { $0.1.mean > $1.1.mean }
        }
    }

    /// A route's pull on the person's held-out draws at each σ level (the assessment's draws).
    public func sigmaBreakdown(routeID: String) throws -> [(sigma: Float, pull: PullEstimate)] {
        guard let bench, let route = routes.first(where: { $0.id == routeID }) else { return [] }
        let set = bench.subject
        let (a, r) = try model.hooks.with(guard: false) {
            (try route.anchors.map { try bench.losses(set, prompt: $0) }, try route.prompts.map { try bench.losses(set, prompt: $0) })
        }
        var offset = 0, out: [(Float, PullEstimate)] = []
        for g in set.groups {
            out.append((g.sigma, PullEstimate.paired(anchor: a, route: r, draws: Array(offset..<(offset + g.rows.count)))))
            offset += g.rows.count
        }
        return out
    }

    // MARK: - Report and figure

    public var evaluations: Int { (bench?.evaluations ?? 0) + (verifyBench?.evaluations ?? 0) + extraEvaluations }

    public func report() -> VeilReport {
        let subject = request.subject
        func record(_ p: EncodedPhoto, role: String, identity: String? = nil) -> PhotoRecord {
            PhotoRecord(sha256: p.photoID, role: role, identity: identity, faceFound: p.weight.faceFound, likeness: p.weight.method,
                        faceArea: p.weight.faceArea)
        }
        var photos = fit.map { record($0, role: "fit") }
        photos += heldOut.map { record($0, role: split?.augmented == true ? "held-out (augmented)" : "held-out") }
        let subjectRecord = SubjectRecord(id: subjectID, names: request.redactNames ? nil : subject.names,
                                          nameHashes: subject.names.map { String(Hashing.sha256Hex("veil.name|" + $0).prefix(12)) },
                                          descriptions: subject.descriptions.count, anchor: subject.anchor, consent: subject.consent,
                                          split: split ?? PhotoSplit(fit: [], heldOut: [], augmented: false), photos: photos)
        let controls = ControlsRecord(count: controlPhotos.count, identities: Array(Set(controlIdentities.compactMap { $0 })).sorted(),
                                      fallbackSamples: fallbackControls,
                                      photos: zip(controlPhotos, controlIdentities).map { record($0, role: "control", identity: $1) })
        let assessmentRecord = assessment.map { a in
            AssessmentRecord(routes: a.measurements.map { m in request.redactNames && m.kind == .name ? m.redacted() : m },
                             null: a.null, capability: a.capability,
                             search: a.search.isEmpty ? nil : a.search.map {
                                 SearchCandidateRecord(hash: Route(kind: .discovered, label: $0.text, prompts: [], anchors: []).labelHash,
                                                       text: request.revealRoutes ? $0.text : nil, fitLoss: $0.fitLoss)
                             },
                             locator: a.locator)
        }
        let guardRecord = guardFit.map {
            GuardRecord(method: $0.method, erased: request.redactNames ? [] : $0.erased, closedForm: $0.closedForm,
                        training: $0.training, file: guardFile?.shareable)
        } ?? guardFile.map { GuardRecord(method: request.method, erased: [], closedForm: nil, training: nil, file: $0.shareable) }

        var verdict = "assessed"
        var reasons: [String] = []
        if let v = verification {
            verdict = v.verdict.rawValue
            reasons = v.reasons
        } else if let a = assessment {
            verdict = a.capability.rawValue
            let reached = a.reached
            reasons = reached.isEmpty ? ["no route reached the person at this budget (not proof that none can)"]
                : reached.map { "reaches: \($0.label.map { "“\($0)”" } ?? $0.labelHash) (\($0.kind.rawValue), p = \($0.pValue.map { String(format: "%.2f", $0) } ?? "–"))" }
        }
        return VeilReport(schema: VeilInfo.reportSchema, version: VeilInfo.version, runID: runID, createdAt: Date(),
                          environment: .current(), mode: "guard", model: model.descriptor.shareable,
                          withAdapter: request.withAdapter.map { $0.contains("://") ? $0 : ($0 as NSString).lastPathComponent },
                          subjects: [subjectRecord], controls: controls, profile: profile, schedule: schedule.info,
                          assessment: assessmentRecord, guardFit: guardRecord, verification: verification, checks: nil,
                          verdict: verdict, reasons: reasons, issues: issues, limits: VeilReport.limits,
                          evaluations: evaluations, seconds: Date().timeIntervalSince(started))
    }

    /// Before/after pull maps of the strongest route on the held-out photos (a local artifact:
    /// it shows the person's photos).
    public func figure() -> CGImage? {
        guard let assessment, let set = (verifyBench ?? bench)?.subject,
              let primary = assessment.reached.first ?? assessment.measurements.first(where: { $0.kind == .name }),
              let route = routes.first(where: { $0.id == primary.id }),
              let (routeC, anchorC) = try? ((verifyBench ?? bench)!.embeddings([route.prompts[0]])[0].value,
                                            (verifyBench ?? bench)!.embeddings([route.anchors[0]])[0].value) else { return nil }
        func maps(guarded: Bool) -> [PullMap] {
            model.hooks.with(guard: guarded) {
                let a = LossEvaluator.evaluate(model, set, conditioning: anchorC, maps: true).maps
                let r = LossEvaluator.evaluate(model, set, conditioning: routeC, maps: true).maps
                return zip(zip(a, r), set.photos).map { pair, p in
                    PullMap.relative(anchor: pair.0, route: pair.1, rows: p.weight.rows, cols: p.weight.cols)
                }
            }
        }
        let base = maps(guarded: false)
        let guarded = guardFit != nil || guardFile != nil ? maps(guarded: true) : base
        let label = primary.label.map { "“\($0)”" } ?? "a discovered route"
        return PullFigure().render(photos: set.photos.compactMap(\.photo), base: base, guarded: guarded,
                                   title: "Where \(label) pulls toward the person",
                                   subtitle: "\(model.descriptor.pinnedName) · verdict \(verification?.verdict.rawValue ?? assessment.capability.rawValue) · contains the person's photos — do not share")
    }
}
