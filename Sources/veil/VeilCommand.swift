//
//  VeilCommand.swift
//  veil
//
//  The composition root: registers executors, parses a run, prints what it measured.
//

import ArgumentParser
import Foundation
import VeilFlux2
import VeilKit

@main
struct VeilCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "veil",
        abstract: "Find the prompts that reach a protected person in a model, and fit a guard LoRA that closes them.",
        discussion: """
            find → block → verify. Photos never leave this machine; reports carry hashes, not pixels.
            A guard blocks routes, not the likeness itself, and binds only where the deployer applies it.
            """,
        version: VeilInfo.version,
        subcommands: [Inspect.self, Assess.self, Guard.self, Protect.self, Verify.self, Forget.self, Research.self],
        defaultSubcommand: Guard.self)
}

// MARK: - Shared options

struct RunOptions: ParsableArguments {
    @Option(help: "Model: a Hugging Face or GitHub link, a local model folder, or \"toy\".")
    var model: String

    @Option(name: .customLong("photos"), help: "Photos of the protected person (files or folders; repeatable).")
    var photos: [String] = []

    @Option(help: "A name the person is known by (repeatable; the first is primary).")
    var name: [String] = []

    @Option(name: .customLong("describe"), help: "A description that might reach the person without naming them (repeatable).")
    var descriptions: [String] = []

    @Option(help: "The neutral phrase a name is replaced with.")
    var anchor = "a person"

    @Option(help: "Photos of other people (look-alikes welcome); subfolder names are identities.")
    var controls: String?

    @Option(help: "Consent basis: self (you are the person) or representative (authorized to act for them).")
    var consent: String?

    @Option(help: "quick | standard | thorough | toy")
    var profile = "standard"

    @Option(help: "Likeness weighting: auto (face via Vision) or off.")
    var face = "auto"

    @Option(name: .customLong("with-adapter"), help: "An adapter that is part of the deployment (path or link).")
    var withAdapter: String?

    @Option(name: .customLong("option"), help: "Executor option key=value (repeatable), e.g. text-encoder=mflux.")
    var executorOptions: [String] = []

    @Flag(help: "Run discrete prompt search for routes that don't name the person.")
    var search = false

    @Flag(help: "Harden the trained guard against soft attacks while training (experimental).")
    var harden = false

    @Flag(name: .customLong("reveal-routes"), help: "Put discovered prompts' texts into the report (they are attack recipes).")
    var revealRoutes = false

    @Flag(name: .customLong("redact-names"), help: "Replace names with hashes in the report.")
    var redactNames = false

    @Option(help: "Write the JSON report here.")
    var json: String?

    @Option(help: "Write the before/after pull figure here (contains the person's photos).")
    var figure: String?

    @Flag(name: .shortAndLong, help: "Quiet: no progress lines.")
    var quiet = false

    var options: [String: String] {
        Dictionary(executorOptions.compactMap { kv -> (String, String)? in
            let parts = kv.split(separator: "=", maxSplits: 1).map(String.init)
            return parts.count == 2 ? (parts[0], parts[1]) : nil
        }, uniquingKeysWith: { _, b in b })
    }

    func validateConsent() throws -> Consent {
        guard let consent, let basis = Consent.Basis(rawValue: consent) else {
            throw ValidationError("""
                --consent is required: self (you are the person in the photos) or representative (you are authorized \
                to act for them). Veil only builds guards for people who asked to be protected.
                """)
        }
        return Consent(basis: basis)
    }

    /// The model and everything that doesn't depend on who is being protected.
    struct Loaded {
        let model: GuardableModel
        let profile: VeilProfile
        let consent: Consent
        let schedule: SeedSchedule
        let isToy: Bool
        let say: @Sendable (VeilProgress) -> Void
    }

    func loadModel() async throws -> Loaded {
        Registration.registerAll()
        let consent = try validateConsent()
        guard var profile = VeilProfile.named(self.profile) else { throw ValidationError("Unknown profile \(self.profile)") }
        let isToy = ToyExecutor.descriptor(for: model) != nil
        if isToy && self.profile == "standard" { profile = .toy }
        if harden { profile = profile.hardened() }
        let fetcher = RangeFetcher()
        let say = progressPrinter(quiet)
        var adapterURL: URL?
        if let withAdapter {
            say(VeilProgress(phase: .preparing, message: "fetching the deployment adapter"))
            adapterURL = try await AdapterSource.fetch(withAdapter, fetcher: fetcher)
        }
        say(VeilProgress(phase: .preparing, message: "loading \(model)"))
        let loaded = try await ExecutorRegistry.shared.load(model, options: options, withAdapter: adapterURL, fetcher: fetcher) {
            say(VeilProgress(phase: .preparing, message: $0))
        }
        let schedule = isToy ? SeedSchedule.research : ((try? SeedSchedule.load()) ?? .ephemeral())
        return Loaded(model: loaded, profile: profile, consent: consent, schedule: schedule, isToy: isToy, say: say)
    }

    func controlSet() throws -> ControlSet {
        guard let controls else { return .empty }
        return try ControlSet.load(URL(fileURLWithPath: (controls as NSString).expandingTildeInPath))
    }

    func subjectPhotos() throws -> [SubjectPhoto] {
        try SubjectPhoto.load(paths: photos.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) })
    }

    func load(method: GuardMethod = .both, format: String? = nil, forceGuard: Bool = false, preserve: [String] = [])
        async throws -> VeilRun {
        let loaded = try await loadModel()

        // Photos: the toy world brings its own when none are given.
        var photos = try subjectPhotos()
        var names = name, descriptions = self.descriptions
        var controls = try controlSet()
        var extraPreserve = preserve
        if loaded.isToy && photos.isEmpty {
            let world = ToyIdentityWorld.shared
            photos = world.subjectPhotos
            if names.isEmpty { names = [ToyIdentityWorld.subjectName] }
            if descriptions.isEmpty { descriptions = [ToyIdentityWorld.subjectDescription] }
            if controls.photos.isEmpty { controls = world.controls }
            extraPreserve.append(ToyIdentityWorld.attributePrompt)
        }
        guard !photos.isEmpty else { throw ValidationError("Give photos of the person with --photos.") }
        let subject = ProtectedSubject(photos: photos, names: names, descriptions: descriptions, anchor: anchor,
                                       consent: loaded.consent)
        let request = VeilRequest(subject: subject, controls: controls, profile: loaded.profile,
                                  faceWeighting: loaded.isToy ? .off : (face == "off" ? .off : .auto), search: search,
                                  method: method, format: format, revealRoutes: revealRoutes, redactNames: redactNames,
                                  forceGuard: forceGuard, preservePrompts: extraPreserve, withAdapter: withAdapter)
        let run = VeilRun(model: loaded.model, request: request, schedule: loaded.schedule)
        run.onProgress = loaded.say
        try run.prepare()
        return run
    }

    func finish(_ run: VeilRun) throws {
        let report = run.report()
        if let json {
            try report.write(to: URL(fileURLWithPath: (json as NSString).expandingTildeInPath))
            if !quiet { FileHandle.standardError.write(Data("· report → \(json)\n".utf8)) }
        }
        if let figure, let image = run.figure() {
            try ImageWriter.writePNG(image, to: URL(fileURLWithPath: (figure as NSString).expandingTildeInPath))
            if !quiet { FileHandle.standardError.write(Data("· figure → \(figure) (contains the person's photos)\n".utf8)) }
        }
        print(Render.summary(report))
    }
}

func progressPrinter(_ quiet: Bool) -> @Sendable (VeilProgress) -> Void {
    let started = Date()
    return { p in
        guard !quiet else { return }
        let t = Int(Date().timeIntervalSince(started))
        FileHandle.standardError.write(Data(String(format: "[%3d:%02d] %@ · %@\n", t / 60, t % 60, p.phase.rawValue, p.message).utf8))
    }
}

// MARK: - Commands

struct Inspect: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Describe a model link: family, variant, components, support. No weights are fetched.")

    @Option(help: "A Hugging Face or GitHub link, or a local model folder.")
    var model: String

    func run() async throws {
        Registration.registerAll()
        let (d, source) = try await ExecutorRegistry.shared.describe(model, fetcher: RangeFetcher())
        print(Render.descriptor(d, source: source, families: ExecutorRegistry.shared.families))
    }
}

struct Assess: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Find which routes reach the person (no guard).")

    @OptionGroup var run: RunOptions

    func run() async throws {
        let r = try await run.load()
        try r.assess()
        try run.finish(r)
    }
}

struct Guard: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Assess, fit a guard LoRA, export it, and verify the exported file.")

    @OptionGroup var run: RunOptions

    @Option(help: "both (closed form, then trained) | closed-form | trained")
    var method = "both"

    @Option(help: "Export format: diffusers (default) or bfl (ComfyUI / BFL names).")
    var format: String?

    @Option(help: "Where to write the guard.")
    var out = "guard.safetensors"

    @Flag(name: .customLong("force-guard"), help: "Fit a guard even when nothing reached the person.")
    var forceGuard = false

    @Option(help: "A prompt the guard must leave alone (repeatable).")
    var preserve: [String] = []

    func run() async throws {
        guard let m = GuardMethod(rawValue: method) else { throw ValidationError("Unknown method \(method)") }
        let r = try await run.load(method: m, format: format, forceGuard: forceGuard, preserve: preserve)
        try r.assess()
        do {
            try r.fitGuard()
        } catch VeilError.nothingToGuard {
            try run.finish(r)
            print("\nNothing reached the person at this budget, so no guard was written (use --force-guard to fit one anyway).")
            return
        }
        let url = URL(fileURLWithPath: (out as NSString).expandingTildeInPath)
        let file = try r.export(to: url)
        if !run.quiet { FileHandle.standardError.write(Data("· guard → \(file.path)\n".utf8)) }
        try r.verify(guardAt: url)
        try run.finish(r)
    }
}

struct Protect: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Fit a guard LoRA for one person or a group, without assessing the model first.",
        discussion: """
            Straight to the block stage: the names and descriptions you give are erased and everything else is \
            held in place. Nothing here measures whether the model could reach these people — `veil guard` does \
            that, and `veil verify` measures the file afterwards. By default a quick check follows the export: \
            each name's pull on held-out photos with the guard off vs on, plus drift on everyday prompts.
            """)

    @OptionGroup var run: RunOptions

    @Option(help: "A folder with one subfolder per person; the subfolder's name is their name (repeatable).")
    var people: [String] = []

    @Option(help: "both (closed form, then trained) | closed-form | trained")
    var method = "both"

    @Option(help: "Export format: diffusers (default) or bfl (ComfyUI / BFL names).")
    var format: String?

    @Option(help: "Where to write the guard.")
    var out = "guard.safetensors"

    @Option(help: "A prompt the guard must leave alone (repeatable).")
    var preserve: [String] = []

    @Option(name: .customLong("train-steps"), help: "Override the profile's training steps.")
    var trainSteps: Int?

    @Option(help: "Name templates per person (default: scaled to the size of the group).")
    var templates: Int?

    @Flag(name: .customLong("no-check"), help: "Write the guard and stop, measuring nothing.")
    var noCheck = false

    @Flag(help: "Run the full verification per person (null calibration, attacks, preservation) instead of the quick check.")
    var fullVerify = false

    func run() async throws {
        guard let m = GuardMethod(rawValue: method) else { throw ValidationError("Unknown method \(method)") }
        if self.run.search {
            throw ValidationError("--search looks for routes nobody named, which needs an assessment: use `veil guard --search`.")
        }
        if self.run.figure != nil {
            throw ValidationError("--figure is drawn from an assessment: use `veil guard --figure`.")
        }
        let loaded = try await self.run.loadModel()
        var profile = loaded.profile
        if let trainSteps { profile.trainSteps = trainSteps }

        var subjects = try people.flatMap {
            try ProtectedSubject.loadPeople(URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath), consent: loaded.consent)
        }
        let photos = try self.run.subjectPhotos()
        if !photos.isEmpty {
            subjects.append(ProtectedSubject(photos: photos, names: self.run.name, descriptions: self.run.descriptions,
                                             anchor: self.run.anchor, consent: loaded.consent))
        }
        var controls = try self.run.controlSet()
        var extraPreserve = preserve
        if loaded.isToy && subjects.isEmpty {
            let world = ToyIdentityWorld.shared
            subjects = [ProtectedSubject(photos: world.subjectPhotos, names: [ToyIdentityWorld.subjectName],
                                         descriptions: [ToyIdentityWorld.subjectDescription], anchor: self.run.anchor,
                                         consent: loaded.consent)]
            if controls.photos.isEmpty { controls = world.controls }
            extraPreserve.append(ToyIdentityWorld.attributePrompt)
        }
        guard !subjects.isEmpty else {
            throw ValidationError("Give the people to protect: --people <folder with one subfolder per person>, or --photos with --name.")
        }

        let request = ProtectRequest(people: subjects, controls: controls, profile: profile,
                                     faceWeighting: loaded.isToy ? .off : (self.run.face == "off" ? .off : .auto),
                                     method: m, format: format,
                                     check: noCheck ? .none : (fullVerify ? .full : .quick),
                                     redactNames: self.run.redactNames, preservePrompts: extraPreserve,
                                     templatesPerPerson: templates, withAdapter: self.run.withAdapter)
        let runner = ProtectRun(model: loaded.model, request: request, schedule: loaded.schedule)
        runner.onProgress = loaded.say
        try runner.prepare()
        try runner.fitGuard()
        let url = URL(fileURLWithPath: (out as NSString).expandingTildeInPath)
        let file = try runner.export(to: url)
        if !self.run.quiet { FileHandle.standardError.write(Data("· guard → \(file.path)\n".utf8)) }
        try runner.check(guardAt: url)

        let report = runner.report()
        if let json = self.run.json {
            try report.write(to: URL(fileURLWithPath: (json as NSString).expandingTildeInPath))
            if !self.run.quiet { FileHandle.standardError.write(Data("· report → \(json)\n".utf8)) }
        }
        print(Render.summary(report))
        if request.check == .none {
            print("\nNothing has measured this guard. To find out what it does:\n  veil verify --model \(self.run.model) --guard \(file.path) --photos <person> --name \"<name>\" --consent \(self.run.consent ?? "representative")")
        }
    }
}

struct Verify: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Verify an existing guard file (path or link) on this model.")

    @OptionGroup var run: RunOptions

    @Option(name: .customLong("guard"), help: "The guard to verify (path or link).")
    var guardFile: String

    func run() async throws {
        let r = try await run.load()
        let url = try await AdapterSource.fetch(guardFile, fetcher: RangeFetcher())
        try r.assess()
        try r.verify(guardAt: url)
        try run.finish(r)
    }
}

struct Forget: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Delete everything Veil holds for a subject (latents, sealed routes).")

    @Argument(help: "The subject id (from a report), or --list.")
    var subject: String?

    @Flag(help: "List subject ids with stored data.")
    var list = false

    func run() throws {
        let store = SubjectStore()
        if list || subject == nil {
            let ids = store.list()
            print(ids.isEmpty ? "No stored subjects." : ids.joined(separator: "\n"))
            return
        }
        print(try store.forget(subject!) ? "Forgot \(subject!)." : "Nothing stored for \(subject!).")
    }
}

struct Research: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Validation where the answer is known.",
                                                    subcommands: [ResearchToy.self, ResearchSampled.self])
}

struct ResearchSampled: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "sampled",
        abstract: "Positive control on a real model without photos: the model's own latent samples of a name are the subject.",
        discussion: "Latents only — nothing is decoded or shown. Controls are the model's samples of other names or descriptions.")

    @Option(help: "Model link or local folder.")
    var model: String

    @Option(help: "The name whose samples stand in for the subject.")
    var name: String

    @Option(help: "Control identities, sampled the same way (repeatable).")
    var control: [String] = []

    @Option(help: "quick | standard | thorough")
    var profile = "quick"

    @Option(name: .customLong("train-steps"), help: "Override the profile's training steps.")
    var trainSteps: Int?

    @Option(help: "Subject samples.")
    var samples = 6

    @Option(name: .customLong("option"), help: "Executor option key=value (repeatable).")
    var executorOptions: [String] = []

    @Option(help: "Where to write the guard.")
    var out = "sampled-guard.safetensors"

    @Option(help: "Write the JSON report here.")
    var json: String?

    @Flag(name: .customLong("assess-only"), help: "Stop after the assessment.")
    var assessOnly = false

    @Flag(name: .customLong("force-guard"), help: "Fit and verify a guard even when nothing reached.")
    var forceGuard = false

    @Option(name: .customLong("held-out"), help: "Subject samples held out for measurement (default 35%).")
    var heldOut: Int?

    @Option(help: "Override the profile's noise draws per σ.")
    var noise: Int?

    @Option(help: "Override the profile's number of σ levels.")
    var sigmas: Int?

    @Option(name: .customLong("per-control"), help: "Samples per control identity.")
    var perControl = 2

    @Option(help: "Classifier-free guidance for sampling (default 4 for base models, 1 for distilled).")
    var guidance: Float?

    func run() async throws {
        Registration.registerAll()
        guard var profile = VeilProfile.named(self.profile) else { throw ValidationError("Unknown profile") }
        if let trainSteps { profile.trainSteps = trainSteps }
        if let noise { profile.noisePerSigma = noise }
        if let sigmas { profile.sigmaCount = sigmas }
        let options = Dictionary(executorOptions.compactMap { kv -> (String, String)? in
            let p = kv.split(separator: "=", maxSplits: 1).map(String.init)
            return p.count == 2 ? (p[0], p[1]) : nil
        }, uniquingKeysWith: { _, b in b })
        let say = progressPrinter(false)
        let loaded = try await ExecutorRegistry.shared.load(model, options: options) { say(VeilProgress(phase: .preparing, message: $0)) }
        let subject = ProtectedSubject(photos: [], names: [name], consent: Consent(basis: .representative))
        let request = VeilRequest(subject: subject, profile: profile, faceWeighting: .off, forceGuard: forceGuard)
        let run = VeilRun(model: loaded, request: request, schedule: .research,
                          store: SubjectStore(root: FileManager.default.temporaryDirectory.appendingPathComponent("veil-sampled")))
        run.onProgress = say
        let controls = (control.isEmpty ? ["a bearded man in a black suit", "a portrait of a woman"] : control)
            .map { (identity: $0, prompt: Templates.fill(Templates.name[0], $0)) }
        try run.prepareSampled(subjectPrompt: Templates.fill(Templates.name[0], name), controls: controls, count: samples,
                               heldOut: heldOut, perControl: perControl, guidance: guidance)
        let assessment = try run.assess()
        for m in assessment.measurements {
            print("\(m.id): pull on the subject \(Render.f(m.subject.mean)) ± \(Render.f(m.subject.standardError)) (\(Render.pct(m.subject.relative)))")
            for (identity, pull) in try run.controlBreakdown(routeID: m.id) {
                print("    on \(Render.pad(identity, 32)) \(Render.f(pull.mean)) ± \(Render.f(pull.standardError))")
            }
            print("    spec \(Render.f(m.specificity)) · p \(m.pValue.map { Render.f($0, 2) } ?? "–") · null q95 \(Render.f(assessment.null.q95)) · reaches \(m.reaches)")
            for (sigma, pull) in try run.sigmaBreakdown(routeID: m.id) {
                print("    σ \(Render.f(Double(sigma), 3)): pull \(Render.f(pull.mean)) ± \(Render.f(pull.standardError)) (\(Render.pct(pull.relative)))")
            }
        }
        guard !assessOnly else {
            print(Render.summary(run.report()))
            return
        }
        do {
            try run.fitGuard()
        } catch VeilError.nothingToGuard {
            print(Render.summary(run.report()))
            print("\nNothing reached the subject, so no guard was fitted.")
            return
        }
        let url = URL(fileURLWithPath: (out as NSString).expandingTildeInPath)
        try run.export(to: url)
        try run.verify(guardAt: url)
        let report = run.report()
        if let json { try report.write(to: URL(fileURLWithPath: (json as NSString).expandingTildeInPath)) }
        print(Render.summary(report))
    }
}

struct ResearchToy: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "toy", abstract: "Run find → block → verify on the analytic toy world and print a go/no-go.")

    @Option(help: "Also write the toy's photos here (subject/ and controls/<name>/) for CLI demos.")
    var write: String?

    @Option(help: "Write the guard here.")
    var out: String?

    func run() async throws {
        let world = ToyIdentityWorld.shared
        if let write {
            let (s, c) = try world.writePhotos(to: URL(fileURLWithPath: (write as NSString).expandingTildeInPath))
            print("toy photos → \(s.path) and \(c.path)")
        }
        let subject = ProtectedSubject(photos: world.subjectPhotos, names: [ToyIdentityWorld.subjectName],
                                       descriptions: [ToyIdentityWorld.subjectDescription], consent: Consent(basis: .selfAttested))
        var rows: [(String, String, String)] = []
        for method in [GuardMethod.closedForm, .both] {
            let request = VeilRequest(subject: subject, controls: world.controls, profile: .toy, faceWeighting: .off, method: method,
                                      preservePrompts: [ToyIdentityWorld.attributePrompt])
            let run = VeilRun(model: ToyModel(), request: request, schedule: .research,
                              store: SubjectStore(root: FileManager.default.temporaryDirectory.appendingPathComponent("veil-toy")))
            try run.prepare()
            try run.assess()
            try run.fitGuard()
            let url = URL(fileURLWithPath: out.map { ($0 as NSString).expandingTildeInPath }
                          ?? FileManager.default.temporaryDirectory.appendingPathComponent("veil-toy-\(method.rawValue).safetensors").path)
            try run.export(to: url)
            let v = try run.verify(guardAt: url)
            let suppressed = v.suppression.filter { $0.base.reaches }.allSatisfy { $0.pass }
            let cost = v.robustness.map { "\($0.guarded.start) \($0.base.reachedAt.map(String.init) ?? ">max")→\($0.guarded.reachedAt.map(String.init) ?? ">max")" }
            rows.append((method.rawValue, v.verdict.rawValue,
                         String(format: "suppressed %@ · max drift %.2f%% · attack steps %@", suppressed ? "yes" : "NO",
                                100 * v.preservation.maxDrift, cost.joined(separator: ", "))))
        }
        print("Veil toy world (known answer: \(ToyIdentityWorld.subjectName) reachable by name and by one description)")
        for (m, verdict, detail) in rows { print("  \(Render.pad(m, 12)) \(Render.pad(verdict.uppercased(), 10)) \(detail)") }
        let go = rows.allSatisfy { $0.1 == GuardVerdict.guarded.rawValue }
        print(go ? "GO: every route that reached is suppressed, the rest of the world is preserved." : "NO-GO: see the rows above.")
    }
}
