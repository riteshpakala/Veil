//
//  VeilViewModel.swift
//  VeilApp
//
//  One run at a time (MLX evaluates models on one thread per process). The run happens in a
//  detached task that Cancel can stop at the next stage boundary.
//
//  Two paths: assess → guard → verify for one person, and protect — straight to the guard, for
//  one person or a group, with a quick check after.
//

import AppKit
import UniformTypeIdentifiers
import VeilKit

/// One protected person as typed into the app.
struct PersonInput: Identifiable {
    let id = UUID()
    var photos: [SubjectPhoto] = []
    var names = ""
    var descriptions = ""
    var anchor = "a person"
}

@MainActor
final class VeilViewModel: ObservableObject {
    // Model
    @Published var modelLink = "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B"
    @Published var executorOptions = ""
    @Published var deploymentAdapter = ""
    @Published var descriptor: ModelDescriptor?

    // People
    @Published var people: [PersonInput] = [PersonInput()]
    @Published var controls: ControlSet = .empty
    @Published var controlsFolder: URL?

    // Consent
    @Published var consentBasis: Consent.Basis = .selfAttested
    @Published var consentAttested = false

    // Run settings
    @Published var profile = "standard"
    @Published var method: GuardMethod = .both
    @Published var format = "diffusers"
    @Published var search = false
    @Published var harden = false
    /// Protect only: measure the guard after writing it.
    @Published var quickCheck = true

    // State
    @Published var isRunning = false
    @Published var phase = ""
    @Published var log: [String] = []
    @Published var report: VeilReport?
    @Published var figure: NSImage?
    @Published var guardURL: URL?
    @Published var errorMessage: String?

    private var task: Task<Void, Never>?

    var isToy: Bool { ToyExecutor.descriptor(for: modelLink) != nil }

    var hasPhotos: Bool { isToy || people.allSatisfy { !$0.photos.isEmpty } }

    var canRun: Bool {
        !isRunning && consentAttested && hasPhotos && !modelLink.trimmingCharacters(in: .whitespaces).isEmpty
    }

    /// Assessment measures one person against controls; a group goes through protect.
    var canAssess: Bool { canRun && people.count == 1 }

    /// "I am this person" cannot attest for anyone else.
    var canProtect: Bool { canRun && (people.count == 1 || consentBasis == .representative) }

    static let anchors = ["a person", "a man", "a woman"]

    // MARK: People

    func addPerson() {
        people.append(PersonInput())
        report = nil
        figure = nil
    }

    func removePerson(_ person: PersonInput) {
        people.removeAll { $0.id == person.id }
        if people.isEmpty { people = [PersonInput()] }
    }

    func index(of person: PersonInput) -> Int? { people.firstIndex { $0.id == person.id } }

    func choosePhotos(for person: PersonInput) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = true
        panel.message = "Photos of the person to protect (5–20 recommended; different days, light and angles)."
        guard panel.runModal() == .OK else { return }
        addPhotos(panel.urls, to: person)
    }

    func addPhotos(_ urls: [URL], to person: PersonInput) {
        guard let i = index(of: person) else { return }
        do {
            let loaded = try SubjectPhoto.load(paths: urls)
            let known = Set(people[i].photos.map(\.id))
            people[i].photos += loaded.filter { !known.contains($0.id) }
            report = nil
            figure = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func removePhoto(_ photo: SubjectPhoto, from person: PersonInput) {
        guard let i = index(of: person) else { return }
        people[i].photos.removeAll { $0.id == photo.id }
    }

    func chooseControls() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Use as Controls"
        panel.message = "Photos of other people. Subfolders named after a person add a check that their own name still works."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            controls = try ControlSet.load(url)
            controlsFolder = url
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func clearControls() {
        controls = .empty
        controlsFolder = nil
    }

    func inspect() {
        let link = modelLink.trimmingCharacters(in: .whitespaces)
        Task.detached {
            do {
                let (d, _) = try await ExecutorRegistry.shared.describe(link, fetcher: RangeFetcher())
                await MainActor.run { self.descriptor = d }
            } catch {
                await MainActor.run { self.errorMessage = error.localizedDescription }
            }
        }
    }

    // MARK: Run

    func lines(_ s: String) -> [String] {
        s.split(whereSeparator: { $0 == "\n" || $0 == "," }).map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }

    var options: [String: String] {
        Dictionary(lines(executorOptions).compactMap { kv -> (String, String)? in
            let p = kv.split(separator: "=", maxSplits: 1).map(String.init)
            return p.count == 2 ? (p[0].trimmingCharacters(in: .whitespaces), p[1].trimmingCharacters(in: .whitespaces)) : nil
        }, uniquingKeysWith: { _, b in b })
    }

    /// The people as Veil sees them, with the toy world's own subject when nothing was dropped in.
    private func subjects() -> (people: [ProtectedSubject], controls: ControlSet, preserve: [String]) {
        let world = ToyIdentityWorld.shared
        var controls = self.controls
        var preserve: [String] = []
        var out: [ProtectedSubject] = []
        for person in people {
            var photos = person.photos
            var names = lines(person.names), descriptions = lines(person.descriptions)
            if isToy && photos.isEmpty {
                photos = world.subjectPhotos
                if names.isEmpty { names = [ToyIdentityWorld.subjectName] }
                if descriptions.isEmpty { descriptions = [ToyIdentityWorld.subjectDescription] }
                if controls.photos.isEmpty { controls = world.controls }
                preserve.append(ToyIdentityWorld.attributePrompt)
            }
            out.append(ProtectedSubject(photos: photos, names: names, descriptions: descriptions, anchor: person.anchor,
                                        consent: Consent(basis: consentBasis)))
        }
        return (out, controls, preserve)
    }

    private func runProfile() -> VeilProfile {
        var p = isToy && profile == "standard" ? VeilProfile.toy : (VeilProfile.named(profile) ?? .standard)
        if harden { p = p.hardened() }
        return p
    }

    private func begin() -> (link: String, options: [String: String], adapter: String, guardURL: URL) {
        isRunning = true
        errorMessage = nil
        report = nil
        figure = nil
        guardURL = nil
        log = []
        phase = "Starting"
        let destination = SeedSchedule.defaultDirectory.appendingPathComponent("guards", isDirectory: true)
            .appendingPathComponent("guard-\(UUID().uuidString.prefix(8)).safetensors")
        return (modelLink.trimmingCharacters(in: .whitespaces), options,
                deploymentAdapter.trimmingCharacters(in: .whitespaces), destination)
    }

    private func progress() -> @Sendable (VeilProgress) -> Void {
        { p in
            Task { @MainActor in
                self.phase = "\(p.phase.rawValue) — \(p.message)"
                self.log.append(self.phase)
                if self.log.count > 200 { self.log.removeFirst(self.log.count - 200) }
            }
        }
    }

    private func finish(_ report: VeilReport, figure: NSImage?, guardURL: URL?) {
        self.report = report
        self.figure = figure
        self.guardURL = guardURL
        isRunning = false
        phase = "Done — \(report.verdict)"
    }

    private func failed(_ error: Error, partial: VeilReport?) {
        let message = (error as? VeilError) == .nothingToGuard
            ? "Nothing reached the person at this budget, so there is nothing to block." : error.localizedDescription
        report = partial
        errorMessage = error is CancellationError ? "Cancelled." : message
        isRunning = false
        phase = ""
    }

    /// Assess, then optionally fit, export and verify — one person.
    func start(guarding: Bool) {
        guard canRun else {
            if !consentAttested { errorMessage = "Confirm the consent attestation first." }
            return
        }
        let (link, options, adapter, guardDestination) = begin()
        let (subjects, controls, preserve) = subjects()
        guard let subject = subjects.first else { return }
        let isToy = self.isToy
        let request = VeilRequest(subject: subject, controls: controls, profile: runProfile(),
                                  faceWeighting: isToy ? .off : .auto, search: search, method: method, format: format,
                                  preservePrompts: preserve, withAdapter: adapter.isEmpty ? nil : adapter)
        let say = progress()

        task = Task.detached(priority: .userInitiated) {
            var run: VeilRun?
            do {
                let fetcher = RangeFetcher()
                let adapterURL = adapter.isEmpty ? nil : try await AdapterSource.fetch(adapter, fetcher: fetcher)
                say(VeilProgress(phase: .preparing, message: "loading \(link)"))
                let model = try await ExecutorRegistry.shared.load(link, options: options, withAdapter: adapterURL, fetcher: fetcher) {
                    say(VeilProgress(phase: .preparing, message: $0))
                }
                let schedule = isToy ? SeedSchedule.research : ((try? SeedSchedule.load()) ?? .ephemeral())
                let r = VeilRun(model: model, request: request, schedule: schedule)
                run = r
                r.onProgress = say
                try r.prepare()
                try r.assess()
                if guarding {
                    try r.fitGuard()
                    try r.export(to: guardDestination)
                    try r.verify(guardAt: guardDestination)
                }
                let report = r.report()
                let image = r.figure().map { NSImage(cgImage: $0, size: NSSize(width: $0.width, height: $0.height)) }
                await MainActor.run { self.finish(report, figure: image, guardURL: guarding ? guardDestination : nil) }
            } catch {
                let partial = run.map { $0.report() }
                await MainActor.run { self.failed(error, partial: partial) }
            }
        }
    }

    /// Straight to the guard, for one person or a group; nothing assesses the model.
    func startProtect() {
        guard canProtect else {
            if !consentAttested { errorMessage = "Confirm the consent attestation first." }
            else if people.count > 1 { errorMessage = "“I am this person” cannot attest for a group — switch to “I act for them”." }
            return
        }
        let (link, options, adapter, guardDestination) = begin()
        let (subjects, controls, preserve) = subjects()
        let isToy = self.isToy
        let request = ProtectRequest(people: subjects, controls: controls, profile: runProfile(),
                                     faceWeighting: isToy ? .off : .auto, method: method, format: format,
                                     check: quickCheck ? .quick : .none, preservePrompts: preserve,
                                     withAdapter: adapter.isEmpty ? nil : adapter)
        let say = progress()

        task = Task.detached(priority: .userInitiated) {
            var run: ProtectRun?
            do {
                let fetcher = RangeFetcher()
                let adapterURL = adapter.isEmpty ? nil : try await AdapterSource.fetch(adapter, fetcher: fetcher)
                say(VeilProgress(phase: .preparing, message: "loading \(link)"))
                let model = try await ExecutorRegistry.shared.load(link, options: options, withAdapter: adapterURL, fetcher: fetcher) {
                    say(VeilProgress(phase: .preparing, message: $0))
                }
                let schedule = isToy ? SeedSchedule.research : ((try? SeedSchedule.load()) ?? .ephemeral())
                let r = ProtectRun(model: model, request: request, schedule: schedule)
                run = r
                r.onProgress = say
                try r.prepare()
                try r.fitGuard()
                try r.export(to: guardDestination)
                try r.check(guardAt: guardDestination)
                let report = r.report()
                await MainActor.run { self.finish(report, figure: nil, guardURL: guardDestination) }
            } catch {
                let partial = run.map { $0.report() }
                await MainActor.run { self.failed(error, partial: partial) }
            }
        }
    }

    func cancel() {
        task?.cancel()
        phase = "Cancelling at the next stage boundary…"
    }

    // MARK: Export

    private var reportName: String { report?.subjects.first?.id.prefix(8).description ?? "" }

    func exportReport() {
        guard let report else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "veil-report-\(reportName).json"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try report.write(to: url) } catch { errorMessage = error.localizedDescription }
    }

    func exportGuard() {
        guard let guardURL else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType(filenameExtension: "safetensors") ?? .data]
        panel.nameFieldStringValue = "veil-guard-\(reportName).safetensors"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            if FileManager.default.fileExists(atPath: url.path) { try FileManager.default.removeItem(at: url) }
            try FileManager.default.copyItem(at: guardURL, to: url)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func exportFigure() {
        guard let figure, let cg = figure.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "veil-pull-\(reportName).png"
        panel.message = "The figure shows the person's photos. Keep it private."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try ImageWriter.writePNG(cg, to: url) } catch { errorMessage = error.localizedDescription }
    }

    func forgetSubjects() {
        guard let ids = report?.subjects.map(\.id), !ids.isEmpty else { return }
        do {
            for id in ids { _ = try SubjectStore().forget(id) }
            phase = "Forgot everything held for \(ids.count) subject(s)."
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

extension VeilError: Equatable {
    public static func == (a: VeilError, b: VeilError) -> Bool { a.localizedDescription == b.localizedDescription }
}
