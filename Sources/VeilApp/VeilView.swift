//
//  VeilView.swift
//  VeilApp
//
//  Inputs on the left (model, the people to protect, controls, consent), results on the right
//  (reachability, where the route pulls, the guard, what was measured). Nothing here generates an
//  image of anyone: the only faces on screen are the photos the user added.
//

import AppKit
import SwiftUI
import UniformTypeIdentifiers
import VeilKit

struct VeilView: View {
    @ObservedObject var model: VeilViewModel

    var body: some View {
        HSplitView {
            InputsPane(model: model)
                .frame(minWidth: 340, idealWidth: 380, maxWidth: 460)
            ResultsPane(model: model)
                .frame(minWidth: 520)
        }
    }
}

// MARK: - Inputs

struct InputsPane: View {
    @ObservedObject var model: VeilViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                section("Model") {
                    TextField("Hugging Face / GitHub link, local folder, or “toy”", text: $model.modelLink)
                        .textFieldStyle(.roundedBorder)
                    HStack {
                        Button("Inspect") { model.inspect() }
                        Button("Toy world") { model.modelLink = "toy" }
                        Spacer()
                    }
                    if let d = model.descriptor {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(d.pinnedName).font(.callout.weight(.semibold))
                            Text("\(d.family) · \(d.detail)\(d.variant.map { " · \($0)" } ?? "")").font(.caption).foregroundStyle(.secondary)
                            Text(d.supported ? "Supported" : "No executor for \(d.family) yet")
                                .font(.caption.weight(.semibold)).foregroundStyle(d.supported ? .green : .orange)
                        }
                    }
                    TextField("Executor options, e.g. text-encoder=mflux", text: $model.executorOptions)
                        .textFieldStyle(.roundedBorder).font(.caption)
                    TextField("Deployment adapter (optional path or link)", text: $model.deploymentAdapter)
                        .textFieldStyle(.roundedBorder).font(.caption)
                }

                section(model.people.count > 1 ? "The people to protect" : "The person to protect") {
                    ForEach(model.people) { person in
                        PersonCard(model: model, person: person)
                    }
                    HStack {
                        Button("Add Person") { model.addPerson() }
                        Spacer()
                    }
                    if model.people.count > 1 {
                        Text("One guard for all of them. A group goes through Build Guard: the assessment measures one person at a time.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }

                section("Controls") {
                    HStack {
                        Button("Choose Folder…") { model.chooseControls() }
                        if model.controlsFolder != nil { Button("Clear") { model.clearControls() } }
                        Spacer()
                    }
                    if let folder = model.controlsFolder {
                        let named = model.controls.named.keys.sorted()
                        Text("\(folder.lastPathComponent): \(model.controls.photos.count) photos" + (named.isEmpty ? "" : " · " + named.joined(separator: ", ")))
                            .font(.caption).foregroundStyle(.secondary)
                    } else {
                        Text("Photos of other people, look-alikes welcome. Without them, the model's own samples stand in (weaker).")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }

                section("Consent") {
                    Picker("", selection: $model.consentBasis) {
                        Text("I am this person").tag(Consent.Basis.selfAttested)
                        Text("I act for them").tag(Consent.Basis.representative)
                    }
                    .pickerStyle(.segmented).labelsHidden()
                    Toggle(isOn: $model.consentAttested) {
                        Text(Consent(basis: model.consentBasis).statement).font(.caption)
                    }
                    if model.people.count > 1 && model.consentBasis == .selfAttested {
                        Text("“I am this person” cannot attest for a group.").font(.caption).foregroundStyle(.orange)
                    }
                }

                section("Run") {
                    Picker("Profile", selection: $model.profile) {
                        ForEach(["quick", "standard", "thorough"], id: \.self) { Text($0).tag($0) }
                    }
                    Picker("Guard", selection: $model.method) {
                        Text("Closed form + trained").tag(GuardMethod.both)
                        Text("Closed form only").tag(GuardMethod.closedForm)
                        Text("Trained only").tag(GuardMethod.trained)
                    }
                    Picker("Format", selection: $model.format) {
                        Text("diffusers / PEFT").tag("diffusers")
                        Text("BFL / ComfyUI").tag("bfl")
                    }
                    Toggle("Search for prompts that don't name them", isOn: $model.search)
                        .disabled(model.people.count > 1)
                    Toggle("Harden against soft attacks (experimental)", isOn: $model.harden)
                    Toggle("Check the guard after building it", isOn: $model.quickCheck)
                    HStack {
                        Button("Assess") { model.start(guarding: false) }
                            .disabled(!model.canAssess)
                        Button("Assess + Guard") { model.start(guarding: true) }
                            .disabled(!model.canAssess)
                        if model.isRunning { Button("Cancel", role: .cancel) { model.cancel() } }
                        Spacer()
                    }
                    HStack {
                        Button("Build Guard (skip assessment)") { model.startProtect() }
                            .keyboardShortcut(.defaultAction)
                            .disabled(!model.canProtect)
                        Spacer()
                    }
                    Text("Assess measures what the model can reach, then blocks it. Build Guard goes straight to the guard — faster, and it says nothing about what the model could reach.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(18)
        }
    }

    @ViewBuilder
    func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased()).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            content()
        }
    }
}

struct PersonCard: View {
    @ObservedObject var model: VeilViewModel
    let person: PersonInput

    private var binding: Binding<PersonInput>? {
        guard let i = model.index(of: person) else { return nil }
        return $model.people[i]
    }

    var body: some View {
        if let person = binding {
            VStack(alignment: .leading, spacing: 6) {
                PhotoStrip(model: model, person: person.wrappedValue)
                TextField("Names and aliases (comma or newline separated)", text: person.names, axis: .vertical)
                    .textFieldStyle(.roundedBorder).lineLimit(1...3)
                TextField("Descriptions that might reach them (one per line)", text: person.descriptions, axis: .vertical)
                    .textFieldStyle(.roundedBorder).lineLimit(1...4)
                HStack {
                    Picker("Anchor", selection: person.anchor) {
                        ForEach(VeilViewModel.anchors, id: \.self) { Text($0).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    if model.people.count > 1 {
                        Button(role: .destructive) { model.removePerson(person.wrappedValue) } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.borderless)
                    }
                }
            }
            .padding(10)
            .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
        }
    }
}

struct PhotoStrip: View {
    @ObservedObject var model: VeilViewModel
    let person: PersonInput
    @State private var targeted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ScrollView(.horizontal) {
                HStack(spacing: 6) {
                    ForEach(person.photos, id: \.id) { photo in
                        Image(nsImage: NSImage(cgImage: photo.image, size: NSSize(width: photo.width, height: photo.height)))
                            .resizable().scaledToFill().frame(width: 64, height: 64).clipped()
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            .contextMenu { Button("Remove") { model.removePhoto(photo, from: person) } }
                    }
                    Button { model.choosePhotos(for: person) } label: {
                        RoundedRectangle(cornerRadius: 6).strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [4]))
                            .frame(width: 64, height: 64).overlay(Image(systemName: "plus"))
                    }
                    .buttonStyle(.plain)
                }
            }
            .frame(height: 70)
            .background(targeted ? Color.accentColor.opacity(0.12) : .clear)
            .onDrop(of: [.fileURL], isTargeted: $targeted) { providers in
                for p in providers {
                    _ = p.loadObject(ofClass: URL.self) { url, _ in
                        if let url { Task { @MainActor in model.addPhotos([url], to: person) } }
                    }
                }
                return true
            }
            Text(person.photos.isEmpty ? "Drop photos here — 5 to 20, different days, light and angles."
                                       : "\(person.photos.count) photo(s). They stay on this Mac; reports carry only their hashes.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

// MARK: - Results

struct ResultsPane: View {
    @ObservedObject var model: VeilViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if model.isRunning || !model.phase.isEmpty {
                    HStack(spacing: 8) {
                        if model.isRunning { ProgressView().controlSize(.small) }
                        Text(model.phase).font(.callout).foregroundStyle(.secondary).lineLimit(2)
                    }
                }
                if let error = model.errorMessage {
                    Text(error).foregroundStyle(.red).textSelection(.enabled)
                }
                if let r = model.report {
                    VerdictHeader(report: r)
                    if let a = r.assessment { ReachabilityCard(assessment: a) }
                    if let figure = model.figure {
                        Card(title: "Where the strongest route pulls") {
                            Image(nsImage: figure).resizable().scaledToFit()
                            Text("Relative pull of the route over its anchor on held-out photos, fixed scale. Contains the person's photos — keep it private.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    if let g = r.guardFit { GuardCard(record: g) }
                    if let v = r.verification { VerificationCard(title: "Verification — the exported file, read back", verification: v) }
                    if let q = r.checks?.quick { QuickCheckCard(check: q) }
                    ForEach(r.checks?.verifications ?? [], id: \.subjectID) { pv in
                        VerificationCard(title: "Verification — \(pv.name ?? "#" + pv.nameHash)", verification: pv.verification)
                    }
                    IssuesCard(report: r)
                    HStack {
                        Button("Save Guard…") { model.exportGuard() }.disabled(model.guardURL == nil)
                        Button("Save Report…") { model.exportReport() }
                        Button("Save Figure…") { model.exportFigure() }.disabled(model.figure == nil)
                        Spacer()
                        Button(r.subjects.count > 1 ? "Forget These People" : "Forget This Person") { model.forgetSubjects() }
                    }
                } else if !model.isRunning {
                    EmptyState()
                }
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

struct EmptyState: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Veil").font(.largeTitle.weight(.semibold))
            Text("Find the prompts that reach a protected person in a model, and fit a guard LoRA that closes them.")
                .font(.title3).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 6) {
                Label("Find — which names, descriptions and searched prompts pull the model toward the person, beyond invented names.", systemImage: "magnifyingglass")
                Label("Block — a closed-form edit of the text input, then a trained text-path LoRA anchored on their photos.", systemImage: "shield")
                Label("Verify — the exported file, read back: suppression, attack cost, and drift on everyone else.", systemImage: "checkmark.seal")
                Label("Or skip the find: Build Guard blocks one person or a group straight away, then checks what it did.", systemImage: "person.2.slash")
            }
            .font(.callout)
            Text("A guard blocks routes, not the likeness itself, and binds only where the deployer applies it.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(.top, 40)
    }
}

struct Card<Content: View>: View {
    let title: String
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.headline)
            content()
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Color.secondary.opacity(0.15)))
    }
}

struct VerdictHeader: View {
    let report: VeilReport

    var color: Color {
        switch report.verdict {
        case "guarded", "not-reachable-at-budget", "not-needed", "quick-checked": return .green
        case "partial", "quick-partial", "name-bound", "description-reachable", "search-reachable": return .orange
        default: return .secondary
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text(report.verdict.uppercased()).font(.title2.weight(.bold)).foregroundStyle(color)
                Spacer()
                Text(report.model.pinnedName).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }
            ForEach(report.reasons, id: \.self) { Text("· " + $0).font(.callout) }
            ForEach(report.subjects, id: \.id) { s in
                Text("\(s.names?.first ?? "#" + (s.nameHashes.first ?? "")) · \(s.id) · \(s.split.fit.count) fit / \(s.photos.count - s.split.fit.count) held-out · faces \(s.photos.filter(\.faceFound).count)/\(s.photos.count) · consent \(s.consent.basis.rawValue)")
                    .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }
        }
    }
}

struct ReachabilityCard: View {
    let assessment: AssessmentRecord

    var body: some View {
        Card(title: "Reachability — \(assessment.capability.rawValue)") {
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 4) {
                GridRow {
                    ForEach(["Route", "Pull", "Nearest control", "Spec", "p", ""], id: \.self) {
                        Text($0).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                    }
                }
                ForEach(assessment.routes, id: \.id) { m in
                    GridRow {
                        Text("\(m.kind.rawValue): \(m.label ?? "#" + m.labelHash)").lineLimit(1).truncationMode(.tail)
                        Text(String(format: "%.1f%%", 100 * m.subject.relative)).monospacedDigit()
                        Text(m.nearestControl ?? "–").lineLimit(1)
                        Text(String(format: "%.3f", m.specificity)).monospacedDigit()
                        Text(m.pValue.map { String(format: "%.2f", $0) } ?? "–").monospacedDigit()
                        Text(m.reaches ? "reaches" : "—").foregroundStyle(m.reaches ? .orange : .secondary)
                    }
                    .font(.callout)
                }
            }
            Text(String(format: "Null: %d invented names, 95th percentile %.3f. Pull is decode-free: how much the prompt helps the model denoise the person's own held-out photos.",
                        assessment.null.count, assessment.null.q95))
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

struct GuardCard: View {
    let record: GuardRecord

    var body: some View {
        Card(title: "Guard") {
            if let cf = record.closedForm {
                Text(String(format: "Closed form on %@ — rank %d, %.1f%% of the edit's energy, truncation %.2f%%", cf.slot, cf.rank,
                            100 * cf.energy, 100 * cf.truncationError)).font(.callout)
            }
            if let t = record.training {
                Text(String(format: "Trained %d text-path slots × %d steps over %d person(s) — erase loss %.4f → %.4f, preserve %.4f → %.4f, %d hardening rounds",
                            t.slots.count, t.steps, t.people, t.eraseLossStart, t.eraseLossEnd, t.preserveLossStart, t.preserveLossEnd, t.hardenings))
                    .font(.callout)
            }
            if let f = record.file {
                Text("\(URL(fileURLWithPath: f.path).lastPathComponent) · \(f.format) · rank ≤ \(f.rank) · \(ByteFormat.string(f.bytes))")
                    .font(.callout)
                Text("sha256 \(f.sha256)").font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
            }
        }
    }
}

struct QuickCheckCard: View {
    let check: QuickCheck

    var body: some View {
        Card(title: "Quick check — the exported file, read back") {
            ForEach(check.people, id: \.subjectID) { p in
                HStack {
                    Image(systemName: p.pass ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                        .foregroundStyle(p.pass ? .green : .orange)
                    Text(p.name ?? "#" + p.nameHash).lineLimit(1)
                    Spacer()
                    Text(p.baseKnew ? String(format: "pull %.3f → %.3f (%.0f%% left)", p.basePull.mean, p.guardedPull.mean, 100 * max(p.residual, 0))
                                    : "base pull already within noise")
                        .monospacedDigit()
                }
                .font(.callout)
            }
            Text(String(format: "Drift on everyday prompts: largest %.2f%% (tolerance %.0f%%)", 100 * check.maxDrift, 100 * check.tolerance))
                .font(.callout).foregroundStyle(check.maxDrift <= check.tolerance ? Color.primary : Color.orange)
            Text(check.caveat).font(.caption).foregroundStyle(.secondary)
        }
    }
}

struct VerificationCard: View {
    let title: String
    let verification: Verification

    var body: some View {
        Card(title: title) {
            ForEach(verification.suppression, id: \.routeID) { s in
                HStack {
                    Image(systemName: s.pass ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                        .foregroundStyle(s.pass ? .green : .orange)
                    Text("\(s.kind.rawValue): \(s.label ?? "#" + s.labelHash)").lineLimit(1)
                    Spacer()
                    Text(String(format: "spec %.3f → %.3f", s.base.specificity, s.guarded.specificity)).monospacedDigit()
                }
                .font(.callout)
            }
            ForEach(verification.robustness, id: \.base.start) { r in
                let max = r.guarded.budgets.max() ?? 0
                Text("Soft attack from the \(r.base.start): crosses the null after \(r.base.reachedAt.map { "\($0)" } ?? "> \(max)") steps on the base, \(r.guarded.reachedAt.map { "\($0)" } ?? "> \(max)") guarded")
                    .font(.callout)
            }
            let p = verification.preservation
            Text(String(format: "Preservation: largest drift %.2f%% (tolerance %.0f%%)", 100 * p.maxDrift, 100 * p.tolerance))
                .font(.callout).foregroundStyle(p.pass ? Color.primary : Color.orange)
            ForEach(p.namedControls, id: \.name) { n in
                Text(String(format: "  %@ keeps %.0f%% of their own-name pull%@", n.name, 100 * n.retained, n.judged ? "" : " (not judged: the model doesn't recognize them)"))
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

struct IssuesCard: View {
    let report: VeilReport

    var body: some View {
        Card(title: "Issues and limits") {
            ForEach(report.issues, id: \.self) { Text("· " + $0).font(.callout).foregroundStyle(.orange) }
            ForEach(report.limits, id: \.self) { Text("· " + $0).font(.caption).foregroundStyle(.secondary) }
        }
    }
}
