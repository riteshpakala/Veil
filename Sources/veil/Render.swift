//
//  Render.swift
//  veil
//
//  Plain-text summaries of descriptors and reports.
//

import Foundation
import VeilFlux2
import VeilKit

enum Registration {
    static func registerAll() { Flux2Registration.register() }
}

enum Render {
    static func pad(_ s: String, _ n: Int) -> String { s.count >= n ? s : s + String(repeating: " ", count: n - s.count) }
    static func lpad(_ s: String, _ n: Int) -> String { s.count >= n ? s : String(repeating: " ", count: n - s.count) + s }
    static func f(_ x: Double, _ digits: Int = 3) -> String { String(format: "%.\(digits)f", x) }
    static func pct(_ x: Double) -> String { String(format: "%.1f%%", 100 * x) }

    static func descriptor(_ d: ModelDescriptor, source: ResolvedSource?, families: [String]) -> String {
        var lines = ["\(d.pinnedName)"]
        if let via = d.via { lines.append("  via       \(via)") }
        lines.append("  family    \(d.family)  (\(d.detail))")
        if let v = d.variant { lines.append("  variant   \(v)") }
        if let s = d.size { lines.append("  size      \(s)") }
        if let l = d.license { lines.append("  license   \(l)") }
        if !d.components.isEmpty { lines.append("  parts     \(d.components.joined(separator: ", "))") }
        if let source { lines.append("  files     \(source.files.count), \(ByteFormat.string(source.weightBytes)) of weights") }
        lines.append(d.supported ? "  support   yes — Veil can assess and guard this model"
                                 : "  support   no executor for \(d.family) yet (supported: \(families.filter { $0 != "toy" }.joined(separator: ", ")))")
        return lines.joined(separator: "\n")
    }

    static func routeLabel(_ m: RouteMeasurement) -> String {
        let text = m.label.map { "“\($0)”" } ?? "#\(m.labelHash)"
        return "\(pad(m.kind.rawValue, 11)) \(text.count > 40 ? String(text.prefix(39)) + "…" : text)"
    }

    /// The verification block, used for a single-person run and for each member of a group.
    static func verification(_ v: Verification) -> [String] {
        var lines: [String] = []
        for s in v.suppression {
            lines.append("  \(pad(routeLabel(s.base), 52)) spec \(f(s.base.specificity)) → \(f(s.guarded.specificity))  p \(s.guarded.pValue.map { f($0, 2) } ?? "–")  \(s.pass ? "suppressed" : "STILL REACHES")")
        }
        for rb in v.robustness {
            let max = rb.guarded.budgets.max() ?? 0
            lines.append("  soft attack from \(rb.base.start): crosses the null after \(rb.base.reachedAt.map { "\($0)" } ?? "> \(max)") steps on the base, \(rb.guarded.reachedAt.map { "\($0)" } ?? "> \(max)") guarded")
        }
        let p = v.preservation
        lines.append("  preservation: largest drift \(pct(p.maxDrift)) (tolerance \(pct(p.tolerance)))" + (p.namedControls.isEmpty ? ""
            : "; own-name pull kept: " + p.namedControls.map { "\($0.name) \(pct($0.retained))\($0.judged ? "" : "*")" }.joined(separator: ", ")))
        return lines
    }

    static func summary(_ r: VeilReport) -> String {
        var lines: [String] = []
        let variant = [r.model.variant, r.model.size].compactMap { $0 }.joined(separator: ", ")
        lines.append("Veil \(r.version) · \(r.mode) · \(r.model.family) · \(r.model.pinnedName)\(variant.isEmpty ? "" : " (\(variant))")")
        for s in r.subjects {
            let faces = s.photos.filter(\.faceFound).count
            let who = s.names?.first ?? "#\(s.nameHashes.first ?? "")"
            lines.append("\(pad(who, 28)) \(s.id) · \(s.split.fit.count) fit / \(s.photos.count - s.split.fit.count) held-out\(s.split.augmented ? " (augmented)" : "") · faces \(faces)/\(s.photos.count) · consent \(s.consent.basis.rawValue)")
        }
        let c = r.controls
        lines.append(c.fallbackSamples ? "Controls: none given — \(c.count) model samples of generic people stand in"
                                       : "Controls: \(c.count) photos\(c.identities.isEmpty ? "" : ", identities: " + c.identities.joined(separator: ", "))")

        if let a = r.assessment {
            lines.append("")
            lines.append("  \(pad("route", 52)) \(lpad("pull", 7)) \(lpad("rel", 7))  \(pad("nearest control", 18)) \(lpad("spec", 7)) \(lpad("p", 5))  reaches")
            for m in a.routes {
                lines.append("  \(pad(routeLabel(m), 52)) \(lpad(f(m.subject.mean), 7)) \(lpad(pct(m.subject.relative), 7))  \(pad(String((m.nearestControl ?? "–").prefix(18)), 18)) \(lpad(f(m.specificity), 7)) \(lpad(m.pValue.map { f($0, 2) } ?? "–", 5))  \(m.reaches ? "yes" : "no")")
            }
            lines.append("  null: \(a.null.count) invented names, 95th percentile spec \(f(a.null.q95)) (single prompt \(f(a.null.q95Single)))")
            lines.append("  capability: \(a.capability.rawValue)")
            if !a.locator.isEmpty {
                lines.append("  where the strongest route lives: " + a.locator.prefix(4).map { "\($0.key) \(pct($0.restoredShare))" }.joined(separator: " · "))
            }
        }
        if let g = r.guardFit {
            lines.append("")
            var parts: [String] = []
            if let cf = g.closedForm {
                parts.append("closed form on \(cf.slot) (rank \(cf.rank), \(pct(cf.energy)) energy, truncation \(pct(cf.truncationError)))")
            }
            if let t = g.training {
                parts.append("trained \(t.slots.count) text-path slots × \(t.steps) steps over \(t.people) person(s) (erase loss \(f(t.eraseLossStart, 4)) → \(f(t.eraseLossEnd, 4)), hardening \(t.hardenings))")
            }
            lines.append("Guard: " + (parts.isEmpty ? g.method.rawValue : parts.joined(separator: " + ")))
            if let file = g.file {
                lines.append("  file \(file.path) · \(file.format) · rank ≤ \(file.rank) · \(ByteFormat.string(file.bytes)) · sha256 \(file.sha256.prefix(16))…")
            }
        }
        if let v = r.verification {
            lines.append("")
            lines.append("Verification (the exported file, read back; independent draws):")
            lines += verification(v)
        }
        if let q = r.checks?.quick {
            lines.append("")
            lines.append("Quick check (the exported file, read back; held-out photos):")
            for p in q.people {
                let who = p.name.map { "“\($0)”" } ?? "#\(p.nameHash)"
                let state = p.baseKnew ? (p.pass ? "ok" : "STILL PULLS") : "base pull already within noise"
                lines.append("  \(pad(who, 32)) pull \(lpad(f(p.basePull.mean), 7)) → \(lpad(f(p.guardedPull.mean), 7))  \(lpad(pct(max(p.residual, 0)), 7)) left  \(state)")
            }
            lines.append(String(format: "  drift on everyday prompts: largest %.2f%% (tolerance %.0f%%)",
                                100 * q.maxDrift, 100 * q.tolerance))
        }
        for pv in r.checks?.verifications ?? [] {
            lines.append("")
            lines.append("Verification — \(pv.name.map { "“\($0)”" } ?? "#\(pv.nameHash)") (the exported file, read back):")
            lines += verification(pv.verification)
        }
        lines.append("")
        lines.append("Verdict: \(r.verdict.uppercased())")
        for reason in r.reasons { lines.append("  · \(reason)") }
        if !r.issues.isEmpty {
            lines.append("Issues:")
            for i in r.issues { lines.append("  · \(i)") }
        }
        lines.append(String(format: "Cost: %d denoiser evaluations, %.0f s", r.evaluations, r.seconds))
        return lines.joined(separator: "\n")
    }
}
