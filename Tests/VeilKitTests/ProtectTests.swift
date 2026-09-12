//
//  ProtectTests.swift
//  VeilKitTests
//
//  Block without find, on the toy world where the answer is known. The world knows all four
//  identities by name, so a group guard can be held to the property that matters: the people it
//  covers are suppressed, and the people it doesn't are untouched — Cleo Marsh especially, who
//  shares Ada Quill's red hair.
//

import Foundation
import MLX
import Testing
@testable import VeilKit

/// Controls made of named toy identities (the world's own `controls` covers everyone but the
/// subject, which is the wrong set when the guard protects more than one of them).
func toyControls(_ names: [String]) -> ControlSet {
    let world = ToyIdentityWorld.shared
    var photos: [SubjectPhoto] = [], identities: [String?] = []
    for name in names {
        let p = world.photos(of: name)
        photos += p
        identities += Array(repeating: name, count: p.count)
    }
    return ControlSet(photos: photos, identities: identities)
}

func toyProtect(names: [String], controls: [String], check: ProtectCheck = .quick, method: GuardMethod = .both,
                profile: VeilProfile = .toy) throws -> ProtectRun {
    let world = ToyIdentityWorld.shared
    let people = names.map {
        ProtectedSubject(photos: world.photos(of: $0), names: [$0], anchor: "a person", consent: Consent(basis: .representative))
    }
    let request = ProtectRequest(people: people, controls: toyControls(controls), profile: profile, faceWeighting: .off,
                                 method: method, check: check, preservePrompts: [ToyIdentityWorld.attributePrompt])
    let run = ProtectRun(model: ToyModel(), request: request, schedule: .research,
                         store: SubjectStore(root: scratchURL("protect-store")))
    try run.prepare()
    return run
}

@Suite(.serialized) struct ProtectTests {
    @Test func protectsTwoPeopleAtOnceAndLeavesTheOthersAlone() throws {
        let run = try toyProtect(names: ["Ada Quill", "Bram Oake"], controls: ["Cleo Marsh", "Dov Reyes"], check: .full)
        let fit = try run.fitGuard()
        #expect(fit.training?.people == 2, "the erase pool covers both people")
        let url = scratchURL("group.safetensors")
        try run.export(to: url)
        try run.check(guardAt: url)

        #expect(run.verifications.count == 2)
        for pv in run.verifications {
            let v = pv.verification
            for s in v.suppression {
                print(String(format: "%@ %@: spec %.3f → %.3f p %.2f %@", pv.name ?? "?", s.routeID, s.base.specificity,
                             s.guarded.specificity, s.guarded.pValue ?? -1, s.pass ? "pass" : "FAIL"))
            }
            for n in v.preservation.namedControls {
                print(String(format: "  %@ keeps %.0f%% of their own pull (judged %@)", n.name, 100 * n.retained,
                             n.judged ? "yes" : "no"))
            }
            print("  verdict \(v.verdict.rawValue) drift \(v.preservation.maxDrift) reasons \(v.reasons)")
            #expect(v.suppression.allSatisfy { $0.pass }, "\(pv.name ?? "?") is still reachable")
            #expect(v.preservation.namedControls.allSatisfy { $0.pass }, "a control lost their own name")
        }
        let report = run.report()
        #expect(report.mode == "protect" && report.subjects.count == 2)
        #expect(report.assessment == nil, "nothing was assessed")
        #expect(report.verdict == GuardVerdict.guarded.rawValue, "verdict \(report.verdict): \(report.reasons)")
        #expect(report.issues.contains { $0.contains("without assessing") })

        // The guard's metadata says what it covers and that it was never assessed.
        let file = try AdapterFile.load(url)
        #expect(file.metadata["veil.people"] == "2" && file.metadata["veil.assessed"] == "no")
        #expect(file.metadata["veil.subject_ids"]?.split(separator: ",").count == 2)
    }

    @Test func theAssessedPathAndProtectFitTheSameClosedForm() throws {
        // One person, names only: both paths build the same plan, so the edit must be the same.
        let world = ToyIdentityWorld.shared
        var profile = VeilProfile.toy
        profile.trainSteps = 0
        let subject = ProtectedSubject(photos: world.subjectPhotos, names: [ToyIdentityWorld.subjectName], anchor: "a person",
                                       consent: Consent(basis: .selfAttested))
        let assessed = VeilRun(model: ToyModel(),
                               request: VeilRequest(subject: subject, controls: world.controls, profile: profile,
                                                    faceWeighting: .off, method: .closedForm, forceGuard: true),
                               schedule: .research, store: SubjectStore(root: scratchURL("assessed-store")))
        try assessed.prepare()
        try assessed.assess()
        let a = try assessed.fitGuard()

        let direct = ProtectRun(model: ToyModel(),
                                request: ProtectRequest(people: [subject], controls: world.controls, profile: profile,
                                                        faceWeighting: .off, method: .closedForm, check: .none),
                                schedule: .research, store: SubjectStore(root: scratchURL("direct-store")))
        try direct.prepare()
        let b = try direct.fitGuard()

        #expect(Set(a.deltas.keys) == Set(b.deltas.keys))
        #expect(a.erased == b.erased)
        for (key, delta) in a.deltas {
            let other = try #require(b.deltas[key])
            let error = abs(delta.dense - other.dense).max().item(Float.self)
            let scale = abs(delta.dense).max().item(Float.self)
            #expect(error / max(scale, 1e-12) < 1e-3, "\(key): the two paths disagree by \(error)")
        }
    }

    @Test func theQuickCheckSeesTheDropAndSaysWhatItIsNot() throws {
        let run = try toyProtect(names: ["Ada Quill"], controls: ["Cleo Marsh", "Dov Reyes"])
        try run.fitGuard()
        let url = scratchURL("one.safetensors")
        let file = try run.export(to: url)
        try run.check(guardAt: url)

        let check = try #require(run.quickCheck)
        let person = try #require(check.people.first)
        print(String(format: "quick check: pull %.3f → %.3f (residual %.2f), drift %.4f", person.basePull.mean,
                     person.guardedPull.mean, person.residual, check.maxDrift))
        #expect(check.guardSHA256 == file.sha256, "measured on the exported file")
        #expect(person.baseKnew, "the toy model knows her by name")
        #expect(person.residual < 0.25 && person.pass)
        #expect(check.maxDrift <= 0.02 && check.pass)

        let report = run.report()
        #expect(report.verdict == "quick-checked")
        #expect(report.checks?.quick != nil && report.verification == nil && report.checks?.verifications == nil)
        #expect(report.reasons.contains { $0.contains("not calibrated") }, "the caveat travels with the result")
    }

    @Test func withoutACheckTheReportSaysNothingWasMeasured() throws {
        let run = try toyProtect(names: ["Ada Quill"], controls: ["Cleo Marsh"], check: .none, method: .closedForm)
        try run.fitGuard()
        try run.export(to: scratchURL("unchecked.safetensors"))
        let report = run.report()
        #expect(report.verdict == "unverified")
        #expect(report.checks == nil)
        #expect(report.reasons.contains { $0.contains("veil verify") })
    }

    @Test func validationRefusesInputAGuardCannotCover() throws {
        let world = ToyIdentityWorld.shared
        func run(_ people: [ProtectedSubject]) -> ProtectRun {
            ProtectRun(model: ToyModel(),
                       request: ProtectRequest(people: people, profile: .toy, faceWeighting: .off, check: .none),
                       schedule: .research, store: SubjectStore(root: scratchURL("validation-store")))
        }
        let noPhotos = run([ProtectedSubject(photos: [], names: ["Ada Quill"], consent: Consent(basis: .representative))])
        #expect(throws: ProtectError.self) { try noPhotos.prepare() }

        let noName = run([ProtectedSubject(photos: world.subjectPhotos, names: [], consent: Consent(basis: .representative))])
        #expect(throws: ProtectError.self) { try noName.prepare() }

        // "I am this person" cannot attest for someone else.
        let group = run([ProtectedSubject(photos: world.subjectPhotos, names: ["Ada Quill"], consent: Consent(basis: .selfAttested)),
                         ProtectedSubject(photos: world.photos(of: "Bram Oake"), names: ["Bram Oake"], consent: Consent(basis: .selfAttested))])
        #expect(throws: ProtectError.self) { try group.prepare() }
    }

    @Test func aFolderPerPersonBecomesTheGroup() throws {
        let world = ToyIdentityWorld.shared
        let directory = scratchURL("people")
        let ada = directory.appendingPathComponent("Ada Quill")
        try FileManager.default.createDirectory(at: ada, withIntermediateDirectories: true)
        for photo in world.subjectPhotos { try ImageWriter.writePNG(photo.image, to: ada.appendingPathComponent(photo.name)) }
        let bram = directory.appendingPathComponent("Bram Oake")
        try FileManager.default.createDirectory(at: bram, withIntermediateDirectories: true)
        for photo in world.photos(of: "Bram Oake") { try ImageWriter.writePNG(photo.image, to: bram.appendingPathComponent(photo.name)) }
        try Data(#"{"names": ["A. Quill"], "describe": ["the red-haired painter from lisbon"], "anchor": "a woman"}"#.utf8)
            .write(to: ada.appendingPathComponent("person.json"))

        let people = try ProtectedSubject.loadPeople(directory, consent: Consent(basis: .representative))
        #expect(people.count == 2)
        #expect(people[0].names == ["Ada Quill", "A. Quill"] && people[0].anchor == "a woman")
        #expect(people[0].descriptions == [ToyIdentityWorld.subjectDescription])
        #expect(people[1].names == ["Bram Oake"] && people[1].anchor == "a person")
        #expect(people.allSatisfy { !$0.photos.isEmpty })

        // A loose image has no name attached, so it is refused rather than guessed at.
        try ImageWriter.writePNG(world.subjectPhotos[0].image, to: directory.appendingPathComponent("loose.png"))
        #expect(throws: SubjectPhotoError.self) {
            try ProtectedSubject.loadPeople(directory, consent: Consent(basis: .representative))
        }
    }
}
