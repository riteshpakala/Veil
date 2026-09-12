//
//  VeilReport.swift
//  VeilKit
//
//  The shareable record of a run (schema veil.guard.v1). Photos appear as SHA-256 hashes, never
//  pixels; discovered prompts appear as hashes unless the user opts in. Numbers are measured,
//  thresholds are stated, and the limits travel with the result.
//

import Foundation

public struct EnvironmentInfo: Codable, Sendable, Hashable {
    public let hardware: String
    public let os: String
    public let executableSHA256: String?

    public static func current() -> EnvironmentInfo {
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var model = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("hw.model", &model, &size, nil, 0)
        let exe = Bundle.main.executableURL.flatMap(Hashing.fileSHA256)
        return EnvironmentInfo(hardware: String(cString: model), os: ProcessInfo.processInfo.operatingSystemVersionString,
                               executableSHA256: exe)
    }
}

public struct PhotoRecord: Codable, Sendable, Hashable {
    public let sha256: String
    public let role: String            // "fit", "held-out", "held-out (augmented)", "control"
    public let identity: String?
    public let faceFound: Bool
    public let likeness: String
    public let faceArea: Double
}

public struct SubjectRecord: Codable, Sendable, Hashable {
    public let id: String
    public let names: [String]?
    public let nameHashes: [String]
    public let descriptions: Int
    public let anchor: String
    public let consent: Consent
    public let split: PhotoSplit
    public let photos: [PhotoRecord]
}

public struct ControlsRecord: Codable, Sendable, Hashable {
    public let count: Int
    public let identities: [String]
    /// No controls were given: base-model samples of generic people stood in (weaker).
    public let fallbackSamples: Bool
    public let photos: [PhotoRecord]
}

public struct AssessmentRecord: Codable, Sendable {
    public let routes: [RouteMeasurement]
    public let null: NullDistribution
    public let capability: CapabilityClass
    /// Discovered prompts (texts only when revealed).
    public let search: [SearchCandidateRecord]?
    public let locator: [LocatorSite]
}

public struct SearchCandidateRecord: Codable, Sendable, Hashable {
    public let hash: String
    public let text: String?
    public let fitLoss: Double
}

public struct ClosedFormSummary: Codable, Sendable, Hashable {
    public let slot: String
    public let rank: Int
    public let energy: Double
    public let truncationError: Double
    public let erasePairs: Int
    public let preserveVectors: Int
    public let relativeSize: Double
}

public struct GuardRecord: Codable, Sendable {
    public let method: GuardMethod
    public let erased: [String]
    public let closedForm: ClosedFormSummary?
    public let training: TrainingReport?
    public let file: GuardFile?
}

public struct PersonVerification: Codable, Sendable {
    public let subjectID: String
    public let name: String?
    public let nameHash: String
    public let verification: Verification

    public init(subjectID: String, name: String?, nameHash: String, verification: Verification) {
        self.subjectID = subjectID
        self.name = name
        self.nameHash = nameHash
        self.verification = verification
    }
}

/// What a `protect` run measured after export: nothing, the quick check, or the full verifier per
/// person.
public struct ProtectChecks: Codable, Sendable {
    public let quick: QuickCheck?
    public let verifications: [PersonVerification]?

    public init(quick: QuickCheck?, verifications: [PersonVerification]?) {
        self.quick = quick
        self.verifications = verifications
    }
}

public struct VeilReport: Codable, Sendable {
    public let schema: String
    public let version: String
    public let runID: String
    public let createdAt: Date
    public let environment: EnvironmentInfo
    /// "guard" — assessed, then blocked and verified; "protect" — blocked without assessing.
    public let mode: String
    public let model: ModelDescriptor
    public let withAdapter: String?
    /// Everyone this run protects: one person for `guard`, one or more for `protect`.
    public let subjects: [SubjectRecord]
    public let controls: ControlsRecord
    public let profile: VeilProfile
    public let schedule: ScheduleInfo
    public let assessment: AssessmentRecord?
    public let guardFit: GuardRecord?
    public let verification: Verification?
    public let checks: ProtectChecks?
    public let verdict: String
    public let reasons: [String]
    public let issues: [String]
    public let limits: [String]
    public let evaluations: Int
    public let seconds: Double

    public static let limits = [
        "A guard blocks routes, not the likeness itself: it covers the routes searched, at the stated attack budgets. Textual-inversion-style attacks recover erased concepts given enough freedom.",
        "It binds only where the deployer runs the model with the guard applied (APIs, hosting platforms, apps). On open weights it can be removed or fine-tuned away.",
        "Pull is a decode-free conditional-likelihood proxy measured on the person's own photos; no image was generated.",
        "Thresholds (α, drift tolerance, retained pull) are uncalibrated until a validation experiment with known answers has run.",
    ]

    public func write(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        encoder.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try encoder.encode(self).write(to: url, options: .atomic)
    }

    public static func read(from url: URL) throws -> VeilReport {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        decoder.nonConformingFloatDecodingStrategy = .convertFromString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        return try decoder.decode(VeilReport.self, from: Data(contentsOf: url))
    }
}
