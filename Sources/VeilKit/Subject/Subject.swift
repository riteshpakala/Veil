//
//  Subject.swift
//  VeilKit
//
//  The protected person, their consent, the controls, and the fit / held-out split.
//
//  Veil only runs with a consent attestation: the person themself, or someone authorized to
//  act for them. It is recorded (basis and time, no personal data) in the report and in the
//  guard's metadata. The photos never leave the machine and are never copied; reports carry
//  their SHA-256 hashes.
//

import CoreGraphics
import Foundation

public struct Consent: Codable, Sendable, Hashable {
    public enum Basis: String, Codable, Sendable, CaseIterable {
        /// The protected person runs Veil for themself.
        case selfAttested = "self"
        /// Someone authorized to act for the person (a representative, an estate, counsel).
        case representative
    }

    public let basis: Basis
    public let attestedAt: Date
    public let statement: String

    public init(basis: Basis, attestedAt: Date = Date()) {
        self.basis = basis
        self.attestedAt = attestedAt
        switch basis {
        case .selfAttested:
            statement = "I am the person in these photos and I ask for a guard that blocks generating my likeness."
        case .representative:
            statement = "I am authorized to act for the person in these photos, who asks for a guard that blocks generating their likeness."
        }
    }
}

public struct ProtectedSubject: Sendable {
    public var photos: [SubjectPhoto]
    /// Names the person is known by; the first is the primary name. May be empty (photos only).
    public var names: [String]
    /// Descriptions that might reach the person without naming them.
    public var descriptions: [String]
    /// The neutral phrase a name is replaced with ("a person", "a man", "a woman").
    public var anchor: String
    public var consent: Consent

    public init(photos: [SubjectPhoto], names: [String] = [], descriptions: [String] = [],
                anchor: String = "a person", consent: Consent) {
        self.photos = photos
        self.names = names.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        self.descriptions = descriptions.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        self.anchor = anchor
        self.consent = consent
    }

    /// Keyed id over the photo hashes: stable for the same photos and key, and reveals nothing
    /// about them to anyone without the key.
    public func id(schedule: SeedSchedule) -> String {
        schedule.keyedID("subject|" + photos.map(\.id).sorted().joined(separator: ","))
    }
}

extension ProtectedSubject {
    /// A group of protected people: one per immediate subfolder, the folder's name their primary
    /// name. An optional `person.json` inside a subfolder adds aliases, descriptions and the
    /// anchor:
    ///
    ///     { "names": ["Honest Abe"], "describe": ["the 16th president"], "anchor": "a man" }
    ///
    /// Images loose in `directory` are an error: a guard has to know whose likeness it closes.
    public static func loadPeople(_ directory: URL, consent: Consent) throws -> [ProtectedSubject] {
        struct Card: Decodable {
            var names: [String]?
            var describe: [String]?
            var descriptions: [String]?
            var anchor: String?
        }
        let items = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isDirectoryKey])
            .sorted { $0.lastPathComponent.naturalLess($1.lastPathComponent) }
        var people: [ProtectedSubject] = []
        var loose = 0
        for item in items {
            var isDir: ObjCBool = false
            FileManager.default.fileExists(atPath: item.path, isDirectory: &isDir)
            if isDir.boolValue {
                let photos = try SubjectPhoto.loadDirectory(item)
                guard !photos.isEmpty else { continue }
                let name = item.lastPathComponent
                let card = (try? Data(contentsOf: item.appendingPathComponent("person.json")))
                    .flatMap { try? JSONDecoder().decode(Card.self, from: $0) }
                let aliases = (card?.names ?? []).filter { $0 != name }
                people.append(ProtectedSubject(photos: photos, names: [name] + aliases,
                                               descriptions: card?.describe ?? card?.descriptions ?? [],
                                               anchor: card?.anchor ?? "a person", consent: consent))
            } else if SubjectPhoto.imageExtensions.contains(item.pathExtension.lowercased()) {
                loose += 1
            }
        }
        guard loose == 0 else { throw SubjectPhotoError.looseImages(directory.path, loose) }
        guard !people.isEmpty else { throw SubjectPhotoError.empty(directory.path) }
        return people
    }
}

/// Photos of other people, the measure of what the guard must leave alone. Files directly in the
/// folder are unnamed; a subfolder's name is the identity of the photos inside it (e.g.
/// `controls/Ulysses S. Grant/*.jpg`), which adds a "their own name still works" check.
public struct ControlSet: Sendable {
    public var photos: [SubjectPhoto]
    public var identities: [String?]

    public init(photos: [SubjectPhoto], identities: [String?]? = nil) {
        self.photos = photos
        self.identities = identities ?? Array(repeating: nil, count: photos.count)
    }

    public static let empty = ControlSet(photos: [])

    public static func load(_ directory: URL) throws -> ControlSet {
        var photos: [SubjectPhoto] = []
        var identities: [String?] = []
        let items = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isDirectoryKey])
            .sorted { $0.lastPathComponent.naturalLess($1.lastPathComponent) }
        for item in items {
            var isDir: ObjCBool = false
            FileManager.default.fileExists(atPath: item.path, isDirectory: &isDir)
            if isDir.boolValue {
                let inner = try SubjectPhoto.loadDirectory(item)
                photos += inner
                identities += Array(repeating: item.lastPathComponent, count: inner.count)
            } else if SubjectPhoto.imageExtensions.contains(item.pathExtension.lowercased()) {
                photos.append(try SubjectPhoto.load(url: item))
                identities.append(nil)
            }
        }
        if photos.isEmpty { throw SubjectPhotoError.empty(directory.path) }
        return ControlSet(photos: photos, identities: identities)
    }

    /// Identity name → indices of its photos.
    public var named: [String: [Int]] {
        var out: [String: [Int]] = [:]
        for (i, name) in identities.enumerated() { if let name { out[name, default: []].append(i) } }
        return out
    }

    /// At most `limit` photos, spread across identities (keyed order within each).
    public func limited(_ limit: Int, schedule: SeedSchedule) -> ControlSet {
        guard photos.count > limit else { return self }
        let order = photos.indices.sorted { schedule.value(.split, photos[$0].id) < schedule.value(.split, photos[$1].id) }
        var picked: [Int] = [], seen: Set<String> = []
        for i in order where !seen.contains(identities[i] ?? "#\(i)") {   // one per identity first
            picked.append(i)
            seen.insert(identities[i] ?? "#\(i)")
            if picked.count == limit { break }
        }
        for i in order where picked.count < limit && !picked.contains(i) { picked.append(i) }
        picked.sort()
        return ControlSet(photos: picked.map { photos[$0] }, identities: picked.map { identities[$0] })
    }
}

/// Which photos fit the guard and which verify it. Keyed, so the split can't be chosen to
/// flatter a result, and reproducible under the same key.
public struct PhotoSplit: Codable, Sendable, Hashable {
    public let fit: [String]
    public let heldOut: [String]
    /// The held-out set is augmentations of the only photo (weak; flagged).
    public let augmented: Bool

    public static func make(photoIDs: [String], schedule: SeedSchedule, heldOutFraction: Double = 0.35,
                            maxHeldOut: Int = 8) -> PhotoSplit {
        let unique = Array(Set(photoIDs))
        guard unique.count > 1 else { return PhotoSplit(fit: unique, heldOut: [], augmented: true) }
        let order = unique.sorted { schedule.value(.split, $0) < schedule.value(.split, $1) }
        let k = min(maxHeldOut, max(1, Int((Double(unique.count) * heldOutFraction).rounded())), unique.count - 1)
        return PhotoSplit(fit: Array(order.dropFirst(k)).sorted(), heldOut: Array(order.prefix(k)).sorted(), augmented: false)
    }
}

// MARK: - Likeness weighting

public enum FaceWeighting: String, Codable, Sendable {
    /// Weight the loss toward the face found by Vision; the whole image when none is found.
    case auto
    /// Uniform weight (e.g. the toy world, or non-photographic references).
    case off
}

/// Per-cell loss weights on a latent grid: face cells ≈ 1, background at a small floor.
public struct LikenessWeight: Sendable {
    public let rows: Int
    public let cols: Int
    public let values: [Float]
    public let faceFound: Bool
    public let method: String
    /// Share of the grid the face mask covers.
    public let faceArea: Double

    public static func uniform(rows: Int, cols: Int, method: String = "uniform") -> LikenessWeight {
        LikenessWeight(rows: rows, cols: cols, values: Array(repeating: 1, count: rows * cols), faceFound: false,
                       method: method, faceArea: 0)
    }
}

public enum LikenessWeighter {
    /// Background weight relative to the face.
    public static var backgroundFloor: Float = 0.05
    private static let lock = NSLock()
    nonisolated(unsafe) private static var masks: [String: FaceSegment?] = [:]

    public static func weight(for photo: SubjectPhoto, rows: Int, cols: Int, policy: FaceWeighting) -> LikenessWeight {
        guard policy == .auto else { return .uniform(rows: rows, cols: cols) }
        let segment: FaceSegment? = lock.withLock { masks[photo.id] } ?? {
            let found = (try? FaceDetector.detect(photo))?.first.map { FaceSegmenter.segment(photo, face: $0) }
            lock.withLock { masks[photo.id] = .some(found) }
            return found
        }()
        guard let segment else { return .uniform(rows: rows, cols: cols, method: "no-face") }
        let grid = segment.mask.grid(rows: rows, cols: cols)
        let area = Double(grid.reduce(0, +)) / Double(max(grid.count, 1))
        guard area > 0.002 else { return .uniform(rows: rows, cols: cols, method: "face-too-small") }
        let values = grid.map { backgroundFloor + (1 - backgroundFloor) * min(max($0, 0), 1) }
        return LikenessWeight(rows: rows, cols: cols, values: values, faceFound: true, method: segment.method, faceArea: area)
    }
}
