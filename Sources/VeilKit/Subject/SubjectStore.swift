//
//  SubjectStore.swift
//  VeilKit
//
//  Custody. Per protected person, Veil keeps only what it derived — latents and the sealed
//  list of discovered prompts — under Application Support/Veil/subjects/<subject id>. The
//  original photos are never copied. `forget` deletes everything held for a subject.
//
//  Discovered prompts are attack recipes (prompts that pull the model toward the person), so
//  they live only here; shareable reports list them by hash unless the user opts in.
//

import Foundation
import MLX

public struct SealedRoute: Codable, Sendable {
    public let hash: String
    public let text: String
    public let kind: String
    public let pull: Double
}

public struct SubjectStore: Sendable {
    public let root: URL

    public static var defaultRoot: URL {
        SeedSchedule.defaultDirectory.appendingPathComponent("subjects", isDirectory: true)
    }

    public init(root: URL = SubjectStore.defaultRoot) { self.root = root }

    public func directory(for subjectID: String) -> URL {
        root.appendingPathComponent(subjectID, isDirectory: true)
    }

    /// Subject ids with stored data.
    public func list() -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []).filter { !$0.hasPrefix(".") }.sorted()
    }

    /// Delete everything held for a subject. Returns false when there was nothing.
    @discardableResult
    public func forget(_ subjectID: String) throws -> Bool {
        let dir = directory(for: subjectID)
        guard FileManager.default.fileExists(atPath: dir.path) else { return false }
        try FileManager.default.removeItem(at: dir)
        return true
    }

    // MARK: Latents

    func latentURL(subjectID: String, encodingKey: String, photoID: String) -> URL {
        directory(for: subjectID).appendingPathComponent("latents", isDirectory: true)
            .appendingPathComponent(String(Hashing.sha256Hex(encodingKey + "|" + photoID).prefix(24)) + ".safetensors")
    }

    public func cachedLatent(subjectID: String, encodingKey: String, photoID: String) -> MLXArray? {
        let url = latentURL(subjectID: subjectID, encodingKey: encodingKey, photoID: photoID)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return (try? loadArrays(url: url))?["latent"]
    }

    public func storeLatent(_ latent: MLXArray, subjectID: String, encodingKey: String, photoID: String) {
        let url = latentURL(subjectID: subjectID, encodingKey: encodingKey, photoID: photoID)
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? save(arrays: ["latent": latent], url: url)
    }

    // MARK: Sealed routes

    public func sealedRoutesURL(for subjectID: String) -> URL {
        directory(for: subjectID).appendingPathComponent("routes.sealed.json")
    }

    public func writeSealedRoutes(_ routes: [SealedRoute], subjectID: String) throws {
        let url = sealedRoutesURL(for: subjectID)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(routes).write(to: url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }

    public func sealedRoutes(for subjectID: String) -> [SealedRoute] {
        guard let data = try? Data(contentsOf: sealedRoutesURL(for: subjectID)) else { return [] }
        return (try? JSONDecoder().decode([SealedRoute].self, from: data)) ?? []
    }
}
