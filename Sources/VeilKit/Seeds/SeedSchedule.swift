//
//  SeedSchedule.swift
//  VeilKit
//  Adapted from Scorpion (github.com/riteshpakala/Scorpion, ScorpionKit), GPL-3.0.
//
//  A keyed, fixed seed schedule. Every random number Veil uses — noise draws ε, the
//  fit/held-out split, training pools, attack starts — comes from HMAC-SHA256(key, stream|index).
//
//  - A fixed schedule gives common random numbers across everything: route vs anchor,
//    base vs guarded, the person's photos vs controls. That pairing is what makes the pull
//    comparisons precise.
//  - A secret key keeps the schedule reproducible for whoever holds it, and unpredictable to
//    anyone tuning a model against known draws.
//  Reports record the schedule id (a hash of the key) and version, never the key.
//

import CryptoKit
import Foundation
import MLX

public struct ScheduleInfo: Codable, Sendable, Hashable {
    /// First 8 bytes of SHA-256(key), hex. Identifies the key without revealing it.
    public let id: String
    public let version: String
}

public struct SeedSchedule: Sendable {
    public static let version = "veil.seed.v1"

    public enum Stream: String, Sendable {
        /// Fit / held-out photo split.
        case split
        /// Noise draws for pull measurements during assessment.
        case pull
        /// Invented names for the null routes.
        case null
        /// Guard training: pool sampling, adapter init, minibatches.
        case train
        /// Soft-embedding attacks.
        case attack
        /// Discrete prompt search.
        case search
        /// Independent noise draws for verifying the exported guard.
        case verify
        /// Base-model latent samples used as fallback controls.
        case control
        /// Single-photo augmentations.
        case augment
        /// The analytic toy world.
        case toy
    }

    /// Stable index for a noise level, keyed by σ's value (not its position in a list), so
    /// every stage that visits a level sees the same draws there.
    public static func levelKey(_ sigma: Float) -> Int { 1_000_000 + Int((sigma * 4096).rounded()) }

    private let key: SymmetricKey
    public let info: ScheduleInfo

    public init(keyData: Data) {
        key = SymmetricKey(data: keyData)
        info = ScheduleInfo(id: String(Hashing.sha256Hex(keyData).prefix(16)), version: Self.version)
    }

    /// Public, fixed key for tests and the toy research harness: results reproduce anywhere.
    /// Not for real guards, where an unpredictable schedule matters.
    public static let research = SeedSchedule(keyData: Data("veil.research.v1".utf8))

    /// A fresh random key held only in memory (when no key file can be written).
    public static func ephemeral() -> SeedSchedule {
        var bytes = [UInt8](repeating: 0, count: 32)
        _ = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        return SeedSchedule(keyData: Data(bytes))
    }

    /// Key from `VEIL_SEED_KEY` (hex, or raw text), else a per-install random key stored with
    /// 0600 permissions (created on first use).
    public static func load(environment: [String: String] = ProcessInfo.processInfo.environment,
                            directory: URL = defaultDirectory) throws -> SeedSchedule {
        if let raw = environment["VEIL_SEED_KEY"], !raw.isEmpty {
            return SeedSchedule(keyData: Data(hex: raw) ?? Data(raw.utf8))
        }
        let url = directory.appendingPathComponent("seed.key")
        if let data = try? Data(contentsOf: url), data.count >= 16 {
            return SeedSchedule(keyData: data)
        }
        var bytes = [UInt8](repeating: 0, count: 32)
        guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
            throw CocoaError(.fileWriteUnknown)
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let data = Data(bytes)
        try data.write(to: url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        return SeedSchedule(keyData: data)
    }

    public static var defaultDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return base.appendingPathComponent("Veil", isDirectory: true)
    }

    /// 64 pseudorandom bits for (stream, index).
    public func value(_ stream: Stream, _ index: Int) -> UInt64 {
        let mac = HMAC<SHA256>.authenticationCode(for: Data("\(Self.version)|\(stream.rawValue)|\(index)".utf8), using: key)
        return Data(mac).prefix(8).reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
    }

    /// 64 pseudorandom bits for (stream, a string label), e.g. a photo id.
    public func value(_ stream: Stream, _ label: String) -> UInt64 {
        let mac = HMAC<SHA256>.authenticationCode(for: Data("\(Self.version)|\(stream.rawValue)|s:\(label)".utf8), using: key)
        return Data(mac).prefix(8).reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
    }

    /// A keyed identifier (hex) — e.g. a subject id that doesn't reveal which photos it covers
    /// to anyone without the key.
    public func keyedID(_ label: String, length: Int = 16) -> String {
        let mac = HMAC<SHA256>.authenticationCode(for: Data("\(Self.version)|id|\(label)".utf8), using: key)
        return String(Data(mac).map { String(format: "%02x", $0) }.joined().prefix(length))
    }

    /// A PRNG seeded from (stream, index), for sequences of choices.
    public func rng(_ stream: Stream, _ index: Int) -> SplitMix64 { SplitMix64(seed: value(stream, index)) }

    public func rng(_ stream: Stream, _ label: String) -> SplitMix64 { SplitMix64(seed: value(stream, label)) }

    /// Standard-normal array for (stream, index).
    public func normal(_ stream: Stream, _ index: Int, shape: [Int]) -> MLXArray {
        MLXRandom.normal(shape, key: MLXRandom.key(value(stream, index)))
    }

    /// Standard-normal array for (stream, label).
    public func normal(_ stream: Stream, _ label: String, shape: [Int]) -> MLXArray {
        MLXRandom.normal(shape, key: MLXRandom.key(value(stream, label)))
    }
}
