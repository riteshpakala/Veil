//
//  AdapterFile.swift
//  VeilKit
//  Adapted from Scorpion (github.com/riteshpakala/Scorpion, ScorpionKit: AdapterDecoder), GPL-3.0.
//
//  Reads adapter files — a community LoRA that is part of the deployment being protected, or a
//  guard read back for verification. Formats are recognized by tensor-name suffix, shared
//  across architectures:
//    LoRA  (kohya lora_up/lora_down/alpha, PEFT lora_B/lora_A)  ΔW = (α/r)·up·down
//    LoHa  (hada_w1_a/b, hada_w2_a/b)                           ΔW = (w1a·w1b) ⊙ (w2a·w2b)·(α/r)
//    LoKr  (lokr_w1[_a,_b], lokr_w2[_a,_b])                     ΔW = kron(w1, w2)·scale
//    diff  (LyCORIS full diff)                                   ΔW = diff
//  Module names are returned as the file spells them; each executor maps them to its layers.
//

import Foundation
import MLX

public enum AdapterFormat: String, Codable, Sendable {
    case lora
    case loha
    case lokr
    case diff
    case mixed
    case none
}

/// A module's weight update, kept factored when possible.
public enum WeightUpdate {
    case lowRank(up: MLXArray, down: MLXArray, scale: Float)   // up (out×r), down (r×in)
    case dense(MLXArray)                                       // (out×in)

    public var materialized: MLXArray {
        switch self {
        case .lowRank(let up, let down, let scale): return matmul(up, down) * scale
        case .dense(let w): return w
        }
    }
}

public enum AdapterFileError: Error, LocalizedError {
    case noModules(String)
    case shapeMismatch(String)

    public var errorDescription: String? {
        switch self {
        case .noModules(let s): return "\(s) holds no adapter modules (LoRA, LoHa, LoKr or diff)"
        case .shapeMismatch(let s): return "Shape mismatch in adapter module \(s)"
        }
    }
}

public struct AdapterFile {
    public let url: URL
    public let sha256: String
    public let metadata: [String: String]
    public let format: AdapterFormat
    /// Module name (file spelling, role suffix stripped) → ΔW.
    public let updates: [String: WeightUpdate]
    /// Modules with DoRA magnitudes (applied direction-only).
    public let dora: [String]
    /// Tensors that belong to no module.
    public let unassigned: [String]

    public static func load(_ url: URL) throws -> AdapterFile {
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        return try AdapterFile(url: url, arrays: arrays, metadata: metadata)
    }

    public init(url: URL, arrays: [String: MLXArray], metadata: [String: String]) throws {
        self.url = url
        self.sha256 = Hashing.fileSHA256(url) ?? ""
        self.metadata = metadata
        var groups: [String: [Role: String]] = [:]
        var unassigned: [String] = []
        for name in arrays.keys {
            if let (module, role) = Self.split(name) { groups[module, default: [:]][role] = name } else { unassigned.append(name) }
        }
        let peftScale = Self.peftScale(metadata)
        func t(_ n: String?) -> MLXArray? { n.flatMap { arrays[$0] }.map { $0.reshaped($0.dim(0), -1).asType(.float32) } }
        func alphaScale(_ n: String?, rank: Int) -> Float? {
            guard let a = n.flatMap({ arrays[$0] }) else { return nil }
            return a.reshaped(-1)[0].item(Float.self) / Float(max(rank, 1))
        }
        var updates: [String: WeightUpdate] = [:]
        var formats: Set<AdapterFormat> = []
        var dora: [String] = []
        for (module, roles) in groups {
            if roles[.dora] != nil { dora.append(module) }
            if let up = t(roles[.loraUp]), let down = t(roles[.loraDown]) {
                guard up.dim(1) == down.dim(0) else { throw AdapterFileError.shapeMismatch(module) }
                let r = down.dim(0)
                updates[module] = .lowRank(up: up, down: down, scale: alphaScale(roles[.alpha], rank: r) ?? peftScale?(r) ?? 1)
                formats.insert(.lora)
            } else if let a = t(roles[.hadaW1A]), let b = t(roles[.hadaW1B]), let c = t(roles[.hadaW2A]), let d = t(roles[.hadaW2B]) {
                let w1 = matmul(a, b), w2 = matmul(c, d)
                guard w1.shape == w2.shape else { throw AdapterFileError.shapeMismatch(module) }
                updates[module] = .dense(w1 * w2 * (alphaScale(roles[.alpha], rank: b.dim(0)) ?? 1))
                formats.insert(.loha)
            } else if roles[.lokrW1] != nil || roles[.lokrW1A] != nil {
                guard let w1 = t(roles[.lokrW1]) ?? Self.zip2(t(roles[.lokrW1A]), t(roles[.lokrW1B])).map({ matmul($0, $1) }),
                      let w2 = t(roles[.lokrW2]) ?? Self.zip2(t(roles[.lokrW2A]), t(roles[.lokrW2B])).map({ matmul($0, $1) }) else { continue }
                let factorized = roles[.lokrW1] == nil || roles[.lokrW2] == nil
                let rank = t(roles[.lokrW2B])?.dim(0) ?? t(roles[.lokrW1B])?.dim(0) ?? 1
                let scale = factorized ? (alphaScale(roles[.alpha], rank: rank) ?? 1) : 1
                updates[module] = .dense(Self.kron(w1, w2) * scale)
                formats.insert(.lokr)
            } else if let diff = t(roles[.diff]) {
                updates[module] = .dense(diff)
                formats.insert(.diff)
            } else {
                unassigned.append(contentsOf: roles.values)
            }
        }
        guard !updates.isEmpty else { throw AdapterFileError.noModules(url.lastPathComponent) }
        self.updates = updates
        self.format = formats.count == 1 ? formats.first! : .mixed
        self.dora = dora.sorted()
        self.unassigned = unassigned.sorted()
    }

    private static func zip2(_ a: MLXArray?, _ b: MLXArray?) -> (MLXArray, MLXArray)? {
        guard let a, let b else { return nil }
        return (a, b)
    }

    enum Role: String {
        case loraUp, loraDown, loraMid, alpha, dora
        case hadaW1A, hadaW1B, hadaW2A, hadaW2B
        case lokrW1, lokrW1A, lokrW1B, lokrW2, lokrW2A, lokrW2B
        case diff
    }

    /// Longest suffixes first so ".lora_A.default.weight" wins over ".weight".
    static let suffixes: [(String, Role)] = [
        (".lora_A.default.weight", .loraDown), (".lora_B.default.weight", .loraUp),
        ("_lora.down.weight", .loraDown), ("_lora.up.weight", .loraUp),
        (".lora.down.weight", .loraDown), (".lora.up.weight", .loraUp),
        (".lora_down.weight", .loraDown), (".lora_up.weight", .loraUp), (".lora_mid.weight", .loraMid),
        (".lora_A.weight", .loraDown), (".lora_B.weight", .loraUp),
        (".lora_magnitude_vector", .dora), (".dora_scale", .dora),
        (".hada_w1_a", .hadaW1A), (".hada_w1_b", .hadaW1B), (".hada_w2_a", .hadaW2A), (".hada_w2_b", .hadaW2B),
        (".lokr_w1_a", .lokrW1A), (".lokr_w1_b", .lokrW1B), (".lokr_w2_a", .lokrW2A), (".lokr_w2_b", .lokrW2B),
        (".lokr_w1", .lokrW1), (".lokr_w2", .lokrW2),
        (".diff", .diff),
        (".alpha", .alpha),
    ]

    static func split(_ name: String) -> (module: String, role: Role)? {
        for (suffix, role) in suffixes where name.hasSuffix(suffix) {
            return (String(name.dropLast(suffix.count)), role)
        }
        return nil
    }

    /// PEFT `lora_alpha / r` from a `lora_alpha` / `r` metadata pair, when present.
    static func peftScale(_ metadata: [String: String]) -> ((Int) -> Float)? {
        guard let alpha = metadata["lora_alpha"].flatMap(Float.init) else { return nil }
        let fixedR = metadata["r"].flatMap(Int.init)
        return { r in alpha / Float(fixedR ?? r) }
    }

    /// Kronecker product of 2-D arrays: (a×c) ⊗ (b×d) → (ab × cd).
    public static func kron(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        let (ar, ac) = (a.dim(0), a.dim(1))
        let (br, bc) = (b.dim(0), b.dim(1))
        return (a.reshaped(ar, 1, ac, 1) * b.reshaped(1, br, 1, bc)).reshaped(ar * br, ac * bc)
    }
}
