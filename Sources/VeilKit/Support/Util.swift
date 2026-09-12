//
//  Util.swift
//  VeilKit
//  Adapted from Scorpion (github.com/riteshpakala/Scorpion, ScorpionKit), GPL-3.0.
//

import CryptoKit
import Foundation

public enum VeilInfo {
    public static let version = "0.1.0"
    /// Report JSON schema.
    public static let reportSchema = "veil.guard.v1"
    /// Adapter metadata schema written into the guard's safetensors header.
    public static let adapterSchema = "veil.adapter.v1"
}

/// Codable JSON value, used for model-card data and report extras.
public enum JSONValue: Codable, Sendable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case null
    case array([JSONValue])
    case object([String: JSONValue])

    public init(any value: Any) {
        switch value {
        case let v as String: self = .string(v)
        case let v as Bool where type(of: value) == type(of: NSNumber(value: true)): self = .bool(v)
        case let v as NSNumber: self = .number(v.doubleValue)
        case let v as [Any]: self = .array(v.map(JSONValue.init(any:)))
        case let v as [String: Any]: self = .object(JSONValue.object(from: v))
        default: self = .null
        }
    }

    public static func object(from dict: [String: Any]) -> [String: JSONValue] {
        dict.mapValues(JSONValue.init(any:))
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let v = try? c.decode(Bool.self) { self = .bool(v) }
        else if let v = try? c.decode(Double.self) { self = .number(v) }
        else if let v = try? c.decode(String.self) { self = .string(v) }
        else if let v = try? c.decode([JSONValue].self) { self = .array(v) }
        else { self = .object(try c.decode([String: JSONValue].self)) }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let v): try c.encode(v)
        case .number(let v): try c.encode(v)
        case .bool(let v): try c.encode(v)
        case .null: try c.encodeNil()
        case .array(let v): try c.encode(v)
        case .object(let v): try c.encode(v)
        }
    }

    public var stringValue: String? {
        switch self {
        case .string(let v): return v
        case .number(let v): return String(v)
        case .bool(let v): return String(v)
        default: return nil
        }
    }

    /// All string leaves (for free-text matching over card data).
    public var flattenedStrings: [String] {
        switch self {
        case .string(let v): return [v]
        case .array(let v): return v.flatMap(\.flattenedStrings)
        case .object(let v): return v.values.flatMap(\.flattenedStrings)
        default: return []
        }
    }
}

/// SplitMix64 — tiny, fast, well-distributed; the seed schedule's only PRNG.
public struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) { state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    public mutating func unit() -> Double { Double(next() >> 11) * 0x1.0p-53 }
}

public enum Hashing {
    public static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    public static func sha256Hex(_ string: String) -> String { sha256Hex(Data(string.utf8)) }

    /// First 8 bytes of SHA-256, big-endian.
    public static func seed(_ data: Data) -> UInt64 {
        SHA256.hash(data: data).prefix(8).reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
    }

    public static func seed(_ string: String) -> UInt64 { seed(Data(string.utf8)) }

    /// SHA-256 of a file's bytes (nil when unreadable).
    public static func fileSHA256(_ url: URL) -> String? {
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe) else { return nil }
        return sha256Hex(data)
    }
}

extension String {
    /// Numeric-aware ordering ("block_2" < "block_10").
    func naturalLess(_ other: String) -> Bool {
        compare(other, options: [.numeric]) == .orderedAscending
    }
}

extension Data {
    /// Parse an even-length hex string; nil if it isn't one.
    init?(hex: String) {
        let s = hex.trimmingCharacters(in: .whitespacesAndNewlines)
        guard s.count >= 2, s.count % 2 == 0, s.allSatisfy(\.isHexDigit) else { return nil }
        var out = Data(capacity: s.count / 2)
        var i = s.startIndex
        while i < s.endIndex {
            let j = s.index(i, offsetBy: 2)
            guard let b = UInt8(s[i..<j], radix: 16) else { return nil }
            out.append(b)
            i = j
        }
        self = out
    }
}

/// Standard normal CDF.
func normalCDF(_ z: Double) -> Double { 0.5 * erfc(-z / 2.0.squareRoot()) }

/// Small statistics used across stages.
public enum Stats {
    public static func mean(_ x: [Double]) -> Double { x.isEmpty ? 0 : x.reduce(0, +) / Double(x.count) }

    /// Standard error of the mean.
    public static func standardError(_ x: [Double]) -> Double {
        guard x.count > 1 else { return 0 }
        let m = mean(x)
        let v = x.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(x.count - 1)
        return (v / Double(x.count)).squareRoot()
    }

    /// One-sided 95% t quantile for `df` degrees of freedom (df ≥ 1).
    public static func t95(_ df: Int) -> Double {
        let table: [Double] = [6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812,
                               1.796, 1.782, 1.771, 1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725]
        if df < 1 { return .infinity }
        if df <= table.count { return table[df - 1] }
        if df <= 30 { return 1.70 }
        if df <= 60 { return 1.67 }
        return 1.645
    }

    /// Exceedance p-value against a null sample: (1 + #null ≥ x) / (1 + N).
    public static func exceedanceP(_ x: Double, null: [Double]) -> Double {
        Double(1 + null.filter { $0 >= x }.count) / Double(1 + null.count)
    }

    /// Empirical quantile (linear interpolation), q in [0, 1].
    public static func quantile(_ x: [Double], _ q: Double) -> Double {
        guard !x.isEmpty else { return .nan }
        let s = x.sorted()
        let pos = q * Double(s.count - 1)
        let lo = Int(pos.rounded(.down)), hi = min(lo + 1, s.count - 1)
        return s[lo] + (s[hi] - s[lo]) * (pos - Double(lo))
    }

    public static func median(_ x: [Double]) -> Double { quantile(x, 0.5) }
}
