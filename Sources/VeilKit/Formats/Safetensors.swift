//
//  Safetensors.swift
//  VeilKit
//  Adapted from Scorpion (github.com/riteshpakala/Scorpion, ScorpionKit), GPL-3.0.
//
//  safetensors layout: 8-byte little-endian header length N, N bytes of JSON header,
//  then the raw tensor data. `data_offsets` in the header are relative to the start of
//  the data section (8 + N). Because the header is self-describing, a remote file can
//  be inventoried with two small HTTP Range requests and any single tensor fetched by
//  its byte range.
//

import Foundation

public enum SafetensorsError: Error, LocalizedError {
    case truncated(needed: Int)
    case headerTooLarge(Int)
    case malformedHeader(String)

    public var errorDescription: String? {
        switch self {
        case .truncated(let n): return "safetensors header truncated (need \(n) bytes)"
        case .headerTooLarge(let n): return "safetensors header length \(n) exceeds the 100 MB spec limit"
        case .malformedHeader(let why): return "Malformed safetensors header: \(why)"
        }
    }
}

public struct SafetensorsHeader: Sendable {
    public struct Entry: Sendable, Hashable {
        public let name: String
        /// Raw dtype string from the header (kept for unknown types).
        public let rawDType: String
        public let shape: [Int]
        /// Byte range relative to the data section.
        public let dataOffsets: Range<Int>

        public var dtype: TensorDType? { TensorDType(rawValue: rawDType) }
        public var elementCount: Int { shape.reduce(1, *) }
        public var byteCount: Int { dataOffsets.count }
    }

    /// Spec limit on the JSON header size.
    public static let maxHeaderLength = 100_000_000

    public let headerLength: Int
    public let metadata: [String: String]
    /// Entries sorted by data offset (physical file order).
    public let entries: [Entry]

    public var dataStart: Int { 8 + headerLength }
    public var totalSize: Int { dataStart + (entries.map(\.dataOffsets.upperBound).max() ?? 0) }

    public func entry(named name: String) -> Entry? { entries.first { $0.name == name } }

    /// Absolute byte range of a tensor within the file.
    public func absoluteRange(of entry: Entry) -> Range<Int> {
        (dataStart + entry.dataOffsets.lowerBound)..<(dataStart + entry.dataOffsets.upperBound)
    }

    /// Header length from the first 8 bytes.
    public static func headerLength(fromPrefix data: Data) throws -> Int {
        guard data.count >= 8 else { throw SafetensorsError.truncated(needed: 8) }
        let n = data.prefix(8).withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) }
        guard n <= UInt64(maxHeaderLength) else { throw SafetensorsError.headerTooLarge(Int(n)) }
        return Int(n)
    }

    /// Parse from a prefix of the file that contains at least the full header.
    public static func parse(prefix data: Data) throws -> SafetensorsHeader {
        let n = try headerLength(fromPrefix: data)
        guard data.count >= 8 + n else { throw SafetensorsError.truncated(needed: 8 + n) }
        let json = data.subdata(in: data.startIndex + 8..<data.startIndex + 8 + n)
        return try parse(json: json, headerLength: n)
    }

    public static func parse(json: Data, headerLength n: Int) throws -> SafetensorsHeader {
        guard let object = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw SafetensorsError.malformedHeader("top level is not an object")
        }
        var metadata: [String: String] = [:]
        var entries: [Entry] = []
        for (key, value) in object {
            if key == "__metadata__" {
                if let dict = value as? [String: Any] {
                    for (k, v) in dict { metadata[k] = (v as? String) ?? String(describing: v) }
                }
                continue
            }
            guard let info = value as? [String: Any],
                  let dtype = info["dtype"] as? String,
                  let shape = info["shape"] as? [Int],
                  let offsets = info["data_offsets"] as? [Int], offsets.count == 2,
                  offsets[0] <= offsets[1] else {
                throw SafetensorsError.malformedHeader("bad entry for \(key)")
            }
            entries.append(Entry(name: key, rawDType: dtype, shape: shape,
                                 dataOffsets: offsets[0]..<offsets[1]))
        }
        entries.sort { ($0.dataOffsets.lowerBound, $0.name) < ($1.dataOffsets.lowerBound, $1.name) }
        return SafetensorsHeader(headerLength: n, metadata: metadata, entries: entries)
    }
}

/// `*.safetensors.index.json` for sharded checkpoints: tensor name → shard file.
public struct SafetensorsIndex: Sendable {
    public let weightMap: [String: String]
    public let metadata: [String: String]

    public static func parse(_ data: Data) throws -> SafetensorsIndex {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let map = object["weight_map"] as? [String: String] else {
            throw SafetensorsError.malformedHeader("index has no weight_map")
        }
        var metadata: [String: String] = [:]
        if let meta = object["metadata"] as? [String: Any] {
            for (k, v) in meta { metadata[k] = String(describing: v) }
        }
        return SafetensorsIndex(weightMap: map, metadata: metadata)
    }

    public var shards: [String] { Array(Set(weightMap.values)).sorted() }
}

/// Minimal writer, used by tests and fixtures. Tensors are written in the given order.
public enum SafetensorsWriter {
    public struct Tensor {
        public let name: String
        public let dtype: TensorDType
        public let shape: [Int]
        public let data: Data

        public init(name: String, dtype: TensorDType, shape: [Int], data: Data) {
            self.name = name
            self.dtype = dtype
            self.shape = shape
            self.data = data
        }

        /// Float values encoded as `dtype` (F32, F16 or BF16).
        public init(name: String, floats: [Float], shape: [Int], dtype: TensorDType = .f32) {
            var data = Data(capacity: floats.count * dtype.byteSize)
            for f in floats {
                switch dtype {
                case .f16:
                    withUnsafeBytes(of: TensorDecoding.floatToHalf(f).littleEndian) { data.append(contentsOf: $0) }
                case .bf16:
                    withUnsafeBytes(of: UInt16(truncatingIfNeeded: f.bitPattern >> 16).littleEndian) { data.append(contentsOf: $0) }
                default:
                    withUnsafeBytes(of: f.bitPattern.littleEndian) { data.append(contentsOf: $0) }
                }
            }
            self.init(name: name, dtype: dtype, shape: shape, data: data)
        }
    }

    public static func encode(_ tensors: [Tensor], metadata: [String: String] = [:]) throws -> Data {
        var header: [String: Any] = [:]
        if !metadata.isEmpty { header["__metadata__"] = metadata }
        var offset = 0
        for t in tensors {
            header[t.name] = ["dtype": t.dtype.rawValue, "shape": t.shape,
                              "data_offsets": [offset, offset + t.data.count]]
            offset += t.data.count
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        // Pad with spaces to 8-byte alignment, as the reference implementation does.
        while (8 + json.count) % 8 != 0 { json.append(0x20) }
        var out = Data()
        withUnsafeBytes(of: UInt64(json.count).littleEndian) { out.append(contentsOf: $0) }
        out.append(json)
        for t in tensors { out.append(t.data) }
        return out
    }
}
