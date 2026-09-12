//
//  AdapterWriter.swift
//  VeilKit
//
//  The deliverable: a standard LoRA `.safetensors` any host can load (names from the executor's
//  `AdapterNaming`, e.g. diffusers/PEFT or BFL/ComfyUI), with Veil's metadata in the header.
//  Written in bf16 like most published LoRAs, then read back through the same importer hosts'
//  names go through — verification runs on those read-back deltas, so what ships is what's
//  measured.
//

import Foundation
import MLX

public struct GuardFile: Codable, Sendable, Hashable {
    public let path: String
    public let sha256: String
    public let format: String
    public let slots: [String]
    public let rank: Int
    public let parameters: Int
    public let bytes: Int
}

extension GuardFile {
    /// The same record with only the file's name: reports are shared, local paths aren't.
    public var shareable: GuardFile {
        GuardFile(path: URL(fileURLWithPath: path).lastPathComponent, sha256: sha256, format: format, slots: slots, rank: rank,
                  parameters: parameters, bytes: bytes)
    }
}

public enum AdapterWriterError: Error, LocalizedError {
    case unknownFormat(String, [String])
    case incomplete([String])
    case dense([String])

    public var errorDescription: String? {
        switch self {
        case .unknownFormat(let f, let all): return "Unknown adapter format \(f); this model writes \(all.joined(separator: ", "))"
        case .incomplete(let m): return "The guard file maps only partly onto the model (unmapped: \(m.prefix(6).joined(separator: ", ")))"
        case .dense(let m): return "Expected a LoRA guard, found dense modules: \(m.prefix(4).joined(separator: ", "))"
        }
    }
}

public enum AdapterWriter {
    /// Write the guard. Returns what was written.
    public static func write(_ deltas: [String: LowRankDelta], model: AdapterNaming, format: String?,
                             metadata: [String: String], to url: URL, dtype: DType = .bfloat16) throws -> GuardFile {
        let chosen = format ?? model.exportFormats.first ?? "diffusers"
        guard model.exportFormats.contains(chosen) else { throw AdapterWriterError.unknownFormat(chosen, model.exportFormats) }
        let tensors = try model.exportTensors(deltas, format: chosen).mapValues { $0.asType(dtype) }
        var meta = metadata
        meta["format"] = "pt"
        meta["veil.format"] = chosen
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try save(arrays: tensors, metadata: meta, url: url)
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
        return GuardFile(path: url.path, sha256: Hashing.fileSHA256(url) ?? "", format: chosen, slots: deltas.keys.sorted(),
                         rank: deltas.values.map(\.rank).max() ?? 0,
                         parameters: deltas.values.reduce(0) { $0 + $1.rank * ($1.inFeatures + $1.outFeatures) }, bytes: size)
    }

    /// Read a guard back through the model's importer: slot deltas, factored.
    public static func readBack(_ url: URL, model: AdapterNaming) throws -> (deltas: [String: LowRankDelta], file: AdapterFile) {
        let file = try AdapterFile.load(url)
        let (deltas, unmapped) = model.importDeltas(file)
        guard unmapped.isEmpty else { throw AdapterWriterError.incomplete(unmapped) }
        var out: [String: LowRankDelta] = [:], dense: [String] = []
        for (key, delta) in deltas {
            switch delta {
            case .lowRank(let d): out[key] = d
            case .dense: dense.append(key)
            }
        }
        guard dense.isEmpty else { throw AdapterWriterError.dense(dense) }
        return (out, file)
    }

    /// Sum of guards for the same slots, exactly, by concatenating ranks: [B₁ B₂]·[A₁; A₂].
    public static func concatenate(_ a: LowRankDelta, _ b: LowRankDelta) -> LowRankDelta {
        LowRankDelta(up: concatenated([a.up, b.up], axis: 1), down: concatenated([a.down, b.down], axis: 0))
    }
}
