//
//  AdapterSource.swift
//  VeilKit
//
//  An adapter given as a local path or a Hugging Face / GitHub link → a local .safetensors
//  file (downloaded once into the cache for pinned links).
//

import Foundation

public enum AdapterSource {
    public enum AdapterSourceError: Error, LocalizedError {
        case noSafetensors(String)
        case ambiguous(String, [String])

        public var errorDescription: String? {
            switch self {
            case .noSafetensors(let s): return "No .safetensors adapter found at \(s)"
            case .ambiguous(let s, let files):
                return "\(s) holds several .safetensors files (\(files.prefix(4).joined(separator: ", "))); link the one to use."
            }
        }
    }

    public static var cacheDirectory: URL {
        FetchCache.defaultDirectory.appendingPathComponent("adapters", isDirectory: true)
    }

    public static func fetch(_ pathOrLink: String, fetcher: RangeFetcher) async throws -> URL {
        let local = URL(fileURLWithPath: (pathOrLink as NSString).expandingTildeInPath)
        if !pathOrLink.contains("://"), FileManager.default.fileExists(atPath: local.path) { return local }
        let source = try await SourceResolver(fetcher: fetcher).resolve(pathOrLink)
        let candidates = source.files.filter { $0.format == .safetensors }
        guard !candidates.isEmpty else { throw AdapterSourceError.noSafetensors(pathOrLink) }
        let file: RemoteFile
        if candidates.count == 1 {
            file = candidates[0]
        } else if let top = candidates.filter({ !$0.path.contains("/") }).first, candidates.filter({ !$0.path.contains("/") }).count == 1 {
            file = top
        } else {
            throw AdapterSourceError.ambiguous(pathOrLink, candidates.map(\.path))
        }
        return try await fetcher.download(file, into: cacheDirectory, pinned: source.pinned)
    }
}
