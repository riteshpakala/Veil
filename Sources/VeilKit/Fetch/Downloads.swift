//
//  Downloads.swift
//  VeilKit
//
//  Whole-file downloads on top of the range fetcher (adapter files, small model repos), and
//  the one extra link hop Veil makes: a GitHub repository that holds no weights but whose
//  README points at a Hugging Face model resolves to that model, with both recorded.
//

import Foundation

extension RangeFetcher {
    /// Download `file` into `directory` (kept across runs for pinned URLs), in ranged chunks.
    public func download(_ file: RemoteFile, into directory: URL, pinned: Bool,
                         chunk: Int = 16 << 20) async throws -> URL {
        let name = Hashing.sha256Hex(file.url.absoluteString).prefix(16) + "-" + (file.path as NSString).lastPathComponent
        let destination = directory.appendingPathComponent(String(name))
        if pinned, FileManager.default.fileExists(atPath: destination.path) { return destination }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let size: Int
        if let known = file.size { size = known } else if let remote = try await remoteSize(file.url) { size = remote } else {
            let data = try await fetchSmallFile(file.url, limit: 2 << 30, cacheable: false)
            try data.write(to: destination, options: .atomic)
            return destination
        }
        let partial = destination.appendingPathExtension("partial")
        FileManager.default.createFile(atPath: partial.path, contents: nil)
        let handle = try FileHandle(forWritingTo: partial)
        defer { try? handle.close() }
        var offset = 0
        while offset < size {
            let upper = min(size, offset + chunk)
            let data = try await fetch(file.url, range: offset..<upper, cacheable: false)
            try handle.write(contentsOf: data)
            offset = upper
        }
        try handle.close()
        if FileManager.default.fileExists(atPath: destination.path) { try FileManager.default.removeItem(at: destination) }
        try FileManager.default.moveItem(at: partial, to: destination)
        return destination
    }
}

extension SourceResolver {
    /// Resolve a link; a GitHub repo without weight files whose README links a Hugging Face
    /// model is followed once to that model. Returns the resolved source and the hop, if any.
    public func resolveFollowingModelCard(_ link: String) async throws -> (source: ResolvedSource, via: String?) {
        let reference = try SourceParser.parse(link)
        let source = try await resolve(reference)
        guard reference.host == .gitHub, source.weightFiles.isEmpty,
              let readme = source.files.first(where: { ($0.path as NSString).lastPathComponent.lowercased() == "readme.md" }),
              let data = try? await fetcher.fetchSmallFile(readme.url, limit: 1 << 20),
              let target = Self.firstHuggingFaceModel(in: String(decoding: data, as: UTF8.self)) else {
            return (source, nil)
        }
        return (try await resolve(target), link)
    }

    /// First huggingface.co/{owner}/{name} model link in free text (skips datasets/spaces).
    static func firstHuggingFaceModel(in text: String) -> String? {
        guard let re = try? NSRegularExpression(pattern: #"https?://(?:www\.)?(?:huggingface\.co|hf\.co)/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)"#) else {
            return nil
        }
        for m in re.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            guard let owner = Range(m.range(at: 1), in: text).map({ String(text[$0]) }),
                  let name = Range(m.range(at: 2), in: text).map({ String(text[$0]) }) else { continue }
            if ["datasets", "spaces", "docs", "blog", "papers", "api"].contains(owner) { continue }
            return "https://huggingface.co/\(owner)/\(name)"
        }
        return nil
    }
}
