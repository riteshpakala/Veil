//
//  SourceResolver.swift
//  VeilKit
//  Adapted from Scorpion (github.com/riteshpakala/Scorpion, ScorpionKit), GPL-3.0.
//
//  Model links → a pinned, listable set of remote files. Supported:
//    huggingface.co/{owner}/{name}[/tree|blob|resolve/{rev}[/{path}]]
//    github.com/{owner}/{repo}[/tree|blob|raw/{ref}[/{path}]]
//    github.com/{owner}/{repo}/releases/(tag|download)/{tag}[/{asset}]
//    raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}
//    media.githubusercontent.com/media/{owner}/{repo}/{ref}/{path}
//  Every resolution pins a commit sha so fetched ranges are cacheable and reports are
//  reproducible.
//

import Foundation

public enum SourceError: Error, LocalizedError {
    case unsupportedURL(String)
    case notFound(String)
    case noWeightFiles(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedURL(let s): return "Unsupported model link: \(s). Use a Hugging Face or GitHub URL."
        case .notFound(let s): return "Not found: \(s)"
        case .noWeightFiles(let s): return "No model weight files found at \(s)"
        }
    }
}

public enum ModelHost: String, Codable, Sendable {
    case huggingFace
    case gitHub
    /// A file on this machine.
    case local
}

/// A parsed model link, before any network access.
public struct ModelReference: Codable, Sendable, Equatable {
    public var host: ModelHost
    /// HF: "owner/name". GitHub: "owner/repo".
    public var repo: String
    public var revision: String?
    /// File or directory path inside the repo, if the link pointed at one.
    public var path: String?
    /// GitHub release tag / asset, for release links.
    public var releaseTag: String?
    public var releaseAsset: String?
    public var original: String
}

public enum WeightFormat: String, Codable, Sendable {
    case safetensors
    case gguf
    case pickle       // .ckpt/.pt/.pth/.bin — never deserialized
    case onnx
    case other

    public init(path: String) {
        let p = path.lowercased()
        if p.hasSuffix(".safetensors") { self = .safetensors }
        else if p.hasSuffix(".gguf") { self = .gguf }
        else if p.hasSuffix(".ckpt") || p.hasSuffix(".pt") || p.hasSuffix(".pth") || p.hasSuffix(".bin") { self = .pickle }
        else if p.hasSuffix(".onnx") { self = .onnx }
        else { self = .other }
    }

    public var isWeights: Bool { self != .other }
}

public struct RemoteFile: Codable, Sendable, Hashable {
    public let path: String
    public let size: Int?
    public let url: URL
    public var format: WeightFormat { WeightFormat(path: path) }
}

public struct ResolvedSource: Codable, Sendable {
    public let reference: ModelReference
    /// Pinned commit sha (nil only for release assets, which are immutable by URL).
    public let commit: String?
    public let files: [RemoteFile]
    /// HF model info: tags, card data (parsed model-card YAML), etc.
    public let hubTags: [String]
    public let cardData: [String: JSONValue]
    /// Whether file URLs are pinned (safe to cache).
    public let pinned: Bool

    public var weightFiles: [RemoteFile] { files.filter { $0.format.isWeights } }
    public var repoBytes: Int { files.compactMap(\.size).reduce(0, +) }
    public var weightBytes: Int { weightFiles.compactMap(\.size).reduce(0, +) }

    public func file(named suffix: String) -> RemoteFile? {
        files.first { $0.path == suffix || $0.path.hasSuffix("/" + suffix) }
    }
}

// MARK: - Parsing

public enum SourceParser {
    public static func parse(_ string: String) throws -> ModelReference {
        let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
        let withScheme = trimmed.contains("://") ? trimmed : "https://" + trimmed
        guard let url = URL(string: withScheme), let host = url.host?.lowercased() else {
            throw SourceError.unsupportedURL(string)
        }
        let parts = url.path.split(separator: "/").map(String.init)

        switch host {
        case "huggingface.co", "www.huggingface.co", "hf.co":
            guard parts.count >= 2, !["datasets", "spaces", "api", "docs"].contains(parts[0]) else {
                throw SourceError.unsupportedURL(string)
            }
            var ref = ModelReference(host: .huggingFace, repo: "\(parts[0])/\(parts[1])", original: trimmed)
            if parts.count >= 4, ["tree", "blob", "resolve"].contains(parts[2]) {
                ref.revision = parts[3]
                if parts.count > 4 { ref.path = parts[4...].joined(separator: "/") }
            }
            return ref

        case "github.com", "www.github.com":
            guard parts.count >= 2 else { throw SourceError.unsupportedURL(string) }
            let repoName = parts[1].hasSuffix(".git") ? String(parts[1].dropLast(4)) : parts[1]
            var ref = ModelReference(host: .gitHub, repo: "\(parts[0])/\(repoName)", original: trimmed)
            if parts.count >= 4, parts[2] == "releases" {
                if parts[3] == "tag", parts.count >= 5 { ref.releaseTag = parts[4] }
                else if parts[3] == "download", parts.count >= 6 {
                    ref.releaseTag = parts[4]
                    ref.releaseAsset = parts[5...].joined(separator: "/")
                } else if parts[3] == "latest" { ref.releaseTag = "latest" }
                return ref
            }
            if parts.count >= 4, ["tree", "blob", "raw"].contains(parts[2]) {
                ref.revision = parts[3]
                if parts.count > 4 { ref.path = parts[4...].joined(separator: "/") }
            }
            return ref

        case "raw.githubusercontent.com":
            guard parts.count >= 4 else { throw SourceError.unsupportedURL(string) }
            return ModelReference(host: .gitHub, repo: "\(parts[0])/\(parts[1])", revision: parts[2],
                                  path: parts[3...].joined(separator: "/"), original: trimmed)

        case "media.githubusercontent.com":
            guard parts.count >= 5, parts[0] == "media" else { throw SourceError.unsupportedURL(string) }
            return ModelReference(host: .gitHub, repo: "\(parts[1])/\(parts[2])", revision: parts[3],
                                  path: parts[4...].joined(separator: "/"), original: trimmed)

        default:
            throw SourceError.unsupportedURL(string)
        }
    }
}

// MARK: - Resolution

public struct SourceResolver: Sendable {
    let fetcher: RangeFetcher
    var huggingFaceAPI = URL(string: "https://huggingface.co")!
    var gitHubAPI = URL(string: "https://api.github.com")!

    public init(fetcher: RangeFetcher) { self.fetcher = fetcher }

    public func resolve(_ reference: ModelReference) async throws -> ResolvedSource {
        switch reference.host {
        case .huggingFace: return try await resolveHuggingFace(reference)
        case .gitHub: return try await resolveGitHub(reference)
        case .local: throw SourceError.unsupportedURL(reference.original)
        }
    }

    public func resolve(_ link: String) async throws -> ResolvedSource {
        try await resolve(try SourceParser.parse(link))
    }

    // MARK: Hugging Face

    func resolveHuggingFace(_ ref: ModelReference) async throws -> ResolvedSource {
        let rev = ref.revision ?? "main"
        let infoURL = huggingFaceAPI.appendingPathComponent("api/models/\(ref.repo)/revision/\(rev)")
        let (infoJSON, _) = try await fetcher.getJSON(infoURL)
        guard let info = infoJSON as? [String: Any], let sha = info["sha"] as? String else {
            throw SourceError.notFound("\(ref.repo)@\(rev)")
        }
        let tags = info["tags"] as? [String] ?? []
        let cardData = (info["cardData"] as? [String: Any]).map { JSONValue.object(from: $0) } ?? [:]

        var files: [RemoteFile] = []
        var next: URL? = {
            var c = URLComponents(url: huggingFaceAPI.appendingPathComponent("api/models/\(ref.repo)/tree/\(sha)"),
                                  resolvingAgainstBaseURL: false)!
            c.queryItems = [URLQueryItem(name: "recursive", value: "true")]
            return c.url
        }()
        while let page = next {
            let (json, headers) = try await fetcher.getJSON(page)
            for item in json as? [[String: Any]] ?? [] where item["type"] as? String == "file" {
                guard let path = item["path"] as? String else { continue }
                let lfsSize = (item["lfs"] as? [String: Any])?["size"] as? Int
                let size = lfsSize ?? item["size"] as? Int
                let url = huggingFaceAPI.appendingPathComponent("\(ref.repo)/resolve/\(sha)/\(path)")
                files.append(RemoteFile(path: path, size: size, url: url))
            }
            next = Self.nextLink(headers)
        }
        files = Self.scope(files, to: ref.path)
        if files.isEmpty { throw SourceError.notFound("\(ref.original) (no files)") }
        return ResolvedSource(reference: ref, commit: sha, files: files.sorted { $0.path < $1.path },
                              hubTags: tags, cardData: cardData, pinned: true)
    }

    static func nextLink(_ headers: [AnyHashable: Any]) -> URL? {
        guard let link = headers.first(where: { ($0.key as? String)?.lowercased() == "link" })?.value as? String else { return nil }
        for part in link.split(separator: ",") where part.contains("rel=\"next\"") {
            if let start = part.firstIndex(of: "<"), let end = part.firstIndex(of: ">") {
                return URL(string: String(part[part.index(after: start)..<end]))
            }
        }
        return nil
    }

    static func scope(_ files: [RemoteFile], to path: String?) -> [RemoteFile] {
        guard let path, !path.isEmpty else { return files }
        let exact = files.filter { $0.path == path }
        if !exact.isEmpty {
            // A file link still pulls in sibling metadata (README, configs) for context.
            let dir = (path as NSString).deletingLastPathComponent
            let context = files.filter { f in
                let name = (f.path as NSString).lastPathComponent.lowercased()
                return (f.path as NSString).deletingLastPathComponent == dir && f.path != path
                    && (name == "readme.md" || name.hasSuffix(".json"))
            }
            return exact + context
        }
        return files.filter { $0.path.hasPrefix(path.hasSuffix("/") ? path : path + "/") }
    }

    // MARK: GitHub

    func resolveGitHub(_ ref: ModelReference) async throws -> ResolvedSource {
        if let tag = ref.releaseTag { return try await resolveGitHubRelease(ref, tag: tag) }

        let revision: String
        if let r = ref.revision {
            revision = r
        } else {
            let (repoJSON, _) = try await fetcher.getJSON(gitHubAPI.appendingPathComponent("repos/\(ref.repo)"))
            revision = (repoJSON as? [String: Any])?["default_branch"] as? String ?? "main"
        }
        let (commitJSON, _) = try await fetcher.getJSON(gitHubAPI.appendingPathComponent("repos/\(ref.repo)/commits/\(revision)"))
        guard let sha = (commitJSON as? [String: Any])?["sha"] as? String else {
            throw SourceError.notFound("\(ref.repo)@\(revision)")
        }
        var treeURL = URLComponents(url: gitHubAPI.appendingPathComponent("repos/\(ref.repo)/git/trees/\(sha)"),
                                    resolvingAgainstBaseURL: false)!
        treeURL.queryItems = [URLQueryItem(name: "recursive", value: "1")]
        let (treeJSON, _) = try await fetcher.getJSON(treeURL.url!)
        let entries = ((treeJSON as? [String: Any])?["tree"] as? [[String: Any]]) ?? []

        var files: [RemoteFile] = []
        for entry in entries where entry["type"] as? String == "blob" {
            guard let path = entry["path"] as? String else { continue }
            let size = entry["size"] as? Int
            let raw = URL(string: "https://raw.githubusercontent.com/\(ref.repo)/\(sha)/\(Self.escape(path))")!
            files.append(RemoteFile(path: path, size: size, url: raw))
        }
        files = Self.scope(files, to: ref.path)

        // Weight files stored in Git LFS appear as ~130-byte pointer blobs; swap in the
        // media URL and the real size from the pointer.
        var resolved: [RemoteFile] = []
        for file in files {
            if file.format.isWeights, let size = file.size, size < 1024,
               let pointer = try? await fetcher.fetchSmallFile(file.url, limit: 4096),
               let lfs = Self.parseLFSPointer(pointer) {
                let media = URL(string: "https://media.githubusercontent.com/media/\(ref.repo)/\(sha)/\(Self.escape(file.path))")!
                resolved.append(RemoteFile(path: file.path, size: lfs.size, url: media))
            } else {
                resolved.append(file)
            }
        }
        if resolved.isEmpty { throw SourceError.notFound("\(ref.original) (no files)") }
        return ResolvedSource(reference: ref, commit: sha, files: resolved.sorted { $0.path < $1.path },
                              hubTags: [], cardData: [:], pinned: true)
    }

    func resolveGitHubRelease(_ ref: ModelReference, tag: String) async throws -> ResolvedSource {
        let path = tag == "latest" ? "repos/\(ref.repo)/releases/latest" : "repos/\(ref.repo)/releases/tags/\(tag)"
        let (json, _) = try await fetcher.getJSON(gitHubAPI.appendingPathComponent(path))
        let assets = ((json as? [String: Any])?["assets"] as? [[String: Any]]) ?? []
        var files: [RemoteFile] = []
        for asset in assets {
            guard let name = asset["name"] as? String,
                  let urlString = asset["browser_download_url"] as? String,
                  let url = URL(string: urlString) else { continue }
            if let wanted = ref.releaseAsset, wanted != name { continue }
            files.append(RemoteFile(path: name, size: asset["size"] as? Int, url: url))
        }
        if files.isEmpty { throw SourceError.notFound("\(ref.original) (no release assets)") }
        // Release asset URLs are immutable per (tag, name) in practice; tags can be moved,
        // so they are treated as unpinned and not cached.
        return ResolvedSource(reference: ref, commit: nil, files: files, hubTags: [], cardData: [:], pinned: false)
    }

    static func escape(_ path: String) -> String {
        path.split(separator: "/").map {
            String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0)
        }.joined(separator: "/")
    }

    static func parseLFSPointer(_ data: Data) -> (oid: String, size: Int)? {
        let text = String(decoding: data, as: UTF8.self)
        guard text.hasPrefix("version https://git-lfs.github.com/spec/") else { return nil }
        var oid: String?, size: Int?
        for line in text.split(separator: "\n") {
            if line.hasPrefix("oid sha256:") { oid = String(line.dropFirst("oid sha256:".count)) }
            if line.hasPrefix("size ") { size = Int(line.dropFirst("size ".count)) }
        }
        guard let oid, let size else { return nil }
        return (oid, size)
    }
}
