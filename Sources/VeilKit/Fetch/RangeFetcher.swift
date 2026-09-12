//
//  RangeFetcher.swift
//  VeilKit
//  Adapted from Scorpion (github.com/riteshpakala/Scorpion, ScorpionKit), GPL-3.0.
//
//  HTTP byte-range fetching over URLSession. Guarantees:
//  - A ranged request only succeeds on `206` with a matching `Content-Range`, or — when a
//    server ignores `Range` and answers `200` — by streaming just the requested bytes and
//    cancelling, capped at `maxFallbackBytes`. It never silently downloads a whole model.
//  - `Range` survives redirects (HF resolve → CDN, GitHub release → object storage);
//    `Authorization` is only ever sent to the token's own host family.
//

import Foundation

public enum FetchError: Error, LocalizedError {
    case http(status: Int, url: URL, body: String)
    case rangeNotSupported(URL)
    case badContentRange(URL, String)
    case budgetExceeded(needed: Int, budget: Int)
    case invalidResponse(URL)

    public var errorDescription: String? {
        switch self {
        case .http(let status, let url, let body):
            let hint = status == 401 || status == 403 ? " (gated/private? set HF_TOKEN or GITHUB_TOKEN)" : ""
            return "HTTP \(status) for \(url.absoluteString)\(hint)\(body.isEmpty ? "" : ": \(body.prefix(200))")"
        case .rangeNotSupported(let url):
            return "Server ignored the byte-range request for \(url.absoluteString) and the file is too large to stream"
        case .badContentRange(let url, let v): return "Unexpected Content-Range '\(v)' from \(url.absoluteString)"
        case .budgetExceeded(let needed, let budget):
            return "Fetch plan needs \(ByteFormat.string(needed)) but the budget is \(ByteFormat.string(budget))"
        case .invalidResponse(let url): return "Invalid response from \(url.absoluteString)"
        }
    }
}

/// Tokens for gated/private repos. Each is only sent to its own hosts.
public struct FetchAuth: Sendable {
    public var huggingFaceToken: String?
    public var gitHubToken: String?

    public init(huggingFaceToken: String? = nil, gitHubToken: String? = nil) {
        self.huggingFaceToken = huggingFaceToken
        self.gitHubToken = gitHubToken
    }

    public static func fromEnvironment(_ env: [String: String] = ProcessInfo.processInfo.environment) -> FetchAuth {
        FetchAuth(huggingFaceToken: env["HF_TOKEN"] ?? env["HUGGING_FACE_HUB_TOKEN"],
                  gitHubToken: env["GITHUB_TOKEN"] ?? env["GH_TOKEN"])
    }

    static let huggingFaceHosts: Set<String> = ["huggingface.co", "hf.co"]
    static let gitHubHosts: Set<String> = [
        "github.com", "api.github.com", "raw.githubusercontent.com", "media.githubusercontent.com",
    ]

    /// Authorization header value for `url`, or nil. Presigned CDN hosts never get a token.
    public func authorization(for url: URL) -> String? {
        guard let host = url.host?.lowercased() else { return nil }
        if Self.huggingFaceHosts.contains(host), let t = huggingFaceToken { return "Bearer \(t)" }
        if Self.gitHubHosts.contains(host), let t = gitHubToken { return "Bearer \(t)" }
        return nil
    }
}

public final class RangeFetcher: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    public let ledger: FetchLedger
    public let auth: FetchAuth
    let cache: FetchCache?
    /// Largest prefix we will stream from a server that ignores `Range`.
    public let maxFallbackBytes: Int

    private var session: URLSession!
    private let lock = NSLock()
    private var states: [Int: TaskState] = [:]

    private final class TaskState {
        let range: Range<Int>?
        let limit: Int?
        let continuation: CheckedContinuation<Response, Error>
        var status = 0
        var contentRange: String?
        var headers: [AnyHashable: Any] = [:]
        var body = Data()
        var streamOffset = 0
        var fallback = false
        var finishedEarly = false

        init(range: Range<Int>?, limit: Int?, continuation: CheckedContinuation<Response, Error>) {
            self.range = range
            self.limit = limit
            self.continuation = continuation
        }
    }

    struct Response {
        let status: Int
        let data: Data
        let headers: [AnyHashable: Any]
        /// Total file size, when the server reported one.
        let totalSize: Int?
        /// Range was ignored by the server and satisfied by streaming a prefix.
        var fallback = false
    }

    public init(auth: FetchAuth = .fromEnvironment(),
                cache: FetchCache? = FetchCache(),
                ledger: FetchLedger = FetchLedger(),
                configuration: URLSessionConfiguration = .default,
                maxFallbackBytes: Int = 64 << 20) {
        self.auth = auth
        self.cache = cache
        self.ledger = ledger
        self.maxFallbackBytes = maxFallbackBytes
        super.init()
        configuration.httpAdditionalHeaders = ["User-Agent": "Veil/\(VeilInfo.version)"]
        self.session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
    }

    /// Break the session ↔ delegate cycle once the fetcher is no longer needed.
    public func invalidate() { session.finishTasksAndInvalidate() }

    // MARK: - Public API

    /// Fetch `range` of `url`. `cacheable` only for URLs pinned to an immutable revision.
    public func fetch(_ url: URL, range: Range<Int>, cacheable: Bool = true) async throws -> Data {
        if range.isEmpty { return Data() }
        if cacheable, let hit = cache?.data(for: url, range: range) {
            ledger.recordCached(hit.count)
            return hit
        }
        let response = try await perform(url: url, range: range, limit: nil)
        try Self.check(response, url: url)
        guard response.data.count == range.count else {
            throw FetchError.badContentRange(url, "got \(response.data.count) bytes for \(range.count)")
        }
        if cacheable { cache?.store(response.data, for: url, range: range) }
        return response.data
    }

    /// Fetch many ranges of one file, merging ranges separated by ≤ `gapTolerance` bytes
    /// into single requests (tensors of one module are usually adjacent).
    public func fetch(_ url: URL, ranges: [Range<Int>], gapTolerance: Int = 64 << 10,
                      cacheable: Bool = true, concurrency: Int = 8) async throws -> [Range<Int>: Data] {
        let merged = Self.coalesce(ranges, gapTolerance: gapTolerance)
        var blobs: [Range<Int>: Data] = [:]
        try await withThrowingTaskGroup(of: (Range<Int>, Data).self) { group in
            var iterator = merged.makeIterator()
            var inFlight = 0
            func launchNext() -> Bool {
                guard let r = iterator.next() else { return false }
                group.addTask { (r, try await self.fetch(url, range: r, cacheable: cacheable)) }
                return true
            }
            while inFlight < concurrency, launchNext() { inFlight += 1 }
            while let (r, data) = try await group.next() {
                blobs[r] = data
                if !launchNext() { inFlight -= 1 }
            }
        }
        var out: [Range<Int>: Data] = [:]
        for r in ranges {
            guard let block = merged.first(where: { $0.contains(r.lowerBound) && $0.upperBound >= r.upperBound }),
                  let data = blobs[block] else { continue }
            let start = r.lowerBound - block.lowerBound
            out[r] = data.subdata(in: data.startIndex + start..<data.startIndex + start + r.count)
        }
        return out
    }

    /// Whole small file (JSON, README) with a hard size cap.
    public func fetchSmallFile(_ url: URL, limit: Int = 4 << 20, cacheable: Bool = true) async throws -> Data {
        if cacheable, let hit = cache?.data(for: url, range: nil) {
            ledger.recordCached(hit.count)
            return hit
        }
        let response = try await perform(url: url, range: nil, limit: limit)
        try Self.check(response, url: url)
        if cacheable { cache?.store(response.data, for: url, range: nil) }
        return response.data
    }

    /// GET + JSON decode for REST APIs (never cached: refs can move).
    public func getJSON(_ url: URL, limit: Int = 32 << 20) async throws -> (Any, [AnyHashable: Any]) {
        let response = try await perform(url: url, range: nil, limit: limit,
                                         accept: "application/json")
        try Self.check(response, url: url)
        return (try JSONSerialization.jsonObject(with: response.data), response.headers)
    }

    /// Size of a remote file from a 1-byte ranged request (Content-Range total).
    public func remoteSize(_ url: URL) async throws -> Int? {
        let response = try await perform(url: url, range: 0..<1, limit: nil)
        try Self.check(response, url: url)
        return response.totalSize
    }

    // MARK: - Helpers

    static func coalesce(_ ranges: [Range<Int>], gapTolerance: Int) -> [Range<Int>] {
        let sorted = ranges.filter { !$0.isEmpty }.sorted { $0.lowerBound < $1.lowerBound }
        var out: [Range<Int>] = []
        for r in sorted {
            if let last = out.last, r.lowerBound <= last.upperBound + gapTolerance {
                out[out.count - 1] = last.lowerBound..<max(last.upperBound, r.upperBound)
            } else {
                out.append(r)
            }
        }
        return out
    }

    static func check(_ response: Response, url: URL) throws {
        guard (200..<300).contains(response.status) else {
            throw FetchError.http(status: response.status, url: url,
                                  body: String(decoding: response.data.prefix(512), as: UTF8.self))
        }
    }

    static func parseContentRange(_ value: String) -> (start: Int, end: Int, total: Int?)? {
        // "bytes 0-7/12345" or "bytes 0-7/*"
        let parts = value.replacingOccurrences(of: "bytes ", with: "").split(separator: "/")
        guard parts.count == 2 else { return nil }
        let span = parts[0].split(separator: "-")
        guard span.count == 2, let s = Int(span[0]), let e = Int(span[1]) else { return nil }
        return (s, e, Int(parts[1]))
    }

    private func perform(url: URL, range: Range<Int>?, limit: Int?,
                         accept: String? = nil) async throws -> Response {
        var request = URLRequest(url: url)
        if let range { request.setValue("bytes=\(range.lowerBound)-\(range.upperBound - 1)", forHTTPHeaderField: "Range") }
        if let accept { request.setValue(accept, forHTTPHeaderField: "Accept") }
        if let authorization = auth.authorization(for: url) {
            request.setValue(authorization, forHTTPHeaderField: "Authorization")
        }
        ledger.recordRequest()
        let response: Response = try await withCheckedThrowingContinuation { continuation in
            let task = session.dataTask(with: request)
            lock.withLock { states[task.taskIdentifier] = TaskState(range: range, limit: limit, continuation: continuation) }
            task.resume()
        }
        if let range, response.status == 206, !response.fallback {
            guard let cr = response.headers.first(where: { ($0.key as? String)?.lowercased() == "content-range" })?.value as? String,
                  let parsed = Self.parseContentRange(cr),
                  parsed.start == range.lowerBound else {
                throw FetchError.badContentRange(url, "missing or mismatched")
            }
        }
        return response
    }

    private func state(_ task: URLSessionTask) -> TaskState? {
        lock.withLock { states[task.taskIdentifier] }
    }

    // MARK: - URLSessionDataDelegate

    public func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                           didReceive response: URLResponse,
                           completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        guard let st = state(dataTask), let http = response as? HTTPURLResponse else {
            completionHandler(.cancel)
            return
        }
        st.status = http.statusCode
        st.headers = http.allHeaderFields
        if let range = st.range, http.statusCode == 200 {
            // Range ignored: stream the prefix we need, if it's small enough.
            guard range.upperBound <= maxFallbackBytes else {
                st.status = -1
                completionHandler(.cancel)
                return
            }
            st.fallback = true
        }
        completionHandler(.allow)
    }

    public func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard let st = state(dataTask) else { return }
        ledger.recordNetwork(data.count)
        if st.fallback, let range = st.range {
            let chunk = st.streamOffset..<(st.streamOffset + data.count)
            st.streamOffset += data.count
            let lo = max(chunk.lowerBound, range.lowerBound), hi = min(chunk.upperBound, range.upperBound)
            if lo < hi {
                st.body.append(data.subdata(in: data.startIndex + (lo - chunk.lowerBound)..<data.startIndex + (hi - chunk.lowerBound)))
            }
            if st.streamOffset >= range.upperBound {
                st.finishedEarly = true
                dataTask.cancel()
            }
            return
        }
        st.body.append(data)
        if let limit = st.limit, st.body.count > limit {
            st.status = -2
            dataTask.cancel()
        }
    }

    public func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let st = lock.withLock({ states.removeValue(forKey: task.taskIdentifier) }) else { return }
        let url = task.originalRequest?.url ?? URL(string: "about:blank")!
        if st.status == -1 { return st.continuation.resume(throwing: FetchError.rangeNotSupported(url)) }
        if st.status == -2 { return st.continuation.resume(throwing: FetchError.budgetExceeded(needed: st.body.count, budget: st.limit ?? 0)) }
        if let error, !st.finishedEarly { return st.continuation.resume(throwing: error) }
        var total: Int?
        if let cr = st.headers.first(where: { ($0.key as? String)?.lowercased() == "content-range" })?.value as? String {
            total = Self.parseContentRange(cr)?.total
        } else if st.fallback, let len = st.headers.first(where: { ($0.key as? String)?.lowercased() == "content-length" })?.value as? String {
            total = Int(len)
        }
        // A satisfied fallback is reported as 206 so callers see uniform semantics.
        let status = st.fallback ? 206 : st.status
        st.continuation.resume(returning: Response(status: status, data: st.body, headers: st.headers,
                                                   totalSize: total, fallback: st.fallback))
    }

    public func urlSession(_ session: URLSession, task: URLSessionTask,
                           willPerformHTTPRedirection response: HTTPURLResponse,
                           newRequest request: URLRequest,
                           completionHandler: @escaping (URLRequest?) -> Void) {
        var next = request
        if let range = task.originalRequest?.value(forHTTPHeaderField: "Range") {
            next.setValue(range, forHTTPHeaderField: "Range")
        }
        // Re-derive auth for the new host: presigned CDN URLs must not receive tokens.
        next.setValue(nil, forHTTPHeaderField: "Authorization")
        if let url = next.url, let authorization = auth.authorization(for: url) {
            next.setValue(authorization, forHTTPHeaderField: "Authorization")
        }
        completionHandler(next)
    }
}

public enum ByteFormat {
    public static func string(_ bytes: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }
}
