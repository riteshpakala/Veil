//
//  FetchCache.swift
//  VeilKit
//  Adapted from Scorpion (github.com/riteshpakala/Scorpion, ScorpionKit), GPL-3.0.
//
//  Content cache for fetched byte ranges, keyed by (URL, range). Callers only cache URLs
//  pinned to an immutable revision (a commit sha), so entries never go stale.
//

import CryptoKit
import Foundation

public final class FetchCache: @unchecked Sendable {
    public let directory: URL
    private let fm = FileManager.default

    public static var defaultDirectory: URL {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return base.appendingPathComponent("Veil", isDirectory: true)
    }

    public init(directory: URL = FetchCache.defaultDirectory) {
        self.directory = directory.appendingPathComponent("ranges", isDirectory: true)
        try? fm.createDirectory(at: self.directory, withIntermediateDirectories: true)
    }

    static func key(_ url: URL, _ range: Range<Int>?) -> String {
        let raw = url.absoluteString + "#" + (range.map { "\($0.lowerBound)-\($0.upperBound)" } ?? "all")
        return SHA256.hash(data: Data(raw.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    private func path(_ key: String) -> URL {
        directory.appendingPathComponent(String(key.prefix(2)), isDirectory: true)
            .appendingPathComponent(key)
    }

    public func data(for url: URL, range: Range<Int>?) -> Data? {
        try? Data(contentsOf: path(Self.key(url, range)), options: .mappedIfSafe)
    }

    public func store(_ data: Data, for url: URL, range: Range<Int>?) {
        let p = path(Self.key(url, range))
        try? fm.createDirectory(at: p.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: p, options: .atomic)
    }
}

/// Byte accounting for one scan: what came over the network vs. from cache.
public final class FetchLedger: @unchecked Sendable {
    private let lock = NSLock()
    private var _networkBytes = 0
    private var _cachedBytes = 0
    private var _requests = 0
    /// Called (off the main thread) with the running network byte count.
    public var onProgress: ((Int) -> Void)?

    public init() {}

    public var networkBytes: Int { lock.withLock { _networkBytes } }
    public var cachedBytes: Int { lock.withLock { _cachedBytes } }
    public var requests: Int { lock.withLock { _requests } }

    func recordNetwork(_ n: Int) {
        let total = lock.withLock { () -> Int in
            _networkBytes += n
            return _networkBytes
        }
        onProgress?(total)
    }

    func recordRequest() { lock.withLock { _requests += 1 } }
    func recordCached(_ n: Int) { lock.withLock { _cachedBytes += n } }
}
