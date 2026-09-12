//
//  ExecutorRegistry.swift
//  VeilKit
//
//  Family → executor. VeilKit knows no model family; the composition roots (CLI, app) register
//  executors (the toy world here, FLUX.2 Klein from VeilFlux2).
//

import Foundation

public struct ExecutorRequest: Sendable {
    public let descriptor: ModelDescriptor
    public let source: ResolvedSource?
    /// Executor options (`--option key=value`).
    public let options: [String: String]
    /// An adapter that is part of the deployment (local path or link), applied to every run.
    public let withAdapter: URL?
    public let fetcher: RangeFetcher?
    public let progress: (@Sendable (String) -> Void)?

    public init(descriptor: ModelDescriptor, source: ResolvedSource?, options: [String: String] = [:],
                withAdapter: URL? = nil, fetcher: RangeFetcher? = nil, progress: (@Sendable (String) -> Void)? = nil) {
        self.descriptor = descriptor
        self.source = source
        self.options = options
        self.withAdapter = withAdapter
        self.fetcher = fetcher
        self.progress = progress
    }

    public func option(_ key: String) -> String? { options[key] }
}

public typealias ExecutorFactory = @Sendable (ExecutorRequest) async throws -> GuardableModel

public enum ExecutorError: Error, LocalizedError {
    case unsupportedFamily(ModelDescriptor, supported: [String])
    case adapterNotBase(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedFamily(let d, let supported):
            return "\(d.pinnedName) is a \(d.family) model (\(d.detail)). Veil has no executor for it yet; supported: \(supported.joined(separator: ", "))."
        case .adapterNotBase(let s):
            return "\(s) is a single adapter file, not a base model. Pass the base model as --model and this file as --with-adapter."
        }
    }
}

public final class ExecutorRegistry: @unchecked Sendable {
    public static let shared: ExecutorRegistry = {
        let r = ExecutorRegistry()
        r.register(family: "toy", factory: ToyExecutor.make)
        return r
    }()

    private var factories: [String: ExecutorFactory] = [:]
    private let lock = NSLock()

    public init() {}

    public func register(family: String, factory: @escaping ExecutorFactory) {
        lock.withLock { factories[family] = factory }
    }

    public var families: [String] { lock.withLock { factories.keys.sorted() } }

    /// Resolve and describe a link without loading weights.
    public func describe(_ link: String, fetcher: RangeFetcher) async throws -> (ModelDescriptor, ResolvedSource?) {
        if let toy = ToyExecutor.descriptor(for: link) {
            var d = toy
            d.supported = true
            return (d, nil)
        }
        let local = URL(fileURLWithPath: (link as NSString).expandingTildeInPath)
        var isDir: ObjCBool = false
        if !link.contains("://"), FileManager.default.fileExists(atPath: local.path, isDirectory: &isDir), isDir.boolValue {
            var d = FamilyDetector.describeLocal(local)
            d.supported = lock.withLock { factories[d.family] != nil }
            return (d, nil)
        }
        let (source, via) = try await SourceResolver(fetcher: fetcher).resolveFollowingModelCard(link)
        var descriptor = await FamilyDetector.describe(source, via: via, fetcher: fetcher)
        descriptor.supported = lock.withLock { factories[descriptor.family] != nil }
        return (descriptor, source)
    }

    /// Resolve, describe and load.
    public func load(_ link: String, options: [String: String] = [:], withAdapter: URL? = nil,
                     fetcher: RangeFetcher = RangeFetcher(), progress: (@Sendable (String) -> Void)? = nil) async throws -> GuardableModel {
        let (descriptor, source) = try await describe(link, fetcher: fetcher)
        if descriptor.family == "adapter" { throw ExecutorError.adapterNotBase(descriptor.pinnedName) }
        guard let factory = lock.withLock({ factories[descriptor.family] }) else {
            throw ExecutorError.unsupportedFamily(descriptor, supported: families.filter { $0 != "toy" })
        }
        return try await factory(ExecutorRequest(descriptor: descriptor, source: source, options: options,
                                                 withAdapter: withAdapter, fetcher: fetcher, progress: progress))
    }
}
