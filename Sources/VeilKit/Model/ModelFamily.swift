//
//  ModelFamily.swift
//  VeilKit
//
//  Link → what the model is. "Any model" holds at this layer: any Hugging Face or GitHub
//  repo resolves, and its configs say which family it belongs to. Running a family needs an
//  executor; families without one are named and reported as unsupported, not guessed at.
//

import Foundation

public struct ModelDescriptor: Codable, Sendable {
    /// The link as given.
    public var link: String
    /// A GitHub link that was followed to this model (its README pointed here).
    public var via: String?
    public var host: ModelHost
    public var repo: String
    /// Pinned commit, when the host has one.
    public var commit: String?
    /// e.g. "flux2-klein", "flux2-dev", "flux1", "sd3", "sdxl", "sd1", "qwen-image", "toy", "unknown".
    public var family: String
    /// What the configs said (class names, key dimensions).
    public var detail: String
    /// "base" (undistilled) or "distilled", when known.
    public var variant: String?
    /// Parameter scale, e.g. "4B", "9B".
    public var size: String?
    public var license: String?
    /// Components present (transformer, text_encoder, vae, tokenizer, …).
    public var components: [String]
    /// Whether an executor is registered for the family.
    public var supported: Bool
    public var notes: [String]

    public init(link: String, via: String? = nil, host: ModelHost, repo: String, commit: String? = nil, family: String,
                detail: String, variant: String? = nil, size: String? = nil, license: String? = nil,
                components: [String] = [], supported: Bool = false, notes: [String] = []) {
        self.link = link
        self.via = via
        self.host = host
        self.repo = repo
        self.commit = commit
        self.family = family
        self.detail = detail
        self.variant = variant
        self.size = size
        self.license = license
        self.components = components
        self.supported = supported
        self.notes = notes
    }

    /// `owner/name@commit` for hosted models; a local folder's name only (never its path, which
    /// carries the user's account name into shared files).
    public var pinnedName: String {
        if host == .local && family != "toy" { return (repo as NSString).lastPathComponent + " (local)" }
        return commit.map { "\(repo)@\(String($0.prefix(12)))" } ?? repo
    }

    /// The same descriptor with local paths reduced to folder names, for reports that get shared.
    public var shareable: ModelDescriptor {
        guard host == .local && family != "toy" else { return self }
        var d = self
        d.link = (link as NSString).lastPathComponent
        d.repo = (repo as NSString).lastPathComponent
        return d
    }
}

public enum FamilyDetector {
    public struct Classification: Sendable, Equatable {
        public var family: String
        public var detail: String
        public var variant: String?
        public var size: String?
    }

    /// Describe a resolved repo by reading its small config files.
    public static func describe(_ source: ResolvedSource, via: String?, fetcher: RangeFetcher) async -> ModelDescriptor {
        func json(_ path: String) async -> [String: Any]? {
            guard let file = source.files.first(where: { $0.path == path }),
                  let data = try? await fetcher.fetchSmallFile(file.url, cacheable: source.pinned) else { return nil }
            return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        }
        let modelIndex = await json("model_index.json")
        let transformer = await json("transformer/config.json")
        let unet = await json("unet/config.json")
        var readme: String?
        if let file = source.files.first(where: { $0.path.lowercased() == "readme.md" }),
           let data = try? await fetcher.fetchSmallFile(file.url, limit: 1 << 20, cacheable: source.pinned) {
            readme = String(decoding: data, as: UTF8.self)
        }
        let paths = source.files.map(\.path)
        let c = classify(modelIndex: modelIndex, transformer: transformer, unet: unet, readme: readme, repo: source.reference.repo,
                         paths: paths)
        let components = ["transformer", "text_encoder", "text_encoder_2", "vae", "tokenizer", "unet"].filter { comp in
            paths.contains { $0.hasPrefix(comp + "/") }
        }
        var license: String?
        if case .string(let l)? = source.cardData["license"] { license = l }
        return ModelDescriptor(link: via ?? source.reference.original, via: via, host: source.reference.host,
                               repo: source.reference.repo, commit: source.commit, family: c.family, detail: c.detail,
                               variant: c.variant, size: c.size, license: license, components: components)
    }

    /// Describe a model folder on this machine (diffusers layout or an mflux export).
    public static func describeLocal(_ directory: URL) -> ModelDescriptor {
        func json(_ path: String) -> [String: Any]? {
            guard let data = try? Data(contentsOf: directory.appendingPathComponent(path)) else { return nil }
            return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        }
        let readme = try? String(contentsOf: directory.appendingPathComponent("README.md"), encoding: .utf8)
        let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)
        var paths: [String] = []
        while let url = enumerator?.nextObject() as? URL, paths.count < 5000 {
            paths.append(String(url.path.dropFirst(directory.path.count + 1)))
        }
        let c = classify(modelIndex: json("model_index.json"), transformer: json("transformer/config.json"), unet: json("unet/config.json"),
                         readme: readme, repo: directory.lastPathComponent, paths: paths)
        let components = ["transformer", "text_encoder", "text_encoder_2", "vae", "tokenizer", "unet"].filter { comp in
            paths.contains { $0.hasPrefix(comp + "/") }
        }
        return ModelDescriptor(link: directory.path, host: .local, repo: directory.path, family: c.family, detail: c.detail,
                               variant: c.variant, size: c.size, components: components)
    }

    /// Pure classification over config contents (testable offline).
    public static func classify(modelIndex: [String: Any]?, transformer: [String: Any]?, unet: [String: Any]?,
                                readme: String?, repo: String, paths: [String]) -> Classification {
        let text = ((readme ?? "") + " " + repo).lowercased()
        let variant: String? = {
            if text.contains("klein-base") || text.contains("klein_base") || text.contains("undistilled") { return "base" }
            if text.contains("distilled") || text.contains("step-distilled") { return "distilled" }
            if repo.lowercased().contains("klein") && !repo.lowercased().contains("base") { return "distilled" }
            return nil
        }()
        if let t = transformer, let cls = t["_class_name"] as? String {
            let joint = t["joint_attention_dim"] as? Int
            switch cls {
            case "Flux2Transformer2DModel":
                // Klein 4B reads Qwen3-4B (3 × 2560), Klein 9B Qwen3-8B (3 × 4096); FLUX.2 [dev]
                // reads Mistral Small (3 × 5120).
                switch joint {
                case 7680: return Classification(family: "flux2-klein", detail: "\(cls) · joint \(7680)", variant: variant, size: "4B")
                case 12288: return Classification(family: "flux2-klein", detail: "\(cls) · joint 12288", variant: variant, size: "9B")
                default: return Classification(family: "flux2-dev", detail: "\(cls) · joint \(joint.map(String.init) ?? "?")", variant: variant, size: nil)
                }
            case "FluxTransformer2DModel": return Classification(family: "flux1", detail: cls, variant: variant, size: "12B")
            case "SD3Transformer2DModel": return Classification(family: "sd3", detail: cls, variant: nil, size: nil)
            case "QwenImageTransformer2DModel": return Classification(family: "qwen-image", detail: cls, variant: nil, size: nil)
            default: return Classification(family: "unknown", detail: cls, variant: variant, size: nil)
            }
        }
        if let u = unet, let cls = u["_class_name"] as? String {
            let cross = u["cross_attention_dim"] as? Int
            switch cross {
            case 2048: return Classification(family: "sdxl", detail: "\(cls) · cross 2048", variant: nil, size: nil)
            case 1024: return Classification(family: "sd2", detail: "\(cls) · cross 1024", variant: nil, size: nil)
            case 768: return Classification(family: "sd1", detail: "\(cls) · cross 768", variant: nil, size: nil)
            default: return Classification(family: "unknown", detail: cls, variant: nil, size: nil)
            }
        }
        if let cls = modelIndex?["_class_name"] as? String {
            if cls.hasPrefix("Flux2") { return Classification(family: "flux2-klein", detail: cls, variant: variant, size: nil) }
            return Classification(family: "unknown", detail: cls, variant: variant, size: nil)
        }
        // mflux-style exports: per-component folders with an index, no diffusers configs.
        if paths.contains(where: { $0.hasPrefix("transformer/") }) && text.contains("klein") {
            return Classification(family: "flux2-klein", detail: "mflux export", variant: variant, size: text.contains("9b") ? "9B" : "4B")
        }
        if paths.contains(where: { $0.hasSuffix(".safetensors") }) && paths.count <= 4 {
            return Classification(family: "adapter", detail: "single weight file (an adapter, not a base model)", variant: nil, size: nil)
        }
        return Classification(family: "unknown", detail: "no diffusers config found", variant: variant, size: nil)
    }
}
