//
//  ToyIdentityWorld.swift
//  VeilKit
//
//  An analytic world where the answer is known, so every stage can be checked exactly.
//
//  - Text: words → keyed token embeddings (E = 96); a causal running mean plays the text
//    encoder (padding positions carry the whole prompt, as in a causal LM), padded to T = 16.
//    The first 32 embedding dimensions are an identity-code space: a known person's name tokens
//    carry that person's code; every other word leaks only a little into it.
//  - Model: `context_embedder` (E → H) and `reader` (H → D) are linear slots, both hooked, so
//    guards apply exactly as on a real model. The conditioning mean is m(c) = reader(mean_t
//    context_embedder(h_t)); the reader turns identity codes into identity directions.
//  - Data: x₀ | c ~ N(m(c), s²·I) on a 3×8×8 latent. The velocity v̂ = E[ε − x₀ | x_σ] is exact
//    and differentiable (Gaussian posterior).
//  - Identities: four people, two of them red-haired (a shared attribute direction). Each known
//    name lands on its person's mean; the subject's description ("…painter from lisbon") carries
//    her code; "red-haired" carries the attribute. Unknown names (strangers, Veil's invented null
//    names) read as a generic person plus a small leak — as a real model treats a name it
//    doesn't know.
//  - Photos: latents drawn around each identity's mean, stored as 64×64 RGB PNGs (8×8
//    blocks), so the whole pipeline runs from image files.
//

import CoreGraphics
import Foundation
import MLX
import MLXLinalg

/// The toy's tokenizer and text encoder.
public struct ToyText: @unchecked Sendable {
    public let words: [String]
    let wordIndex: [String: Int]
    public let bucketCount = 256
    public let tokenDim: Int
    public let sequence: Int
    public let beginID = 0, endID = 1, padID = 2
    /// (V, E) token embeddings.
    public let table: MLXArray

    public var vocabularySize: Int { words.count + bucketCount }

    init(words: [String], table: MLXArray, sequence: Int) {
        self.words = words
        wordIndex = Dictionary(uniqueKeysWithValues: words.enumerated().map { ($1, $0) })
        self.tokenDim = table.dim(1)
        self.sequence = sequence
        self.table = table
    }

    static func split(_ text: String) -> [String] {
        text.lowercased().components(separatedBy: CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-'")).inverted)
            .map { $0.trimmingCharacters(in: CharacterSet(charactersIn: "'")) }
            .filter { !$0.isEmpty }
    }

    static func isBucketWord(_ w: String) -> Bool { w.hasPrefix("zq") && Int(w.dropFirst(2)) != nil }

    public func tokenID(_ word: String) -> Int {
        if let i = wordIndex[word] { return i }
        if Self.isBucketWord(word), let n = Int(word.dropFirst(2)), n < bucketCount { return words.count + n }
        return words.count + Int(Hashing.seed("veil.toy.word|" + word) % UInt64(bucketCount))
    }

    public func word(_ id: Int) -> String { id < words.count ? words[id] : "zq\(id - words.count)" }

    /// Real token ids: <u> + words (truncated) + <e>.
    public func tokenize(_ text: String) -> [Int] {
        [beginID] + Self.split(text).prefix(sequence - 2).map(tokenID) + [endID]
    }

    /// Causal running mean over the real tokens (1, n, E) → (1, T, E); padding carries the full mean.
    public func contextualize(_ embeddings: MLXArray) -> MLXArray {
        let n = embeddings.dim(1)
        let counts = MLXArray((1...n).map { Float($0) }).reshaped(1, n, 1)
        let running = cumsum(embeddings.asType(.float32), axis: 1) / counts
        guard n < sequence else { return running[0..., ..<sequence, 0...] }
        let last = running[0..., (n - 1)..<n, 0...]
        return concatenated([running, broadcast(last, to: [embeddings.dim(0), sequence - n, tokenDim])], axis: 1)
    }

    public func encode(_ text: String) -> PromptEmbedding {
        let ids = tokenize(text)
        let e = table[MLXArray(ids.map(Int32.init))].expandedDimensions(axis: 0)
        return PromptEmbedding(text: text, value: contextualize(e), tokenIDs: ids)
    }
}

public final class ToyIdentityWorld: @unchecked Sendable {
    public struct Identity: Sendable {
        public let name: String
        public let redHaired: Bool
        public let photoCount: Int
    }

    public static let subjectName = "Ada Quill"
    /// A description that reaches the subject specifically.
    public static let subjectDescription = "the red-haired painter from lisbon"
    /// An attribute prompt that must keep working for everyone with the attribute.
    public static let attributePrompt = "a photo of a red-haired person"

    public static let identities: [Identity] = [
        Identity(name: subjectName, redHaired: true, photoCount: 6),
        Identity(name: "Cleo Marsh", redHaired: true, photoCount: 4),
        Identity(name: "Bram Oake", redHaired: false, photoCount: 4),
        Identity(name: "Dov Reyes", redHaired: false, photoCount: 4),
    ]

    public static let shared = ToyIdentityWorld()

    /// Names the toy knows only as "a person" (distinct syllables from Veil's null names).
    static let strangers: [String] = {
        let first = ["Yuna", "Oskar", "Ines", "Tomas", "Wren", "Pavel", "Nia", "Felix", "Greta", "Iker", "Suki", "Lev"]
        let last = ["Holm", "Quade", "Birch", "Nakamura", "Ferro", "Lindqvist", "Okafor", "Brandt", "Castell", "Mireles"]
        return (0..<30).map { "\(first[$0 % first.count]) \(last[($0 * 7) % last.count])" }
    }()

    public var identities: [Identity] { Self.identities }
    public let contextDim: Int
    public let channels = 3
    public let side = 8
    public var latentDim: Int { channels * side * side }
    /// Photo noise around an identity's mean (the data std s).
    public let photoStd: Float = 0.4

    public let text: ToyText
    /// Latent means: identity name → (D); "<attribute>" is the shared red-hair direction.
    public let means: [String: MLXArray]
    /// (H, E)
    public let contextWeight: MLXArray
    /// (D, H), fitted.
    public let readerWeight: MLXArray
    /// How closely the fitted reader lands the design prompts (1 − residual/target energy).
    public let readerFit: Double

    /// Identity-code space: the first `codeDim` embedding and context dimensions.
    public static let codeDim = 32
    /// How strongly an ordinary word leaks into the identity-code space (relative to a name).
    public static let leak: Float = 0.12

    public init(schedule: SeedSchedule = .research) {
        let latentDim = 3 * 8 * 8, tokenDim = 96, codeDim = Self.codeDim, contextDim = 64
        let identities = Self.identities

        // Vocabulary and structured embeddings: [identity code (32) | everything else (64)].
        var vocab = ["<u>", "<e>", "<pad>"]
        var seen = Set(vocab)
        for t in Templates.name.map({ Templates.fill($0, "") }) + Templates.generic + Templates.people + identities.map(\.name)
            + Self.strangers + [Self.subjectDescription, Self.attributePrompt, "a person a man a woman"] {
            for w in ToyText.split(t) where !seen.contains(w) && !ToyText.isBucketWord(w) {
                seen.insert(w)
                vocab.append(w)
            }
        }
        // Orthonormal codes: one per identity, one for the attribute.
        let q = MLXLinalg.qr(schedule.normal(.toy, 4, shape: [codeDim, identities.count + 1]), stream: .cpu).0
        eval(q)
        let codes = q.transposed()
        let attributeCode = codes[identities.count]
        var codeOf: [String: MLXArray] = ["red-haired": attributeCode]
        for (k, identity) in identities.enumerated() {
            for w in ToyText.split(identity.name) { codeOf[w] = codes[k] }
        }
        let subjectIndex = identities.firstIndex { $0.name == Self.subjectName }!
        for w in ["painter", "lisbon"] { codeOf[w] = codes[subjectIndex] }

        let v = vocab.count + 256
        let rest = schedule.normal(.toy, 1, shape: [v, tokenDim - codeDim]) / Float(tokenDim).squareRoot()
        let leaks = schedule.normal(.toy, 5, shape: [v, codeDim]) * (Self.leak / Float(codeDim).squareRoot())
        var idRows: [MLXArray] = []
        for i in 0..<v {
            idRows.append(i < vocab.count ? (codeOf[vocab[i]] ?? leaks[i]) : leaks[i])
        }
        let table = concatenated([stacked(idRows, axis: 0), rest], axis: 1)
        let text = ToyText(words: vocab, table: table, sequence: 16)

        // context_embedder: identity codes pass through; the rest is a random projection.
        let projection = schedule.normal(.toy, 2, shape: [contextDim - codeDim, tokenDim - codeDim]) / Float(tokenDim - codeDim).squareRoot()
        let contextWeight = concatenated([
            concatenated([MLXArray.eye(codeDim), MLXArray.zeros([codeDim, tokenDim - codeDim])], axis: 1),
            concatenated([MLXArray.zeros([contextDim - codeDim, codeDim]), projection], axis: 1),
        ], axis: 0)

        // Means: an identity direction each, plus the shared attribute for the red-haired.
        let attribute = schedule.normal(.toy, 3, shape: [latentDim]) * 0.6
        var own: [MLXArray] = []
        var means: [String: MLXArray] = ["<attribute>": attribute]
        for (k, identity) in identities.enumerated() {
            own.append(schedule.normal(.toy, 10 + k, shape: [latentDim]) * 0.6)
            means[identity.name] = identity.redHaired ? own[k] + attribute : own[k]
        }

        // Reader: identity code → that person's mean, attribute code → the attribute, scaled so
        // each known name lands on its mean on average over the templates (the pooled mean dilutes
        // a name by the length of its template).
        func pooledCode(_ prompt: String) -> MLXArray { text.encode(prompt).value[0].mean(axis: 0)[..<codeDim] }
        var columns = MLXArray.zeros([latentDim, codeDim])
        var fitError = 0.0, fitCount = 0
        for (k, identity) in identities.enumerated() {
            let weights = Templates.name.map { t in (pooledCode(Templates.fill(t, identity.name)) * codes[k]).sum().item(Float.self) }
            let scale = 1 / max(weights.reduce(0, +) / Float(weights.count), 1e-6)
            columns = columns + scale * matmul(means[identity.name]!.reshaped(latentDim, 1), codes[k].reshaped(1, codeDim))
        }
        let attrWeights = Templates.name.prefix(8).map { t in (pooledCode(Templates.fill(t, "a red-haired person")) * attributeCode).sum().item(Float.self) }
        let attrScale = 1 / max(attrWeights.reduce(0, +) / Float(attrWeights.count), 1e-6)
        columns = columns + attrScale * matmul(attribute.reshaped(latentDim, 1), attributeCode.reshaped(1, codeDim))
        let readerWeight = concatenated([columns, MLXArray.zeros([latentDim, contextDim - codeDim])], axis: 1)
        for identity in identities {
            let mean = means[identity.name]!
            for t in Templates.name {
                let m = matmul(readerWeight, matmul(contextWeight, text.encode(Templates.fill(t, identity.name)).value[0].transposed()).mean(axis: 1))
                fitError += Double(((m - mean).square().sum() / mean.square().sum()).item(Float.self))
                fitCount += 1
            }
        }

        self.text = text
        self.means = means
        self.contextWeight = contextWeight
        self.readerWeight = readerWeight
        self.readerFit = 1 - fitError / Double(max(fitCount, 1))
        self.contextDim = contextDim
    }

    // MARK: Photos

    /// Keyed photos of one identity as images (64×64 RGB, 8×8 blocks).
    public func photos(of name: String) -> [SubjectPhoto] {
        guard let k = identities.firstIndex(where: { $0.name == name }), let mean = means[name] else { return [] }
        return (0..<identities[k].photoCount).map { j in
            let latent = mean + SeedSchedule.research.normal(.toy, 1000 + 100 * k + j, shape: [latentDim]) * photoStd
            return SubjectPhoto(image: Self.image(latent.reshaped(channels, side, side)), name: "\(name) \(j + 1).png")
        }
    }

    static let range: Float = 4

    /// (3, 8, 8) latent → 64×64 RGB image.
    static func image(_ latent: MLXArray) -> CGImage {
        let values = latent.asType(.float32).asArray(Float.self)
        let side = latent.dim(1), scale = 8, w = side * scale
        var px = [UInt8](repeating: 255, count: w * w * 4)
        for y in 0..<w {
            for x in 0..<w {
                let (i, j) = (y / scale, x / scale)
                for c in 0..<3 {
                    let v = values[c * side * side + i * side + j]
                    px[(y * w + x) * 4 + c] = UInt8(max(0, min(255, ((v + range) / (2 * range) * 255).rounded())))
                }
            }
        }
        return ImageWriter.image(rgba: px, width: w, height: w)!
    }

    /// Image → (3, 8, 8) latent (block centers).
    func latent(of photo: SubjectPhoto) -> MLXArray {
        let w = side * 8
        let px = photo.rgba(width: w, height: w)
        var out = [Float](repeating: 0, count: latentDim)
        for c in 0..<3 {
            for i in 0..<side {
                for j in 0..<side {
                    let p = ((i * 8 + 4) * w + (j * 8 + 4)) * 4 + c
                    out[c * side * side + i * side + j] = Float(px[p]) / 255 * 2 * Self.range - Self.range
                }
            }
        }
        return MLXArray(out, [channels, side, side])
    }

    /// Write the subject's photos and the controls (one folder per identity) for CLI demos.
    @discardableResult
    public func writePhotos(to directory: URL) throws -> (subject: URL, controls: URL) {
        let subject = directory.appendingPathComponent("subject", isDirectory: true)
        let controls = directory.appendingPathComponent("controls", isDirectory: true)
        for identity in identities {
            let dir = identity.name == Self.subjectName ? subject : controls.appendingPathComponent(identity.name, isDirectory: true)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            for photo in photos(of: identity.name) {
                try ImageWriter.writePNG(photo.image, to: dir.appendingPathComponent(photo.name))
            }
        }
        return (subject, controls)
    }

    public var subjectPhotos: [SubjectPhoto] { photos(of: Self.subjectName) }

    public var controls: ControlSet {
        var photos: [SubjectPhoto] = [], ids: [String?] = []
        for identity in identities where identity.name != Self.subjectName {
            let p = self.photos(of: identity.name)
            photos += p
            ids += Array(repeating: identity.name, count: p.count)
        }
        return ControlSet(photos: photos, identities: ids)
    }
}
