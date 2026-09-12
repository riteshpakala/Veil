//
//  ClosedFormEditor.swift
//  VeilKit
//
//  Guard, stage A: a closed-form edit of the slot that reads the text encoder's output (UCE-
//  style, Gandikota et al. 2024). The slot W should send every token of an erase prompt where
//  it sends the matching token of the anchor prompt, and leave preserve prompts where they are:
//
//      min_Δ  1/Nₑ Σ ‖(W+Δ)e − We*‖²  +  λ/Nₚ Σ ‖(W+Δ)p − Wp‖²  +  γ‖Δ·S‖²      (S² = channel variances)
//      Δ = W·D·A⁻¹,   D = 1/Nₑ Σ (e* − e)eᵀ,   A = 1/Nₑ Σ eeᵀ + λ/Nₚ Σ ppᵀ + γ·diag(…)
//
//  Exact on this slot, because its input is the text encoder's output: no image, no timestep.
//  Δ then becomes a LoRA by reduced-rank regression: the truncation is done in the data's own
//  metric A (SVD of Δ·L with A = L·Lᵀ), so the rank is spent where prompts actually live, not
//  on directions of the embedding space no prompt uses. The truncation error is reported in
//  that metric — the error on the edit's own inputs.
//
//  Token pairing between a prompt and its anchor (real tokens, then padding):
//    - the shared prefix is skipped (a causal encoder gives it identical embeddings);
//    - the differing span maps proportionally onto the anchor's span;
//    - the shared suffix and the padding align by offset from the span's end (in a causal
//      encoder they carry the name, so they must move too).
//

import Foundation
import MLX
import MLXLinalg

public struct ClosedFormEdit: @unchecked Sendable {
    public let key: String
    public let delta: LowRankDelta
    /// ‖(Δ − Δ_r)·L‖_F / ‖Δ·L‖_F: the rank-r error on the edit's own inputs (A = L·Lᵀ).
    public let truncationError: Double
    /// Share of ‖Δ·L‖²_F the kept rank captures.
    public let energy: Double
    public let erasePairs: Int
    public let preserveVectors: Int
    /// ‖Δ‖_F / ‖W‖_F.
    public let relativeSize: Double
}

public enum TokenAlignment {
    /// (prompt position, anchor position) pairs to edit, over `sequence` positions.
    public static func pairs(prompt: [Int]?, anchor: [Int]?, sequence: Int) -> [(Int, Int)] {
        guard let prompt, let anchor, !prompt.isEmpty, !anchor.isEmpty else {
            return (0..<sequence).map { ($0, $0) }
        }
        var prefix = 0
        while prefix < min(prompt.count, anchor.count), prompt[prefix] == anchor[prefix] { prefix += 1 }
        var suffix = 0
        while suffix < min(prompt.count, anchor.count) - prefix,
              prompt[prompt.count - 1 - suffix] == anchor[anchor.count - 1 - suffix] { suffix += 1 }
        let pSpan = prefix..<(prompt.count - suffix), aSpan = prefix..<(anchor.count - suffix)
        var out: [(Int, Int)] = []
        if !pSpan.isEmpty && !aSpan.isEmpty {
            for i in pSpan {
                let j = aSpan.lowerBound + (i - pSpan.lowerBound) * aSpan.count / pSpan.count
                out.append((i, j))
            }
        }
        // Shared suffix and padding, aligned from the span ends.
        var k = 0
        while pSpan.upperBound + k < sequence && aSpan.upperBound + k < sequence {
            out.append((pSpan.upperBound + k, aSpan.upperBound + k))
            k += 1
        }
        return out.filter { $0.0 < sequence && $0.1 < sequence }
    }
}

public enum ClosedFormEditor {
    /// - Parameters:
    ///   - weight: the deployed slot weight (out, in), float32.
    ///   - erase: prompt/anchor embedding pairs, (1, T, in) each.
    ///   - preserve: prompts whose every token should map as before.
    public static func fit(key: String, weight: MLXArray, erase: [(prompt: PromptEmbedding, anchor: PromptEmbedding)],
                           preserve: [PromptEmbedding], preserveWeight: Float, ridge: Float, maxRank: Int, energy: Double,
                           schedule: SeedSchedule) -> ClosedFormEdit {
        let inFeatures = weight.dim(1), outFeatures = weight.dim(0)
        // Accumulated per prompt, never concatenated: the Gram and the right-hand side are the
        // same arithmetic either way, but memory then doesn't grow with the number of people the
        // guard covers (7680 channels × a group's worth of padded prompts is gigabytes).
        let weightT = weight.transposed().asType(.float32)
        var gram = MLXArray.zeros([inFeatures, inFeatures])
        var rhs = MLXArray.zeros([inFeatures, outFeatures])      // (in, out)
        var rows = 0
        for (p, a) in erase {
            let t = p.value.dim(1)
            let pairs = TokenAlignment.pairs(prompt: p.tokenIDs, anchor: a.tokenIDs, sequence: min(t, a.value.dim(1)))
            guard !pairs.isEmpty else { continue }
            let pi = MLXArray(pairs.map { Int32($0.0) }), ai = MLXArray(pairs.map { Int32($0.1) })
            let e = p.value[0].asType(.float32)[pi]              // (n, in)
            let eStar = a.value[0].asType(.float32)[ai]
            gram = gram + matmul(e.transposed(), e)
            rhs = rhs + matmul(e.transposed(), matmul(eStar - e, weightT))
            rows += e.dim(0)
            eval(gram, rhs)
        }
        guard rows > 0 else {
            return ClosedFormEdit(key: key, delta: .zero(out: outFeatures, in: inFeatures, rank: 1), truncationError: 0,
                                  energy: 0, erasePairs: 0, preserveVectors: 0, relativeSize: 0)
        }
        let ne = Float(rows)
        gram = gram / ne
        rhs = rhs / ne
        var np = 0
        if !preserve.isEmpty {
            var pg = MLXArray.zeros([inFeatures, inFeatures])
            for p in preserve {
                let x = p.value[0].asType(.float32)
                pg = pg + matmul(x.transposed(), x)
                np += x.dim(0)
                eval(pg)
            }
            gram = gram + (preserveWeight / Float(max(np, 1))) * pg
        }
        // Per-channel ridge γ·diag(A) (= an isotropic ridge on standardized channels): LLM text
        // encoders have a few massive-activation channels, and a scalar ridge either lets the fit
        // exploit them (small ridge) or swamps every other channel (large ridge).
        let diagonal = gram.diagonal()
        let floor = diagonal.asArray(Float.self).sorted()[inFeatures / 2] * 1e-6
        gram = gram + diag(ridge * diagonal + floor)
        eval(gram, rhs)      // Δᵀ = A⁻¹ · (1/Nₑ) Eᵀ M,  M = (E* − E) Wᵀ   → (in, out)
        // CPU-stream results are materialized before any GPU op reads them: a GPU command buffer
        // waiting seconds on a CPU solve trips the Metal watchdog.
        let deltaT = MLXLinalg.solve(gram, rhs, stream: .cpu)
        eval(deltaT)
        let delta = deltaT.transposed()                         // (out, in)
        eval(delta)

        let (factor, captured, truncation) = LowRank.factor(delta, metric: gram, maxRank: maxRank, energy: energy, schedule: schedule)
        let wNorm = weight.square().sum().sqrt().item(Float.self)
        let dNorm = delta.square().sum().sqrt().item(Float.self)
        return ClosedFormEdit(key: key, delta: factor, truncationError: truncation, energy: captured,
                              erasePairs: rows, preserveVectors: np, relativeSize: Double(dNorm / max(wNorm, 1e-12)))
    }
}

public enum LowRank {
    /// Reduced-rank factorization of Δ in the metric A (symmetric positive definite): minimizes
    /// ‖(Δ − Δ_r)·L‖_F with A = L·Lᵀ, i.e. the error Δ_r makes on inputs distributed like A.
    public static func factor(_ delta: MLXArray, metric: MLXArray, maxRank: Int, energy: Double,
                              schedule: SeedSchedule) -> (LowRankDelta, energy: Double, truncation: Double) {
        let l = MLXLinalg.cholesky(metric, upper: false, stream: .cpu)       // A = L·Lᵀ
        eval(l)
        let m = matmul(delta, l)
        eval(m)
        let (f, captured, truncation) = factor(m, maxRank: maxRank, energy: energy, schedule: schedule)
        // Δ_r = up · downₘ · L⁻¹  →  down = (L⁻ᵀ · downₘᵀ)ᵀ
        let solved = MLXLinalg.solveTriangular(l.transposed(), f.down.transposed(), upper: true, stream: .cpu)
        eval(solved)
        let down = solved.transposed()
        eval(down)
        return (LowRankDelta(up: f.up, down: down), captured, truncation)
    }

    /// Randomized SVD (Halko et al.): Δ ≈ U_r·diag(s_r)·V_rᵀ with r the smallest rank capturing
    /// `energy` of ‖Δ‖²_F (capped at `maxRank`). Returns (up = U_r·diag(s_r), down = V_rᵀ).
    public static func factor(_ delta: MLXArray, maxRank: Int, energy: Double, schedule: SeedSchedule,
                              oversample: Int = 16, powerIterations: Int = 2) -> (LowRankDelta, energy: Double, truncation: Double) {
        let out = delta.dim(0), inF = delta.dim(1)
        let k = min(maxRank + oversample, min(out, inF))
        let total = Double(delta.square().sum().item(Float.self))
        guard total > 0 else { return (LowRankDelta.zero(out: out, in: inF, rank: 1), 1, 0) }
        let omega = schedule.normal(.train, "lowrank|\(out)x\(inF)", shape: [inF, k])
        var y = matmul(delta, omega)                           // (out, k)
        eval(y)
        for _ in 0..<powerIterations {
            y = MLXLinalg.qr(y, stream: .cpu).0
            eval(y)
            y = matmul(delta, matmul(delta.transposed(), y))
            eval(y)
        }
        let q = MLXLinalg.qr(y, stream: .cpu).0                 // (out, k)
        eval(q)
        let b = matmul(q.transposed(), delta)                   // (k, in)
        // SVD of the small k×k Gram b·bᵀ = U·diag(s²)·Uᵀ avoids a full (in × in) Vᵀ.
        let (ub, s2, _) = MLXLinalg.svd(matmul(b, b.transposed()), stream: .cpu)
        eval(ub, s2)
        let sv = s2.asArray(Float.self).map { Double(max($0, 0)).squareRoot() }
        var cumulative = 0.0, r = 0
        while r < min(maxRank, sv.count) {
            cumulative += sv[r] * sv[r]
            r += 1
            if cumulative / total >= energy { break }
        }
        while r > 1 && sv[r - 1] < 1e-9 { r -= 1 }
        r = max(r, 1)
        let ur = ub[0..., ..<r]                                  // (k, r)
        let s = MLXArray(sv.prefix(r).map { Float($0) })
        let up = matmul(q, ur) * s.expandedDimensions(axis: 0)   // (out, r)
        let down = matmul(ur.transposed(), b) / maximum(s, 1e-12).expandedDimensions(axis: 1)   // (r, in)
        eval(up, down)
        let residual = Double((delta - matmul(up, down)).square().sum().item(Float.self))
        return (LowRankDelta(up: up, down: down), min(cumulative / total, 1), (residual / total).squareRoot())
    }
}
