//
//  Optimization.swift
//  VeilKit
//
//  Adam over a flat list of arrays (guard factors, soft-prompt perturbations), and the
//  weighted denoising loss every optimizer in Veil minimizes.
//

import Foundation
import MLX

struct AdamState {
    var m: [MLXArray]
    var v: [MLXArray]
    var t = 0
    let learningRate: Float
    let beta1: Float = 0.9
    let beta2: Float = 0.999
    let eps: Float = 1e-8
    let weightDecay: Float

    init(like params: [MLXArray], learningRate: Float, weightDecay: Float = 0) {
        m = params.map { MLXArray.zeros(like: $0) }
        v = params.map { MLXArray.zeros(like: $0) }
        self.learningRate = learningRate
        self.weightDecay = weightDecay
    }

    /// One AdamW step; returns the updated parameters (evaluated).
    mutating func step(_ params: [MLXArray], _ grads: [MLXArray]) -> [MLXArray] {
        t += 1
        let c1 = 1 - pow(beta1, Float(t)), c2 = 1 - pow(beta2, Float(t))
        var out: [MLXArray] = []
        for i in params.indices {
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
            v[i] = beta2 * v[i] + (1 - beta2) * grads[i].square()
            let update = (m[i] / c1) / ((v[i] / c2).sqrt() + eps)
            out.append(params[i] * (1 - learningRate * weightDecay) - learningRate * update)
        }
        eval(out + m + v)
        return out
    }
}

enum Losses {
    /// Mean over rows of the likeness-weighted squared error: v, target (B, C, H, W), w (B, 1, H, W).
    static func weighted(_ v: MLXArray, _ target: MLXArray, _ w: MLXArray) -> MLXArray {
        let perPosition = (v.asType(.float32) - target).square().mean(axis: 1, keepDims: true)
        return ((perPosition * w).sum(axes: [1, 2, 3]) / w.sum(axes: [1, 2, 3])).mean()
    }

    /// Global-norm clipping; returns the clipped gradients and the norm before clipping.
    static func clip(_ grads: [MLXArray], maxNorm: Float) -> ([MLXArray], Float) {
        let norm = grads.reduce(MLXArray(Float(0))) { $0 + $1.square().sum() }.sqrt()
        let n = norm.item(Float.self)
        guard n.isFinite, n > maxNorm else { return (grads, n) }
        return (grads.map { $0 * (maxNorm / n) }, n)
    }
}
