//
//  Profile.swift
//  VeilKit
//
//  Every knob of a run, fixed before it starts and written into the report.
//

import Foundation

public struct VeilProfile: Codable, Sendable, Hashable {
    public var name: String

    // Assessment
    /// Name templates per name (and per null name).
    public var templatesPerName: Int
    /// Invented names forming the null. 19 gives p = 0.05 at best.
    public var nullCount: Int
    /// Sampler σ levels measured (evenly spaced over the deployed schedule).
    public var sigmaCount: Int
    public var noisePerSigma: Int
    public var maxHeldOut: Int
    public var maxControls: Int
    /// Significance level for "reaches".
    public var alpha: Double

    // Guard, stage A (closed form on the text-input slot)
    public var closedFormTemplates: Int
    public var closedFormMaxRank: Int
    public var closedFormEnergy: Double
    public var closedFormPreserve: Float
    public var closedFormRidge: Float

    // Guard, stage B (trained on the text path)
    public var trainSteps: Int
    public var trainRank: Int
    public var learningRate: Float
    public var eraseBatch: Int
    public var preserveBatch: Int
    public var erasePool: Int
    public var preservePool: Int
    public var preserveWeight: Float
    public var trainSigmaCount: Int
    /// Every this many steps, harden against a soft attack (0 = off). Experimental and off by
    /// default: on the toy world, hardening against a ±50% ball of conditionings overwhelmed a
    /// small LoRA and left a description route unsuppressed.
    public var hardenEvery: Int
    public var hardenSteps: Int

    // Attacks and search
    public var attackBudgets: [Int]
    public var attackLearningRate: Float
    /// Perturbation bound, relative to the conditioning's RMS.
    public var attackRadius: Float
    public var searchSteps: Int
    public var searchTokens: Int

    // Verdict tolerances (uncalibrated until the validation experiment runs)
    /// Largest relative loss drift allowed on controls and generic prompts.
    public var preservationTolerance: Double
    /// Share of a named control's own-name pull that must survive.
    public var retainedNamePull: Double
    /// Quick check only: the share of a name's own pull that may remain after the guard. Not a
    /// verdict — the quick check has no invented-name null to calibrate against.
    public var residualTolerance: Double

    public static let quick = VeilProfile(
        name: "quick", templatesPerName: 3, nullCount: 9, sigmaCount: 3, noisePerSigma: 1, maxHeldOut: 3, maxControls: 6,
        alpha: 0.1, closedFormTemplates: 12, closedFormMaxRank: 32, closedFormEnergy: 0.99, closedFormPreserve: 1,
        closedFormRidge: 0.1, trainSteps: 0, trainRank: 8, learningRate: 3e-4, eraseBatch: 2, preserveBatch: 2,
        erasePool: 32, preservePool: 32, preserveWeight: 1, trainSigmaCount: 4, hardenEvery: 0, hardenSteps: 0,
        attackBudgets: [0, 25, 50], attackLearningRate: 0.05, attackRadius: 0.25, searchSteps: 40, searchTokens: 4,
        preservationTolerance: 0.02, retainedNamePull: 0.8, residualTolerance: 0.25)

    public static let standard = VeilProfile(
        name: "standard", templatesPerName: 6, nullCount: 19, sigmaCount: 4, noisePerSigma: 1, maxHeldOut: 4, maxControls: 8,
        alpha: 0.05, closedFormTemplates: 24, closedFormMaxRank: 64, closedFormEnergy: 0.99, closedFormPreserve: 1,
        closedFormRidge: 0.1, trainSteps: 300, trainRank: 16, learningRate: 3e-4, eraseBatch: 2, preserveBatch: 2,
        erasePool: 128, preservePool: 128, preserveWeight: 1, trainSigmaCount: 6, hardenEvery: 0, hardenSteps: 0,
        attackBudgets: [0, 25, 50, 100], attackLearningRate: 0.05, attackRadius: 0.25, searchSteps: 100, searchTokens: 6,
        preservationTolerance: 0.02, retainedNamePull: 0.8, residualTolerance: 0.25)

    public static let thorough = VeilProfile(
        name: "thorough", templatesPerName: 12, nullCount: 19, sigmaCount: 6, noisePerSigma: 2, maxHeldOut: 6, maxControls: 12,
        alpha: 0.05, closedFormTemplates: 24, closedFormMaxRank: 64, closedFormEnergy: 0.995, closedFormPreserve: 1,
        closedFormRidge: 0.1, trainSteps: 800, trainRank: 16, learningRate: 3e-4, eraseBatch: 2, preserveBatch: 2,
        erasePool: 256, preservePool: 256, preserveWeight: 1, trainSigmaCount: 8, hardenEvery: 0, hardenSteps: 0,
        attackBudgets: [0, 25, 50, 100, 200], attackLearningRate: 0.05, attackRadius: 0.25, searchSteps: 200, searchTokens: 8,
        preservationTolerance: 0.02, retainedNamePull: 0.8, residualTolerance: 0.25)

    /// The analytic toy world: exact and cheap, so it runs everything at full strength.
    public static let toy = VeilProfile(
        name: "toy", templatesPerName: 6, nullCount: 19, sigmaCount: 4, noisePerSigma: 4, maxHeldOut: 2, maxControls: 12,
        alpha: 0.05, closedFormTemplates: 24, closedFormMaxRank: 16, closedFormEnergy: 0.999, closedFormPreserve: 1,
        closedFormRidge: 1e-3, trainSteps: 300, trainRank: 8, learningRate: 5e-4, eraseBatch: 8, preserveBatch: 8,
        erasePool: 256, preservePool: 256, preserveWeight: 1, trainSigmaCount: 6, hardenEvery: 0, hardenSteps: 0,
        attackBudgets: [0, 25, 50, 100, 200], attackLearningRate: 0.05, attackRadius: 0.5, searchSteps: 60, searchTokens: 3,
        preservationTolerance: 0.02, retainedNamePull: 0.8, residualTolerance: 0.25)

    /// The same profile with soft-attack hardening switched on (experimental).
    public func hardened(every: Int = 50, steps: Int = 10) -> VeilProfile {
        var p = self
        p.hardenEvery = every
        p.hardenSteps = steps
        return p
    }

    public static func named(_ name: String) -> VeilProfile? {
        switch name.lowercased() {
        case "quick": return .quick
        case "standard": return .standard
        case "thorough": return .thorough
        case "toy": return .toy
        default: return nil
        }
    }
}
