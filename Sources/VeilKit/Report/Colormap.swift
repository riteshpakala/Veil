//
//  Colormap.swift
//  VeilKit
//  Adapted from Scorpion (github.com/riteshpakala/Scorpion, ScorpionKit), GPL-3.0.
//
//  One hue, light → dark, on a fixed absolute scale: lightness carries magnitude, so the
//  ramp stays ordered for colour-blind readers and in greyscale print. No rainbow. Values
//  below τ are transparent — "no pull" shows the photo itself, not a colour.
//

import Foundation

public struct Colormap: Sendable {
    public struct Stop: Sendable {
        public let t: Float
        public let r: Float, g: Float, b: Float

        public init(_ t: Float, hex: UInt32) {
            self.t = t
            r = Float((hex >> 16) & 0xff) / 255
            g = Float((hex >> 8) & 0xff) / 255
            b = Float(hex & 0xff) / 255
        }
    }

    public let stops: [Stop]
    /// Opacity at τ and at the top of the scale.
    public let alphaRange: (low: Float, high: Float)

    /// Orange, light (weak pull) → dark (strong pull).
    public static let pull = Colormap(stops: [
        Stop(0, hex: 0xFDE3D0), Stop(0.25, hex: 0xF6AE86), Stop(0.5, hex: 0xEB6834),
        Stop(0.75, hex: 0xB9461A), Stop(1, hex: 0x6B240A),
    ], alphaRange: (0.55, 0.92))

    /// Colour at normalized position t ∈ [0, 1].
    public func color(_ t: Float) -> (r: Float, g: Float, b: Float) {
        let x = min(max(t, 0), 1)
        guard let upper = stops.firstIndex(where: { $0.t >= x }), upper > 0 else {
            let s = stops[0]
            return (s.r, s.g, s.b)
        }
        let a = stops[upper - 1], b = stops[upper]
        let f = (x - a.t) / max(b.t - a.t, 1e-6)
        return (a.r + (b.r - a.r) * f, a.g + (b.g - a.g) * f, a.b + (b.b - a.b) * f)
    }

    /// RGBA for a value on the absolute scale [threshold, scaleMax]; nil (transparent) below the threshold.
    public func rgba(_ value: Float, threshold: Float, scaleMax: Float) -> (r: Float, g: Float, b: Float, a: Float)? {
        guard value >= threshold, value.isFinite else { return nil }
        let t = (value - threshold) / max(scaleMax - threshold, 1e-6)
        let c = color(t)
        return (c.r, c.g, c.b, alphaRange.low + (alphaRange.high - alphaRange.low) * min(max(t, 0), 1))
    }
}
